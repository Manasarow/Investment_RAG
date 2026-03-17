"""
BGE-M3 embedder for the RAG investment pipeline.
Produces dense + sparse vectors for every chunk.
"""

import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
MODEL_MAX_LENGTH = 8192


def _select_device() -> str:
    """Select the best available device, with env override support."""
    override = os.getenv("EMBED_DEVICE", "").strip().lower()
    if override:
        log.info("EMBED_DEVICE override: '%s'", override)
        return override

    try:
        import torch

        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            vram = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
            log.info("GPU detected: %s VRAM=%.1f GB -> using %s", name, vram, device)
            return device

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("Apple MPS detected -> using mps")
            return "mps"

    except ImportError:
        log.warning("torch not importable -> falling back to cpu")

    log.info("No GPU found -> using cpu")
    return "cpu"


def _use_fp16_for_device(device: str) -> bool:
    """Enable FP16 on CUDA by default, unless overridden by env."""
    env = os.getenv("EMBED_FP16", "").strip()
    if env in ("0", "1"):
        return env == "1"
    return device.startswith("cuda")


def _default_batch_size(device: str) -> int:
    """Choose a default batch size based on device type."""
    env = os.getenv("EMBED_BATCH_SIZE", "").strip()
    if env:
        return int(env)
    return 64 if device.startswith("cuda") or device == "mps" else 32


class Embedder:
    """BGE-M3 wrapper for chunk and query embedding."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = _select_device()
        self.use_fp16 = _use_fp16_for_device(self.device)
        self.batch_size = _default_batch_size(self.device)

        log.info(
            "Loading BGE-M3 model: %s device=%s fp16=%s default_batch=%s",
            model_name,
            self.device,
            self.use_fp16,
            self.batch_size,
        )

        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
        except ImportError as exc:
            raise ImportError("FlagEmbedding not installed. Run: pip install FlagEmbedding") from exc

        log.info("BGE-M3 loaded.")

        self._torch = None
        if self.device.startswith("cuda"):
            try:
                import torch
                self._torch = torch
            except ImportError:
                pass

    @staticmethod
    def _embed_text(chunk: dict) -> str:
        """Build the text sent to the embedder, prepending key metadata when useful."""
        chunk_type = chunk.get("chunk_type", "prose")
        ticker = chunk.get("ticker", "")
        fiscal_year = chunk.get("fiscal_year", "")
        form_type = chunk.get("form_type", "")
        statement_type = chunk.get("statement_type", "") or ""
        text = chunk.get("text", "")

        if chunk_type == "row":
            prefix = f"{ticker} {form_type} FY{fiscal_year}"
            if statement_type and statement_type != "other_table":
                prefix += f" {statement_type}"
            return f"{prefix} | {text}" if prefix.strip() else text

        if chunk_type in {"table", "micro_block"}:
            return text

        prefix = f"{ticker} {form_type} FY{fiscal_year}"
        return f"{prefix} | {text}" if prefix.strip() else text

    def _maybe_free_vram(self) -> None:
        """Release cached CUDA memory between batches when running on GPU."""
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    def embed_chunks(
        self,
        chunks: list[dict],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[dict]:
        """Embed chunk payloads and return dense+sparse vectors in input order."""
        if not chunks:
            return []

        bs = batch_size if batch_size is not None else self.batch_size
        texts = [self._embed_text(chunk) for chunk in chunks]
        results: list[dict] = []

        try:
            from tqdm import tqdm

            batch_iter = tqdm(
                range(0, len(texts), bs),
                desc=f"Embedding [{self.device}]",
                unit="batch",
                disable=not show_progress,
            )
        except ImportError:
            batch_iter = range(0, len(texts), bs)

        for start in batch_iter:
            batch_texts = texts[start:start + bs]
            batch_chunks = chunks[start:start + bs]

            try:
                output = self._model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    max_length=MODEL_MAX_LENGTH,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
            except Exception as exc:
                log.error("Embedding failed for batch starting at %d: %s", start, exc)
                for chunk in batch_chunks:
                    results.append(
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "dense": [0.0] * 1024,
                            "sparse": {"indices": [], "values": []},
                            "_embed_error": str(exc),
                        }
                    )
                self._maybe_free_vram()
                continue

            dense_vecs = output["dense_vecs"]
            sparse_dicts = output["lexical_weights"]

            for i, chunk in enumerate(batch_chunks):
                dense = dense_vecs[i]
                if isinstance(dense, np.ndarray):
                    dense = dense.tolist()

                sparse_weights = sparse_dicts[i] if i < len(sparse_dicts) else {}
                sparse_indices: list[int] = []
                sparse_values: list[float] = []

                for token_id, weight in sparse_weights.items():
                    try:
                        sparse_indices.append(int(token_id))
                        sparse_values.append(float(weight))
                    except (ValueError, TypeError):
                        continue

                results.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "dense": dense,
                        "sparse": {"indices": sparse_indices, "values": sparse_values},
                    }
                )

            self._maybe_free_vram()

        return results

    def embed_query(self, query: str) -> dict:
        """Embed a single query for retrieval."""
        output = self._model.encode(
            [query],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = output["dense_vecs"][0]
        if isinstance(dense, np.ndarray):
            dense = dense.tolist()

        sparse_weights = output["lexical_weights"][0] if output["lexical_weights"] else {}
        sparse_indices = [int(token_id) for token_id in sparse_weights.keys()]
        sparse_values = [float(weight) for weight in sparse_weights.values()]

        return {
            "dense": dense,
            "sparse": {"indices": sparse_indices, "values": sparse_values},
        }


_EMBEDDER: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Return the module-level embedder singleton."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER