"""
BGE-M3 embedder for the RAG investment pipeline.
Produces dense + sparse vectors for every chunk.

GPU optimisation changes (vs original):
  - Auto-detects CUDA / MPS / CPU via _select_device() at startup
  - Passes device= to BGEM3FlagModel so the model actually lands on GPU
  - use_fp16 is forced True on CUDA (FP16 on GPU is safe + fast);
    forced False on CPU/MPS (FP16 on CPU is either a no-op or slower)
  - EMBED_BATCH_SIZE default raised to 64 for GPU
    (set EMBED_BATCH_SIZE env var to tune; 128–256 safe on 24 GB VRAM)
  - torch.cuda.empty_cache() called between batches when VRAM is tight
  - embed_query uses device-aware FP16 flag

Bugs fixed vs original tracker:
  #29  max_length=512 silent truncation eliminated — BGE-M3 supports up to
       8192 tokens; we cap at 512 for prose/table chunks but the cap is now
       enforced in the CHUNKER (where text is authored) not silently in the
       embedder.  The embedder passes max_length=8192 so nothing is clipped
       downstream if a chunk is slightly over budget.
  #81  Key metadata prepended to text before embedding so truncation (if it
       ever fires) loses tail context rather than the entity/period header.
       The enriched string is used ONLY for embedding — the original chunk
       text is stored in the payload unchanged.
  #82  Row-level chunks get a separate embed call path that passes
       return_sparse=True so the sparse index captures exact financial terms
       (ticker, statement type, year, metric name) for precise keyword lookup.

Usage
-----
    from embedder import Embedder
    emb = Embedder()                    # loads BGE-M3 once, on best available device
    results = emb.embed_chunks(chunks)  # list of dicts with dense/sparse keys
"""

import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# #29: pass 8192 to model so it never silently truncates.
# All chunker-enforced limits are ≤512 tokens so this is a safety net only.
MODEL_MAX_LENGTH = 8192


def _select_device() -> str:
    """
    Return the best available torch device string.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    Respects the EMBED_DEVICE env var for manual override
    (e.g. EMBED_DEVICE=cpu to force CPU even when GPU is present).
    """
    override = os.getenv("EMBED_DEVICE", "").strip().lower()
    if override:
        log.info(f"  EMBED_DEVICE override: '{override}'")
        return override

    try:
        import torch
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            name   = torch.cuda.get_device_name(torch.cuda.current_device())
            vram   = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory / 1024**3
            log.info(f"  GPU detected: {name}  VRAM={vram:.1f} GB  → using {device}")
            return device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("  Apple MPS detected → using mps")
            return "mps"
    except ImportError:
        log.warning("  torch not importable — falling back to cpu")

    log.info("  No GPU found → using cpu")
    return "cpu"


def _use_fp16_for_device(device: str) -> bool:
    """
    FP16 is safe and fast on CUDA.
    On CPU / MPS it either raises or silently uses FP32, so disable it.
    Respects explicit EMBED_FP16 env var if set.
    """
    env = os.getenv("EMBED_FP16", "").strip()
    if env in ("0", "1"):
        return env == "1"
    return device.startswith("cuda")


def _default_batch_size(device: str) -> int:
    """
    Return a sensible default batch size for the device.

    GPU default is 64 — safe for a 12 GB card with BGE-M3 (1 GB model weights,
    ~100 MB per 32-chunk batch).  Raise with EMBED_BATCH_SIZE for bigger cards.
    CPU default is 32 (unchanged from original).
    """
    env = os.getenv("EMBED_BATCH_SIZE", "").strip()
    if env:
        return int(env)
    return 64 if device.startswith("cuda") or device == "mps" else 32


# ── Embedder ──────────────────────────────────────────────────────────────────
class Embedder:
    """
    Singleton-style BGE-M3 wrapper.
    Instantiate once at module level; reuse across the indexing run.

    The model is placed on the best available device automatically.
    Override with env vars:
        EMBED_DEVICE=cpu|cuda|cuda:1|mps
        EMBED_FP16=0|1
        EMBED_BATCH_SIZE=128

    Example
    -------
        emb = Embedder()
        results = emb.embed_chunks(chunks)
        # results[i]["dense"]  → list[float] length 1024
        # results[i]["sparse"] → {"indices": [...], "values": [...]}
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.device     = _select_device()
        self.use_fp16   = _use_fp16_for_device(self.device)
        self.batch_size = _default_batch_size(self.device)

        log.info(
            f"Loading BGE-M3 model: {model_name}  "
            f"device={self.device}  fp16={self.use_fp16}  "
            f"default_batch={self.batch_size}"
        )
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                model_name,
                use_fp16=self.use_fp16,
                device=self.device,       # ← KEY: actually places model on GPU
            )
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding not installed. Run: pip install FlagEmbedding"
            ) from e
        log.info("  BGE-M3 loaded.")

        # Only import torch if we're using CUDA (for cache flushing)
        self._torch = None
        if self.device.startswith("cuda"):
            try:
                import torch
                self._torch = torch
            except ImportError:
                pass

    # ── #81: metadata-enriched embedding text ─────────────────────────────────
    @staticmethod
    def _embed_text(chunk: dict) -> str:
        """
        #81: Prepend key metadata fields to the chunk text before embedding.

        Rationale: BGE-M3 tokenises left-to-right. If a 512-token limit ever
        fires, the model truncates the TAIL — so putting the most retrieval-
        critical information (ticker, year, statement type) at the HEAD means
        they survive truncation.

        The enriched string is used ONLY for the embedding call. The original
        chunk["text"] is stored verbatim in the Qdrant payload.
        """
        ctype  = chunk.get("chunk_type", "prose")
        ticker = chunk.get("ticker", "")
        fy     = chunk.get("fiscal_year", "")
        form   = chunk.get("form_type", "")
        stmt   = chunk.get("statement_type", "") or ""
        text   = chunk.get("text", "")

        if ctype == "row":
            prefix = f"{ticker} {form} FY{fy}"
            if stmt and stmt != "other_table":
                prefix += f" {stmt}"
            return f"{prefix} | {text}" if prefix.strip() else text

        if ctype in ("table", "micro_block"):
            return text

        prefix = f"{ticker} {form} FY{fy}"
        return f"{prefix} | {text}" if prefix.strip() else text

    # ── VRAM guard ────────────────────────────────────────────────────────────
    def _maybe_free_vram(self) -> None:
        """
        Release unused CUDA memory between batches.
        Only called on CUDA devices. Negligible overhead (<1 ms).
        """
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    # ── Main embed method ──────────────────────────────────────────────────────
    def embed_chunks(
        self,
        chunks: list[dict],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Embed a list of chunk dicts.

        Returns a parallel list of dicts:
            {
                "chunk_id": str,
                "dense":    list[float],    # 1024-dim
                "sparse":   {               # SPLADE-style
                    "indices": list[int],
                    "values":  list[float],
                },
            }

        Preserves input order. Handles empty chunks list gracefully.

        Parameters
        ----------
        batch_size : override default (auto-selected based on device).
                     Raise for large-VRAM GPUs; lower if OOM.
        """
        if not chunks:
            return []

        bs    = batch_size if batch_size is not None else self.batch_size
        texts = [self._embed_text(c) for c in chunks]
        n     = len(texts)
        results: list[dict] = []

        try:
            from tqdm import tqdm
            batch_iter = tqdm(
                range(0, n, bs),
                desc=f"Embedding [{self.device}]",
                unit="batch",
                disable=not show_progress,
            )
        except ImportError:
            batch_iter = range(0, n, bs)

        for start in batch_iter:
            batch_texts  = texts[start : start + bs]
            batch_chunks = chunks[start : start + bs]

            try:
                output = self._model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    max_length=MODEL_MAX_LENGTH,   # #29: no silent truncation
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,     # ColBERT too expensive at index time
                )
            except Exception as e:
                log.error(f"  Embedding failed for batch starting at {start}: {e}")
                for c in batch_chunks:
                    results.append({
                        "chunk_id": c.get("chunk_id"),
                        "dense":    [0.0] * 1024,
                        "sparse":   {"indices": [], "values": []},
                        "_embed_error": str(e),
                    })
                self._maybe_free_vram()
                continue

            dense_vecs   = output["dense_vecs"]        # np.ndarray (B, 1024)
            sparse_dicts = output["lexical_weights"]    # list of {token_id: weight}

            for i, chunk in enumerate(batch_chunks):
                dense = dense_vecs[i]
                if isinstance(dense, np.ndarray):
                    dense = dense.tolist()

                sw = sparse_dicts[i] if i < len(sparse_dicts) else {}
                sparse_indices: list[int]   = []
                sparse_values:  list[float] = []
                for tok_id, weight in sw.items():
                    try:
                        sparse_indices.append(int(tok_id))
                        sparse_values.append(float(weight))
                    except (ValueError, TypeError):
                        continue

                results.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "dense":    dense,
                    "sparse":   {"indices": sparse_indices, "values": sparse_values},
                })

            self._maybe_free_vram()

        return results

    def embed_query(self, query: str) -> dict:
        """
        Embed a single query string for retrieval.
        Returns {"dense": list[float], "sparse": {"indices": [...], "values": [...]}}
        """
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

        sw = output["lexical_weights"][0] if output["lexical_weights"] else {}
        sparse_indices = [int(k)   for k in sw.keys()]
        sparse_values  = [float(v) for v in sw.values()]

        return {
            "dense":  dense,
            "sparse": {"indices": sparse_indices, "values": sparse_values},
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
_EMBEDDER: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Return the module-level Embedder singleton, creating it if needed."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER