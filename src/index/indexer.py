"""
Qdrant indexer for the RAG investment pipeline.
Reads chunk JSONL files, embeds them, and upserts into Qdrant.
"""

import hashlib
import json
import logging
import logging.handlers
import os
import struct
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, PointStruct, SparseVector

from embedder import get_embedder
from qdrant_setup import COLLECTION_NAME, filing_date_to_ts, setup_collection

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
MANIFEST = BASE_DIR / "data_manifest" / "manifest.jsonl"

UPSERT_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "128"))
LOG_GPU_EVERY = int(os.getenv("LOG_GPU_EVERY", "5"))
LOG_PROGRESS_EVERY = int(os.getenv("LOG_PROGRESS_EVERY", "10"))


def _setup_logging() -> None:
    """Configure console and rotating file logging once."""
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"indexer_{timestamp}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging to %s", log_file)


def _assert_gpu() -> dict:
    """Verify CUDA is available and return active GPU details."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is not installed. Run: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Possible causes:\n"
            "  1. PyTorch CPU-only build — reinstall with CUDA.\n"
            "  2. NVIDIA drivers not installed or outdated.\n"
            "  3. Running inside a VM without GPU passthrough.\n"
            "Set EMBED_DEVICE=cpu to skip this check."
        )

    device_idx = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_idx)
    props = torch.cuda.get_device_properties(device_idx)
    vram_gb = props.total_memory / 1024**3

    log.info("=" * 60)
    log.info("GPU VERIFICATION")
    log.info("  Device   : cuda:%s", device_idx)
    log.info("  Name     : %s", device_name)
    log.info("  VRAM     : %.1f GB", vram_gb)
    log.info("  Compute  : %s.%s", props.major, props.minor)
    log.info("  CUDA_VISIBLE_DEVICES : %s", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    log.info("  CUDA_DEVICE_ORDER    : %s", os.environ.get("CUDA_DEVICE_ORDER", "unset"))
    log.info("=" * 60)

    if "intel" in device_name.lower() or "uhd" in device_name.lower():
        raise RuntimeError(
            f"cuda:0 resolved to '{device_name}', not the NVIDIA card. "
            "Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=<nvidia index>."
        )

    return {
        "device": f"cuda:{device_idx}",
        "name": device_name,
        "vram_gb": vram_gb,
        "torch": torch,
    }


def _log_gpu_stats(torch_module, prefix: str = "") -> None:
    """Log current CUDA memory usage when torch is available."""
    if torch_module is None:
        return

    try:
        allocated = torch_module.cuda.memory_allocated() / 1024**3
        reserved = torch_module.cuda.memory_reserved() / 1024**3
        log.debug("%sGPU VRAM: allocated=%.2f GB reserved=%.2f GB", prefix, allocated, reserved)
    except Exception:
        pass


def chunk_id_to_point_id(chunk_id: str) -> int:
    """Convert a chunk_id into a deterministic uint64 Qdrant point ID."""
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return struct.unpack(">Q", digest[:8])[0]


def _safe_int(value):
    """Convert common numeric payload values to int when possible."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_fiscal_quarter(value) -> str | None:
    """Normalize fiscal quarter into keyword-friendly string form."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    upper = text.upper()
    if upper in {"Q1", "Q2", "Q3", "Q4"}:
        return upper[-1]
    if text in {"1", "2", "3", "4"}:
        return text

    return text


def _source_class(chunk: dict) -> str:
    """Derive a normalized source class from filing metadata and content."""
    form_type = (chunk.get("form_type") or "").upper()
    period_type = (chunk.get("period_type") or "").lower()
    section = (chunk.get("section") or "").lower()
    text = (chunk.get("text") or "")[:500].lower()

    if "10-K" in form_type:
        return "10-K"
    if "10-Q" in form_type:
        return "10-Q"
    if "8-K" in form_type:
        earnings_signals = (
            period_type == "event"
            and any(signal in section for signal in ("2.02", "7.01", "results of operations"))
        ) or any(
            signal in text
            for signal in (
                "earnings release",
                "financial results",
                "quarterly results",
                "annual results",
            )
        )
        return "8-K-earnings" if earnings_signals else "8-K"
    if "DEF" in form_type or "PROXY" in form_type:
        return "proxy"
    return form_type or "unknown"


def build_payload(chunk: dict) -> dict:
    """Build the Qdrant payload for one chunk."""
    filing_date = chunk.get("filing_date", "")

    payload = {
        "chunk_id": chunk.get("chunk_id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "ticker": chunk.get("ticker", ""),
        "company": chunk.get("company", ""),
        "form_type": chunk.get("form_type", ""),
        "filing_date": filing_date,
        "filing_date_ts": filing_date_to_ts(filing_date),
        "fiscal_year": _safe_int(chunk.get("fiscal_year")),
        "fiscal_quarter": _normalize_fiscal_quarter(chunk.get("fiscal_quarter")),
        "period_type": chunk.get("period_type", ""),
        "period_end_date": chunk.get("period_end_date", ""),
        "report_priority": _safe_int(chunk.get("report_priority")),
        "sector": chunk.get("sector", ""),
        "industry": chunk.get("industry", ""),
        "source_url": chunk.get("source_url", ""),
        "source_class": _source_class(chunk),
        "chunk_type": chunk.get("chunk_type", ""),
        "chunk_index": _safe_int(chunk.get("chunk_index")),
        "page": _safe_int(chunk.get("page")),
        "page_is_surrogate": bool(chunk.get("page_is_surrogate", False)),
        "block_order": _safe_int(chunk.get("block_order")),
        "citation_key": chunk.get("citation_key", ""),
        "section": chunk.get("section", ""),
        "parent_chunk_id": chunk.get("parent_chunk_id"),
        "next_chunk_id": chunk.get("next_chunk_id"),
        "prev_chunk_id": chunk.get("prev_chunk_id"),
        "text": chunk.get("text", ""),
        "statement_type": chunk.get("statement_type", ""),
        "table_title": chunk.get("table_title", ""),
        "period_signals": chunk.get("period_signals", []),
        "periods": chunk.get("periods", []),
        "units": chunk.get("units", []),
        "is_financial_statement": bool(chunk.get("is_financial_statement", False)),
        "is_cover_page": bool(chunk.get("is_cover_page", False)),
        "row_label": chunk.get("row_label", ""),
        "col_header": chunk.get("col_header", ""),
        "value": chunk.get("value", ""),
        "table_json": json.dumps(chunk["table_json"]) if chunk.get("table_json") is not None else None,
    }

    return {key: value for key, value in payload.items() if value is not None}


def load_manifest() -> list[dict]:
    """Load manifest entries from JSONL."""
    if not MANIFEST.exists():
        return []

    entries: list[dict] = []
    with open(MANIFEST, encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Manifest parse error at line %d: %s", line_no, exc)

    return entries


def _write_manifest(entries: list[dict]) -> None:
    """Write the manifest atomically after indexing completes."""
    tmp = MANIFEST.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as file:
            for entry in entries:
                file.write(json.dumps(entry) + "\n")
        tmp.replace(MANIFEST)
    except Exception as exc:
        log.error("Manifest write failed: %s", exc)
        tmp.unlink(missing_ok=True)
        raise


def delete_doc_points(
    client: QdrantClient,
    collection_name: str,
    doc_ids: list[str],
) -> None:
    """Delete existing points for the given doc_ids before reindexing."""
    if not doc_ids:
        return

    log.info("Deleting existing points for %d doc(s)...", len(doc_ids))
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=doc_ids),
                )
            ]
        ),
    )


def index_chunks(
    chunks: list[dict],
    client: QdrantClient,
    collection_name: str,
    embedder,
    batch_size: int = UPSERT_BATCH_SIZE,
    torch_module=None,
) -> int:
    """Embed chunks and upsert them into Qdrant in batches."""
    if not chunks:
        return 0

    total_upserted = 0
    batch_count = 0
    started_at = time.perf_counter()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        batch_started_at = time.perf_counter()

        log.debug(
            "Embed batch %d: chunks %d-%d of %d",
            batch_count + 1,
            start,
            start + len(batch) - 1,
            len(chunks),
        )

        embeddings = embedder.embed_chunks(batch, show_progress=False)

        embed_ms = (time.perf_counter() - batch_started_at) * 1000
        if embed_ms > 0:
            log.debug("Embed done in %.0f ms (%.0f chunks/s)", embed_ms, len(batch) / embed_ms * 1000)

        if batch_count % LOG_GPU_EVERY == 0:
            _log_gpu_stats(torch_module, prefix=f"[batch {batch_count}] ")

        points: list[PointStruct] = []
        for chunk, emb in zip(batch, embeddings):
            if emb.get("_embed_error"):
                log.warning("Skipping chunk %s due to embed error", chunk.get("chunk_id"))
                continue

            chunk_id = chunk.get("chunk_id", "")
            if not chunk_id:
                log.warning("Skipping chunk with empty chunk_id")
                continue

            dense = emb.get("dense")
            sparse = emb.get("sparse")
            if dense is None or sparse is None:
                log.warning("Skipping chunk %s due to missing dense/sparse vector", chunk_id)
                continue

            points.append(
                PointStruct(
                    id=chunk_id_to_point_id(chunk_id),
                    vector={
                        "dense": dense,
                        "sparse": SparseVector(
                            indices=sparse["indices"],
                            values=sparse["values"],
                        ),
                    },
                    payload=build_payload(chunk),
                )
            )

        if points:
            upsert_started_at = time.perf_counter()
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            upsert_ms = (time.perf_counter() - upsert_started_at) * 1000
            log.debug("Upserted %d points in %.0f ms", len(points), upsert_ms)
            total_upserted += len(points)

        batch_count += 1

    elapsed = time.perf_counter() - started_at
    if elapsed > 0 and total_upserted > 0:
        log.debug(
            "index_chunks complete: %d points in %.1fs (%.0f pts/s)",
            total_upserted,
            elapsed,
            total_upserted / elapsed,
        )

    return total_upserted


def index_document(
    doc_id: str,
    chunk_path: Path,
    client: QdrantClient,
    collection_name: str,
    embedder,
    reindex: bool = False,
    torch_module=None,
) -> int:
    """Load one chunk file, embed its chunks, and upsert them."""
    started_at = time.perf_counter()
    chunks: list[dict] = []

    with open(chunk_path, encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSON in %s line %d: %s", chunk_path.name, line_no, exc)

    if not chunks:
        log.warning("No chunks loaded from %s", chunk_path.name)
        return 0

    log.debug("Loaded %d chunks from %s", len(chunks), chunk_path.name)

    if reindex:
        delete_doc_points(client, collection_name, [doc_id])

    points_upserted = index_chunks(
        chunks,
        client,
        collection_name,
        embedder,
        torch_module=torch_module,
    )
    elapsed = time.perf_counter() - started_at

    rate = points_upserted / elapsed if elapsed > 0 else 0
    log.info(
        "%-55s %5d points %4d chunks %5.1fs (%.0f pts/s)",
        doc_id,
        points_upserted,
        len(chunks),
        elapsed,
        rate,
    )
    return points_upserted


def run(
    tickers: list[str] | None = None,
    reindex: bool = False,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Run the full indexing pipeline and update manifest statuses once."""
    _setup_logging()
    run_started_at = time.perf_counter()

    allow_cpu = os.getenv("EMBED_DEVICE", "").strip().lower() == "cpu"
    torch_module = None

    if not allow_cpu:
        gpu_info = _assert_gpu()
        torch_module = gpu_info["torch"]
    else:
        log.warning("EMBED_DEVICE=cpu set — GPU check skipped.")

    client = setup_collection(
        collection_name=collection_name,
        recreate=False,
        recreate_on_mismatch=False,
    )
    embedder = get_embedder()

    if torch_module is not None:
        log.info("Running GPU warmup...")
        warmup_started_at = time.perf_counter()
        embedder._model.encode(
            ["GPU warmup"] * min(8, embedder.batch_size),
            batch_size=8,
            max_length=16,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        torch_module.cuda.synchronize()
        log.info("GPU warmup complete in %.0f ms", (time.perf_counter() - warmup_started_at) * 1000)
        _log_gpu_stats(torch_module, prefix="Post-warmup ")

    all_entries = load_manifest()
    id_to_idx = {entry.get("doc_id"): i for i, entry in enumerate(all_entries)}

    working = all_entries
    if tickers:
        tickers_upper = {ticker.strip().upper() for ticker in tickers}
        working = [
            entry
            for entry in all_entries
            if entry.get("ticker", "").upper() in tickers_upper
        ]

    to_index = [
        entry
        for entry in working
        if entry.get("chunk_status") == "chunked"
        and (reindex or entry.get("index_status") != "indexed")
    ]

    log.info("=" * 60)
    log.info(
        "Indexing %d documents (reindex=%s, collection=%s)",
        len(to_index),
        reindex,
        collection_name,
    )
    log.info("=" * 60)

    total_points = 0
    success = 0
    failed = 0
    progress_started_at = time.perf_counter()

    for doc_num, entry in enumerate(to_index, 1):
        doc_id = entry["doc_id"]
        chunk_path = CHUNKS_DIR / f"{doc_id}.jsonl"

        if not chunk_path.exists():
            log.warning("[%d/%d] Chunk file missing: %s", doc_num, len(to_index), chunk_path.name)
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["index_status"] = "index_failed"
                all_entries[idx]["indexed_points"] = 0
            failed += 1
            continue

        log.info("[%d/%d] %s", doc_num, len(to_index), doc_id)

        try:
            points_upserted = index_document(
                doc_id=doc_id,
                chunk_path=chunk_path,
                client=client,
                collection_name=collection_name,
                embedder=embedder,
                reindex=reindex,
                torch_module=torch_module,
            )

            idx = id_to_idx.get(doc_id)

            if points_upserted <= 0:
                log.warning("Indexing produced 0 points for %s", doc_id)
                if idx is not None:
                    all_entries[idx]["index_status"] = "index_failed"
                    all_entries[idx]["indexed_points"] = 0
                failed += 1
                continue

            if idx is not None:
                all_entries[idx]["index_status"] = "indexed"
                all_entries[idx]["indexed_points"] = points_upserted

            total_points += points_upserted
            success += 1

        except Exception as exc:
            log.error("Indexing failed for %s: %s", doc_id, exc, exc_info=True)
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["index_status"] = "index_failed"
                all_entries[idx]["indexed_points"] = 0
            failed += 1

        if doc_num % LOG_PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - progress_started_at
            docs_done = success + failed
            rate = docs_done / elapsed if elapsed > 0 else 0
            remaining = len(to_index) - doc_num
            eta_seconds = remaining / rate if rate > 0 else 0

            _log_gpu_stats(torch_module, prefix="[progress] ")
            log.info(
                "Progress: %d/%d docs success=%d failed=%d pts=%d rate=%.1f docs/s ETA=%.1f min",
                doc_num,
                len(to_index),
                success,
                failed,
                total_points,
                rate,
                eta_seconds / 60,
            )

    _write_manifest(all_entries)

    total_elapsed = time.perf_counter() - run_started_at
    log.info("")
    log.info("=" * 60)
    log.info("INDEXING COMPLETE")
    log.info("  Documents : success=%d failed=%d total=%d", success, failed, len(to_index))
    log.info("  Points    : %d", total_points)
    log.info("  Wall time : %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    if total_elapsed > 0 and total_points > 0:
        log.info("  Throughput: %.0f pts/s", total_points / total_elapsed)
    log.info("=" * 60)

    try:
        info = client.get_collection(collection_name)
        log.info(
            "Collection '%s': vectors_count=%s status=%s",
            collection_name,
            info.vectors_count,
            info.status,
        )
    except Exception:
        pass

    if torch_module is not None:
        _log_gpu_stats(torch_module, prefix="Final ")


if __name__ == "__main__":
    import argparse

    _setup_logging()

    parser = argparse.ArgumentParser(description="Index chunked SEC filings into Qdrant.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Subset of tickers to index. Omit to index all chunked documents.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete existing points for affected docs before upserting.",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME}).",
    )
    args = parser.parse_args()

    run(
        tickers=args.tickers,
        reindex=args.reindex,
        collection_name=args.collection,
    )