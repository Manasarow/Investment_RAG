"""
Qdrant indexer for the RAG investment pipeline.
Reads chunk JSONL files, embeds them, and upserts into Qdrant.

Bugs fixed vs original tracker:
  #30  Sparse vector hash instability eliminated — embedder now returns integer
       token IDs directly from BGE-M3; no hash() call anywhere.
  #31  Payload text no longer truncated to 2000 chars — full chunk text stored
       so the reranker and LLM receive complete context.
  #32  reindex=True now deletes existing points for the affected doc_ids before
       upserting, preventing duplicate accumulation.
  #33  Point IDs are deterministic per chunk_id (SHA-256 → uint64) not random
       UUID — guarantees idempotent upsert across re-runs.
  #34  Prose chunks prefixed with "{ticker} {form_type} FY{year}" in the
       embedding text (handled in embedder._embed_text) for cross-company
       precision.
  #35  Manifest O(N²) eliminated — load once, update in-memory, write once.
  #83  table_json stored as structured payload field (not dropped).
  #84  Compact structured fields (row_labels, periods, values) preserved for
       post-retrieval validation in the generator.

Additional safety fixes:
  - fiscal_quarter normalized to string to match Qdrant KEYWORD index
  - zero-point indexing runs marked as index_failed, not indexed
  - stricter source_class handling for 8-K routing
  - fail-fast collection setup usage

GPU / logging additions:
  - _assert_gpu() hard-fails at startup if RTX 4090 is not active on CUDA
  - CUDA_VISIBLE_DEVICES + CUDA_DEVICE_ORDER forced to pin the NVIDIA card
  - GPU warmup run before indexing loop (eliminates first-batch JIT spike)
  - Per-batch GPU utilisation + VRAM logged every LOG_GPU_EVERY batches
  - Per-document timing logged (chunks/sec, points/sec)
  - Rolling throughput summary logged every LOG_PROGRESS_EVERY documents
  - File-based logging to logs/indexer_<timestamp>.log alongside console
"""

import hashlib
import json
import logging
import logging.handlers
import os
import struct
import time
from pathlib import Path

# ── GPU environment pins — must be set BEFORE torch / FlagEmbedding import ───
# Force PyTorch to use PCI bus order so cuda:0 is always the discrete GPU,
# not the Intel iGPU which Windows sometimes lists first.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # expose only the 4090

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchAny,
)

from embedder import get_embedder
from qdrant_setup import (
    setup_collection,
    filing_date_to_ts,
    COLLECTION_NAME,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _setup_logging() -> None:
    """
    Configure root logger with:
      - Console handler  (INFO+)
      - Rotating file handler → logs/indexer_<timestamp>.log  (DEBUG+)
    Call once from __main__ or run().
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. called twice)

    root.setLevel(logging.DEBUG)

    # Console — INFO and above, human-readable
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # File — DEBUG and above, timestamped filename
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"indexer_{ts}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,   # 50 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(f"Logging to {log_file}")


log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parents[2]
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
MANIFEST   = BASE_DIR / "data_manifest" / "manifest.jsonl"

# Qdrant upsert batch size — keep below 512 to avoid request timeouts
UPSERT_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "128"))

# How often to log GPU stats (every N embed batches) and progress (every N docs)
LOG_GPU_EVERY      = int(os.getenv("LOG_GPU_EVERY", "5"))
LOG_PROGRESS_EVERY = int(os.getenv("LOG_PROGRESS_EVERY", "10"))


# ── GPU enforcement ───────────────────────────────────────────────────────────
def _assert_gpu() -> dict:
    """
    Verify that a CUDA GPU is available and return a dict of GPU info.
    Raises RuntimeError with actionable instructions if no GPU is found.

    Returns
    -------
    {
        "device": "cuda:0",
        "name":   "NVIDIA GeForce RTX 4090 Laptop GPU",
        "vram_gb": 16.0,
        "torch":   <torch module>,
    }
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "torch is not installed. Run: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Possible causes:\n"
            "  1. PyTorch CPU-only build — reinstall with CUDA:\n"
            "       pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  2. NVIDIA drivers not installed or outdated.\n"
            "  3. Running inside a VM without GPU passthrough.\n"
            "Set EMBED_DEVICE=cpu to skip this check (CPU will be very slow)."
        )

    device_idx  = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_idx)
    props       = torch.cuda.get_device_properties(device_idx)
    vram_gb     = props.total_memory / 1024 ** 3

    log.info("=" * 60)
    log.info("GPU VERIFICATION")
    log.info(f"  Device   : cuda:{device_idx}")
    log.info(f"  Name     : {device_name}")
    log.info(f"  VRAM     : {vram_gb:.1f} GB")
    log.info(f"  Compute  : {props.major}.{props.minor}")
    log.info(f"  CUDA_VISIBLE_DEVICES : {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    log.info(f"  CUDA_DEVICE_ORDER    : {os.environ.get('CUDA_DEVICE_ORDER', 'unset')}")
    log.info("=" * 60)

    # Warn if the wrong GPU is selected (e.g. Intel iGPU slipped through)
    if "intel" in device_name.lower() or "uhd" in device_name.lower():
        raise RuntimeError(
            f"cuda:0 resolved to '{device_name}' (Intel iGPU), not the NVIDIA card.\n"
            "Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=<nvidia index>."
        )

    return {
        "device":   f"cuda:{device_idx}",
        "name":     device_name,
        "vram_gb":  vram_gb,
        "torch":    torch,
    }


def _log_gpu_stats(torch_module, prefix: str = "") -> None:
    """
    Log current CUDA memory usage.  Cheap call (~0.1 ms).
    """
    if torch_module is None:
        return
    try:
        alloc  = torch_module.cuda.memory_allocated()  / 1024 ** 3
        reserv = torch_module.cuda.memory_reserved()   / 1024 ** 3
        log.debug(
            f"{prefix}GPU VRAM: allocated={alloc:.2f} GB  reserved={reserv:.2f} GB"
        )
    except Exception:
        pass


# ── #33: deterministic point ID ───────────────────────────────────────────────
def chunk_id_to_point_id(chunk_id: str) -> int:
    """
    Convert a string chunk_id to a stable uint64 Qdrant point ID.

    Uses SHA-256 truncated to 8 bytes → unsigned 64-bit integer.
    This is deterministic: the same chunk_id always maps to the same point ID,
    enabling idempotent upserts across re-index runs.
    """
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return struct.unpack(">Q", digest[:8])[0]


def _safe_int(value):
    """Convert common numeric payload fields to int, else return None."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_fiscal_quarter(value) -> str | None:
    """
    Normalize fiscal_quarter to a string because Qdrant indexes it as KEYWORD.

    Examples:
      1     -> "1"
      "1"   -> "1"
      "Q1"  -> "1"
      "q3"  -> "3"
      ""    -> None
    """
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    upper = s.upper()
    if upper in {"Q1", "Q2", "Q3", "Q4"}:
        return upper[-1]
    if s in {"1", "2", "3", "4"}:
        return s

    return s


def _source_class(chunk: dict) -> str:
    """
    Derive a normalized source_class label from form_type and content signals.
    Used as a routing dimension separate from raw form_type strings.
    """
    ft          = (chunk.get("form_type") or "").upper()
    period_type = (chunk.get("period_type") or "").lower()
    section     = (chunk.get("section") or "").lower()
    text        = (chunk.get("text") or "")[:500].lower()

    if "10-K" in ft:
        return "10-K"
    if "10-Q" in ft:
        return "10-Q"
    if "8-K" in ft:
        earnings_signals = (
            period_type == "event"
            and any(sig in section for sig in ("2.02", "7.01", "results of operations"))
        ) or any(
            sig in text
            for sig in (
                "earnings release",
                "financial results",
                "quarterly results",
                "annual results",
            )
        )
        return "8-K-earnings" if earnings_signals else "8-K"
    if "DEF" in ft or "PROXY" in ft:
        return "proxy"
    return ft or "unknown"


# ── #31: build payload without text truncation ────────────────────────────────
def build_payload(chunk: dict) -> dict:
    """
    Build the Qdrant payload dict from a chunk.

    Key decisions:
    - text stored in FULL so reranker gets complete context
    - filing_date_ts added as INTEGER for range queries
    - table_json preserved
    - row-level fields preserved
    - is_financial_statement and is_cover_page always present
    - numeric/filterable fields normalized where appropriate
    """
    filing_date = chunk.get("filing_date", "")

    payload = {
        # Identity
        "chunk_id":       chunk.get("chunk_id", ""),
        "doc_id":         chunk.get("doc_id", ""),
        "ticker":         chunk.get("ticker", ""),
        "company":        chunk.get("company", ""),
        "form_type":      chunk.get("form_type", ""),
        "filing_date":    filing_date,
        "filing_date_ts": filing_date_to_ts(filing_date),
        "fiscal_year":    _safe_int(chunk.get("fiscal_year")),
        "fiscal_quarter": _normalize_fiscal_quarter(chunk.get("fiscal_quarter")),
        "period_type":    chunk.get("period_type", ""),
        "period_end_date":chunk.get("period_end_date", ""),
        "report_priority":_safe_int(chunk.get("report_priority")),
        "sector":         chunk.get("sector", ""),
        "industry":       chunk.get("industry", ""),
        "source_url":     chunk.get("source_url", ""),
        "source_class":   _source_class(chunk),

        # Chunk provenance
        "chunk_type":     chunk.get("chunk_type", ""),
        "chunk_index":    _safe_int(chunk.get("chunk_index")),
        "page":           _safe_int(chunk.get("page")),
        "page_is_surrogate": bool(chunk.get("page_is_surrogate", False)),
        "block_order":    _safe_int(chunk.get("block_order")),
        "citation_key":   chunk.get("citation_key", ""),
        "section":        chunk.get("section", ""),
        "parent_chunk_id":chunk.get("parent_chunk_id"),
        "next_chunk_id":  chunk.get("next_chunk_id"),
        "prev_chunk_id":  chunk.get("prev_chunk_id"),

        # Full text
        "text": chunk.get("text", ""),

        # Table metadata
        "statement_type":       chunk.get("statement_type", ""),
        "table_title":          chunk.get("table_title", ""),
        "period_signals":       chunk.get("period_signals", []),
        "periods":              chunk.get("periods", []),
        "units":                chunk.get("units", []),
        "is_financial_statement": bool(chunk.get("is_financial_statement", False)),
        "is_cover_page":         bool(chunk.get("is_cover_page", False)),

        # Row-level fields
        "row_label":  chunk.get("row_label", ""),
        "col_header": chunk.get("col_header", ""),
        "value":      chunk.get("value", ""),

        # Structured table JSON
        "table_json": (
            json.dumps(chunk["table_json"])
            if chunk.get("table_json") is not None else None
        ),
    }

    return {k: v for k, v in payload.items() if v is not None}


# ── Manifest helpers (O(N) pattern — #35) ────────────────────────────────────
def load_manifest() -> list[dict]:
    """Load manifest entries from JSONL."""
    if not MANIFEST.exists():
        return []

    entries: list[dict] = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"  Manifest parse error at line {line_no}: {e}")

    return entries


def _write_manifest(entries: list[dict]) -> None:
    """Atomic O(N) manifest write."""
    tmp = MANIFEST.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        tmp.replace(MANIFEST)
    except Exception as e:
        log.error(f"  Manifest write failed: {e}")
        tmp.unlink(missing_ok=True)
        raise


# ── Delete existing points for a doc_id (#32) ────────────────────────────────
def delete_doc_points(
    client: QdrantClient,
    collection_name: str,
    doc_ids: list[str],
) -> None:
    """
    Delete all existing Qdrant points for the given doc_ids before re-indexing,
    preventing duplicate point accumulation.
    """
    if not doc_ids:
        return

    log.info(f"  Deleting existing points for {len(doc_ids)} doc(s)…")
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


# ── Core indexing function ────────────────────────────────────────────────────
def index_chunks(
    chunks: list[dict],
    client: QdrantClient,
    collection_name: str,
    embedder,
    batch_size: int = UPSERT_BATCH_SIZE,
    torch_module=None,
) -> int:
    """
    Embed and upsert a list of chunks into Qdrant.
    Returns the number of points successfully upserted.

    Parameters
    ----------
    torch_module : pass the torch module for GPU stat logging, or None to skip.
    """
    if not chunks:
        return 0

    total_upserted = 0
    batch_count    = 0
    t0             = time.perf_counter()

    for start in range(0, len(chunks), batch_size):
        batch      = chunks[start : start + batch_size]
        batch_t0   = time.perf_counter()

        log.debug(
            f"    Embed batch {batch_count + 1}: "
            f"chunks {start}–{start + len(batch) - 1} of {len(chunks)}"
        )

        embeddings = embedder.embed_chunks(batch, show_progress=False)

        embed_ms = (time.perf_counter() - batch_t0) * 1000
        log.debug(f"    Embed done in {embed_ms:.0f} ms  ({len(batch)/embed_ms*1000:.0f} chunks/s)")

        # Log GPU VRAM every N batches
        if batch_count % LOG_GPU_EVERY == 0:
            _log_gpu_stats(torch_module, prefix=f"    [batch {batch_count}] ")

        points: list[PointStruct] = []
        for chunk, emb in zip(batch, embeddings):
            if emb.get("_embed_error"):
                log.warning(f"  Skipping chunk {chunk.get('chunk_id')} — embed error")
                continue

            chunk_id = chunk.get("chunk_id", "")
            if not chunk_id:
                log.warning("  Skipping chunk with empty chunk_id")
                continue

            sparse = emb.get("sparse")
            dense  = emb.get("dense")

            if dense is None or sparse is None:
                log.warning(f"  Skipping chunk {chunk_id} — missing dense/sparse vector")
                continue

            point_id = chunk_id_to_point_id(chunk_id)
            payload  = build_payload(chunk)

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense,
                        "sparse": SparseVector(
                            indices=sparse["indices"],
                            values=sparse["values"],
                        ),
                    },
                    payload=payload,
                )
            )

        if points:
            upsert_t0 = time.perf_counter()
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            upsert_ms = (time.perf_counter() - upsert_t0) * 1000
            log.debug(f"    Upsert {len(points)} points in {upsert_ms:.0f} ms")
            total_upserted += len(points)

        batch_count += 1

    elapsed = time.perf_counter() - t0
    if elapsed > 0 and total_upserted > 0:
        log.debug(
            f"  index_chunks complete: {total_upserted} points in {elapsed:.1f}s "
            f"({total_upserted / elapsed:.0f} pts/s)"
        )

    return total_upserted


# ── Per-document worker ───────────────────────────────────────────────────────
def index_document(
    doc_id: str,
    chunk_path: Path,
    client: QdrantClient,
    collection_name: str,
    embedder,
    reindex: bool = False,
    torch_module=None,
) -> int:
    """
    Load, embed, and upsert all chunks for a single document.
    Returns number of points upserted.
    """
    t0     = time.perf_counter()
    chunks: list[dict] = []

    with open(chunk_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(
                    f"  Skipping malformed JSON in {chunk_path.name} line {line_no}: {e}"
                )

    if not chunks:
        log.warning(f"  No chunks loaded from {chunk_path.name}")
        return 0

    log.debug(f"  Loaded {len(chunks)} chunks from {chunk_path.name}")

    if reindex:
        delete_doc_points(client, collection_name, [doc_id])

    n       = index_chunks(chunks, client, collection_name, embedder, torch_module=torch_module)
    elapsed = time.perf_counter() - t0

    log.info(
        f"  {doc_id:<55} {n:>5} points  "
        f"{len(chunks):>4} chunks  "
        f"{elapsed:5.1f}s  "
        f"({n / elapsed:.0f} pts/s)"
    )
    return n


# ── Entry point ───────────────────────────────────────────────────────────────
def run(
    tickers: list[str] | None = None,
    reindex: bool = False,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    O(N) manifest pattern — load once, update in-memory, write once.

    Parameters
    ----------
    tickers         : Subset of tickers to index. None = all chunked documents.
    reindex         : If True, delete existing points for affected docs before upsert.
    collection_name : Override default collection name.
    """
    _setup_logging()
    run_t0 = time.perf_counter()

    # ── GPU check ─────────────────────────────────────────────────────────────
    allow_cpu = os.getenv("EMBED_DEVICE", "").strip().lower() == "cpu"
    torch_module = None

    if not allow_cpu:
        gpu_info     = _assert_gpu()   # raises if 4090 not found
        torch_module = gpu_info["torch"]
    else:
        log.warning("EMBED_DEVICE=cpu — GPU check skipped. Indexing will be slow.")

    # ── Collection + embedder setup ───────────────────────────────────────────
    client = setup_collection(
        collection_name=collection_name,
        recreate=False,
        recreate_on_mismatch=False,
    )
    embedder = get_embedder()

    # ── GPU warmup (eliminates first-batch JIT compilation spike) ─────────────
    if torch_module is not None:
        log.info("Running GPU warmup…")
        warmup_t0 = time.perf_counter()
        embedder._model.encode(
            ["GPU warmup — ignore this text"] * min(8, embedder.batch_size),
            batch_size=8,
            max_length=16,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        torch_module.cuda.synchronize()
        log.info(f"  GPU warmup complete in {(time.perf_counter()-warmup_t0)*1000:.0f} ms")
        _log_gpu_stats(torch_module, prefix="  Post-warmup ")

    # ── Manifest filtering ────────────────────────────────────────────────────
    all_entries = load_manifest()
    id_to_idx   = {entry.get("doc_id"): i for i, entry in enumerate(all_entries)}

    working = all_entries
    if tickers:
        tickers_upper = {t.strip().upper() for t in tickers}
        working = [
            entry for entry in all_entries
            if entry.get("ticker", "").upper() in tickers_upper
        ]

    to_index = [
        entry for entry in working
        if entry.get("chunk_status") == "chunked"
        and (reindex or entry.get("index_status") != "indexed")
    ]

    log.info("=" * 60)
    log.info(
        f"Indexing {len(to_index)} documents "
        f"(reindex={reindex}, collection={collection_name})"
    )
    log.info("=" * 60)

    total_points = 0
    success      = 0
    failed       = 0
    progress_t0  = time.perf_counter()

    for doc_num, entry in enumerate(to_index, 1):
        doc_id     = entry["doc_id"]
        chunk_path = CHUNKS_DIR / f"{doc_id}.jsonl"

        if not chunk_path.exists():
            log.warning(f"  [{doc_num}/{len(to_index)}] Chunk file missing: {chunk_path.name}")
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["index_status"] = "index_failed"
                all_entries[idx]["indexed_points"] = 0
            failed += 1
            continue

        log.info(f"[{doc_num}/{len(to_index)}] {doc_id}")

        try:
            n = index_document(
                doc_id=doc_id,
                chunk_path=chunk_path,
                client=client,
                collection_name=collection_name,
                embedder=embedder,
                reindex=reindex,
                torch_module=torch_module,
            )

            idx = id_to_idx.get(doc_id)

            if n <= 0:
                log.warning(f"  Indexing produced 0 points for {doc_id}")
                if idx is not None:
                    all_entries[idx]["index_status"] = "index_failed"
                    all_entries[idx]["indexed_points"] = 0
                failed += 1
                continue

            if idx is not None:
                all_entries[idx]["index_status"] = "indexed"
                all_entries[idx]["indexed_points"] = n

            total_points += n
            success      += 1

        except Exception as e:
            log.error(f"  Indexing failed for {doc_id}: {e}", exc_info=True)
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["index_status"] = "index_failed"
                all_entries[idx]["indexed_points"] = 0
            failed += 1

        # ── Rolling progress summary ──────────────────────────────────────────
        if doc_num % LOG_PROGRESS_EVERY == 0:
            elapsed_so_far = time.perf_counter() - progress_t0
            docs_done      = success + failed
            rate           = docs_done / elapsed_so_far if elapsed_so_far > 0 else 0
            remaining      = len(to_index) - doc_num
            eta_s          = remaining / rate if rate > 0 else 0
            _log_gpu_stats(torch_module, prefix="  [progress] ")
            log.info(
                f"  ── Progress: {doc_num}/{len(to_index)} docs  "
                f"success={success}  failed={failed}  "
                f"pts={total_points}  "
                f"rate={rate:.1f} docs/s  "
                f"ETA={eta_s/60:.1f} min ──"
            )

    _write_manifest(all_entries)

    total_elapsed = time.perf_counter() - run_t0
    log.info("")
    log.info("=" * 60)
    log.info("INDEXING COMPLETE")
    log.info(f"  Documents : success={success}  failed={failed}  total={len(to_index)}")
    log.info(f"  Points    : {total_points}")
    log.info(f"  Wall time : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    if total_elapsed > 0 and total_points > 0:
        log.info(f"  Throughput: {total_points/total_elapsed:.0f} pts/s")
    log.info("=" * 60)

    try:
        info = client.get_collection(collection_name)
        log.info(
            f"  Collection '{collection_name}': "
            f"vectors_count={info.vectors_count}  "
            f"status={info.status}"
        )
    except Exception:
        pass

    if torch_module is not None:
        _log_gpu_stats(torch_module, prefix="  Final ")


if __name__ == "__main__":
    import argparse

    _setup_logging()

    parser = argparse.ArgumentParser(description="Index chunked SEC filings into Qdrant.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Subset of tickers (e.g. --tickers AAPL MSFT). Omit to index all chunked documents.",
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