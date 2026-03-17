"""
Qdrant collection setup for the RAG investment pipeline.
Safer version: validates existing schema before reuse.
"""

import logging
import os
import time
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    PayloadSchemaType,
)

log = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_investment")

DENSE_DIM = 1024
HNSW_M = 16
HNSW_EF_CONSTRUCT = 256

_KEYWORD_INDEXES = [
    "ticker",
    "form_type",
    "doc_id",
    "chunk_type",
    "period_type",
    "fiscal_quarter",
    "sector",
    "industry",
    "source_class",
    "statement_type",
    "table_title",
    "section",
]

_INTEGER_INDEXES = [
    "fiscal_year",
    "page",
    "block_order",
    "report_priority",
    "filing_date_ts",
]

_BOOL_INDEXES = [
    "is_financial_statement",
    "is_cover_page",
    "page_is_surrogate",
]


def get_client() -> QdrantClient:
    """Return a connected Qdrant client. Retries 3× with backoff on startup."""
    for attempt in range(3):
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
            client.get_collections()
            return client
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(
                    f"Cannot connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} "
                    f"after 3 attempts. Is Docker running?\n"
                    f"  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant\n"
                    f"Error: {e}"
                ) from e
            log.warning(f"Qdrant not ready (attempt {attempt + 1}/3): {e}. Retrying in 3s…")
            time.sleep(3)


def _expected_vectors_config() -> dict:
    return {
        "dense": VectorParams(
            size=DENSE_DIM,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(
                m=HNSW_M,
                ef_construct=HNSW_EF_CONSTRUCT,
                full_scan_threshold=10_000,
            ),
            on_disk=False,
        )
    }


def _expected_sparse_vectors_config() -> dict:
    return {
        "sparse": SparseVectorParams(
            index=SparseIndexParams(
                on_disk=False,
            ),
        )
    }


def _expected_quantization_config() -> ScalarQuantization:
    return ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    )


def _create_collection(client: QdrantClient, collection_name: str) -> None:
    log.info(f"Creating collection '{collection_name}'…")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=_expected_vectors_config(),
        sparse_vectors_config=_expected_sparse_vectors_config(),
        quantization_config=_expected_quantization_config(),
        on_disk_payload=True,
        optimizers_config=OptimizersConfigDiff(default_segment_number=4),
    )
    log.info(f"  Collection '{collection_name}' created.")


def _validate_collection_schema(client: QdrantClient, collection_name: str) -> None:
    """
    Raise RuntimeError if the existing collection does not match what the indexer expects.
    """
    info = client.get_collection(collection_name)

    # qdrant-client response structure varies slightly by version, so keep this defensive.
    params = info.config.params

    vector_cfg = getattr(params, "vectors", None)
    sparse_cfg = getattr(params, "sparse_vectors", None)

    errors: list[str] = []

    dense = None
    if hasattr(vector_cfg, "get"):
        dense = vector_cfg.get("dense")
    elif isinstance(vector_cfg, dict):
        dense = vector_cfg.get("dense")
    elif hasattr(vector_cfg, "size"):
        # single unnamed-vector collection — definitely wrong for this pipeline
        dense = None

    if dense is None:
        errors.append("missing named dense vector 'dense'")
    else:
        dense_size = getattr(dense, "size", None)
        dense_distance = getattr(dense, "distance", None)
        if dense_size != DENSE_DIM:
            errors.append(f"dense.size={dense_size} != expected {DENSE_DIM}")
        if str(dense_distance).upper() != str(Distance.COSINE).upper():
            errors.append(f"dense.distance={dense_distance} != COSINE")

    sparse = None
    if hasattr(sparse_cfg, "get"):
        sparse = sparse_cfg.get("sparse")
    elif isinstance(sparse_cfg, dict):
        sparse = sparse_cfg.get("sparse")

    if sparse is None:
        errors.append("missing named sparse vector 'sparse'")

    on_disk_payload = getattr(params, "on_disk_payload", None)
    if on_disk_payload is not True:
        errors.append(f"on_disk_payload={on_disk_payload} != True")

    if errors:
        raise RuntimeError(
            f"Existing collection '{collection_name}' schema mismatch:\n  - "
            + "\n  - ".join(errors)
            + "\nUse --recreate or recreate_on_mismatch=True."
        )


def setup_collection(
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
    recreate_on_mismatch: bool = False,
) -> QdrantClient:
    """
    Create or verify the Qdrant collection with the exact schema the indexer expects.
    """
    if client is None:
        client = get_client()

    existing = {c.name for c in client.get_collections().collections}

    if recreate and collection_name in existing:
        log.warning(f"recreate=True — deleting collection '{collection_name}'")
        client.delete_collection(collection_name)
        existing.discard(collection_name)

    if collection_name not in existing:
        _create_collection(client, collection_name)
    else:
        log.info(f"Collection '{collection_name}' already exists — validating schema…")
        try:
            _validate_collection_schema(client, collection_name)
            log.info("  Existing collection schema is compatible.")
        except RuntimeError as e:
            if recreate_on_mismatch:
                log.warning(str(e))
                log.warning(f"Recreating incompatible collection '{collection_name}'…")
                client.delete_collection(collection_name)
                _create_collection(client, collection_name)
            else:
                raise

    log.info("  Creating payload indexes (idempotent)…")
    _create_payload_indexes(client, collection_name)

    log.info(f"  setup_collection complete for '{collection_name}'.")
    return client


def _create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create all payload indexes. Each call is idempotent."""
    for field in _KEYWORD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                log.warning(f"    Could not create KEYWORD index for '{field}': {e}")

    for field in _INTEGER_INDEXES:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                log.warning(f"    Could not create INTEGER index for '{field}': {e}")

    for field in _BOOL_INDEXES:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.BOOL,
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                log.warning(f"    Could not create BOOL index for '{field}': {e}")

    log.info(
        f"    {len(_KEYWORD_INDEXES)} KEYWORD + {len(_INTEGER_INDEXES)} INTEGER "
        f"+ {len(_BOOL_INDEXES)} BOOL payload indexes ensured."
    )


def filing_date_to_ts(filing_date: str) -> int | None:
    """
    Convert filing_date string to Unix epoch seconds (UTC midnight).
    """
    if not filing_date:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y%m%d"):
        try:
            dt = datetime.strptime(filing_date, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    log.warning(f"  Could not parse filing_date: '{filing_date}'")
    return None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Set up Qdrant collection for RAG pipeline.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection.",
    )
    parser.add_argument(
        "--recreate-on-mismatch",
        action="store_true",
        help="Recreate automatically if an existing collection schema is incompatible.",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Collection name (default: {COLLECTION_NAME}).",
    )
    args = parser.parse_args()

    client = setup_collection(
        collection_name=args.collection,
        recreate=args.recreate,
        recreate_on_mismatch=args.recreate_on_mismatch,
    )
    info = client.get_collection(args.collection)
    points = getattr(info, "points_count", None) or getattr(info, "vectors_count", 0)
    log.info(f"Collection info: points_count={points}  status={info.status}")