"""
Retrieval pipeline for the investment RAG system.

This revision fixes the retrieval-stage bugs from the tracker:
  #40  Uses BGE-M3 token IDs directly for sparse query vectors; no hash()
  #41  Context assembly defaults are widened and intent-aware
  #42  Reranker sees full payload text (already stored by indexer) and richer metadata
  #43  Dedup no longer collapses table row-groups by text prefix
  #44  Page-level dedup removed; multiple valid chunks from one page are allowed
  #45  Missing ticker/year filters can be auto-derived from the query via planner
  #46  Multi-hop retrieval added for multi-entity / broad synthesis queries
  #47  RRF is configurable and weighted instead of hardcoded/equal-only
  #48  Reranker model is configurable; default remains finance-friendly BGE reranker
  #49  Filter-relaxation fallback recovers from zero-hit filtered searches
  #50  Qdrant client includes health-check and reconnect logic
  #91  Numeric queries boost table/row evidence
  #92  Annual / fiscal-year queries boost annual sources
  #93  Reranker input now includes period / source metadata to reduce wrong-period wins
  #94  Metadata bonus scoring layer added after reranking
  #95  Context assembly enforces diversity across company + year + evidence type

BUGFIXES (post-review):
  BUG-3  retrieve() no longer silently discards the planner's filters when the caller
         passes explicit filters={}. The caller in node_retrieve now always passes
         plan=plan so the full plan object is available. The filters priority logic
         is clarified: explicit non-None filters take precedence; otherwise plan filters
         are used. An empty dict {} is treated as "no override" to avoid accidentally
         clearing plan-derived constraints.
  BUG-4  _metadata_pre_bonus() was defined but never called. It is now wired into
         _post_rrf_rescore() which runs immediately after RRF fusion, giving pre-rerank
         boosting to numeric/annual/source-preferred candidates.
  BUG-5  _relaxed_filter_variants() crashed with TypeError: unhashable type: 'list'
         when a filter value was a list (e.g. fiscal_year: [2022, 2023, 2024] for
         trend queries). Fixed by converting each variant to a hashable key via a
         recursive helper before adding to the seen set.

Public API remains compatible:
    retrieve(query, filters=None, dense_top_k=..., sparse_top_k=..., reranker_top_k=..., final_top_k=...)

You may also pass an explicit planner output:
    retrieve(query, plan=plan_retrieval(query))
"""

from __future__ import annotations

import copy
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, SparseVector

from src.index.embedder import get_embedder
from src.index.qdrant_setup import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT

log = logging.getLogger(__name__)

RRF_K = int(os.getenv("RETRIEVAL_RRF_K", "40"))
DENSE_WEIGHT = float(os.getenv("RETRIEVAL_DENSE_WEIGHT", "1.0"))
SPARSE_WEIGHT = float(os.getenv("RETRIEVAL_SPARSE_WEIGHT", "1.1"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_USE_FP16 = os.getenv("RERANKER_FP16", "1") == "1"
CLIENT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))

NUMERIC_QUERY_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"\brevenue\b",
        r"\bsegment(?:s)?\b",
        r"\bbreak\s*down\b",
        r"\bnet\s+income\b",
        r"\bgross\s+margin\b",
        r"\bmargin(?:s)?\b",
        r"\bdebt\b",
        r"\bcash\b",
        r"\bassets?\b",
        r"\bliabilit(?:y|ies)\b",
        r"\beps\b",
        r"\bcapex\b",
        r"\bbuyback(?:s)?\b",
        r"\br&d\b",
        r"\boperating\s+income\b",
    ]
]
LATEST_PATTERNS = [re.compile(p, re.I) for p in [r"\blatest\b", r"\bmost\s+recent\b", r"\blast\s+reported\b", r"\bcurrent\b"]]
ANNUAL_PATTERNS = [re.compile(p, re.I) for p in [r"\bfiscal\s+year\b", r"\bannual\b", r"\b10-?k\b", r"\bfy\s*20\d{2}\b"]]
TREND_PATTERNS = [re.compile(p, re.I) for p in [r"\btrend(?:ed|ing)?\b", r"\bpast\s+\d+\s+(?:fiscal\s+)?years?\b", r"\bover\s+time\b", r"\byoy\b"]]

_client: Optional[QdrantClient] = None
_reranker = None


# ---------------------------------------------------------------------------
# Client / health check
# ---------------------------------------------------------------------------
def _new_client() -> QdrantClient:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=CLIENT_TIMEOUT)
    client.get_collections()
    return client


def get_client(force_refresh: bool = False) -> QdrantClient:
    global _client
    if _client is None or force_refresh:
        _client = _new_client()
        return _client

    try:
        _client.get_collections()
        return _client
    except Exception as e:
        log.warning(f"Qdrant health check failed; reconnecting: {e}")
        _client = _new_client()
        return _client


# ---------------------------------------------------------------------------
# Planner integration (#45)
# ---------------------------------------------------------------------------
def _auto_plan(query: str) -> dict:
    from src.generate.query_planner import plan_retrieval
    return plan_retrieval(query)


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------
def _match_condition(field: str, value: Any) -> Optional[FieldCondition]:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, list):
        return FieldCondition(key=field, match=MatchAny(any=value))
    return FieldCondition(key=field, match=MatchValue(value=value))


SUPPORTED_FILTER_FIELDS = (
    "ticker",
    "fiscal_year",
    "form_type",
    "chunk_type",
    "source_class",
    "statement_type",
    "sector",
    "industry",
    "period_type",
    "fiscal_quarter",
)


def build_filter(filters: Optional[dict]) -> Optional[Filter]:
    if not filters:
        return None
    must = []
    for field in SUPPORTED_FILTER_FIELDS:
        cond = _match_condition(field, filters.get(field))
        if cond is not None:
            must.append(cond)
    return Filter(must=must) if must else None


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def _contains_any(query: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(query) for p in patterns)


def _query_flags(query: str, plan: Optional[dict] = None) -> dict:
    q = query.lower()
    report_scope = (plan or {}).get("report_scope", {})
    evidence = (plan or {}).get("evidence_profile", {})
    return {
        "numeric": bool(evidence.get("numeric")) or _contains_any(q, NUMERIC_QUERY_PATTERNS),
        "latest": bool((plan or {}).get("retrieval_hints", {}).get("latest_bias")) or _contains_any(q, LATEST_PATTERNS),
        "annual": bool(report_scope.get("prefer_annual")) or _contains_any(q, ANNUAL_PATTERNS),
        "trend": bool((plan or {}).get("dimensions", {}).get("trend")) or _contains_any(q, TREND_PATTERNS),
    }


# ---------------------------------------------------------------------------
# Sparse query vector (#40)
# ---------------------------------------------------------------------------
def _embed_query(query: str) -> dict:
    return get_embedder().embed_query(query)


def _sparse_query_vector(query_emb: dict) -> SparseVector:
    sparse = query_emb["sparse"]
    return SparseVector(indices=list(sparse["indices"]), values=list(sparse["values"]))


# ---------------------------------------------------------------------------
# Stage A: candidate generation
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(result_lists: list[list], *, k: int = RRF_K, weights: Optional[list[float]] = None) -> list[dict]:
    """Weighted Reciprocal Rank Fusion over Qdrant scored-point lists."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    if weights is None:
        weights = [1.0] * len(result_lists)

    for list_weight, hits in zip(weights, result_lists):
        for rank, hit in enumerate(hits):
            hit_id = str(hit.id)
            scores[hit_id] = scores.get(hit_id, 0.0) + list_weight * (1.0 / (k + rank + 1))
            payload = dict(hit.payload or {})
            payload["_point_id"] = hit_id
            payloads[hit_id] = payload

    fused = []
    for hit_id in sorted(scores.keys(), key=lambda x: -scores[x]):
        payload = payloads[hit_id]
        payload["_rrf_score"] = scores[hit_id]
        fused.append(payload)
    return fused


def _metadata_pre_bonus(candidate: dict, flags: dict, plan: Optional[dict]) -> float:
    """
    Scoring bonus applied immediately after RRF fusion, before the neural reranker.

    BUG-4 FIX: This function was defined but never called. It is now invoked inside
    _post_rrf_rescore() so that numeric/annual/source preferences actually influence
    the candidate ranking passed to the reranker.
    """
    bonus = 0.0
    chunk_type = candidate.get("chunk_type", "")
    source_class = candidate.get("source_class", "")
    statement_type = candidate.get("statement_type", "")
    periods = [str(p) for p in candidate.get("periods", [])]
    hints = (plan or {}).get("retrieval_hints", {})

    if flags["numeric"]:
        if chunk_type in {"row", "table", "micro_block"}:
            bonus += 0.12
        if statement_type in {"segment_table", "income_statement", "balance_sheet", "cash_flow_statement", "debt_table", "eps_table"}:
            bonus += 0.08

    if flags["annual"]:
        if source_class == "10-K":
            bonus += 0.12
        if candidate.get("period_type") == "annual":
            bonus += 0.06

    if flags["latest"]:
        report_priority = candidate.get("report_priority")
        if report_priority == 1 and source_class == "10-K":
            bonus += 0.05
        elif report_priority == 2 and source_class == "10-Q":
            bonus += 0.04

    preferred_sources = hints.get("prefer_source_classes") or []
    if preferred_sources and source_class in preferred_sources:
        bonus += 0.10

    preferred_statement_types = hints.get("prefer_statement_types") or []
    if preferred_statement_types and statement_type in preferred_statement_types:
        bonus += 0.08

    if periods and flags["trend"]:
        bonus += 0.03

    # Row-label boost: prioritise rows whose label contains the queried metric.
    # This ensures "Revenue" rows rank above "Cost of revenue" or "Operating income"
    # rows when the query explicitly asks for revenue, margins, etc.
    row_label = (candidate.get("row_label") or "").lower()
    if row_label and flags["numeric"]:
        query_lower = "" if plan is None else (plan.get("query", "")).lower()
        if "revenue" in query_lower and "revenue" in row_label and "cost" not in row_label:
            bonus += 0.15
        elif "margin" in query_lower and ("margin" in row_label or "gross profit" in row_label):
            bonus += 0.15
        elif "net income" in query_lower and "net income" in row_label:
            bonus += 0.15
        elif "eps" in query_lower and ("eps" in row_label or "earnings per share" in row_label):
            bonus += 0.15
        elif "operating income" in query_lower and "operating income" in row_label:
            bonus += 0.15

    return bonus


def _post_rrf_rescore(candidates: list[dict], query: str, plan: Optional[dict]) -> list[dict]:
    """
    Apply pre-rerank metadata bonuses on top of RRF scores.

    BUG-4 FIX: Now correctly calls _metadata_pre_bonus() with computed flags,
    so numeric/annual/source hints actually affect pre-rerank candidate ordering.
    """
    flags = _query_flags(query, plan)
    rescored = []
    for cand in candidates:
        cand = dict(cand)
        cand["_score"] = float(cand.get("_rrf_score", 0.0)) + _metadata_pre_bonus(cand, flags, plan)
        rescored.append(cand)
    return sorted(rescored, key=lambda x: -x.get("_score", 0.0))


def _run_single_search(
    *,
    client: QdrantClient,
    query: str,
    filters: Optional[dict],
    dense_top_k: int,
    sparse_top_k: int,
    plan: Optional[dict],
) -> list[dict]:
    q_emb = _embed_query(query)
    dense_vec = q_emb["dense"]
    sparse_vec = _sparse_query_vector(q_emb)
    qfilter = build_filter(filters)

    dense_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vec,
        using="dense",
        query_filter=qfilter,
        limit=dense_top_k,
        with_payload=True,
    ).points

    sparse_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vec,
        using="sparse",
        query_filter=qfilter,
        limit=sparse_top_k,
        with_payload=True,
    ).points

    fused = reciprocal_rank_fusion(
        [dense_hits, sparse_hits],
        k=RRF_K,
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )
    rescored = _post_rrf_rescore(fused, query, plan)
    log.info(
        "Hybrid search: dense=%d sparse=%d fused=%d filters=%s",
        len(dense_hits), len(sparse_hits), len(rescored), filters,
    )
    return rescored


def _multi_hop_candidates(
    *,
    client: QdrantClient,
    query: str,
    plan: dict,
    filters: Optional[dict],
    dense_top_k: int,
    sparse_top_k: int,
) -> list[dict]:
    tickers = plan.get("tickers") or []
    results: list[list[dict]] = []

    if len(tickers) > 1:
        for ticker in tickers:
            per_filters = copy.deepcopy(filters or {})
            per_filters["ticker"] = ticker
            results.append(
                _run_single_search(
                    client=client,
                    query=query,
                    filters=per_filters,
                    dense_top_k=max(18, dense_top_k // max(1, len(tickers))),
                    sparse_top_k=max(18, sparse_top_k // max(1, len(tickers))),
                    plan=plan,
                )
            )

    # Also run a global pass so cross-company / thematic bridges can surface.
    results.append(
        _run_single_search(
            client=client,
            query=query,
            filters=filters,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            plan=plan,
        )
    )

    # Merge by point id, keeping max score.
    merged: dict[str, dict] = {}
    for res in results:
        for cand in res:
            pid = str(cand.get("_point_id") or cand.get("chunk_id") or id(cand))
            if pid not in merged or cand.get("_score", 0.0) > merged[pid].get("_score", 0.0):
                merged[pid] = cand
    return sorted(merged.values(), key=lambda x: -x.get("_score", 0.0))


def _make_filter_hashable(filters: Optional[dict]) -> Any:
    """
    BUG-5 FIX: Convert a filter dict to a hashable key for deduplication.

    The original code did `tuple(sorted(variant.items()))` which raises
    TypeError when any value is a list (e.g. fiscal_year: [2022, 2023, 2024]).
    This helper recursively converts lists to tuples so the result is hashable.
    """
    if filters is None:
        return None
    return tuple(
        sorted(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in filters.items()
        )
    )


def _relaxed_filter_variants(filters: Optional[dict], plan: Optional[dict]) -> list[Optional[dict]]:
    """
    Generate progressively looser filter sets for zero-hit recovery.

    BUG-5 FIX: Uses _make_filter_hashable() instead of bare tuple(sorted(...items()))
    so that list-valued filter fields (e.g. fiscal_year=[2022,2023,2024] on trend
    queries) no longer raise TypeError: unhashable type: 'list'.
    """
    base = copy.deepcopy(filters or {})
    variants: list[Optional[dict]] = [base]
    if not base:
        return variants

    # 1) remove statement-specific constraint
    if "statement_type" in base:
        v = copy.deepcopy(base)
        v.pop("statement_type", None)
        variants.append(v)

    # 2) widen source_class for quarter/event ambiguity
    if "source_class" in base:
        v = copy.deepcopy(base)
        v.pop("source_class", None)
        variants.append(v)

    # 3) remove fiscal year but keep ticker/sector
    if "fiscal_year" in base:
        v = copy.deepcopy(base)
        v.pop("fiscal_year", None)
        variants.append(v)

    # 4) remove sector / industry gates
    if "sector" in base or "industry" in base:
        v = copy.deepcopy(base)
        v.pop("sector", None)
        v.pop("industry", None)
        variants.append(v)

    # 5) remove everything except explicit ticker when present
    if base.get("ticker"):
        variants.append({"ticker": base["ticker"]})

    # 6) full fallback
    variants.append(None)

    deduped: list[Optional[dict]] = []
    seen: set = set()
    for variant in variants:
        # BUG-5 FIX: use _make_filter_hashable() to safely handle list values.
        key = _make_filter_hashable(variant)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def hybrid_search(
    query: str,
    filters: Optional[dict] = None,
    dense_top_k: int = 50,
    sparse_top_k: int = 50,
    *,
    plan: Optional[dict] = None,
) -> list[dict]:
    """
    Run dense + sparse retrieval and fuse with weighted RRF.

    BUG-3 FIX: effective_filters resolution clarified.
    - If the caller passes explicit non-None filters, those take precedence.
    - An empty dict {} from the caller is treated as "no override" (same as None)
      to avoid accidentally wiping plan-derived filter constraints on retries.
    - The plan object is always passed through to _run_single_search / multi-hop
      so scoring hints are available regardless of how filters were derived.
    """
    if plan is None:
        plan = _auto_plan(query)

    # BUG-3 FIX: treat explicit empty dict the same as None — it means the caller
    # didn't intend to override, so fall back to plan filters.
    if filters is not None and len(filters) > 0:
        effective_filters = copy.deepcopy(filters)
    else:
        effective_filters = copy.deepcopy(plan.get("filters", {}))

    client = get_client()
    hints = plan.get("retrieval_hints", {})

    last_results: list[dict] = []
    for variant in _relaxed_filter_variants(effective_filters, plan):
        try:
            if hints.get("enable_multi_hop"):
                results = _multi_hop_candidates(
                    client=client,
                    query=query,
                    plan=plan,
                    filters=variant,
                    dense_top_k=dense_top_k,
                    sparse_top_k=sparse_top_k,
                )
            else:
                results = _run_single_search(
                    client=client,
                    query=query,
                    filters=variant,
                    dense_top_k=dense_top_k,
                    sparse_top_k=sparse_top_k,
                    plan=plan,
                )
        except Exception:
            # Reconnect once if query path trips on a stale client.
            client = get_client(force_refresh=True)
            results = _run_single_search(
                client=client,
                query=query,
                filters=variant,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                plan=plan,
            )

        last_results = results
        if results:
            if variant != effective_filters:
                log.info("Filter fallback succeeded with relaxed filters=%s", variant)
            return results

    return last_results


# ---------------------------------------------------------------------------
# Stage B: reranking
# ---------------------------------------------------------------------------
def get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker

        log.info("Loading reranker: %s", RERANKER_MODEL)
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=RERANKER_USE_FP16)
        log.info("Reranker loaded.")
    return _reranker


def _reranker_text(candidate: dict) -> str:
    """
    Give the reranker the text plus the metadata that disambiguates period/source.
    This addresses wrong-period wins on semantically similar passages.
    """
    prefix_parts = [
        candidate.get("ticker", ""),
        candidate.get("source_class") or candidate.get("form_type", ""),
        f"FY{candidate.get('fiscal_year')}" if candidate.get("fiscal_year") else "",
        f"Q{candidate.get('fiscal_quarter')}" if candidate.get("fiscal_quarter") else "",
        candidate.get("period_type", ""),
        candidate.get("statement_type", ""),
        candidate.get("section", ""),
        candidate.get("table_title", ""),
        candidate.get("citation_key", ""),
    ]
    prefix = " | ".join(p for p in prefix_parts if p)
    text = candidate.get("text", "")
    return f"{prefix}\n{text}" if prefix else text


def _metadata_post_bonus(candidate: dict, query: str, plan: Optional[dict]) -> float:
    flags = _query_flags(query, plan)
    hints = (plan or {}).get("retrieval_hints", {})
    bonus = 0.0

    chunk_type = candidate.get("chunk_type", "")
    source_class = candidate.get("source_class", "")
    fiscal_year = candidate.get("fiscal_year")
    periods = set(str(p) for p in candidate.get("periods", []))

    if flags["numeric"]:
        if chunk_type == "row":
            bonus += 0.20
        elif chunk_type in {"table", "micro_block"}:
            bonus += 0.14

    if flags["annual"] and source_class == "10-K":
        bonus += 0.16
    if hints.get("prefer_source_classes") and source_class in hints["prefer_source_classes"]:
        bonus += 0.10

    target_years = plan.get("fiscal_years") if plan else None
    if target_years:
        if isinstance(target_years, int):
            target_years = [target_years]
        if fiscal_year in set(target_years):
            bonus += 0.12
        elif periods and any(str(y) in periods for y in target_years):
            bonus += 0.06

    return bonus


def rerank(query: str, candidates: list[dict], top_k: int = 10, *, plan: Optional[dict] = None) -> list[dict]:
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, _reranker_text(c)] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if not hasattr(scores, "__iter__"):
        scores = [scores]

    reranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["_rerank_score"] = float(score)
        item["_final_score"] = float(score) + _metadata_post_bonus(item, query, plan)
        reranked.append(item)

    reranked.sort(key=lambda x: -x.get("_final_score", 0.0))
    log.info("Reranked %d -> top %d", len(candidates), top_k)
    return reranked[:top_k]


# ---------------------------------------------------------------------------
# Stage C: context assembly
# ---------------------------------------------------------------------------
def _normalized_text_key(chunk: dict) -> str:
    text = re.sub(r"\s+", " ", (chunk.get("text") or "").strip().lower())
    return f"{chunk.get('chunk_type','')}|{chunk.get('citation_key','')}|{text}"


def assemble_context(
    reranked: list[dict],
    max_chunks: int = 8,
    max_per_doc: int = 6,
    max_per_page: int = 3,
    *,
    plan: Optional[dict] = None,
) -> list[dict]:
    """
    Deduplicate with provenance, then enforce diversity across:
      - document
      - page within document (max_per_page cap prevents same-page chunk flooding)
      - company + year
      - evidence type
    """
    if not reranked:
        return []

    hints = (plan or {}).get("retrieval_hints", {})
    require_company_diversity = hints.get("require_company_diversity", False)
    require_multi_year = hints.get("require_multi_year", False)

    # Bug #41: intent-aware larger defaults.
    if require_company_diversity:
        max_chunks = max(max_chunks, 10)
    if require_multi_year:
        max_chunks = max(max_chunks, 8)

    seen_content: set[str] = set()
    seen_row_labels: set[tuple] = set()  # (doc_id, row_label) for row dedup
    seen_row_labels: set[tuple[str, str]] = set()  # (doc_id, row_label) for row dedup
    doc_counts: dict[str, int] = defaultdict(int)
    # Page-level dedup: prevents multiple row chunks from same page dominating context.
    page_counts: dict[tuple[str, Any], int] = defaultdict(int)
    company_year_counts: dict[tuple[str, Any], int] = defaultdict(int)
    evidence_counts: dict[str, int] = defaultdict(int)
    assembled: list[dict] = []

    for chunk in reranked:
        content_key = _normalized_text_key(chunk)
        if content_key in seen_content:
            continue

        doc_id = chunk.get("doc_id", "")
        page = chunk.get("page")
        ticker = chunk.get("ticker", "")
        fiscal_year = chunk.get("fiscal_year")
        evidence = chunk.get("chunk_type", "")

        if doc_counts[doc_id] >= max_per_doc:
            continue

        # Page-level cap: at most max_per_page chunks from any (doc_id, page) pair.
        if page is not None and page_counts[(doc_id, page)] >= max_per_page:
            continue

        # Row-label dedup: for row-type chunks, prevent the same row_label from
        # the same document appearing more than once (e.g. 6 "Revenue" rows from
        # p.41 all scoring highly). Different row_labels (Revenue, Net income,
        # Gross margin) are all useful; identical labels are pure duplicates.
        row_label = chunk.get("row_label", "") if evidence == "row" else ""
        if row_label and (doc_id, row_label) in seen_row_labels:
            continue

        # Diversity: don't let one company-year dominate unless single-company intent.
        if require_company_diversity and company_year_counts[(ticker, fiscal_year)] >= 2:
            continue

        # Diversity: cap each evidence type a bit so prose/table/row can coexist.
        if evidence_counts[evidence] >= max(2, max_chunks // 2) and len(assembled) < max_chunks - 2:
            continue

        seen_content.add(content_key)
        doc_counts[doc_id] += 1
        if page is not None:
            page_counts[(doc_id, page)] += 1
        if row_label:
            seen_row_labels.add((doc_id, row_label))
        company_year_counts[(ticker, fiscal_year)] += 1
        evidence_counts[evidence] += 1
        assembled.append(chunk)

        if len(assembled) >= max_chunks:
            break

    # If diversity rules were too strict, backfill from the reranked list.
    if len(assembled) < min(max_chunks, 4):
        for chunk in reranked:
            if len(assembled) >= max_chunks:
                break
            content_key = _normalized_text_key(chunk)
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            assembled.append(chunk)

    log.info(
        "Context assembled: %d chunks | docs=%d | evidence_types=%d",
        len(assembled), len(doc_counts), len(evidence_counts),
    )
    return assembled


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    filters: Optional[dict] = None,
    dense_top_k: int = 50,
    sparse_top_k: int = 50,
    reranker_top_k: int = 12,
    final_top_k: int = 8,
    *,
    plan: Optional[dict] = None,
) -> list[dict]:
    """
    Full 3-stage retrieval pipeline.
    Returns assembled context chunks ready for generation.

    BUG-3 FIX: plan is now a first-class parameter. When node_retrieve passes
    plan=plan, the full plan object (with retrieval_hints, fiscal_years, etc.)
    flows through to hybrid_search, rerank, and assemble_context — so diversity
    enforcement, multi-year requirements, and scoring hints all work correctly.
    """
    if plan is None:
        plan = _auto_plan(query)

    dense_top_k = max(dense_top_k, int(plan.get("dense_top_k", dense_top_k)))
    sparse_top_k = max(sparse_top_k, int(plan.get("sparse_top_k", sparse_top_k)))
    reranker_top_k = max(reranker_top_k, int(plan.get("reranker_k", reranker_top_k)))
    final_top_k = max(final_top_k, int(plan.get("final_k", final_top_k)))

    candidates = hybrid_search(
        query=plan.get("query", query),
        filters=filters,  # BUG-3 FIX: pass filters as-is; hybrid_search handles precedence
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        plan=plan,  # BUG-3 FIX: always pass the full plan object
    )

    reranked = rerank(query, candidates, top_k=reranker_top_k, plan=plan)
    context = assemble_context(reranked, max_chunks=final_top_k, plan=plan)

    # YEAR-DISAMBIGUATION FIX: When the assembled context contains row-type chunks
    # with no col_header (common for multi-year income statements where three annual
    # values share the same row_label and page), the LLM cannot determine which value
    # belongs to which year. Fix: fetch the parent table chunk, which contains the
    # full retrieval_text with all column headers intact, and prepend it to context.
    # The parent table chunk is always more useful for year-specific queries than
    # the individual headerless row chunks.
    context = _inject_parent_table_chunks(context, plan=plan)

    return context


def _inject_parent_table_chunks(context: list[dict], *, plan: Optional[dict] = None) -> list[dict]:
    """
    For row chunks with no col_header, fetch their parent table chunk from Qdrant
    and prepend it to the context. The parent table chunk contains the full
    multi-year table text with column headers, allowing the LLM to correctly
    identify which value belongs to which fiscal year.

    Only fires when:
    - At least one row chunk in context has an empty col_header
    - The row chunk has a parent_chunk_id pointing to a table chunk
    - The parent chunk is not already in context
    """
    # Find row chunks that are missing col_header
    ambiguous_parent_ids: list[str] = []
    context_chunk_ids: set[str] = {c.get("chunk_id", "") for c in context}

    for chunk in context:
        if chunk.get("chunk_type") != "row":
            continue
        if chunk.get("col_header", "").strip():
            continue  # col_header present — no ambiguity
        parent_id = chunk.get("parent_chunk_id")
        if parent_id and parent_id not in context_chunk_ids and parent_id not in ambiguous_parent_ids:
            ambiguous_parent_ids.append(parent_id)

    if not ambiguous_parent_ids:
        return context

    try:
        client = get_client()
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(
                    key="chunk_id",
                    match=MatchAny(any=ambiguous_parent_ids),
                )
            ]),
            with_payload=True,
            with_vectors=False,
            limit=len(ambiguous_parent_ids) * 2,
        )

        injected = []
        for rec in results:
            p = rec.payload or {}
            if p.get("chunk_type") not in ("table", "micro_block"):
                continue
            if p.get("chunk_id") in context_chunk_ids:
                continue
            injected.append(p)
            context_chunk_ids.add(p.get("chunk_id", ""))

        if injected:
            log.info(
                "Injected %d parent table chunk(s) to resolve year-ambiguous row chunks",
                len(injected),
            )
            # Prepend parent table chunks so the LLM sees the structured table
            # with column headers before the individual row chunks.
            return injected + context

    except Exception as e:
        log.warning("Parent table chunk injection failed: %s", e)

    return context