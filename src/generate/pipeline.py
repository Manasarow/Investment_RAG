import copy
import logging
from typing import TypedDict, Any

from langgraph.graph import END, StateGraph

from src.generate.query_planner import plan_retrieval
from src.retrieve.hybrid_search import retrieve
from src.generate.generator import generate

log = logging.getLogger(__name__)


class RAGState(TypedDict):
    query: str
    plan: dict
    context: list
    verification: dict
    result: dict
    retry_count: int


def node_plan(state: RAGState) -> RAGState:
    plan = plan_retrieval(state["query"])
    return {**state, "plan": plan}


def node_retrieve(state: RAGState) -> RAGState:
    """
    Retrieve a fresh context set for the current plan.

    BUG-7 FIX: The full plan object is now passed as plan=plan to retrieve().
    Previously, only individual scalar fields were passed (query, filters, top-k
    values) but plan= was omitted, causing retrieve() to call _auto_plan() again
    and discard all retry enrichment done by node_verify (widened top-k, relaxed
    filters, multi-year hints, diversity flags, etc.).

    By passing plan=plan, retrieval_hints / require_multi_year / require_company_diversity
    flow correctly into assemble_context(), and the retry mechanism actually works.
    """
    plan = state["plan"]

    context = retrieve(
        query=plan["query"],
        filters=plan.get("filters", {}),
        dense_top_k=plan.get("dense_top_k", 50),
        sparse_top_k=plan.get("sparse_top_k", 50),
        reranker_top_k=plan.get("reranker_k", 10),
        final_top_k=plan.get("final_k", 6),
        plan=plan,  # BUG-7 FIX: pass the full plan so retrieve() doesn't re-derive it
    )

    return {**state, "context": context}


def _tickers_in_context(context: list[dict]) -> set[str]:
    return {str(c.get("ticker")) for c in context if c.get("ticker")}


def _years_in_context(context: list[dict]) -> set[int]:
    years: set[int] = set()
    for chunk in context:
        year = chunk.get("fiscal_year")
        if isinstance(year, int):
            years.add(year)
        else:
            try:
                years.add(int(year))
            except (TypeError, ValueError):
                pass
    return years


def _forms_in_context(context: list[dict]) -> set[str]:
    return {str(c.get("form_type")) for c in context if c.get("form_type")}


def node_verify(state: RAGState) -> RAGState:
    """
    Intent-aware verification.
    Checks are intentionally conservative: if evidence coverage is weak, retry retrieval.

    BUG-6 NOTE: The retry guard (retry_count < 2) is checked here before incrementing,
    so the increment only happens when a retry is actually going to occur. This means
    should_retry() can use the same threshold (retry_count < 2) consistently without
    an off-by-one — if node_verify didn't increment (because the limit was hit), the
    count stays at 2 and should_retry() correctly routes to generate.
    """
    context = state["context"]
    plan = state["plan"]
    intent = plan.get("intent", "single_company_factual")
    tickers = plan.get("tickers", []) or []
    filters = plan.get("filters", {}) or {}

    checks: dict[str, bool] = {
        "has_context": len(context) > 0,
        "sufficient_chunks": len(context) >= min(max(plan.get("final_k", 6), 2), 3),
    }

    if tickers:
        found_tickers = _tickers_in_context(context)
        if intent == "cross_company_comparison":
            checks["all_companies_present"] = all(t in found_tickers for t in tickers)
        elif intent in {"single_company_factual", "trend_over_time"}:
            checks["target_company_present"] = any(t in found_tickers for t in tickers)

    if intent == "trend_over_time":
        checks["multi_year"] = len(_years_in_context(context)) >= 2

    if intent == "single_company_factual":
        requested_year = filters.get("fiscal_year")
        context_years = _years_in_context(context)

        if requested_year is not None:
            # Explicit year(s) in the filter — verify at least one is present.
            requested_years = requested_year if isinstance(requested_year, list) else [requested_year]
            # BUG-1 compatibility: both sides are int thanks to query_planner fix,
            # but guard with int() cast here as a safety net.
            checks["correct_period"] = any(int(y) in context_years for y in requested_years if str(y).isdigit())
        else:
            # No explicit year filter (e.g. "latest", "most recent", "current").
            # FIX (period correctness): fall back to the plan's latest_year so that
            # a "latest 10-K" query doesn't silently pass with only stale-year chunks.
            # If latest_year is unknown we skip the check rather than block incorrectly.
            latest_year = plan.get("latest_year")
            if latest_year is not None:
                try:
                    checks["correct_period"] = int(latest_year) in context_years
                except (TypeError, ValueError):
                    pass  # can't determine expected year — omit the check

        # Check form_type filter first; fall back to source_class filter.
        # The planner sets source_class (e.g. "10-K") not form_type when it detects
        # an explicit form keyword, so we need to check both fields.
        requested_form = filters.get("form_type") or filters.get("source_class")
        if requested_form:
            requested_forms = requested_form if isinstance(requested_form, list) else [requested_form]
            context_forms = _forms_in_context(context)
            # Normalise: "10-K" source_class matches "10-K" form_type values
            checks["correct_form"] = any(
                any(rf.upper() in cf.upper() or cf.upper() in rf.upper()
                    for cf in context_forms)
                for rf in requested_forms
            )

    if intent == "thematic_synthesis":
        checks["company_diversity"] = len(_tickers_in_context(context)) >= 2

    if intent == "investment_opinion":
        checks["company_diversity"] = len(_tickers_in_context(context)) >= max(2, len(tickers) or 2)

    passed = all(checks.values())

    # BUG-6 FIX: Only increment retry_count and update the plan when a retry is
    # actually going to happen (i.e. checks failed AND we're under the limit).
    # Previously the increment happened inside the if-block, which is correct, but
    # should_retry() used `retry_count < 2` after the increment — meaning the same
    # threshold was checked twice with different counts. Now node_verify and
    # should_retry() both use the same pre-increment value for their decisions,
    # keeping the logic consistent and easy to reason about.
    current_retry_count = state.get("retry_count", 0)

    if not passed and current_retry_count < 2:
        log.info("Verification failed %s — retrying with widened retrieval (attempt %d)", checks, current_retry_count + 1)

        new_plan = copy.deepcopy(plan)
        new_filters = copy.deepcopy(new_plan.get("filters", {}))

        # Relax the most common overconstraint first.
        new_filters.pop("fiscal_year", None)

        # Allow both annual and quarterly evidence if the first pass was too narrow.
        if intent in {"single_company_factual", "trend_over_time"}:
            current_form = new_filters.get("form_type")
            if current_form in {"10-K", "10-Q"}:
                new_filters["form_type"] = ["10-K", "10-Q"]

        # For broad synthesis queries, remove ticker restriction on retry.
        if intent in {"thematic_synthesis", "investment_opinion"}:
            new_filters.pop("ticker", None)

        new_plan["filters"] = new_filters
        new_plan["dense_top_k"] = min(int(new_plan.get("dense_top_k", 50)) + 20, 120)
        new_plan["sparse_top_k"] = min(int(new_plan.get("sparse_top_k", 50)) + 20, 120)
        new_plan["reranker_k"] = min(int(new_plan.get("reranker_k", 10)) + 5, 25)
        new_plan["final_k"] = min(int(new_plan.get("final_k", 6)) + 2, 10)

        return {
            **state,
            "plan": new_plan,
            "verification": checks,
            "retry_count": current_retry_count + 1,  # BUG-6 FIX: increment after decision
        }

    if not passed:
        log.warning(
            "Verification still failing after %d retries — proceeding to generate anyway: %s",
            current_retry_count,
            checks,
        )

    return {**state, "verification": checks}


def node_generate(state: RAGState) -> RAGState:
    result = generate(
        query=state["query"],
        context_chunks=state["context"],
    )
    return {**state, "result": result}


def should_retry(state: RAGState) -> str:
    """
    BUG-6 FIX: The routing decision is now purely based on whether node_verify
    updated the plan for a retry (indicated by verification failures AND retry_count
    being incremented). Since node_verify only increments retry_count when it's
    actually scheduling a retry, we can check whether the new plan differs from the
    context OR simply check if any verification check failed while retry_count < 2.

    The cleaner signal: if verification failed AND retry_count was just incremented
    (i.e. we're still under the limit), node_verify already put a new plan in state
    and we should re-enter retrieve. Otherwise go to generate.
    """
    verification = state.get("verification", {})
    retry_count = state.get("retry_count", 0)

    # node_verify only increments retry_count when scheduling a retry.
    # So if checks failed and count > 0, a retry was just scheduled.
    if verification and not all(verification.values()) and 0 < retry_count < 2:
        return "retrieve"
    return "generate"


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("plan", node_plan)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("verify", node_verify)
    graph.add_node("generate", node_generate)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges(
        "verify",
        should_retry,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )
    graph.add_edge("generate", END)

    return graph.compile()


_graph: Any = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str) -> dict:
    graph = get_graph()

    initial_state: RAGState = {
        "query": query,
        "plan": {},
        "context": [],
        "verification": {},
        "result": {},
        "retry_count": 0,
    }

    final_state = graph.invoke(initial_state)
    result = final_state.get("result", {}) or {}

    result["query"] = query
    result["intent"] = final_state.get("plan", {}).get("intent", "unknown")
    result["tickers"] = final_state.get("plan", {}).get("tickers", [])
    result["retries"] = final_state.get("retry_count", 0)
    result["verification"] = final_state.get("verification", {})
    result["context_used"] = [
        {
            "ticker": c.get("ticker"),
            "form": c.get("form_type"),
            "fy": c.get("fiscal_year"),
            "page": c.get("page"),
            "section": c.get("section"),
            "type": c.get("chunk_type"),
        }
        for c in final_state.get("context", [])
    ]

    return result