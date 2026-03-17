import copy
import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.generate.generator import generate
from src.generate.query_planner import plan_retrieval
from src.retrieve.hybrid_search import retrieve

log = logging.getLogger(__name__)


class RAGState(TypedDict):
    query: str
    plan: dict
    context: list
    verification: dict
    result: dict
    retry_count: int


def node_plan(state: RAGState) -> RAGState:
    """Create a retrieval plan for the incoming query."""
    return {**state, "plan": plan_retrieval(state["query"])}


def node_retrieve(state: RAGState) -> RAGState:
    """Run retrieval using the current plan."""
    plan = state["plan"]

    context = retrieve(
        query=plan["query"],
        filters=plan.get("filters", {}),
        dense_top_k=plan.get("dense_top_k", 50),
        sparse_top_k=plan.get("sparse_top_k", 50),
        reranker_top_k=plan.get("reranker_k", 10),
        final_top_k=plan.get("final_k", 6),
        plan=plan,
    )

    return {**state, "context": context}


def _tickers_in_context(context: list[dict]) -> set[str]:
    """Collect tickers present in retrieved chunks."""
    return {str(chunk.get("ticker")) for chunk in context if chunk.get("ticker")}


def _years_in_context(context: list[dict]) -> set[int]:
    """Collect fiscal years present in retrieved chunks."""
    years: set[int] = set()
    for chunk in context:
        year = chunk.get("fiscal_year")
        if isinstance(year, int):
            years.add(year)
            continue
        try:
            years.add(int(year))
        except (TypeError, ValueError):
            pass
    return years


def _forms_in_context(context: list[dict]) -> set[str]:
    """Collect form types present in retrieved chunks."""
    return {str(chunk.get("form_type")) for chunk in context if chunk.get("form_type")}


def node_verify(state: RAGState) -> RAGState:
    """Check whether retrieved context is sufficient for the query intent."""
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
            checks["all_companies_present"] = all(ticker in found_tickers for ticker in tickers)
        elif intent in {"single_company_factual", "trend_over_time"}:
            checks["target_company_present"] = any(ticker in found_tickers for ticker in tickers)

    if intent == "trend_over_time":
        checks["multi_year"] = len(_years_in_context(context)) >= 2

    if intent == "single_company_factual":
        requested_year = filters.get("fiscal_year")
        context_years = _years_in_context(context)

        if requested_year is not None:
            requested_years = requested_year if isinstance(requested_year, list) else [requested_year]
            checks["correct_period"] = any(
                int(year) in context_years
                for year in requested_years
                if str(year).isdigit()
            )
        else:
            latest_year = plan.get("latest_year")
            if latest_year is not None:
                try:
                    checks["correct_period"] = int(latest_year) in context_years
                except (TypeError, ValueError):
                    pass

        requested_form = filters.get("form_type") or filters.get("source_class")
        if requested_form:
            requested_forms = requested_form if isinstance(requested_form, list) else [requested_form]
            context_forms = _forms_in_context(context)
            checks["correct_form"] = any(
                any(
                    requested.upper() in context_form.upper() or context_form.upper() in requested.upper()
                    for context_form in context_forms
                )
                for requested in requested_forms
            )

    if intent == "thematic_synthesis":
        checks["company_diversity"] = len(_tickers_in_context(context)) >= 2

    if intent == "investment_opinion":
        checks["company_diversity"] = len(_tickers_in_context(context)) >= max(2, len(tickers) or 2)

    passed = all(checks.values())
    current_retry_count = state.get("retry_count", 0)

    if not passed and current_retry_count < 2:
        log.info(
            "Verification failed %s — retrying with widened retrieval (attempt %d)",
            checks,
            current_retry_count + 1,
        )

        new_plan = copy.deepcopy(plan)
        new_filters = copy.deepcopy(new_plan.get("filters", {}))

        # Relax narrow year constraints first.
        new_filters.pop("fiscal_year", None)

        # Broaden annual/quarterly form constraints on retry.
        if intent in {"single_company_factual", "trend_over_time"}:
            current_form = new_filters.get("form_type")
            if current_form in {"10-K", "10-Q"}:
                new_filters["form_type"] = ["10-K", "10-Q"]

        # Remove ticker restriction for broader synthesis questions.
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
            "retry_count": current_retry_count + 1,
        }

    if not passed:
        log.warning(
            "Verification still failing after %d retries — proceeding to generate anyway: %s",
            current_retry_count,
            checks,
        )

    return {**state, "verification": checks}


def node_generate(state: RAGState) -> RAGState:
    """Generate the final answer from retrieved context."""
    return {
        **state,
        "result": generate(
            query=state["query"],
            context_chunks=state["context"],
        ),
    }


def should_retry(state: RAGState) -> str:
    """Route back to retrieval only when verification failed and a retry was scheduled."""
    verification = state.get("verification", {})
    retry_count = state.get("retry_count", 0)

    if verification and not all(verification.values()) and 0 < retry_count < 2:
        return "retrieve"
    return "generate"


def build_graph():
    """Build and compile the RAG workflow graph."""
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
    """Return a cached compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str) -> dict:
    """Execute the full RAG workflow for a single query."""
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
            "ticker": chunk.get("ticker"),
            "form": chunk.get("form_type"),
            "fy": chunk.get("fiscal_year"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "type": chunk.get("chunk_type"),
        }
        for chunk in final_state.get("context", [])
    ]

    return result