"""
Query planner for the investment RAG pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.index.qdrant_setup import COLLECTION_NAME, get_client

log = logging.getLogger(__name__)

COMPANY_TICKER_MAP: dict[str, str] = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "microsoft": "MSFT",
    "microsoft corp": "MSFT",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "amazon.com": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "jpmorgan chase": "JPM",
    "visa": "V",
    "unitedhealth": "UNH",
    "unitedhealth group": "UNH",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "mastercard": "MA",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "broadcom": "AVGO",
    "johnson": "JNJ",
    "johnson & johnson": "JNJ",
    "procter": "PG",
    "procter & gamble": "PG",
    "home depot": "HD",
    "costco": "COST",
    "merck": "MRK",
    "abbvie": "ABBV",
    "chevron": "CVX",
    "salesforce": "CRM",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "netflix": "NFLX",
    "thermo fisher": "TMO",
    "accenture": "ACN",
    "linde": "LIN",
    "mcdonald": "MCD",
    "bank of america": "BAC",
    "walmart": "WMT",
    "pepsi": "PEP",
    "oracle": "ORCL",
    "cisco": "CSCO",
    "abbott": "ABT",
    "ge aerospace": "GE",
    "general electric": "GE",
    "texas instruments": "TXN",
    "qualcomm": "QCOM",
    "ibm": "IBM",
    "servicenow": "NOW",
    "morgan stanley": "MS",
    "goldman sachs": "GS",
    "goldman": "GS",
    "intuit": "INTU",
    "applied materials": "AMAT",
    "s&p global": "SPGI",
    "sp global": "SPGI",
    "blackrock": "BLK",
    "caterpillar": "CAT",
    "rtx": "RTX",
    "honeywell": "HON",
}

ALL_TICKERS = set(COMPANY_TICKER_MAP.values())
_SINGLE_CHAR_TICKERS = {ticker for ticker in ALL_TICKERS if len(ticker) == 1}

_ALIAS_PATTERNS: list[tuple[re.Pattern[str], str]] = sorted(
    [
        (
            re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.I),
            ticker,
        )
        for alias, ticker in COMPANY_TICKER_MAP.items()
    ],
    key=lambda item: -len(item[0].pattern),
)

SECTOR_ALIASES: dict[str, str] = {
    "technology": "Technology",
    "tech": "Technology",
    "software": "Technology",
    "semiconductor": "Technology",
    "semiconductors": "Technology",
    "chip": "Technology",
    "chips": "Technology",
    "financial sector": "Financial Services",
    "financial services": "Financial Services",
    "financial": "Financial Services",
    "bank": "Financial Services",
    "banks": "Financial Services",
    "banking": "Financial Services",
    "healthcare": "Healthcare",
    "health care": "Healthcare",
    "pharma": "Healthcare",
    "biotech": "Healthcare",
    "energy": "Energy",
    "oil": "Energy",
    "gas": "Energy",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "communication services": "Communication Services",
    "media": "Communication Services",
    "consumer staples": "Consumer Staples",
    "consumer discretionary": "Consumer Discretionary",
    "retail": "Consumer Discretionary",
    "industrials": "Industrials",
    "industrial": "Industrials",
    "materials": "Basic Materials",
    "basic materials": "Basic Materials",
}

NUMERIC_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\brevenue\b",
        r"\bsegment(?:s)?\b",
        r"\bnet\s+income\b",
        r"\bgross\s+margin\b",
        r"\bmargin(?:s)?\b",
        r"\bdebt\b",
        r"\bcash\b",
        r"\bassets?\b",
        r"\bliabilit(?:y|ies)\b",
        r"\beps\b",
        r"\bearn(?:ing|ings)?\b",
        r"\bcapex\b",
        r"\bdividend(?:s)?\b",
        r"\bbuyback(?:s)?\b",
        r"\br&d\b",
        r"\bresearch\s+and\s+development\b",
        r"\boperating\s+income\b",
        r"\bcash\s+flow(?:s)?\b",
        r"\bbalance\s+sheet\b",
    ]
]

COMPARISON_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bbetween\b",
        r"\brelative\s+to\b",
        r"\bcompared\s+with\b",
        r"\bpeer(?:s)?\b",
        r"\bamong\b",
    ]
]

TREND_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\btrend(?:ed|ing)?\b",
        r"\bover\s+time\b",
        r"\bpast\s+\d+\s+(?:fiscal\s+)?years?\b",
        r"\blast\s+\d+\s+(?:fiscal\s+)?years?\b",
        r"\bhistor(?:y|ical|ically)\b",
        r"\byoy\b",
        r"\byear\s+over\s+year\b",
        r"\bgrow(?:th|ing|n)?\b",
        r"\bdeclin(?:e|ed|ing)\b",
        r"\bincreas(?:e|ed|ing)\b",
        r"\bchang(?:e|ed|ing)\b",
        r"\btrended\b",
    ]
]

THEMATIC_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bhow\s+are\b",
        r"\bdiscuss(?:ing|ed)?\b",
        r"\btalk(?:ing)?\s+about\b",
        r"\bmention(?:ed|ing)?\b",
        r"\bcommentary\b",
        r"\btheme(?:s)?\b",
        r"\bai\b",
        r"\bartificial\s+intelligence\b",
        r"\badoption\b",
        r"\bdemand\b",
        r"\boutlook\b",
        r"\bguidance\b",
        r"\bmacro\b",
        r"\bstrategy\b",
        r"\bacross\b",
        r"\bsector\b",
        r"\bindustry\b",
        r"\bcompanies\b",
        r"\brecent\s+performance\b",
    ]
]

INVESTMENT_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bshould\s+i\s+invest\b",
        r"\bwhich\s+should\s+i\s+invest\b",
        r"\bworth\s+investing\b",
        r"\binvest(?:ing|ment)?\b",
        r"\bbuy\b",
        r"\bsell\b",
        r"\brecommend(?:ation|ed)?\b",
        r"\bportfolio\b",
        r"\bvaluation\b",
        r"\bbest\s+stock\b",
    ]
]

SEGMENT_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bsegment(?:s)?\b",
        r"\bbreak\s*down\b",
        r"\bbreakdown\b",
        r"\bby\s+segment\b",
        r"\bgeograph(?:y|ic)\b",
        r"\bbusiness\s+unit(?:s)?\b",
    ]
]

ANNUAL_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bfy\s*20\d{2}\b",
        r"\bfiscal\s+year\b",
        r"\bannual\b",
        r"\b10-?k\b",
        r"\blast\s+reported\s+(?:fiscal\s+year|fy)\b",
        r"\blatest\s+(?:fiscal\s+year|fy|annual)\b",
        r"\b(?:past|last)\s+\d+\s+fiscal\s+years?\b",
        r"\bover\s+the\s+(?:past|last)\s+\d+\s+(?:fiscal\s+)?years?\b",
    ]
]

QUARTERLY_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bquarter(?:ly)?\b",
        r"\bq[1-4]\b",
        r"\b10-?q\b",
        r"\bthree\s+months\s+ended\b",
        r"\bnine\s+months\s+ended\b",
        r"\blatest\s+quarter\b",
        r"\bcurrent\s+quarter\b",
    ]
]

EARNINGS_RELEASE_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bearnings\s+release\b",
        r"\bresults\s+of\s+operations\b",
        r"\b8-?k\b",
        r"\brecent\s+performance\b",
    ]
]

LATEST_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\blatest\b",
        r"\bmost\s+recent\b",
        r"\bcurrent\b",
        r"\blast\s+reported\b",
        r"\brecent\b",
    ]
]

LAST_N_YEARS_RE = re.compile(r"\b(?:last|past)\s+(\d+)\s+(?:fiscal\s+)?years?\b", re.I)
EXPLICIT_YEAR_RE = re.compile(r"\b(?:FY\s*)?(20\d{2})\b", re.I)


def _contains_any(query: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    """Return True if any pattern matches the query."""
    return any(pattern.search(query) for pattern in patterns)


def _extract_sector(query: str) -> Optional[str]:
    """Extract a normalized sector label from the query."""
    q = query.lower()
    for alias, sector in sorted(SECTOR_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", q, re.I):
            return sector
    return None


def _available_fiscal_years(
    tickers_key: tuple[str, ...] = (),
    sector: Optional[str] = None,
) -> tuple[int, ...]:
    """Read available fiscal years from the live Qdrant index."""
    try:
        client = get_client()
    except Exception as exc:
        log.warning("Could not connect to Qdrant for fiscal-year discovery: %s", exc)
        return tuple()

    must = []
    if tickers_key:
        if len(tickers_key) == 1:
            must.append(FieldCondition(key="ticker", match=MatchValue(value=tickers_key[0])))
        else:
            must.append(FieldCondition(key="ticker", match=MatchAny(any=list(tickers_key))))
    if sector:
        must.append(FieldCondition(key="sector", match=MatchValue(value=sector)))

    qfilter = Filter(must=must) if must else None
    years: set[int] = set()
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qfilter,
            with_vectors=False,
            with_payload=["fiscal_year"],
            limit=256,
            offset=offset,
        )
        for record in records:
            fiscal_year = record.payload.get("fiscal_year") if record.payload else None
            try:
                if fiscal_year is not None:
                    years.add(int(fiscal_year))
            except (TypeError, ValueError):
                continue

        if offset is None or len(years) >= 8:
            break

    return tuple(sorted(years, reverse=True))


def _latest_year_for_scope(tickers: list[str], sector: Optional[str]) -> Optional[int]:
    """Return the latest available fiscal year for the current scope."""
    years = _available_fiscal_years(tuple(sorted(tickers)), sector)
    return years[0] if years else None


def extract_tickers(query: str) -> list[str]:
    """Extract ticker symbols and company aliases from the query."""
    found: set[str] = set()

    for pattern, ticker in _ALIAS_PATTERNS:
        if pattern.search(query):
            found.add(ticker)

    for token in re.findall(r"\b[A-Z]{1,5}(?:-[A-Z])?\b", query):
        if token in ALL_TICKERS:
            if len(token) == 1 and token not in _SINGLE_CHAR_TICKERS:
                continue
            found.add(token)

    return sorted(found)


def extract_fiscal_years(
    query: str,
    *,
    tickers: Optional[list[str]] = None,
    sector: Optional[str] = None,
) -> list[int]:
    """Extract explicit or relative fiscal-year references from the query."""
    years = sorted({int(match.group(1)) for match in EXPLICIT_YEAR_RE.finditer(query)}, reverse=True)
    latest_year = _latest_year_for_scope(tickers or [], sector)

    match = LAST_N_YEARS_RE.search(query)
    if match:
        count = max(1, min(10, int(match.group(1))))
        if years:
            base = max(years)
            return sorted([base - i for i in range(count)], reverse=True)

        available = list(_available_fiscal_years(tuple(sorted(tickers or [])), sector))
        return available[:count]

    if years:
        return years

    if _contains_any(query, LATEST_PATTERNS):
        if latest_year is not None:
            return [int(latest_year)]

        available = list(_available_fiscal_years(tuple(sorted(tickers or [])), sector))
        if available:
            return [int(available[0])]

    return []


def _detect_report_scope(query: str) -> dict:
    """Infer whether the query prefers annual, quarterly, or event-driven filings."""
    q = query.lower()

    annual = _contains_any(q, ANNUAL_PATTERNS)
    quarterly = _contains_any(q, QUARTERLY_PATTERNS)
    earnings_release = _contains_any(q, EARNINGS_RELEASE_PATTERNS)

    scope = "either"
    preferred_sources: list[str] = []
    preferred_forms: list[str] = []

    if annual and not quarterly:
        scope = "annual"
        preferred_sources = ["10-K"]
        preferred_forms = ["10-K", "10-K405", "10-KSB"]
    elif quarterly and not annual:
        scope = "quarterly"
        preferred_sources = ["10-Q", "8-K-earnings"] if earnings_release else ["10-Q"]
        preferred_forms = ["10-Q", "8-K"] if earnings_release else ["10-Q"]
    elif earnings_release:
        scope = "event"
        preferred_sources = ["8-K-earnings"]
        preferred_forms = ["8-K"]

    return {
        "scope": scope,
        "preferred_sources": preferred_sources,
        "preferred_forms": preferred_forms,
        "prefer_annual": scope == "annual",
        "prefer_quarterly": scope == "quarterly",
    }


def _detect_evidence_profile(query: str) -> dict:
    """Infer what kind of evidence the retriever should bias toward."""
    q = query.lower()
    numeric = _contains_any(q, NUMERIC_PATTERNS)
    segment = _contains_any(q, SEGMENT_PATTERNS)
    thematic = _contains_any(q, THEMATIC_PATTERNS)

    preferred_chunk_types: list[str] = []
    preferred_statement_types: list[str] = []

    if numeric:
        preferred_chunk_types.extend(["row", "table", "micro_block"])
    if segment:
        preferred_statement_types.append("segment_table")
    if thematic:
        preferred_chunk_types.append("prose")

    return {
        "numeric": numeric,
        "segment": segment,
        "thematic": thematic,
        "preferred_chunk_types": list(dict.fromkeys(preferred_chunk_types)),
        "preferred_statement_types": list(dict.fromkeys(preferred_statement_types)),
    }


def _derive_intent(query: str, tickers: list[str], evidence: dict) -> str:
    """Map query signals into the legacy intent label used downstream."""
    q = query.lower()
    investment = _contains_any(q, INVESTMENT_PATTERNS)
    comparison = _contains_any(q, COMPARISON_PATTERNS)
    trend = _contains_any(q, TREND_PATTERNS)
    thematic = _contains_any(q, THEMATIC_PATTERNS)

    if investment:
        return "investment_opinion"

    if thematic and (len(tickers) == 0 or len(tickers) > 2 or "across" in q or "sector" in q or "industry" in q):
        return "thematic_synthesis"

    if trend:
        return "trend_over_time"

    if len(tickers) > 1 and (comparison or evidence.get("numeric")):
        return "cross_company_comparison"

    if thematic:
        return "thematic_synthesis"

    return "single_company_factual"


def classify_intent(query: str) -> str:
    """Public helper for intent-only classification."""
    tickers = extract_tickers(query)
    evidence = _detect_evidence_profile(query)
    return _derive_intent(query, tickers, evidence)


def _generic_enrichment(query: str, evidence: dict) -> str:
    """Append retrieval hints that improve finance-specific recall."""
    additions: list[str] = []

    if evidence.get("segment"):
        additions.append("segment revenue operating income business units geography")
    if evidence.get("numeric"):
        additions.append("financial statement table row values")

    if not additions:
        return query

    return f"{query} {' '.join(additions)}"


def _latest_years_by_ticker(tickers: list[str]) -> dict[str, int]:
    """Return the latest indexed fiscal year for each ticker."""
    result: dict[str, int] = {}
    for ticker in tickers:
        years = _available_fiscal_years((ticker,), None)
        if years:
            result[ticker] = years[0]
    return result


def plan_retrieval(query: str) -> dict:
    """Build the retrieval plan consumed by the retriever."""
    tickers = extract_tickers(query)
    sector = _extract_sector(query)
    report_scope = _detect_report_scope(query)
    evidence = _detect_evidence_profile(query)
    fiscal_years = extract_fiscal_years(query, tickers=tickers, sector=sector)
    intent = _derive_intent(query, tickers, evidence)

    latest_year = _latest_year_for_scope(tickers, sector)
    latest_years_by_ticker = _latest_years_by_ticker(tickers) if tickers else {}
    retrieval_query = _generic_enrichment(query, evidence)

    filters: dict = {}
    if tickers:
        filters["ticker"] = tickers if len(tickers) > 1 else tickers[0]
    if fiscal_years:
        filters["fiscal_year"] = fiscal_years if len(fiscal_years) > 1 else fiscal_years[0]
    if sector:
        filters["sector"] = sector
    if report_scope["preferred_sources"]:
        filters["source_class"] = (
            report_scope["preferred_sources"]
            if len(report_scope["preferred_sources"]) > 1
            else report_scope["preferred_sources"][0]
        )

    if evidence["preferred_statement_types"] and intent == "single_company_factual":
        filters["statement_type"] = (
            evidence["preferred_statement_types"]
            if len(evidence["preferred_statement_types"]) > 1
            else evidence["preferred_statement_types"][0]
        )

    dense_top_k = 50
    sparse_top_k = 50
    reranker_k = 12
    final_k = 8

    if intent == "single_company_factual":
        dense_top_k = 36
        sparse_top_k = 36
        reranker_k = 10
        final_k = 6
    elif intent == "cross_company_comparison":
        dense_top_k = 72
        sparse_top_k = 72
        reranker_k = 18
        final_k = max(8, 2 * max(2, len(tickers)))
    elif intent == "trend_over_time":
        dense_top_k = 60
        sparse_top_k = 60
        reranker_k = 16
        final_k = 8
        if not fiscal_years:
            available = list(_available_fiscal_years(tuple(sorted(tickers)), sector))
            if available:
                filters["fiscal_year"] = available[:3]
    elif intent == "thematic_synthesis":
        dense_top_k = 90
        sparse_top_k = 90
        reranker_k = 22
        final_k = 12
    elif intent == "investment_opinion":
        dense_top_k = 96
        sparse_top_k = 96
        reranker_k = 24
        final_k = max(10, 2 * max(2, len(tickers)))

    q_lower = query.lower()
    dimensions = {
        "intent": intent,
        "entity_scope": "multi" if len(tickers) > 1 else ("single" if tickers else "broad"),
        "report_scope": report_scope["scope"],
        "time_scope": "specific_years" if fiscal_years else ("latest" if _contains_any(q_lower, LATEST_PATTERNS) else "unspecified"),
        "comparison": _contains_any(q_lower, COMPARISON_PATTERNS),
        "trend": _contains_any(q_lower, TREND_PATTERNS),
        "thematic": evidence["thematic"],
        "numeric": evidence["numeric"],
        "segment": evidence["segment"],
        "investment": _contains_any(q_lower, INVESTMENT_PATTERNS),
        "sector": sector,
    }

    plan = {
        "query": retrieval_query,
        "intent": intent,
        "tickers": tickers,
        "fiscal_years": fiscal_years,
        "filters": filters,
        "dense_top_k": dense_top_k,
        "sparse_top_k": sparse_top_k,
        "reranker_k": reranker_k,
        "final_k": final_k,
        "dimensions": dimensions,
        "report_scope": report_scope,
        "evidence_profile": evidence,
        "entity_scope": dimensions["entity_scope"],
        "sector": sector,
        "latest_year": latest_year,
        "latest_years_by_ticker": latest_years_by_ticker,
        "retrieval_hints": {
            "prefer_chunk_types": evidence["preferred_chunk_types"],
            "prefer_statement_types": evidence["preferred_statement_types"],
            "prefer_source_classes": report_scope["preferred_sources"],
            "prefer_form_types": report_scope["preferred_forms"],
            "enable_multi_hop": intent in {"cross_company_comparison", "thematic_synthesis", "investment_opinion"} or len(tickers) > 1,
            "require_company_diversity": intent in {"cross_company_comparison", "thematic_synthesis", "investment_opinion"},
            "require_multi_year": intent == "trend_over_time",
            "latest_bias": _contains_any(q_lower, LATEST_PATTERNS),
            "numeric_bias": evidence["numeric"],
        },
    }

    log.info(
        "Query plan: intent=%s | tickers=%s | years=%s | sector=%s | filters=%s",
        intent,
        tickers,
        fiscal_years,
        sector,
        filters,
    )
    return plan