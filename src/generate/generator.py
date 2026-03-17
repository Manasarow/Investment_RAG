import os
import re
import logging
from typing import Any, Optional

from openai import OpenAI

log = logging.getLogger(__name__)

_client: OpenAI | None = None

INSUFFICIENT_EVIDENCE = "Insufficient evidence in retrieved documents to answer this question."

# Accepts:
# [AAPL 10-K FY2024, p.32]
# [AAPL 10-K/A FY2024, pp.32-33]
# [BRK-B 10-Q FY2025, p32]
CITATION_PATTERN = re.compile(
    r"\["
    r"(?P<ticker>[A-Z][A-Z0-9\-]*)\s+"
    r"(?P<form>[A-Za-z0-9\-\/]+)\s+"
    r"FY(?P<year>\d{4}),\s*"
    r"p{1,2}\.?\s*(?P<page_start>\d+)"
    r"(?:\s*[-–]\s*(?P<page_end>\d+))?"
    r"\]",
    re.IGNORECASE,
)

# Restrict faithfulness checks to financial-looking numbers to avoid false positives
# on years, page numbers, chunk counts, etc.
FINANCIAL_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:"
    r"\$?\d[\d,]*(?:\.\d+)?\s*(?:B|M|K|T)\b"
    r"|"
    r"\$?\d[\d,]*(?:\.\d+)?\s*(?:billion|million|thousand|trillion)\b"
    r"|"
    r"\d[\d,]*(?:\.\d+)?\s*%"
    r")",
    re.IGNORECASE,
)

# BUG-9 FIX: Additional normalisation patterns to catch semantically equivalent
# representations of the same number (e.g. "$2.3B" vs "2,300M" vs "2.3 billion").
# We normalise both the answer number and the context text before comparing, so
# that unit-scaled variants don't produce false hallucination alerts.
_UNIT_MULTIPLIERS = {
    "t": 1_000_000_000_000,
    "trillion": 1_000_000_000_000,
    "b": 1_000_000_000,
    "billion": 1_000_000_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "k": 1_000,
    "thousand": 1_000,
}

_NORMALISE_NUM_RE = re.compile(
    r"\$?([\d,]+(?:\.\d+)?)\s*(t|trillion|b|billion|m|million|k|thousand)?",
    re.IGNORECASE,
)


def _normalise_financial_number(raw: str) -> Optional[float]:
    """
    BUG-9 FIX: Convert a raw financial number string to a canonical float value
    so that "$2.3B" and "2,300M" both normalise to 2_300_000_000.0, preventing
    false hallucination positives when the LLM uses a different scale than the source.
    Returns None if the string cannot be parsed.
    """
    m = _NORMALISE_NUM_RE.search(raw.replace(",", ""))
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = (m.group(2) or "").lower()
    multiplier = _UNIT_MULTIPLIERS.get(unit, 1)
    return value * multiplier


# Bring Optional into scope for the type hint above (Python 3.9 compat)
from typing import Optional


SYSTEM_PROMPT = """You are a financial research assistant for investment professionals.

Rules:
1. Use ONLY the provided context. Do not use outside knowledge.
2. Every numeric claim MUST include a citation in the format [TICKER FORM FY{year}, p.{page}].
   When a source contains multiple rows for the same metric (e.g. Total revenue for different years),
   each row's header shows "metric: <name> (period ended YYYY-MM-DD)" — use the row whose period
   matches the year being asked about. If no period label is present, use the largest value as the
   most recent year unless context clearly indicates otherwise.
3. Preserve exact units and period labels from the source when possible.
4. For comparisons, address each company separately and cite each material claim.
5. For trend questions, present the evidence chronologically and cite each year/value.
6. If the context is insufficient, reply exactly:
"Insufficient evidence in retrieved documents to answer this question."
7. For investment-opinion style questions, provide only a factual comparison and end with:
"This is factual information only and not investment advice."
"""


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=key)
    return _client


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_context_block(chunks: list[dict], max_context_chars: int = 24000) -> str:
    """
    Format retrieved chunks into a bounded context block for the prompt.

    BUG-10 FIX: When truncation occurs, a warning is logged and the number of
    included vs total chunks is reported. Previously truncation was completely
    silent, making it impossible to diagnose cases where the model returned
    INSUFFICIENT_EVIDENCE because nearly all context had been dropped.
    """
    blocks: list[str] = []
    total = 0

    for i, chunk in enumerate(chunks):
        ticker = _clean_text(chunk.get("ticker", "?")) or "?"
        form = _clean_text(chunk.get("form_type", "?")) or "?"
        fy = _clean_text(chunk.get("fiscal_year", "?")) or "?"
        page = _clean_text(chunk.get("page", "?")) or "?"
        section = _clean_text(chunk.get("section", ""))
        chunk_type = _clean_text(chunk.get("chunk_type", ""))
        text = _clean_text(chunk.get("text", ""))

        label = f"[{ticker} {form} FY{fy}, p.{page}]"
        meta_bits = []
        if section and section.lower() != "general":
            meta_bits.append(section)
        if chunk_type:
            meta_bits.append(chunk_type)
        # For row-type chunks, include the row_label in the header so the LLM
        # can identify which metric each chunk contains without reading the full text.
        # This prevents "Insufficient evidence" when multiple same-page rows are
        # present but the LLM can't distinguish Revenue from Net Income by header alone.
        row_label = _clean_text(chunk.get("row_label", ""))
        col_header = _clean_text(chunk.get("col_header", ""))
        if row_label and chunk_type == "row":
            metric_label = f"metric: {row_label}"
            if col_header:
                # Use the column header directly when available (e.g. "Year ended June 30, 2024")
                metric_label += f" ({col_header})"
            else:
                # When col_header is missing (income statements store 3 years as separate
                # row chunks with no column label), extract the fiscal year from the chunk's
                # own fiscal_year field and append it so the LLM can distinguish
                # FY2022 / FY2023 / FY2024 rows that otherwise look identical.
                chunk_fy = _clean_text(chunk.get("fiscal_year", ""))
                chunk_period = _clean_text(chunk.get("period_end_date", ""))
                if chunk_period:
                    # period_end_date like "2024-06-30" — extract the year
                    year_hint = chunk_period[:4]
                    metric_label += f" (period ended {chunk_period})"
                elif chunk_fy:
                    metric_label += f" (FY{chunk_fy} filing)"
            meta_bits.append(metric_label)

        header = f"Source {i + 1} {label}"
        if meta_bits:
            header += " — " + " | ".join(meta_bits)

        block = f"{header}\n{text}"
        projected = total + len(block) + (6 if blocks else 0)
        if projected > max_context_chars:
            # BUG-10 FIX: log a warning so truncation is visible in the run logs
            # and can be diagnosed when the model returns INSUFFICIENT_EVIDENCE.
            log.warning(
                "Context truncated: included %d of %d chunks (%d chars used, limit %d). "
                "Consider increasing max_context_chars or reducing chunk verbosity.",
                len(blocks),
                len(chunks),
                total,
                max_context_chars,
            )
            break

        blocks.append(block)
        total = projected

    return "\n\n---\n\n".join(blocks)


def extract_citations(answer: str, context_chunks: list[dict]) -> list[dict]:
    """
    Parse citations from the answer and map them back to source chunks.

    BUG-8 FIX: Page numbers extracted from the citation regex are always strings
    (e.g. "41"), but chunk payloads store page as int (e.g. 41). The original
    code compared them with == which always returned False (str != int), so every
    citation had matched=False and chunk_id/source_url were always None.

    Fix: normalise both sides to str before comparing, so "41" == str(41) is True.
    """
    citations: list[dict] = []
    seen: set[tuple[str, str, str, str, str | None]] = set()

    for match in CITATION_PATTERN.finditer(answer):
        ticker = match.group("ticker").upper()
        form = match.group("form")
        year = match.group("year")
        page_start = match.group("page_start")
        page_end = match.group("page_end")

        dedup_key = (ticker, form.upper(), year, page_start, page_end)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        source = next(
            (
                chunk
                for chunk in context_chunks
                if _clean_text(chunk.get("ticker")).upper() == ticker
                and _clean_text(chunk.get("form_type")).upper() == form.upper()
                and _clean_text(chunk.get("fiscal_year")) == year
                and (
                    # BUG-8 FIX: compare as str on both sides — payload page may be
                    # an int while regex captures are always str.
                    str(chunk.get("page", "")) == str(page_start)
                    or (
                        page_end is not None
                        and str(chunk.get("page", "")) in {str(page_start), str(page_end)}
                    )
                )
            ),
            None,
        )

        label_page = page_start if page_end is None else f"{page_start}-{page_end}"
        citations.append(
            {
                "label": f"{ticker} {form} FY{year}, p.{label_page}",
                "chunk_id": source.get("chunk_id") if source else None,
                "source_url": source.get("source_url") if source else None,
                "matched": source is not None,
            }
        )

    return citations


# ---------------------------------------------------------------------------
# Unit-header and bare-number scaling helpers
# ---------------------------------------------------------------------------

# SEC '(In millions, except ...)' style headers.
_UNIT_HEADER_RE = re.compile(
    r"\(In\s+(millions?|billions?|thousands?)",
    re.IGNORECASE,
)

_HEADER_UNIT_MULTIPLIERS: dict[str, float] = {
    "million": 1_000_000.0, "millions": 1_000_000.0,
    "billion": 1_000_000_000.0, "billions": 1_000_000_000.0,
    "thousand": 1_000.0, "thousands": 1_000.0,
}

# Currency marker: Apple stores '| $ |' to signal values are in millions.
_CURRENCY_MARKER_RE = re.compile(r"\|\s*\$\s*\|", re.IGNORECASE)

# Bare integers with no unit suffix -- raw SEC table cell values.
# IMPORTANT: must use raw string r"" so \b is a word boundary, not backspace 0x08.
_BARE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z$%])\b(\d[\d,]*(?:\.\d+)?)\b"
    r"(?!\s*(?:B|M|K|T|%|billion|million|thousand|trillion)\b)",
    re.IGNORECASE,
)

_IMPLICIT_SCALES = [1_000_000.0, 1_000_000_000.0, 1_000.0]


def _extract_context_unit_scale(text: str) -> Optional[float]:
    """Return multiplier from an SEC '(In millions/billions/...)' header, or None."""
    m = _UNIT_HEADER_RE.search(text)
    if not m:
        return None
    return _HEADER_UNIT_MULTIPLIERS.get(m.group(1).lower())


def _build_scaled_context_values(context_chunks: list[dict]) -> set[float]:
    """
    Build canonical float values from bare integers in chunk text.

    Strategy 1 -- explicit '(In millions)' header: scale bare integers by the
    declared multiplier. Handles MSFT-style chunks.

    Strategy 2 -- no header: chunks with '| $ |' or large bare integers (>=1000)
    are tried at all common financial scales. Handles Apple-style chunks.
    """
    scaled: set[float] = set()
    for chunk in context_chunks:
        text = _clean_text(chunk.get("text", ""))
        explicit_scale = _extract_context_unit_scale(text)
        if explicit_scale is not None:
            for m in _BARE_NUMBER_RE.finditer(text):
                try:
                    scaled.add(float(m.group(1).replace(",", "")) * explicit_scale)
                except ValueError:
                    continue
        else:
            has_currency = bool(_CURRENCY_MARKER_RE.search(text))
            bare_matches = list(_BARE_NUMBER_RE.finditer(text))
            has_large = any(
                float(m.group(1).replace(",", "")) >= 1000
                for m in bare_matches
                if m.group(1).replace(",", "").replace(".", "").isdigit()
            )
            if not (has_currency or has_large):
                continue
            for m in bare_matches:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if val < 100:
                        continue
                    for scale in _IMPLICIT_SCALES:
                        scaled.add(val * scale)
                except ValueError:
                    continue
    return scaled


def verify_numeric_faithfulness(answer: str, context_chunks: list[dict]) -> list[str]:
    """
    Return financial-looking numbers in the answer that cannot be matched to
    any value in the retrieved context.

    Matching passes:
    1. Exact substring in raw context text.
    2. Normalised magnitude: '$2.3B' == '2,300M'.
    3. Header-scaled bare integer: '(In millions) 245,122' matches '$245,122 million'.
    4. Implicit-scale bare integer: Apple '| $ | 169,148' tried at all common scales.
    5. Percentage: '15%' matches bare '15' in context.
    """
    answer_matches = list(FINANCIAL_NUMBER_PATTERN.finditer(answer))
    if not answer_matches:
        return []

    context_text = " ".join(_clean_text(chunk.get("text", "")) for chunk in context_chunks)

    context_normalised: set[float] = set()
    for m in FINANCIAL_NUMBER_PATTERN.finditer(context_text):
        val = _normalise_financial_number(m.group(0))
        if val is not None:
            context_normalised.add(val)

    context_normalised |= _build_scaled_context_values(context_chunks)

    hallucinated: list[str] = []
    for m in answer_matches:
        raw = m.group(0).strip()
        if raw in context_text:
            continue
        normalised = _normalise_financial_number(raw)
        if normalised is not None and normalised in context_normalised:
            continue
        if raw.endswith("%"):
            bare = raw.rstrip("%").strip()
            if bare in context_text or f"{bare}%" in context_text:
                continue
            bare_norm = _normalise_financial_number(bare)
            if bare_norm is not None and bare_norm in context_normalised:
                continue
        hallucinated.append(raw)

    return sorted(set(hallucinated))

_GROUNDING_REMINDER = (
    "One or more numbers in your previous answer could not be verified against "
    "the provided context. Do NOT invent or infer numbers. If a number is not "
    "explicitly present in the context, omit it and say the context does not "
    "contain that figure. Revise your answer to include ONLY numbers that appear "
    "verbatim or equivalently in the context above."
)

_MAX_GENERATION_RETRIES = 2


def _call_llm(
    client, model: str, messages: list[dict], max_tokens: int, temperature: float
) -> str:
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def generate(
    query: str,
    context_chunks: list[dict],
    model: str = "gpt-4o",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> dict:
    """
    Generate a grounded answer from retrieved context.

    HALLUCINATION BLOCKING: If verify_numeric_faithfulness() flags unverifiable
    numbers, the LLM is re-prompted with a grounding reminder up to
    _MAX_GENERATION_RETRIES times. If numbers survive all retries, the answer
    is replaced with INSUFFICIENT_EVIDENCE.
    """
    if not context_chunks:
        return {"answer": INSUFFICIENT_EVIDENCE, "citations": [], "hallucinated_numbers": [], "context_count": 0, "model": model}

    context_block = format_context_block(context_chunks)
    if not context_block:
        return {"answer": INSUFFICIENT_EVIDENCE, "citations": [], "hallucinated_numbers": [], "context_count": 0, "model": model}

    user_prompt = (
        f"Question: {query}\n\nContext:\n{context_block}\n\n"
        "Answer the question using only the context above. "
        "Include citations for every material numeric claim."
    )

    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    answer = ""
    hallucinated: list[str] = []

    for attempt in range(_MAX_GENERATION_RETRIES + 1):
        try:
            answer = _call_llm(client, model, messages, max_tokens, temperature)
        except Exception as exc:
            log.exception("Generation failed on attempt %d", attempt)
            return {"answer": f"Generation error: {exc}", "citations": [], "hallucinated_numbers": [], "context_count": len(context_chunks), "model": model}

        hallucinated = verify_numeric_faithfulness(answer, context_chunks)
        if not hallucinated:
            break
        if attempt < _MAX_GENERATION_RETRIES:
            log.warning(
                "Hallucinated numbers detected on attempt %d: %s -- re-prompting with grounding reminder.",
                attempt, hallucinated,
            )
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": _GROUNDING_REMINDER})
        else:
            log.error(
                "Hallucinated numbers still present after %d retries: %s -- returning INSUFFICIENT_EVIDENCE.",
                _MAX_GENERATION_RETRIES, hallucinated,
            )
            citations = extract_citations(answer, context_chunks)
            return {"answer": INSUFFICIENT_EVIDENCE, "citations": citations, "hallucinated_numbers": hallucinated, "context_count": len(context_chunks), "model": model}

    citations = extract_citations(answer, context_chunks)
    return {
        "answer": answer or INSUFFICIENT_EVIDENCE,
        "citations": citations,
        "hallucinated_numbers": hallucinated,
        "context_count": len(context_chunks),
        "model": model,
    }