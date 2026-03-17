import os
import re
import logging
from typing import Any, Optional

from openai import OpenAI

log = logging.getLogger(__name__)

_client: OpenAI | None = None

INSUFFICIENT_EVIDENCE = "Insufficient evidence in retrieved documents to answer this question."

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

_UNIT_HEADER_RE = re.compile(
    r"\(In\s+(millions?|billions?|thousands?)",
    re.IGNORECASE,
)

_HEADER_UNIT_MULTIPLIERS: dict[str, float] = {
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
    "thousand": 1_000.0,
    "thousands": 1_000.0,
}

_CURRENCY_MARKER_RE = re.compile(r"\|\s*\$\s*\|", re.IGNORECASE)

_BARE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z$%])\b(\d[\d,]*(?:\.\d+)?)\b"
    r"(?!\s*(?:B|M|K|T|%|billion|million|thousand|trillion)\b)",
    re.IGNORECASE,
)

_IMPLICIT_SCALES = [1_000_000.0, 1_000_000_000.0, 1_000.0]

_GROUNDING_REMINDER = (
    "One or more numbers in your previous answer could not be verified against "
    "the provided context. Do NOT invent or infer numbers. If a number is not "
    "explicitly present in the context, omit it and say the context does not "
    "contain that figure. Revise your answer to include ONLY numbers that appear "
    "verbatim or equivalently in the context above."
)

_MAX_GENERATION_RETRIES = 2


def get_client() -> OpenAI:
    """Return a cached OpenAI client."""
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=key)
    return _client


def _clean_text(value: Any) -> str:
    """Normalize nullable values into stripped strings."""
    if value is None:
        return ""
    return str(value).strip()


def _normalise_financial_number(raw: str) -> Optional[float]:
    """Convert a financial number string into a canonical float."""
    match = _NORMALISE_NUM_RE.search(raw.replace(",", ""))
    if not match:
        return None

    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None

    unit = (match.group(2) or "").lower()
    multiplier = _UNIT_MULTIPLIERS.get(unit, 1)
    return value * multiplier


def format_context_block(chunks: list[dict], max_context_chars: int = 24000) -> str:
    """Format retrieved chunks into a bounded prompt context."""
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

        meta_bits: list[str] = []
        if section and section.lower() != "general":
            meta_bits.append(section)
        if chunk_type:
            meta_bits.append(chunk_type)

        row_label = _clean_text(chunk.get("row_label", ""))
        col_header = _clean_text(chunk.get("col_header", ""))

        if row_label and chunk_type == "row":
            metric_label = f"metric: {row_label}"
            if col_header:
                metric_label += f" ({col_header})"
            else:
                chunk_fy = _clean_text(chunk.get("fiscal_year", ""))
                chunk_period = _clean_text(chunk.get("period_end_date", ""))
                if chunk_period:
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
            log.warning(
                "Context truncated: included %d of %d chunks (%d chars used, limit %d).",
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
    """Parse answer citations and map them back to retrieved chunks."""
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


def _extract_context_unit_scale(text: str) -> Optional[float]:
    """Extract numeric scale from SEC unit headers like '(In millions)'."""
    match = _UNIT_HEADER_RE.search(text)
    if not match:
        return None
    return _HEADER_UNIT_MULTIPLIERS.get(match.group(1).lower())


def _build_scaled_context_values(context_chunks: list[dict]) -> set[float]:
    """Build normalized numeric values from chunk text, including scaled bare numbers."""
    scaled: set[float] = set()

    for chunk in context_chunks:
        text = _clean_text(chunk.get("text", ""))
        explicit_scale = _extract_context_unit_scale(text)

        if explicit_scale is not None:
            for match in _BARE_NUMBER_RE.finditer(text):
                try:
                    scaled.add(float(match.group(1).replace(",", "")) * explicit_scale)
                except ValueError:
                    continue
            continue

        has_currency = bool(_CURRENCY_MARKER_RE.search(text))
        bare_matches = list(_BARE_NUMBER_RE.finditer(text))
        has_large = any(
            float(match.group(1).replace(",", "")) >= 1000
            for match in bare_matches
            if match.group(1).replace(",", "").replace(".", "").isdigit()
        )

        if not (has_currency or has_large):
            continue

        for match in bare_matches:
            try:
                value = float(match.group(1).replace(",", ""))
            except ValueError:
                continue

            if value < 100:
                continue

            for scale in _IMPLICIT_SCALES:
                scaled.add(value * scale)

    return scaled


def verify_numeric_faithfulness(answer: str, context_chunks: list[dict]) -> list[str]:
    """Return answer numbers that cannot be verified from the retrieved context."""
    answer_matches = list(FINANCIAL_NUMBER_PATTERN.finditer(answer))
    if not answer_matches:
        return []

    context_text = " ".join(_clean_text(chunk.get("text", "")) for chunk in context_chunks)

    context_normalised: set[float] = set()
    for match in FINANCIAL_NUMBER_PATTERN.finditer(context_text):
        value = _normalise_financial_number(match.group(0))
        if value is not None:
            context_normalised.add(value)

    context_normalised |= _build_scaled_context_values(context_chunks)

    hallucinated: list[str] = []
    for match in answer_matches:
        raw = match.group(0).strip()

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


def _call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    """Call the chat completions API and return the assistant text."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def generate(
    query: str,
    context_chunks: list[dict],
    model: str = "gpt-4o",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> dict:
    """Generate a grounded answer from retrieved chunks."""
    if not context_chunks:
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "citations": [],
            "hallucinated_numbers": [],
            "context_count": 0,
            "model": model,
        }

    context_block = format_context_block(context_chunks)
    if not context_block:
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "citations": [],
            "hallucinated_numbers": [],
            "context_count": 0,
            "model": model,
        }

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
            return {
                "answer": f"Generation error: {exc}",
                "citations": [],
                "hallucinated_numbers": [],
                "context_count": len(context_chunks),
                "model": model,
            }

        hallucinated = verify_numeric_faithfulness(answer, context_chunks)
        if not hallucinated:
            break

        if attempt < _MAX_GENERATION_RETRIES:
            log.warning(
                "Hallucinated numbers detected on attempt %d: %s",
                attempt,
                hallucinated,
            )
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": _GROUNDING_REMINDER})
        else:
            log.error(
                "Hallucinated numbers still present after %d retries: %s",
                _MAX_GENERATION_RETRIES,
                hallucinated,
            )
            return {
                "answer": INSUFFICIENT_EVIDENCE,
                "citations": extract_citations(answer, context_chunks),
                "hallucinated_numbers": hallucinated,
                "context_count": len(context_chunks),
                "model": model,
            }

    return {
        "answer": answer or INSUFFICIENT_EVIDENCE,
        "citations": extract_citations(answer, context_chunks),
        "hallucinated_numbers": hallucinated,
        "context_count": len(context_chunks),
        "model": model,
    }