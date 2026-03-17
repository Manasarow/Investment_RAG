"""
Hierarchical, layout-aware chunker for parsed SEC filing documents.
Handles prose, tables, footnotes, and figure captions as distinct chunk types.

── Bug-tracker fixes (P2 priority) ──────────────────────────────────────────
  #19  Manifest O(N²) full rewrite eliminated: load once / mutate in-memory /
       write once at end of run(), following the pattern from sec_downloader.py
       and docling_parser.py.  update_manifest_status() is now an in-memory
       operation on the shared entries list; the file is written once by run().

  #77  Section parser is now form-aware.  10-K and 10-Q use different Item
       numbering (e.g. 10-Q Item 2 = MD&A, not Properties).  SEC_SECTION_PATTERNS
       is replaced by per-form tables (SECTION_MAP_10K, SECTION_MAP_10Q,
       SECTION_MAP_8K) and a form-agnostic fallback for subsection phrases.
       The form is derived from doc_meta and passed into infer_section().

  #20  AVG_CHARS_PER_TOK removed.  Prose splitting now uses a real tiktoken
       encoder (cl100k_base) when available, falling back to a conservative
       4.5-char estimate only when tiktoken is not installed.  Token counts
       are computed on the actual text being split so truncation at the
       embedder is eliminated.

  #21  assign_pseudo_pages() is disabled for real PDF files.
       page_is_surrogate=False means the block already has correct page
       numbers from Docling provenance.  Pseudo-page assignment now only
       fires when page_is_surrogate=True (HTM files), and it is applied to
       BOTH prose_blocks and table_records together so block_order ordering
       is preserved.  Real PDF pages are never overwritten.

  #22  Table header duplication fixed.  The table_header string no longer
       contains form_type twice.  Fixed from:
           "{company} | {form_type} {form_type} FY{year} | {section}"
       to:
           "{company} ({ticker}) | {form_type} FY{year} | {section}"

  #23  Large table splits now carry the column-header line in every row-group.
       When retrieval_text is split into row-groups, the first line (header line)
       is extracted separately and prepended to every group so group 2+ are not
       headerless.

  #24  Section inference (infer_section) is now gated: it only fires on blocks
       whose text is SHORT (≤200 chars) OR whose first token is an Item/PART
       heading prefix.  Body-text paragraphs that happen to contain the words
       "risk factors" deep in a sentence will no longer hijack current_section.

── Bug-tracker fixes (P5 priority) ──────────────────────────────────────────
  #79  Row-level chunks added.  For every table record that is a financial
       statement (is_financial_statement=True) and has a non-empty cell_grid,
       each cell_grid entry is emitted as a normalised row-level chunk with
       schema: {chunk_type='row', row_label, col_header, value, unit,
       statement_type, period_type, ...doc_meta}.  These tiny chunks are
       separately indexable for precise numeric lookup queries.

  #80  Period labels are preserved for every row-level chunk.  The table
       record's period_type, period_signals, and periods fields are carried
       into each row chunk so the retriever can filter by period without
       scanning the full table chunk.

  #78  HTM citation key changed from synthetic page number to
       section + block_order.  When page_is_surrogate=True the chunk receives
       citation_key = "{section} §{block_order}" instead of "p.{page}".
       This makes citations meaningful and reproducible for HTM filings.

  #25  Overlap logic fixed for long sentences.  The previous code would carry
       zero overlap when the last sentence exceeded overlap_chars on its own.
       The new logic always carries the LAST sentence regardless of its length,
       capped at 1.5× overlap_chars, so no chunk is ever context-free.

  #26  Parent-child chunk relationships added.  Every prose split-chunk carries
       parent_chunk_id pointing to the first chunk produced from that source
       block.  Table row-group chunks carry parent_chunk_id pointing to the
       first row-group of the same table.  Row-level chunks carry
       parent_chunk_id pointing to the table chunk they were extracted from.

── Bug-tracker fixes (P7 priority) ──────────────────────────────────────────
  #27  Intra-document chunk linking: each chunk carries a next_chunk_id and
       prev_chunk_id field populated in a post-processing pass over the ordered
       chunk list so downstream agents can walk the document linearly.

  #28  chunk_index now reflects true document reading order.  Prose blocks and
       table records are interleaved by block_order before chunking so that
       chunk_index monotonically follows the order a reader would encounter
       content in the source document.

── Post-review fixes (reviewer critique, 8 issues) ──────────────────────────
  R-C1  Prose splitter fallback for SEC-style text lacking sentence punctuation.
        split_prose() now has a three-stage cascade:
          1. Split by sentence-ending punctuation (existing)
          2. If any resulting segment still exceeds max_tokens, sub-split by
             semicolons / colons / newlines (catches bullet lists, long legal
             paragraphs with semicolons, enumerations without full stops)
          3. If a sub-segment still exceeds max_tokens, hard-split by tokens
             using a greedy word-boundary split
        This prevents any chunk from silently exceeding the embedding limit.

  R-C2  Table row-grouping is now token-budget-aware, not fixed row-count.
        _split_rows_by_token_budget() accumulates rows until the group would
        exceed TABLE_CHUNK_MAX_TOKENS (default 400), then starts a new group.
        ROW_GROUP_SIZE (30) is retained only as a hard backstop guard against
        pathological tables.  Table-chunk embedding length is now stable.

  R-C3  ROW_CHUNK_MIN_ROWS heuristic replaced with a richer eligibility rule.
        Row chunks are now emitted when ANY of the following is true:
          (a) is_financial_statement=True  (covers income/balance/cash/equity)
          (b) statement_type is in a set of high-value note types
              (debt_table, segment_table, eps_table, compensation_table)
          (c) cell_grid has numeric density >= 0.4 and at least 1 row
        This ensures compact but important tables (debt maturities, EPS,
        segment revenue mini-tables) still emit row-level chunks.

  R-C4  Prose and table chunk deduplication added.
        _content_hash() computes a normalised hash (lowercased, whitespace-
        collapsed) of each chunk's text before it is appended to the output.
        Duplicate hashes are skipped with a debug log.  This removes repeated
        safe-harbour boilerplate, HTML-parser duplicate blocks, and repeated
        exhibit headers without touching distinct but thematically similar text.

  R-C5  Table section propagation made more conservative.
        chunk_document() only allows a table to advance current_section when
        the table record carries an explicit parser-supplied section that is
        not "General" or empty.  Tables that inherit current_section from
        context (section == current_section) do NOT update current_section
        so a poorly-labelled table cannot contaminate subsequent prose sections.

  R-C6  Footnote detection extended to multi-paragraph and note-reference patterns.
        is_footnote() now recognises:
          (a) Classic prefix patterns: (1), 1., a)  (existing, extended to 600 chars)
          (b) Parenthetical note references inline: text that is ≤120 chars and
              consists entirely of a parenthetical qualifier like "(in millions)"
          (c) Superscript-style short footnotes: lines beginning with ¹²³ or
              Unicode superscripts followed by text
        The length cap is raised from 400 to 600 chars to capture multi-sentence
        footnotes that appear frequently in balance-sheet notes.

  R-C7  Surrogate pagination now inspects both prose and table records to detect
        HTM mode, not just the first prose block.
        If ALL blocks (prose + table) have page_is_surrogate=True the document
        is treated as HTM.  If even one block has page_is_surrogate=False the
        document is treated as PDF and pseudo-page assignment is skipped.
        This handles table-heavy HTM filings that have no prose blocks.

── Second-round review fixes (8 issues) ─────────────────────────────────────
  Fix-1  Deduplication moved BEFORE adjacency linking.
         Previously: link → dedup.  If B was deduped away, A.next still pointed
         to B and C.prev still pointed to B — broken traversal.
         Now: dedup → link.  Links are built on the survivor list only.

  Fix-2  micro_block excluded from linear adjacency chain.
         _link_adjacent_chunks() now filters out both "row" and "micro_block".
         Micro-blocks are parallel table-derived lookup targets, not narrative
         reading-order content.  An agent walking next_chunk_id will no longer
         traverse prose → table → micro_block → prose.

  Fix-3  _hard_split_by_tokens() handles single ultra-long "words".
         If one whitespace-delimited token exceeds max_tokens (malformed OCR,
         giant pipe-delimited artefact, broken HTML extraction), it is now
         character-sliced: token-ID sliced via tiktoken when available, or
         character-sliced via the fallback estimate.  The "guarantees ≤ max_tokens"
         claim is now actually true.

  Fix-4  _NUM_VAL_RE replaced with _is_numeric_cell() — a broader detector.
         Old regex missed: (1,234.5), $ (1,234), 1.2x, 3.5 million, 12 months,
         dash placeholders (—, -), basis-point formats with spaces.
         New detector: any cell containing at least one digit with ≤ 12 chars of
         alphabetic noise is classified as numeric.  High recall, minimal false
         negatives.

  Fix-5  _content_hash() now includes chunk_type and citation_key alongside text.
         Previously: two chunks with identical text but different provenance
         (same disclosure in different sections, same safe-harbor text on two
         pages) would collapse into one.  Now provenance is part of the key so
         only true duplicates (same text + same type + same location) are removed.

  Fix-6  Row chunk text enriched with period semantics.
         Old: "Company (TICK) FORM FY2024 | statement_type | row_label | col: val"
         New: adds a period_context field derived from period_type, fiscal_quarter,
         period_end_date, and units.  Example addition: "quarterly Q3 | period
         ended 2024-06-29 | $ millions".  Helps retrieval for "latest quarter",
         "FY2024", "three months ended", and similar temporal queries.

  Fix-7  split_prose() chunk overflow fixed at both emission sites.
         There were two sources of over-budget chunks:
         (a) Overlap carry-forward: `overlap_segs + [seg]` could exceed max_tokens
             when seg was near the limit.  Fixed by trimming overlap_segs from the
             front until the assembled window fits.
         (b) Join-induced token inflation: `" ".join(segs)` can measure higher than
             the sum of individual segment token counts because joining with spaces
             adds tokens at segment boundaries (especially with the character-estimate
             fallback where each segment is floor-rounded).  Fixed by measuring the
             assembled string at every flush point via _flush(), which trims from the
             front until the joined text fits.
         The sanity-check warning now fires at exactly max_tokens (not × 1.5) since
         the budget is genuinely enforced at both sites.

  Fix-8  (acknowledged as "good enough for now" per reviewer)
         _group_related_rows() micro-block heuristic is documented as v1 with
         known limitations: it does not model row indentation hierarchy, subtotal
         boundaries, or note-table structures.  Deferred to a future revision.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parents[2]
PARSED_DIR = BASE_DIR / "data" / "parsed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
MANIFEST   = BASE_DIR / "data_manifest" / "manifest.jsonl"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
PROSE_MAX_TOKENS  = 512
PROSE_OVERLAP     = 64
MIN_CHUNK_CHARS   = 50

# #20: conservative char-per-token fallback when tiktoken is unavailable.
# 4.5 is deliberately conservative (real average ~4.1) to prevent silent
# over-length chunks from reaching the embedder.
_AVG_CHARS_PER_TOK_FALLBACK = 4.5

# R-C2: token budget per table row-group chunk (header line + data rows).
# 400 tokens leaves headroom for the prepended table_header line (~30 tokens)
# and keeps table chunks well within the 512-token embedding window.
TABLE_CHUNK_MAX_TOKENS = 400

# Hard backstop: even token-aware grouping won't produce groups > this many rows.
# Guards against degenerate tables where every row is very short.
ROW_GROUP_SIZE = 30

# R-C3: statement types that always warrant row-level chunks regardless of size.
_ROW_CHUNK_NOTE_TYPES: frozenset[str] = frozenset({
    "debt_table",
    "segment_table",
    "eps_table",
    "compensation_table",
})

# R-C3: minimum numeric-cell density for unnamed tables to qualify for row chunks.
_ROW_CHUNK_NUMERIC_DENSITY_THRESHOLD = 0.4

# R-C8: micro-block parameters.
MICRO_BLOCK_MIN_ROWS = 6   # only produce micro-blocks when table has >= this many rows
MICRO_BLOCK_MAX_ROWS = 8   # maximum rows per micro-block


# ── Tokenizer (optional) ───────────────────────────────────────────────────────
# #20: use tiktoken when available for accurate token counts.

_TOKENIZER = None
_TOKENIZER_LOADED = False

def _get_tokenizer():
    global _TOKENIZER, _TOKENIZER_LOADED
    if _TOKENIZER_LOADED:
        return _TOKENIZER
    _TOKENIZER_LOADED = True
    try:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        log.info("tiktoken loaded — using accurate token counts for chunking.")
    except Exception:
        log.info("tiktoken not available — using character-estimate fallback (4.5 chars/token).")
        _TOKENIZER = None
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(text))
    return int(len(text) / _AVG_CHARS_PER_TOK_FALLBACK)


def _tokens_to_chars_approx(tokens: int) -> int:
    """Used only for overlap_chars when tiktoken is unavailable."""
    return int(tokens * _AVG_CHARS_PER_TOK_FALLBACK)


# ── Form-aware SEC section tables ─────────────────────────────────────────────
# #77: separate maps per form type so Item 2 resolves correctly for both
# 10-K (Properties) and 10-Q (MD&A).

# 10-K Item map  (annual report)
SECTION_MAP_10K: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^item\s+1a[\.\s]", re.I),  "Item 1A. Risk Factors"),
    (re.compile(r"^item\s+1b[\.\s]", re.I),  "Item 1B. Unresolved Staff Comments"),
    (re.compile(r"^item\s+1[\.\s]",  re.I),  "Item 1. Business"),
    (re.compile(r"^item\s+2[\.\s]",  re.I),  "Item 2. Properties"),
    (re.compile(r"^item\s+3[\.\s]",  re.I),  "Item 3. Legal Proceedings"),
    (re.compile(r"^item\s+4[\.\s]",  re.I),  "Item 4. Mine Safety"),
    (re.compile(r"^item\s+5[\.\s]",  re.I),  "Item 5. Market for Registrant"),
    (re.compile(r"^item\s+6[\.\s]",  re.I),  "Item 6. Selected Financial Data"),
    (re.compile(r"^item\s+7a[\.\s]", re.I),  "Item 7A. Quantitative and Qualitative Disclosures"),
    (re.compile(r"^item\s+7[\.\s]",  re.I),  "Item 7. MD&A"),
    (re.compile(r"^item\s+8[\.\s]",  re.I),  "Item 8. Financial Statements"),
    (re.compile(r"^item\s+9a[\.\s]", re.I),  "Item 9A. Controls and Procedures"),
    (re.compile(r"^item\s+9[\.\s]",  re.I),  "Item 9. Changes in Disagreements"),
    (re.compile(r"^item\s+10[\.\s]", re.I),  "Item 10. Directors and Officers"),
    (re.compile(r"^item\s+11[\.\s]", re.I),  "Item 11. Executive Compensation"),
    (re.compile(r"^item\s+12[\.\s]", re.I),  "Item 12. Security Ownership"),
    (re.compile(r"^item\s+13[\.\s]", re.I),  "Item 13. Certain Relationships"),
    (re.compile(r"^item\s+14[\.\s]", re.I),  "Item 14. Principal Accountant Fees"),
    (re.compile(r"^item\s+15[\.\s]", re.I),  "Item 15. Exhibits"),
]

# 10-Q Item map  (quarterly report)
# Key difference: Item 2 = MD&A, Item 3 = Quantitative Disclosures, Item 4 = Controls
SECTION_MAP_10Q: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^item\s+1a[\.\s]", re.I),  "Item 1A. Risk Factors"),
    (re.compile(r"^item\s+1[\.\s]",  re.I),  "Item 1. Financial Statements"),
    (re.compile(r"^item\s+2[\.\s]",  re.I),  "Item 2. MD&A"),
    (re.compile(r"^item\s+3[\.\s]",  re.I),  "Item 3. Quantitative and Qualitative Disclosures"),
    (re.compile(r"^item\s+4[\.\s]",  re.I),  "Item 4. Controls and Procedures"),
    (re.compile(r"^item\s+5[\.\s]",  re.I),  "Item 5. Other Information"),
    (re.compile(r"^item\s+6[\.\s]",  re.I),  "Item 6. Exhibits"),
]

# 8-K Item map  (current report)
SECTION_MAP_8K: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^item\s+1\.01", re.I),  "Item 1.01. Entry into Material Agreement"),
    (re.compile(r"^item\s+1\.02", re.I),  "Item 1.02. Termination of Agreement"),
    (re.compile(r"^item\s+2\.02", re.I),  "Item 2.02. Results of Operations"),
    (re.compile(r"^item\s+2\.03", re.I),  "Item 2.03. Creation of Direct Financial Obligation"),
    (re.compile(r"^item\s+5\.02", re.I),  "Item 5.02. Director/Officer Changes"),
    (re.compile(r"^item\s+7\.01", re.I),  "Item 7.01. Regulation FD Disclosure"),
    (re.compile(r"^item\s+8\.01", re.I),  "Item 8.01. Other Events"),
    (re.compile(r"^item\s+9\.01", re.I),  "Item 9.01. Financial Statements and Exhibits"),
]

# Form-agnostic subsection phrases (match anywhere in short blocks)
SECTION_PHRASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"management.{0,10}discussion\s+and\s+analysis", re.I), "MD&A"),
    (re.compile(r"results\s+of\s+operations",   re.I), "Results of Operations"),
    (re.compile(r"liquidity\s+and\s+capital",   re.I), "Liquidity and Capital Resources"),
    (re.compile(r"critical\s+accounting",       re.I), "Critical Accounting Policies"),
    (re.compile(r"segment\s+information",       re.I), "Segment Information"),
    (re.compile(r"notes?\s+to\s+(the\s+)?condensed", re.I), "Notes to Financial Statements"),
    (re.compile(r"notes?\s+to\s+(the\s+)?financial", re.I), "Notes to Financial Statements"),
    (re.compile(r"risk\s+factors",              re.I), "Risk Factors"),
    (re.compile(r"financial\s+statements",      re.I), "Financial Statements"),
]

# #24: heading-prefix anchor — infer_section only fires on these anchors when
# the text exceeds the short-block threshold.
_HEADING_ANCHOR_RE = re.compile(
    r"^(?:PART\s+[IVX]+|Item\s+\d|ITEM\s+\d|Note\s+\d)", re.I
)


def _get_section_map(form_type: str) -> list[tuple[re.Pattern, str]]:
    """Return the per-form Item map based on the filing form type."""
    ft = (form_type or "").upper()
    if "10-Q" in ft:
        return SECTION_MAP_10Q
    if "8-K" in ft:
        return SECTION_MAP_8K
    # Default: 10-K map covers DEF 14A, 10-K, 10-K405, etc.
    return SECTION_MAP_10K


def infer_section(text: str, form_type: str = "") -> Optional[str]:
    """
    #24: Try to identify which SEC section this text belongs to.

    Guard: only apply Item-number patterns when:
      (a) the text is short (≤200 chars), OR
      (b) the text starts with an Item/PART/Note heading anchor.
    This prevents body paragraphs from hijacking current_section.

    Form phrases (MD&A, Results of Operations, etc.) are always checked
    but only in the first 200 chars of the text.
    """
    stripped = text.strip()
    text_head = stripped[:200].lower()
    is_short  = len(stripped) <= 200
    has_anchor = bool(_HEADING_ANCHOR_RE.match(stripped))

    if is_short or has_anchor:
        section_map = _get_section_map(form_type)
        for pattern, label in section_map:
            if pattern.search(text_head):
                return label

    # Subsection phrases: always check but only in the first 200 chars
    for pattern, label in SECTION_PHRASES:
        if pattern.search(text_head):
            return label

    return None


# ── Footnote detection ─────────────────────────────────────────────────────────
# R-C6: extended from simple prefix + length cap to three pattern families.

# Classic footnote prefix: (1), 1., a) at the start of the block
_FOOTNOTE_PREFIX_RE = re.compile(r"^\s*(\(\d+\)|\d+\.|[a-z]\))\s+", re.MULTILINE)
# Parenthetical qualifiers that stand alone as short annotation blocks
_FOOTNOTE_PAREN_RE  = re.compile(r"^\s*\([^)]{3,80}\)\s*$")
# Unicode superscript footnote markers (¹²³ or ⁴⁵⁶⁷⁸⁹⁰)
_FOOTNOTE_SUPER_RE  = re.compile(r"^[\u00B9\u00B2\u00B3\u2070-\u2079]\s+\S")

def is_footnote(text: str) -> bool:
    """
    R-C6: Return True for footnote-like blocks.
    Covers:
      (a) Classic prefix patterns (1), 1., a)  — up to 600 chars (was 400)
      (b) Short parenthetical qualifier blocks "(in millions, except per share)"
      (c) Superscript-prefixed note markers
    """
    stripped = text.strip()
    if _FOOTNOTE_PREFIX_RE.match(stripped) and len(stripped) < 600:
        return True
    if _FOOTNOTE_PAREN_RE.match(stripped) and len(stripped) <= 120:
        return True
    if _FOOTNOTE_SUPER_RE.match(stripped) and len(stripped) < 300:
        return True
    return False


# ── Prose splitter ─────────────────────────────────────────────────────────────
# R-C1: Three-stage cascade handles SEC-style text that lacks sentence punctuation.
# Stage 1: sentence-boundary split (existing, primary path)
# Stage 2: if any segment still exceeds max_tokens, sub-split by ; / : / \n
# Stage 3: if a sub-segment still exceeds max_tokens, hard word-boundary split

def _hard_split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """
    R-C1 stage 3 + Fix-7: greedy word-boundary hard split.

    Measures the ASSEMBLED string at each step (not the sum of per-word estimates)
    so join-induced token inflation cannot cause overflow.  With the character-estimate
    fallback, summing floor-rounded per-word counts underestimates; the assembled
    string measurement is always the ground truth.

    If a single whitespace-delimited "word" itself exceeds max_tokens (malformed OCR,
    giant pipe-delimited artefact), it is character-sliced via tiktoken (when available)
    or character-sliced via the fallback estimate.
    """
    def _char_slice(token_str: str, max_tok: int) -> list[str]:
        tok = _get_tokenizer()
        if tok is not None:
            ids = tok.encode(token_str)
            slices = []
            for i in range(0, len(ids), max_tok):
                slices.append(tok.decode(ids[i: i + max_tok]))
            return [s for s in slices if s.strip()]
        chars_per_slice = max(1, int(max_tok * _AVG_CHARS_PER_TOK_FALLBACK))
        return [token_str[i: i + chars_per_slice]
                for i in range(0, len(token_str), chars_per_slice)]

    words = text.split()
    parts: list[str] = []
    current_words: list[str] = []

    for word in words:
        # Check if this single word alone exceeds the budget
        if _count_tokens(word) > max_tokens:
            if current_words:
                parts.append(" ".join(current_words))
                current_words = []
            parts.extend(_char_slice(word, max_tokens))
            continue

        candidate = current_words + [word]
        # Measure the ASSEMBLED candidate string, not the sum of estimates
        if current_words and _count_tokens(" ".join(candidate)) > max_tokens:
            parts.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words = candidate

    if current_words:
        parts.append(" ".join(current_words))

    return [p for p in parts if p.strip()]


def _sub_split_segment(segment: str, max_tokens: int) -> list[str]:
    """
    R-C1 stage 2: split a too-long segment by secondary delimiters
    (semicolons, colons, newlines).  Falls through to hard split if needed.
    """
    # Try secondary delimiters in order of preference
    for delimiter_re in (
        re.compile(r'(?<=;)\s+'),
        re.compile(r'(?<=:)\s+'),
        re.compile(r'\n+'),
    ):
        parts = [p.strip() for p in delimiter_re.split(segment) if p.strip()]
        if len(parts) > 1:
            # Merge small parts back up to max_tokens to avoid over-fragmentation
            merged: list[str] = []
            current_parts: list[str] = []
            current_tok = 0
            for part in parts:
                p_tok = _count_tokens(part)
                if current_tok + p_tok > max_tokens and current_parts:
                    merged.append(" ".join(current_parts))
                    current_parts = [part]
                    current_tok   = p_tok
                else:
                    current_parts.append(part)
                    current_tok += p_tok
            if current_parts:
                merged.append(" ".join(current_parts))
            # If any merged part still exceeds, recurse to hard split
            result: list[str] = []
            for m in merged:
                if _count_tokens(m) > max_tokens:
                    result.extend(_hard_split_by_tokens(m, max_tokens))
                else:
                    result.append(m)
            return [r for r in result if r.strip()]
    # No delimiter worked — fall straight to hard split
    return _hard_split_by_tokens(segment, max_tokens)


def split_prose(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    #20/#25 + R-C1: Split long prose, measuring in tokens.

    Three-stage cascade:
      1. Sentence-boundary split (primary, works for normal financial prose)
      2. Secondary-delimiter sub-split for oversized segments (semicolons, colons,
         newlines — catches SEC bullet lists and long legal/accounting sentences)
      3. Hard word-boundary split as final fallback (guarantees no segment
         exceeds max_tokens regardless of input pathology)

    #25: overlap always carries at least the last sentence/segment regardless
    of its length (capped at 1.5× overlap_tokens).
    """
    # Stage 1: sentence split
    raw_segments = re.split(r'(?<=[.!?])\s+', text.strip())

    # Stage 2/3: sub-split any segment that already exceeds max_tokens
    segments: list[str] = []
    for seg in raw_segments:
        if _count_tokens(seg) > max_tokens:
            segments.extend(_sub_split_segment(seg, max_tokens))
        else:
            segments.append(seg)

    # Stage 4 (join-inflation guard): re-measure each segment as a joined string.
    # _hard_split_by_tokens() sums per-word token estimates (floor-rounded), so
    # the joined result can measure higher than the sum.  Re-split anything that
    # still overshoots after joining.
    re_expanded: list[str] = []
    for seg in segments:
        if _count_tokens(seg) > max_tokens:
            re_expanded.extend(_hard_split_by_tokens(seg, max_tokens))
        else:
            re_expanded.append(seg)
    segments = re_expanded

    max_overlap_tokens = int(overlap_tokens * 1.5)  # #25: generous cap

    chunks: list[str] = []
    current: list[str] = []

    def _assembled(segs: list[str]) -> int:
        """Token count of the assembled string — the ground-truth measure."""
        return _count_tokens(" ".join(segs)) if segs else 0

    def _flush_trim(segs: list[str]) -> list[str]:
        """
        Trim overlap from the front of segs until the assembled string fits
        max_tokens.  If even a single segment overshoots (fallback floor-rounding
        on a word-boundary segment), hard-split it and return all pieces.
        Always returns at least one non-empty string.
        """
        while len(segs) > 1 and _assembled(segs) > max_tokens:
            segs.pop(0)
        if _assembled(segs) > max_tokens:
            # Single segment still too large — hard-split and return all pieces
            pieces = _hard_split_by_tokens(" ".join(segs), max_tokens)
            return [p for p in pieces if p.strip()] or segs
        return segs

    for seg in segments:
        candidate = current + [seg]
        if current and _assembled(candidate) > max_tokens:
            # Flush current window
            flushed = _flush_trim(list(current))
            chunks.extend(flushed)

            # #25: always carry last segment as overlap seed
            overlap_segs: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                s_tok = _count_tokens(s)
                if overlap_len == 0 or overlap_len + s_tok <= max_overlap_tokens:
                    overlap_segs.insert(0, s)
                    overlap_len += s_tok
                else:
                    break

            current = overlap_segs + [seg]
            # Trim overlap until window fits
            while len(current) > 1 and _assembled(current) > max_tokens:
                current.pop(0)
        else:
            current = candidate

    if current:
        chunks.extend(_flush_trim(list(current)))

    result = [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]

    # Sanity-check guard: with the overlap-trimming fix above, no chunk should exceed
    # max_tokens.  Log a warning (not just DEBUG) if one does — it signals an edge case
    # worth investigating (e.g. a single seg that was itself barely at the limit plus
    # char-estimate rounding on the fallback tokeniser).
    for c in result:
        tok = _count_tokens(c)
        if tok > max_tokens:
            log.warning(
                f"  split_prose: chunk of {tok} tokens exceeds budget of {max_tokens} "
                f"(fallback estimator rounding or single-segment edge case; "
                f"first 60 chars: {c[:60]!r})"
            )

    return result


# ── Pseudo-page assignment (HTM only) ─────────────────────────────────────────
# #21: only called when page_is_surrogate=True; real PDF pages are never touched.

def _assign_pseudo_pages_to_items(
    items: list[dict],
    form_type: str = "",
) -> list[dict]:
    """
    Assign pseudo-page numbers to HTM items based on their block_order position.
    Items must be sorted by block_order before this is called.

    BLOCKS_PER_PAGE is computed dynamically:
        BPP = max(MIN_BLOCKS_PER_PAGE, total_items // target_pages_for_form)

    Target pages: 10-K→80, 10-Q→40, 8-K→10, DEF14A→60, default→30.
    This ensures a 707-block 10-K maps to ~88 pseudo-pages instead of 15.
    """
    if not items:
        return items

    _TARGET: dict[str, int] = {
        "10-K": 80, "10-K405": 80, "10-KSB": 80,
        "10-Q": 40, "8-K": 10, "DEF14A": 60,
    }
    ft = (form_type or "").upper()
    target_pages = next((v for k, v in _TARGET.items() if ft.startswith(k)), 30)
    blocks_per_page = max(3, len(items) // target_pages)

    for i, item in enumerate(items):
        item["page"] = max(1, (i // blocks_per_page) + 1)
    return items


# ── Citation key helper ────────────────────────────────────────────────────────
# #78: HTM files have no meaningful page number; use section + block_order instead.

def _citation_key(page: int, section: str, block_order: int, page_is_surrogate: bool) -> str:
    """
    Return the citation key to embed in the chunk metadata.
    - Real PDF: "p.{page}"
    - HTM surrogate: "{section} §{block_order}"
    """
    if page_is_surrogate:
        safe_section = (section or "General").replace("|", "-").strip()
        return f"{safe_section} §{block_order}"
    return f"p.{page}"


# ── Interleaved block ordering ─────────────────────────────────────────────────
# #28: merge prose_blocks and table_records by block_order so chunk_index
# follows true document reading order.

def _interleave_by_block_order(
    prose_blocks: list[dict],
    table_records: list[dict],
) -> list[dict]:
    """
    Return a single list of items sorted by block_order.
    Each item is tagged with an 'item_kind' field: 'prose' or 'table'.
    Items without block_order are appended at the end in their original order.
    """
    tagged: list[dict] = []
    for b in prose_blocks:
        item = dict(b)
        item["item_kind"] = "prose"
        tagged.append(item)
    for t in table_records:
        item = dict(t)
        item["item_kind"] = "table"
        tagged.append(item)

    with_order    = [x for x in tagged if x.get("block_order") is not None]
    without_order = [x for x in tagged if x.get("block_order") is None]

    with_order.sort(key=lambda x: x["block_order"])
    return with_order + without_order


# ── Main chunker ───────────────────────────────────────────────────────────────
def chunk_document(parsed_doc: dict) -> list[dict]:
    """
    Convert a parsed document dict into a flat list of chunk dicts.
    Applies all P2/P5/P7 fixes.
    """
    doc_meta = {
        "doc_id":        parsed_doc["doc_id"],
        "ticker":        parsed_doc["ticker"],
        "company":       parsed_doc["company"],
        "form_type":     parsed_doc["form_type"],
        "filing_date":   parsed_doc["filing_date"],
        "fiscal_year":   parsed_doc["fiscal_year"],
        "fiscal_quarter": parsed_doc.get("fiscal_quarter"),
        "period_type":   parsed_doc.get("period_type"),       # annual/quarterly/event
        "period_end_date": parsed_doc.get("period_end_date"),
        "sector":        parsed_doc.get("sector", "Unknown"),
        "industry":      parsed_doc.get("industry", ""),
        "report_priority": parsed_doc.get("report_priority"),
        "source_url":    parsed_doc["source_url"],
    }

    prose_blocks  = list(parsed_doc.get("prose_blocks",  []))
    table_records = list(parsed_doc.get("table_records", []))
    form_type     = doc_meta["form_type"]

    # ── R-C7: detect surrogate pagination from ALL blocks, not just first prose ─
    # If the document has no prose blocks but has table-only HTM content the old
    # "first_prose_surrogate" check would miss it entirely.
    all_blocks_for_surrogate = prose_blocks + table_records
    is_htm_doc = (
        len(all_blocks_for_surrogate) > 0
        and all(b.get("page_is_surrogate", False) for b in all_blocks_for_surrogate)
    )

    # ── #21: pseudo-page assignment for HTM files only ────────────────────────
    if is_htm_doc:
        # Interleave prose + tables by block_order for coherent pseudo-page numbering
        all_items_for_pages = sorted(
            [dict(b, item_kind="prose")  for b in prose_blocks] +
            [dict(t, item_kind="table") for t in table_records],
            key=lambda x: x.get("block_order", 0)
        )
        all_items_for_pages = _assign_pseudo_pages_to_items(all_items_for_pages, form_type=form_type)
        # Write pseudo pages back by matching on identity (list position), not block_order.
        # block_order may not be unique (e.g. all tables default to 0), so we build
        # separate ordered index lists for prose and table records.
        prose_sorted_order  = sorted(range(len(prose_blocks)),
                                     key=lambda i: prose_blocks[i].get("block_order", 0))
        table_sorted_order  = sorted(range(len(table_records)),
                                     key=lambda i: table_records[i].get("block_order", 0))
        prose_ptr = 0
        table_ptr = 0
        for item in all_items_for_pages:
            new_page = item["page"]
            if item["item_kind"] == "prose" and prose_ptr < len(prose_sorted_order):
                orig_idx = prose_sorted_order[prose_ptr]
                if prose_blocks[orig_idx].get("page_is_surrogate"):
                    prose_blocks[orig_idx]["page"] = new_page
                prose_ptr += 1
            elif item["item_kind"] == "table" and table_ptr < len(table_sorted_order):
                orig_idx = table_sorted_order[table_ptr]
                if table_records[orig_idx].get("page_is_surrogate"):
                    table_records[orig_idx]["page"] = new_page
                table_ptr += 1

    # ── #28: interleave prose and tables by block_order ───────────────────────
    ordered_items = _interleave_by_block_order(prose_blocks, table_records)

    chunks: list[dict] = []
    chunk_index     = 0
    current_section = "General"

    # ── Process items in document reading order ───────────────────────────────
    for item in ordered_items:
        kind = item.get("item_kind")

        if kind == "prose":
            chunks_from_block = _chunk_prose_block(
                item, doc_meta, current_section, form_type, chunk_index
            )
            if chunks_from_block:
                # Update current_section from the block if it carried one
                section_from_block = chunks_from_block[0].get("section", current_section)
                if section_from_block and section_from_block != "General":
                    current_section = section_from_block
                chunk_index += len(chunks_from_block)
            chunks.extend(chunks_from_block)

        elif kind == "table":
            # Skip cover/admin tables entirely — they are not retrieval targets
            if item.get("statement_type") == "cover_admin_table":
                continue
            if item.get("is_cover_page") and not item.get("is_financial_statement"):
                continue

            table_chunks, row_chunks = _chunk_table_record(
                item, doc_meta, current_section, chunk_index
            )
            if table_chunks:
                # R-C5: only advance current_section when the table has an
                # explicit parser-supplied section (not inherited from context).
                # Tables that received current_section as a default must NOT
                # push that same value back and potentially contaminate later prose.
                tbl_section = table_chunks[0].get("section", "")
                if tbl_section and tbl_section not in ("General", "", current_section):
                    current_section = tbl_section
                chunk_index += len(table_chunks)
            chunks.extend(table_chunks)

            # Row-level chunks are appended after all table chunks
            # They carry parent_chunk_id back to the first table chunk
            if row_chunks:
                first_table_chunk_id = table_chunks[0]["chunk_id"] if table_chunks else None
                for rc in row_chunks:
                    if first_table_chunk_id:
                        rc["parent_chunk_id"] = first_table_chunk_id
                    rc["chunk_id"]    = f"{doc_meta['doc_id']}_chunk_{chunk_index:04d}"
                    rc["chunk_index"] = chunk_index
                    chunk_index += 1
                chunks.extend(row_chunks)

    # ── R-C4: deduplicate BEFORE linking so prev/next IDs are never stale ──────
    # If we linked first (A→B→C) then removed B, A.next_chunk_id would point to a
    # deleted chunk and C.prev_chunk_id would point to a deleted chunk.
    # Deduplication must come first; linking is then built on the survivor list.
    chunks = _deduplicate_chunks(chunks)

    # ── #27: intra-document prev/next linking (on deduplicated list) ──────────
    _link_adjacent_chunks(chunks)

    return chunks


# ── Prose block chunker ────────────────────────────────────────────────────────
def _chunk_prose_block(
    block: dict,
    doc_meta: dict,
    current_section: str,
    form_type: str,
    start_index: int,
) -> list[dict]:
    """
    Produce one or more chunks from a single prose block.
    Returns a list of chunk dicts (may be empty if block is filtered out).
    """
    text       = block.get("text", "").strip()
    page       = block.get("page", 1)
    block_order = block.get("block_order", 0)
    page_is_surrogate = block.get("page_is_surrogate", False)

    if not text or len(text) < MIN_CHUNK_CHARS:
        return []

    # Determine section: prefer parser-supplied section, then infer.
    # #24: infer_section is gated inside the function — won't fire on body text.
    docling_section = block.get("section", "")
    if docling_section and docling_section not in ("General", ""):
        section = docling_section
    else:
        inferred = infer_section(text, form_type)
        section  = inferred if inferred else current_section

    # Determine chunk type
    block_type = block.get("type", "prose")
    if block_type == "figure_caption":
        chunk_type = "figure_caption"
    elif is_footnote(text):
        chunk_type = "footnote"
    else:
        chunk_type = "prose"

    # Split long blocks
    if _count_tokens(text) > PROSE_MAX_TOKENS:
        splits = split_prose(text, PROSE_MAX_TOKENS, PROSE_OVERLAP)
    else:
        splits = [text]

    result: list[dict] = []
    parent_chunk_id: Optional[str] = None

    for i, split_text in enumerate(splits):
        chunk_id = f"{doc_meta['doc_id']}_chunk_{start_index + i:04d}"
        if parent_chunk_id is None:
            parent_chunk_id = chunk_id  # first split is its own parent

        # #78: citation key based on page vs block_order
        cit_key = _citation_key(page, section, block_order, page_is_surrogate)

        result.append({
            "chunk_id":          chunk_id,
            "chunk_index":       start_index + i,
            "chunk_type":        chunk_type,
            "text":              split_text,
            "page":              page,
            "page_is_surrogate": page_is_surrogate,
            "citation_key":      cit_key,      # #78
            "block_order":       block_order,
            "section":           section,
            "parent_chunk_id":   parent_chunk_id,  # #26
            "next_chunk_id":     None,   # filled in by _link_adjacent_chunks (#27)
            "prev_chunk_id":     None,   # filled in by _link_adjacent_chunks (#27)
            "table_json":        None,
            **doc_meta,
        })

    return result


# ── Token-aware table row grouping (R-C2) ─────────────────────────────────────
def _split_rows_by_token_budget(
    data_lines: list[str],
    header_line: str,
    table_header: str,
    max_tokens: int = TABLE_CHUNK_MAX_TOKENS,
) -> list[list[str]]:
    """
    R-C2: Accumulate rows into groups until the embedded text (table_header +
    header_line + rows) would exceed max_tokens, then start a new group.
    ROW_GROUP_SIZE is a hard backstop against degenerate tiny-row tables.
    """
    if not data_lines:
        return []

    # Pre-compute the fixed token overhead per group (same for every group)
    overhead = _count_tokens(f"{table_header}\n{header_line}\n") if header_line else _count_tokens(f"{table_header}\n")

    groups: list[list[str]] = []
    current_group: list[str] = []
    current_tok = overhead

    for line in data_lines:
        line_tok = _count_tokens(line) + 1  # +1 for the \n separator
        if (current_tok + line_tok > max_tokens or len(current_group) >= ROW_GROUP_SIZE) \
                and current_group:
            groups.append(current_group)
            current_group = [line]
            current_tok   = overhead + line_tok
        else:
            current_group.append(line)
            current_tok += line_tok

    if current_group:
        groups.append(current_group)

    return groups


# ── Row-chunk eligibility (R-C3) ──────────────────────────────────────────────
# Fix 4: _NUM_VAL_RE replaced with a broader numeric-presence test that catches
# realistic financial value formats the old regex missed:
#   (1,234.5)   $ (1,234)   1.2x   3.5 million   12 months   —   -   basis points
#
# Strategy: a cell is "numeric" if it contains at least one digit, and the
# non-digit, non-symbol remainder is short (≤ 12 chars of alphabetic noise).
# This is intentionally broad — we want high recall for density detection.
# False positives (e.g. "12 months") are fine; they represent genuine financial data.

_HAS_DIGIT_RE   = re.compile(r'\d')
_ALPHA_NOISE_RE = re.compile(r'[A-Za-z]')

def _is_numeric_cell(value: str) -> bool:
    """
    Return True if value looks like a financial numeric cell.
    Handles: integers, decimals, negatives, parenthetical negatives,
    dollar amounts, percentages, multipliers (1.2x), dash placeholders,
    magnitude suffixes (B/M/K), and short mixed values like "3.5 million".
    """
    v = value.strip()
    if not v:
        return False
    # Pure dash / em-dash placeholders are numeric context in financial tables
    if v in ("-", "—", "–", "N/A", "n/a", "NM", "nm"):
        return True
    # Must contain at least one digit
    if not _HAS_DIGIT_RE.search(v):
        return False
    # Allow up to 12 alphabetic chars (handles "3.5 million", "12 months", "1.2x")
    alpha_count = len(_ALPHA_NOISE_RE.findall(v))
    return alpha_count <= 12

def _should_emit_row_chunks(tbl: dict) -> bool:
    """
    R-C3 + Fix 4: Emit row-level chunks when ANY of these is true:
      (a) is_financial_statement=True  (core income/balance/cash/equity statements)
      (b) statement_type is in the high-value note type set
      (c) cell_grid has numeric density >= threshold (uses broader _is_numeric_cell)
    """
    if tbl.get("is_financial_statement"):
        return True
    if tbl.get("statement_type", "") in _ROW_CHUNK_NOTE_TYPES:
        return True
    cell_grid = tbl.get("cell_grid", [])
    if not cell_grid:
        return False
    numeric = sum(1 for c in cell_grid if _is_numeric_cell(str(c.get("value", ""))))
    density = numeric / len(cell_grid)
    return density >= _ROW_CHUNK_NUMERIC_DENSITY_THRESHOLD


# ── Micro-block grouping (R-C8) ───────────────────────────────────────────────
def _group_related_rows(cell_grid: list[dict]) -> list[list[dict]]:
    """
    R-C8: Group cell_grid entries into semantically related micro-blocks of
    MICRO_BLOCK_MIN_ROWS to MICRO_BLOCK_MAX_ROWS rows.

    Grouping heuristic: accumulate rows until one of these breakpoints is hit:
      - row_label is empty (often a subtotal / section separator)
      - row_label ends with ':' (section header in financial statements)
      - accumulated group reaches MICRO_BLOCK_MAX_ROWS

    This naturally clusters "Current assets" rows together, "Liabilities" rows
    together, etc., which is what comparison queries need.
    """
    if not cell_grid:
        return []

    # Group by unique row_labels first (each row_label may have multiple col_headers)
    seen_labels: list[str] = []
    for cell in cell_grid:
        lbl = cell.get("row_label", "")
        if not seen_labels or seen_labels[-1] != lbl:
            seen_labels.append(lbl)

    if len(seen_labels) < MICRO_BLOCK_MIN_ROWS:
        return []

    # Build label-indexed groups
    label_to_cells: dict[str, list[dict]] = {}
    for cell in cell_grid:
        lbl = cell.get("row_label", "")
        label_to_cells.setdefault(lbl, []).append(cell)

    # Walk labels and slice into micro-blocks
    micro_groups: list[list[dict]] = []
    current_labels: list[str] = []

    for lbl in seen_labels:
        is_break = (
            (not lbl or lbl.rstrip().endswith(":"))
            and len(current_labels) >= 2     # only break when we already have rows to flush
        ) or len(current_labels) >= MICRO_BLOCK_MAX_ROWS
        if is_break:
            group_cells = [c for l in current_labels for c in label_to_cells.get(l, [])]
            if len(group_cells) >= 2:
                micro_groups.append(group_cells)
            current_labels = []
        current_labels.append(lbl)

    # Flush remaining
    if len(current_labels) >= 2:
        group_cells = [c for l in current_labels for c in label_to_cells.get(l, [])]
        micro_groups.append(group_cells)

    return [g for g in micro_groups if len(g) >= 2]


# ── Content hash deduplication (R-C4 + Fix 5) ────────────────────────────────
import hashlib

def _content_hash(chunk: dict) -> str:
    """
    Provenance-aware normalised hash.

    Fix 5: hash includes chunk_type and citation_key alongside text so that two
    chunks with identical text but different provenance are NOT collapsed:
      - same safe-harbor text appearing in different sections
      - same disclosure repeated across multiple filing periods
      - same row text from genuinely different statement/period contexts

    citation_key already encodes page (PDF) or section+block_order (HTM), so it
    distinguishes repeated text in different locations within the same document.
    chunk_type distinguishes e.g. a prose block whose text happens to match a
    row chunk's text.
    """
    text_norm   = re.sub(r'\s+', ' ', chunk.get("text", "").lower().strip())
    chunk_type  = chunk.get("chunk_type", "")
    citation    = chunk.get("citation_key", "")
    fingerprint = f"{chunk_type}|{citation}|{text_norm}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    R-C4 + Fix 5: Remove chunks with identical provenance-aware content hash.
    Preserves the first occurrence; logs skipped duplicates at DEBUG level.
    Called BEFORE _link_adjacent_chunks() so prev/next IDs are never stale.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for chunk in chunks:
        h = _content_hash(chunk)
        if h in seen:
            log.debug(f"  Dedup: skipping duplicate chunk {chunk.get('chunk_id')} "
                      f"(type={chunk.get('chunk_type')})")
            continue
        seen.add(h)
        result.append(chunk)
    return result


# ── Table record chunker ───────────────────────────────────────────────────────
def _chunk_table_record(
    tbl: dict,
    doc_meta: dict,
    current_section: str,
    start_index: int,
) -> tuple[list[dict], list[dict]]:
    """
    Produce table-level chunks (one per token-bounded row-group) and row-level
    chunks (including micro-blocks).
    Returns (table_chunks, row_chunks).

    R-C2: row-grouping is now token-budget-aware (not fixed at 30 rows).
    R-C3: row-chunk eligibility uses a richer rule than ROW_CHUNK_MIN_ROWS.
    R-C8: micro-block chunks added as a middle granularity.
    """
    retrieval_text = tbl.get("retrieval_text", "").strip()
    if not retrieval_text or len(retrieval_text) < MIN_CHUNK_CHARS:
        return [], []

    page              = tbl.get("page", 1)
    block_order       = tbl.get("block_order", 0)
    page_is_surrogate = tbl.get("page_is_surrogate", False)
    section           = tbl.get("section") or current_section

    # ── #22: fixed table_header — no more duplicated form_type ────────────────
    table_header = (
        f"{doc_meta['company']} ({doc_meta['ticker']}) | "
        f"{doc_meta['form_type']} FY{doc_meta['fiscal_year']} | "
        f"{section}"
    )

    # ── #23: extract the column-header line so every row-group carries it ─────
    rows = retrieval_text.split("\n")
    if len(rows) > 1:
        header_line = rows[0]
        data_lines  = rows[1:]
    else:
        header_line = ""
        data_lines  = rows

    # ── R-C2: token-budget-aware row grouping ─────────────────────────────────
    row_groups = _split_rows_by_token_budget(
        data_lines, header_line, table_header, TABLE_CHUNK_MAX_TOKENS
    )
    if not row_groups:
        row_groups = [data_lines]   # fallback: emit as single group

    # #78: citation key
    cit_key = _citation_key(page, section, block_order, page_is_surrogate)

    # Shared metadata for every chunk from this table
    _tbl_meta = {
        "page":              page,
        "page_is_surrogate": page_is_surrogate,
        "citation_key":      cit_key,
        "block_order":       block_order,
        "section":           section,
        "table_title":       tbl.get("table_title", ""),
        "statement_type":    tbl.get("statement_type", ""),
        "period_type":       tbl.get("period_type", ""),
        "period_signals":    tbl.get("period_signals", []),
        "periods":           tbl.get("periods", []),
        "units":             tbl.get("units", []),
        "is_financial_statement": tbl.get("is_financial_statement", False),
    }

    table_chunks: list[dict] = []
    parent_chunk_id: Optional[str] = None

    for group_idx, row_group in enumerate(row_groups):
        # #23: prepend header line to EVERY group so group 2+ are self-contained
        group_content = "\n".join(row_group).strip()
        if not group_content or len(group_content) < MIN_CHUNK_CHARS:
            continue

        group_text    = f"{header_line}\n{group_content}" if header_line else group_content
        embedded_text = f"{table_header}\n{group_text}"

        chunk_id = f"{doc_meta['doc_id']}_chunk_{start_index + len(table_chunks):04d}"
        if parent_chunk_id is None:
            parent_chunk_id = chunk_id

        table_chunks.append({
            "chunk_id":        chunk_id,
            "chunk_index":     start_index + len(table_chunks),
            "chunk_type":      "table",
            "text":            embedded_text,
            "table_group":     group_idx,
            "parent_chunk_id": parent_chunk_id,  # #26
            "next_chunk_id":   None,
            "prev_chunk_id":   None,
            "table_json":      tbl.get("cell_grid"),
            **_tbl_meta,
            **doc_meta,
        })

    # ── R-C3/#79/#80: row-level chunks ────────────────────────────────────────
    row_chunks: list[dict] = []
    cell_grid = tbl.get("cell_grid", [])

    if _should_emit_row_chunks(tbl) and cell_grid:
        # Build period context string once for all rows in this table.
        # Fix 6: embed period semantics directly in the row text so the embedder
        # captures "annual FY2024" or "quarterly Q3 2024" rather than a bare column
        # header like "Sep 2024" whose temporal meaning depends on the full table.
        period_type_str   = tbl.get("period_type", "") or ""
        units_str         = ", ".join(tbl.get("units", [])) if tbl.get("units") else ""
        period_end_str    = doc_meta.get("period_end_date", "") or ""
        fiscal_quarter    = doc_meta.get("fiscal_quarter")
        # Compose a concise period qualifier: e.g. "annual | period ended 2024-09-28"
        # or "quarterly Q3 | period ended 2024-06-29 | $ millions"
        period_parts: list[str] = []
        if period_type_str:
            if period_type_str == "quarterly" and fiscal_quarter:
                period_parts.append(f"quarterly Q{fiscal_quarter}")
            else:
                period_parts.append(period_type_str)
        if period_end_str:
            period_parts.append(f"period ended {period_end_str}")
        if units_str:
            period_parts.append(units_str)
        period_context = " | ".join(period_parts) if period_parts else ""

        seen_cells: set[tuple] = set()
        for cell in cell_grid:
            row_label  = cell.get("row_label", "")
            col_header = cell.get("col_header", "")
            value      = cell.get("value", "")
            key = (row_label, col_header, value)
            if key in seen_cells:
                continue
            seen_cells.add(key)

            # Fix 6: row text includes period context so the embedding captures
            # temporal meaning for queries like "FY2024", "latest quarter", etc.
            row_text_parts = [
                f"{doc_meta['company']} ({doc_meta['ticker']})",
                f"{doc_meta['form_type']} FY{doc_meta['fiscal_year']}",
                tbl.get("statement_type", ""),
            ]
            if period_context:
                row_text_parts.append(period_context)
            row_text_parts.append(f"{row_label} | {col_header}: {value}")
            row_text = " | ".join(p for p in row_text_parts if p)

            row_chunks.append({
                "chunk_id":    None,   # assigned by caller
                "chunk_index": None,
                "chunk_type":  "row",
                "text":        row_text,
                "parent_chunk_id": None,   # set to first table chunk id by caller
                "next_chunk_id":   None,
                "prev_chunk_id":   None,
                "row_label":   row_label,
                "col_header":  col_header,
                "value":       value,
                "table_json":  None,
                **_tbl_meta,
                **doc_meta,
            })

    # ── R-C8: micro-block chunks ───────────────────────────────────────────────
    if _should_emit_row_chunks(tbl) and len(cell_grid) >= MICRO_BLOCK_MIN_ROWS:
        micro_groups = _group_related_rows(cell_grid)
        seen_mb_texts: set[str] = set()   # dedup within this table's micro-blocks
        for mb_cells in micro_groups:
            # Build micro-block text: header + one line per (row_label, col_header, value)
            lines = [table_header]
            if header_line:
                lines.append(header_line)
            seen_mb: set[tuple] = set()
            for cell in mb_cells:
                rl, ch, v = cell.get("row_label",""), cell.get("col_header",""), cell.get("value","")
                k = (rl, ch, v)
                if k in seen_mb:
                    continue
                seen_mb.add(k)
                lines.append(f"{rl} | {ch}: {v}" if ch else rl)
            mb_text = "\n".join(lines)
            if len(mb_text) < MIN_CHUNK_CHARS:
                continue
            # Intra-table dedup: skip if this exact micro-block was already emitted
            # from an overlapping group produced by _group_related_rows().
            mb_norm = re.sub(r'\s+', ' ', mb_text.lower().strip())
            if mb_norm in seen_mb_texts:
                log.debug(f"  micro_block intra-table dedup: skipping duplicate at {cit_key}")
                continue
            seen_mb_texts.add(mb_norm)
            row_chunks.append({
                "chunk_id":    None,
                "chunk_index": None,
                "chunk_type":  "micro_block",
                "text":        mb_text,
                "parent_chunk_id": parent_chunk_id,
                "next_chunk_id":   None,
                "prev_chunk_id":   None,
                "row_label":   "",
                "col_header":  "",
                "value":       "",
                "table_json":  mb_cells,
                **_tbl_meta,
                **doc_meta,
            })

    return table_chunks, row_chunks


# ── Intra-document linking ─────────────────────────────────────────────────────
# #27: populate next_chunk_id / prev_chunk_id on every chunk.
# Row-level chunks are NOT linked into the linear chain (they are parallel
# to the table chunks, not sequential narrative content).

def _link_adjacent_chunks(chunks: list[dict]) -> None:
    """
    #27: Link sequential prose/table/footnote/figure_caption chunks.

    Row-level chunks (chunk_type='row') and micro-blocks (chunk_type='micro_block')
    are both excluded from the linear chain.  They are parallel table-derived lookup
    targets, not narrative reading-order content.  Including them in the linear chain
    would make an agent walking next_chunk_id traverse:
        prose → table → micro_block → prose
    which does not reflect document reading order.

    Mutates chunks in place.
    """
    _NONLINEAR = frozenset({"row", "micro_block"})
    linear = [c for c in chunks if c.get("chunk_type") not in _NONLINEAR]
    for i, chunk in enumerate(linear):
        if i > 0:
            chunk["prev_chunk_id"] = linear[i - 1]["chunk_id"]
        if i < len(linear) - 1:
            chunk["next_chunk_id"] = linear[i + 1]["chunk_id"]


# ── Save chunks ────────────────────────────────────────────────────────────────
def save_chunks(chunks: list[dict], doc_id: str) -> Path:
    out_path = CHUNKS_DIR / f"{doc_id}.jsonl"
    tmp_path = out_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        tmp_path.replace(out_path)
    except Exception as e:
        log.error(f"  save_chunks failed for {doc_id}: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return out_path


# ── Manifest helpers (O(N) pattern) ───────────────────────────────────────────
# #19: manifest is loaded once, mutated in memory, written once at end of run().
# update_manifest_entry() operates on the in-memory list only.

def load_manifest() -> list[dict]:
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
    """Atomic O(N) manifest write — called once at end of run()."""
    tmp = MANIFEST.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        tmp.replace(MANIFEST)
    except Exception as e:
        log.error(f"  Manifest write failed: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


# ── Entry point ────────────────────────────────────────────────────────────────
def run(tickers: list[str] | None = None, rechunk: bool = False) -> None:
    """
    O(N) manifest pattern: load once, update in-memory, write once.
    #19: eliminates the O(N²) rewrite that occurred when update_manifest_status()
    was called inside the per-document loop.
    """
    all_entries = load_manifest()
    id_to_idx   = {e.get("doc_id"): i for i, e in enumerate(all_entries)}

    working = all_entries
    if tickers:
        tickers_upper = {t.strip().upper() for t in tickers}
        working = [e for e in all_entries if e.get("ticker", "").upper() in tickers_upper]

    to_chunk = [
        e for e in working
        if e.get("parse_status") == "parsed"
        and (rechunk or e.get("chunk_status") != "chunked")
    ]

    log.info(f"Chunking {len(to_chunk)} documents (rechunk={rechunk})")

    success      = 0
    failed       = 0
    total_chunks = 0

    for entry in to_chunk:
        doc_id      = entry["doc_id"]
        parsed_path = PARSED_DIR / f"{doc_id}.json"

        if not parsed_path.exists():
            log.warning(f"  Parsed file missing: {doc_id}")
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["chunk_status"] = "chunk_failed"
            failed += 1
            continue

        # Already chunked check
        chunk_path = CHUNKS_DIR / f"{doc_id}.jsonl"
        if chunk_path.exists() and not rechunk:
            log.info(f"  Already chunked: {doc_id}")
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["chunk_status"] = "chunked"
            success += 1
            continue

        with open(parsed_path, encoding="utf-8") as f:
            parsed_doc = json.load(f)

        # Inject manifest-level fields the parser doesn't write into parsed JSON
        # (sector, industry, fiscal_quarter, period_type, period_end_date,
        #  report_priority) so chunk_document can carry them into every chunk.
        for field in ("sector", "industry", "fiscal_quarter", "period_type",
                      "period_end_date", "report_priority"):
            if field not in parsed_doc and field in entry:
                parsed_doc[field] = entry[field]

        try:
            chunks = chunk_document(parsed_doc)
            if not chunks:
                log.warning(f"  No chunks produced for {doc_id}")
                idx = id_to_idx.get(doc_id)
                if idx is not None:
                    all_entries[idx]["chunk_status"] = "chunk_failed"
                failed += 1
                continue

            save_chunks(chunks, doc_id)
            total_chunks += len(chunks)

            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["chunk_status"] = "chunked"

            success += 1
            log.info(f"  {doc_id:<50} -> {len(chunks):>5} chunks")

        except Exception as e:
            log.error(f"  Chunking failed for {doc_id}: {e}", exc_info=True)
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["chunk_status"] = "chunk_failed"
            failed += 1

    # #19: single atomic manifest write at end of run
    _write_manifest(all_entries)

    log.info("\nChunking complete.")
    log.info(f"  Documents : success={success}  failed={failed}")
    log.info(f"  Total chunks produced: {total_chunks}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chunk parsed SEC filings.")
    parser.add_argument("--tickers", nargs="+", help="Subset of tickers to chunk")
    parser.add_argument("--rechunk", action="store_true", help="Re-chunk already chunked docs")
    args = parser.parse_args()
    run(tickers=args.tickers, rechunk=args.rechunk)