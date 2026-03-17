"""
Hierarchical, layout-aware chunker for parsed SEC filing documents.
Handles prose, tables, footnotes, and figure captions as distinct chunk types.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PARSED_DIR = BASE_DIR / "data" / "parsed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
MANIFEST = BASE_DIR / "data_manifest" / "manifest.jsonl"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

PROSE_MAX_TOKENS = 512
PROSE_OVERLAP = 64
MIN_CHUNK_CHARS = 50

_AVG_CHARS_PER_TOK_FALLBACK = 4.5
TABLE_CHUNK_MAX_TOKENS = 400
ROW_GROUP_SIZE = 30

_ROW_CHUNK_NOTE_TYPES: frozenset[str] = frozenset({
    "debt_table",
    "segment_table",
    "eps_table",
    "compensation_table",
})

_ROW_CHUNK_NUMERIC_DENSITY_THRESHOLD = 0.4

MICRO_BLOCK_MIN_ROWS = 6
MICRO_BLOCK_MAX_ROWS = 8

_TOKENIZER = None
_TOKENIZER_LOADED = False


def _get_tokenizer():
    """Load tiktoken once and reuse it for token counting."""
    global _TOKENIZER, _TOKENIZER_LOADED
    if _TOKENIZER_LOADED:
        return _TOKENIZER

    _TOKENIZER_LOADED = True
    try:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        log.info("tiktoken loaded — using accurate token counts for chunking.")
    except Exception:
        log.info("tiktoken not available — using character-estimate fallback.")
        _TOKENIZER = None
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    """Return token count using tiktoken if available, else a char-based estimate."""
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(text))
    return int(len(text) / _AVG_CHARS_PER_TOK_FALLBACK)


def _tokens_to_chars_approx(tokens: int) -> int:
    """Approximate chars from token count when no tokenizer is available."""
    return int(tokens * _AVG_CHARS_PER_TOK_FALLBACK)


SECTION_MAP_10K: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^item\s+1a[\.\s]", re.I), "Item 1A. Risk Factors"),
    (re.compile(r"^item\s+1b[\.\s]", re.I), "Item 1B. Unresolved Staff Comments"),
    (re.compile(r"^item\s+1[\.\s]", re.I), "Item 1. Business"),
    (re.compile(r"^item\s+2[\.\s]", re.I), "Item 2. Properties"),
    (re.compile(r"^item\s+3[\.\s]", re.I), "Item 3. Legal Proceedings"),
    (re.compile(r"^item\s+4[\.\s]", re.I), "Item 4. Mine Safety"),
    (re.compile(r"^item\s+5[\.\s]", re.I), "Item 5. Market for Registrant"),
    (re.compile(r"^item\s+6[\.\s]", re.I), "Item 6. Selected Financial Data"),
    (re.compile(r"^item\s+7a[\.\s]", re.I), "Item 7A. Quantitative and Qualitative Disclosures"),
    (re.compile(r"^item\s+7[\.\s]", re.I), "Item 7. MD&A"),
    (re.compile(r"^item\s+8[\.\s]", re.I), "Item 8. Financial Statements"),
    (re.compile(r"^item\s+9a[\.\s]", re.I), "Item 9A. Controls and Procedures"),
    (re.compile(r"^item\s+9[\.\s]", re.I), "Item 9. Changes in Disagreements"),
    (re.compile(r"^item\s+10[\.\s]", re.I), "Item 10. Directors and Officers"),
    (re.compile(r"^item\s+11[\.\s]", re.I), "Item 11. Executive Compensation"),
    (re.compile(r"^item\s+12[\.\s]", re.I), "Item 12. Security Ownership"),
    (re.compile(r"^item\s+13[\.\s]", re.I), "Item 13. Certain Relationships"),
    (re.compile(r"^item\s+14[\.\s]", re.I), "Item 14. Principal Accountant Fees"),
    (re.compile(r"^item\s+15[\.\s]", re.I), "Item 15. Exhibits"),
]

SECTION_MAP_10Q: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^item\s+1a[\.\s]", re.I), "Item 1A. Risk Factors"),
    (re.compile(r"^item\s+1[\.\s]", re.I), "Item 1. Financial Statements"),
    (re.compile(r"^item\s+2[\.\s]", re.I), "Item 2. MD&A"),
    (re.compile(r"^item\s+3[\.\s]", re.I), "Item 3. Quantitative and Qualitative Disclosures"),
    (re.compile(r"^item\s+4[\.\s]", re.I), "Item 4. Controls and Procedures"),
    (re.compile(r"^item\s+5[\.\s]", re.I), "Item 5. Other Information"),
    (re.compile(r"^item\s+6[\.\s]", re.I), "Item 6. Exhibits"),
]

SECTION_MAP_8K: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^item\s+1\.01", re.I), "Item 1.01. Entry into Material Agreement"),
    (re.compile(r"^item\s+1\.02", re.I), "Item 1.02. Termination of Agreement"),
    (re.compile(r"^item\s+2\.02", re.I), "Item 2.02. Results of Operations"),
    (re.compile(r"^item\s+2\.03", re.I), "Item 2.03. Creation of Direct Financial Obligation"),
    (re.compile(r"^item\s+5\.02", re.I), "Item 5.02. Director/Officer Changes"),
    (re.compile(r"^item\s+7\.01", re.I), "Item 7.01. Regulation FD Disclosure"),
    (re.compile(r"^item\s+8\.01", re.I), "Item 8.01. Other Events"),
    (re.compile(r"^item\s+9\.01", re.I), "Item 9.01. Financial Statements and Exhibits"),
]

SECTION_PHRASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"management.{0,10}discussion\s+and\s+analysis", re.I), "MD&A"),
    (re.compile(r"results\s+of\s+operations", re.I), "Results of Operations"),
    (re.compile(r"liquidity\s+and\s+capital", re.I), "Liquidity and Capital Resources"),
    (re.compile(r"critical\s+accounting", re.I), "Critical Accounting Policies"),
    (re.compile(r"segment\s+information", re.I), "Segment Information"),
    (re.compile(r"notes?\s+to\s+(the\s+)?condensed", re.I), "Notes to Financial Statements"),
    (re.compile(r"notes?\s+to\s+(the\s+)?financial", re.I), "Notes to Financial Statements"),
    (re.compile(r"risk\s+factors", re.I), "Risk Factors"),
    (re.compile(r"financial\s+statements", re.I), "Financial Statements"),
]

_HEADING_ANCHOR_RE = re.compile(r"^(?:PART\s+[IVX]+|Item\s+\d|ITEM\s+\d|Note\s+\d)", re.I)


def _get_section_map(form_type: str) -> list[tuple[re.Pattern, str]]:
    """Return the section map that matches the filing form."""
    ft = (form_type or "").upper()
    if "10-Q" in ft:
        return SECTION_MAP_10Q
    if "8-K" in ft:
        return SECTION_MAP_8K
    return SECTION_MAP_10K


def infer_section(text: str, form_type: str = "") -> Optional[str]:
    """Infer SEC section label from headings or common section phrases."""
    stripped = text.strip()
    text_head = stripped[:200].lower()
    is_short = len(stripped) <= 200
    has_anchor = bool(_HEADING_ANCHOR_RE.match(stripped))

    if is_short or has_anchor:
        for pattern, label in _get_section_map(form_type):
            if pattern.search(text_head):
                return label

    for pattern, label in SECTION_PHRASES:
        if pattern.search(text_head):
            return label

    return None


_FOOTNOTE_PREFIX_RE = re.compile(r"^\s*(\(\d+\)|\d+\.|[a-z]\))\s+", re.MULTILINE)
_FOOTNOTE_PAREN_RE = re.compile(r"^\s*\([^)]{3,80}\)\s*$")
_FOOTNOTE_SUPER_RE = re.compile(r"^[\u00B9\u00B2\u00B3\u2070-\u2079]\s+\S")


def is_footnote(text: str) -> bool:
    """Return True when a block looks like a footnote or annotation."""
    stripped = text.strip()
    if _FOOTNOTE_PREFIX_RE.match(stripped) and len(stripped) < 600:
        return True
    if _FOOTNOTE_PAREN_RE.match(stripped) and len(stripped) <= 120:
        return True
    if _FOOTNOTE_SUPER_RE.match(stripped) and len(stripped) < 300:
        return True
    return False


def _hard_split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split oversized text by words, with char slicing for very long tokens."""
    def _char_slice(token_str: str, max_tok: int) -> list[str]:
        tok = _get_tokenizer()
        if tok is not None:
            ids = tok.encode(token_str)
            return [
                tok.decode(ids[i:i + max_tok])
                for i in range(0, len(ids), max_tok)
                if tok.decode(ids[i:i + max_tok]).strip()
            ]

        chars_per_slice = max(1, int(max_tok * _AVG_CHARS_PER_TOK_FALLBACK))
        return [
            token_str[i:i + chars_per_slice]
            for i in range(0, len(token_str), chars_per_slice)
        ]

    words = text.split()
    parts: list[str] = []
    current_words: list[str] = []

    for word in words:
        if _count_tokens(word) > max_tokens:
            if current_words:
                parts.append(" ".join(current_words))
                current_words = []
            parts.extend(_char_slice(word, max_tokens))
            continue

        candidate = current_words + [word]
        if current_words and _count_tokens(" ".join(candidate)) > max_tokens:
            parts.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words = candidate

    if current_words:
        parts.append(" ".join(current_words))

    return [p for p in parts if p.strip()]


def _sub_split_segment(segment: str, max_tokens: int) -> list[str]:
    """Split long segments by secondary delimiters before using hard splits."""
    for delimiter_re in (
        re.compile(r"(?<=;)\s+"),
        re.compile(r"(?<=:)\s+"),
        re.compile(r"\n+"),
    ):
        parts = [p.strip() for p in delimiter_re.split(segment) if p.strip()]
        if len(parts) > 1:
            merged: list[str] = []
            current_parts: list[str] = []
            current_tok = 0

            for part in parts:
                p_tok = _count_tokens(part)
                if current_tok + p_tok > max_tokens and current_parts:
                    merged.append(" ".join(current_parts))
                    current_parts = [part]
                    current_tok = p_tok
                else:
                    current_parts.append(part)
                    current_tok += p_tok

            if current_parts:
                merged.append(" ".join(current_parts))

            result: list[str] = []
            for merged_part in merged:
                if _count_tokens(merged_part) > max_tokens:
                    result.extend(_hard_split_by_tokens(merged_part, max_tokens))
                else:
                    result.append(merged_part)
            return [r for r in result if r.strip()]

    return _hard_split_by_tokens(segment, max_tokens)


def split_prose(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split prose into overlapping token-bounded chunks."""
    raw_segments = re.split(r"(?<=[.!?])\s+", text.strip())

    segments: list[str] = []
    for seg in raw_segments:
        if _count_tokens(seg) > max_tokens:
            segments.extend(_sub_split_segment(seg, max_tokens))
        else:
            segments.append(seg)

    re_expanded: list[str] = []
    for seg in segments:
        if _count_tokens(seg) > max_tokens:
            re_expanded.extend(_hard_split_by_tokens(seg, max_tokens))
        else:
            re_expanded.append(seg)
    segments = re_expanded

    max_overlap_tokens = int(overlap_tokens * 1.5)
    chunks: list[str] = []
    current: list[str] = []

    def _assembled(segs: list[str]) -> int:
        """Measure tokens on the fully joined string."""
        return _count_tokens(" ".join(segs)) if segs else 0

    def _flush_trim(segs: list[str]) -> list[str]:
        """Trim the front until the assembled segment fits the token limit."""
        while len(segs) > 1 and _assembled(segs) > max_tokens:
            segs.pop(0)
        if _assembled(segs) > max_tokens:
            pieces = _hard_split_by_tokens(" ".join(segs), max_tokens)
            return [p for p in pieces if p.strip()] or segs
        return segs

    for seg in segments:
        candidate = current + [seg]
        if current and _assembled(candidate) > max_tokens:
            chunks.extend(_flush_trim(list(current)))

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
            while len(current) > 1 and _assembled(current) > max_tokens:
                current.pop(0)
        else:
            current = candidate

    if current:
        chunks.extend(_flush_trim(list(current)))

    result = [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]

    for chunk in result:
        tok = _count_tokens(chunk)
        if tok > max_tokens:
            log.warning(
                f"split_prose produced chunk of {tok} tokens over limit {max_tokens} "
                f"(first 60 chars: {chunk[:60]!r})"
            )

    return result


def _assign_pseudo_pages_to_items(items: list[dict], form_type: str = "") -> list[dict]:
    """Assign pseudo-page numbers for HTM documents using block order."""
    if not items:
        return items

    target_pages_map: dict[str, int] = {
        "10-K": 80,
        "10-K405": 80,
        "10-KSB": 80,
        "10-Q": 40,
        "8-K": 10,
        "DEF14A": 60,
    }

    ft = (form_type or "").upper()
    target_pages = next((v for k, v in target_pages_map.items() if ft.startswith(k)), 30)
    blocks_per_page = max(3, len(items) // target_pages)

    for i, item in enumerate(items):
        item["page"] = max(1, (i // blocks_per_page) + 1)

    return items


def _citation_key(page: int, section: str, block_order: int, page_is_surrogate: bool) -> str:
    """Build citation key from real page or HTM pseudo-location."""
    if page_is_surrogate:
        safe_section = (section or "General").replace("|", "-").strip()
        return f"{safe_section} §{block_order}"
    return f"p.{page}"


def _interleave_by_block_order(prose_blocks: list[dict], table_records: list[dict]) -> list[dict]:
    """Merge prose and tables into reading order using block_order."""
    tagged: list[dict] = []

    for block in prose_blocks:
        item = dict(block)
        item["item_kind"] = "prose"
        tagged.append(item)

    for table in table_records:
        item = dict(table)
        item["item_kind"] = "table"
        tagged.append(item)

    with_order = [x for x in tagged if x.get("block_order") is not None]
    without_order = [x for x in tagged if x.get("block_order") is None]

    with_order.sort(key=lambda x: x["block_order"])
    return with_order + without_order


def chunk_document(parsed_doc: dict) -> list[dict]:
    """Convert a parsed document into chunk records."""
    doc_meta = {
        "doc_id": parsed_doc["doc_id"],
        "ticker": parsed_doc["ticker"],
        "company": parsed_doc["company"],
        "form_type": parsed_doc["form_type"],
        "filing_date": parsed_doc["filing_date"],
        "fiscal_year": parsed_doc["fiscal_year"],
        "fiscal_quarter": parsed_doc.get("fiscal_quarter"),
        "period_type": parsed_doc.get("period_type"),
        "period_end_date": parsed_doc.get("period_end_date"),
        "sector": parsed_doc.get("sector", "Unknown"),
        "industry": parsed_doc.get("industry", ""),
        "report_priority": parsed_doc.get("report_priority"),
        "source_url": parsed_doc["source_url"],
    }

    prose_blocks = list(parsed_doc.get("prose_blocks", []))
    table_records = list(parsed_doc.get("table_records", []))
    form_type = doc_meta["form_type"]

    all_blocks_for_surrogate = prose_blocks + table_records
    is_htm_doc = (
        len(all_blocks_for_surrogate) > 0
        and all(block.get("page_is_surrogate", False) for block in all_blocks_for_surrogate)
    )

    if is_htm_doc:
        all_items_for_pages = sorted(
            [dict(block, item_kind="prose") for block in prose_blocks]
            + [dict(table, item_kind="table") for table in table_records],
            key=lambda x: x.get("block_order", 0),
        )
        all_items_for_pages = _assign_pseudo_pages_to_items(all_items_for_pages, form_type=form_type)

        prose_sorted_order = sorted(
            range(len(prose_blocks)),
            key=lambda i: prose_blocks[i].get("block_order", 0),
        )
        table_sorted_order = sorted(
            range(len(table_records)),
            key=lambda i: table_records[i].get("block_order", 0),
        )

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

    ordered_items = _interleave_by_block_order(prose_blocks, table_records)

    chunks: list[dict] = []
    chunk_index = 0
    current_section = "General"

    for item in ordered_items:
        kind = item.get("item_kind")

        if kind == "prose":
            chunks_from_block = _chunk_prose_block(
                item, doc_meta, current_section, form_type, chunk_index
            )
            if chunks_from_block:
                section_from_block = chunks_from_block[0].get("section", current_section)
                if section_from_block and section_from_block != "General":
                    current_section = section_from_block
                chunk_index += len(chunks_from_block)
            chunks.extend(chunks_from_block)

        elif kind == "table":
            if item.get("statement_type") == "cover_admin_table":
                continue
            if item.get("is_cover_page") and not item.get("is_financial_statement"):
                continue

            table_chunks, row_chunks = _chunk_table_record(
                item, doc_meta, current_section, chunk_index
            )

            if table_chunks:
                tbl_section = table_chunks[0].get("section", "")
                if tbl_section and tbl_section not in ("General", "", current_section):
                    current_section = tbl_section
                chunk_index += len(table_chunks)

            chunks.extend(table_chunks)

            if row_chunks:
                first_table_chunk_id = table_chunks[0]["chunk_id"] if table_chunks else None
                for row_chunk in row_chunks:
                    if first_table_chunk_id:
                        row_chunk["parent_chunk_id"] = first_table_chunk_id
                    row_chunk["chunk_id"] = f"{doc_meta['doc_id']}_chunk_{chunk_index:04d}"
                    row_chunk["chunk_index"] = chunk_index
                    chunk_index += 1
                chunks.extend(row_chunks)

    chunks = _deduplicate_chunks(chunks)
    _link_adjacent_chunks(chunks)
    return chunks


def _chunk_prose_block(
    block: dict,
    doc_meta: dict,
    current_section: str,
    form_type: str,
    start_index: int,
) -> list[dict]:
    """Chunk a prose block into one or more prose-like chunks."""
    text = block.get("text", "").strip()
    page = block.get("page", 1)
    block_order = block.get("block_order", 0)
    page_is_surrogate = block.get("page_is_surrogate", False)

    if not text or len(text) < MIN_CHUNK_CHARS:
        return []

    docling_section = block.get("section", "")
    if docling_section and docling_section not in ("General", ""):
        section = docling_section
    else:
        inferred = infer_section(text, form_type)
        section = inferred if inferred else current_section

    block_type = block.get("type", "prose")
    if block_type == "figure_caption":
        chunk_type = "figure_caption"
    elif is_footnote(text):
        chunk_type = "footnote"
    else:
        chunk_type = "prose"

    splits = split_prose(text, PROSE_MAX_TOKENS, PROSE_OVERLAP) if _count_tokens(text) > PROSE_MAX_TOKENS else [text]

    result: list[dict] = []
    parent_chunk_id: Optional[str] = None

    for i, split_text in enumerate(splits):
        chunk_id = f"{doc_meta['doc_id']}_chunk_{start_index + i:04d}"
        if parent_chunk_id is None:
            parent_chunk_id = chunk_id

        result.append({
            "chunk_id": chunk_id,
            "chunk_index": start_index + i,
            "chunk_type": chunk_type,
            "text": split_text,
            "page": page,
            "page_is_surrogate": page_is_surrogate,
            "citation_key": _citation_key(page, section, block_order, page_is_surrogate),
            "block_order": block_order,
            "section": section,
            "parent_chunk_id": parent_chunk_id,
            "next_chunk_id": None,
            "prev_chunk_id": None,
            "table_json": None,
            **doc_meta,
        })

    return result


def _split_rows_by_token_budget(
    data_lines: list[str],
    header_line: str,
    table_header: str,
    max_tokens: int = TABLE_CHUNK_MAX_TOKENS,
) -> list[list[str]]:
    """Group table rows so each group stays within the token budget."""
    if not data_lines:
        return []

    overhead = (
        _count_tokens(f"{table_header}\n{header_line}\n")
        if header_line
        else _count_tokens(f"{table_header}\n")
    )

    groups: list[list[str]] = []
    current_group: list[str] = []
    current_tok = overhead

    for line in data_lines:
        line_tok = _count_tokens(line) + 1
        if (current_tok + line_tok > max_tokens or len(current_group) >= ROW_GROUP_SIZE) and current_group:
            groups.append(current_group)
            current_group = [line]
            current_tok = overhead + line_tok
        else:
            current_group.append(line)
            current_tok += line_tok

    if current_group:
        groups.append(current_group)

    return groups


_HAS_DIGIT_RE = re.compile(r"\d")
_ALPHA_NOISE_RE = re.compile(r"[A-Za-z]")


def _is_numeric_cell(value: str) -> bool:
    """Return True if a table cell looks like numeric financial data."""
    v = value.strip()
    if not v:
        return False
    if v in ("-", "—", "–", "N/A", "n/a", "NM", "nm"):
        return True
    if not _HAS_DIGIT_RE.search(v):
        return False
    return len(_ALPHA_NOISE_RE.findall(v)) <= 12


def _should_emit_row_chunks(tbl: dict) -> bool:
    """Decide whether a table should produce row-level chunks."""
    if tbl.get("is_financial_statement"):
        return True
    if tbl.get("statement_type", "") in _ROW_CHUNK_NOTE_TYPES:
        return True

    cell_grid = tbl.get("cell_grid", [])
    if not cell_grid:
        return False

    numeric = sum(1 for cell in cell_grid if _is_numeric_cell(str(cell.get("value", ""))))
    density = numeric / len(cell_grid)
    return density >= _ROW_CHUNK_NUMERIC_DENSITY_THRESHOLD


def _group_related_rows(cell_grid: list[dict]) -> list[list[dict]]:
    """Group nearby table rows into micro-blocks for semantically related retrieval."""
    if not cell_grid:
        return []

    seen_labels: list[str] = []
    for cell in cell_grid:
        label = cell.get("row_label", "")
        if not seen_labels or seen_labels[-1] != label:
            seen_labels.append(label)

    if len(seen_labels) < MICRO_BLOCK_MIN_ROWS:
        return []

    label_to_cells: dict[str, list[dict]] = {}
    for cell in cell_grid:
        label = cell.get("row_label", "")
        label_to_cells.setdefault(label, []).append(cell)

    micro_groups: list[list[dict]] = []
    current_labels: list[str] = []

    for label in seen_labels:
        is_break = (
            ((not label or label.rstrip().endswith(":")) and len(current_labels) >= 2)
            or len(current_labels) >= MICRO_BLOCK_MAX_ROWS
        )

        if is_break:
            group_cells = [c for lbl in current_labels for c in label_to_cells.get(lbl, [])]
            if len(group_cells) >= 2:
                micro_groups.append(group_cells)
            current_labels = []

        current_labels.append(label)

    if len(current_labels) >= 2:
        group_cells = [c for lbl in current_labels for c in label_to_cells.get(lbl, [])]
        micro_groups.append(group_cells)

    return [group for group in micro_groups if len(group) >= 2]


def _content_hash(chunk: dict) -> str:
    """Hash chunk content with type and citation so only true duplicates collapse."""
    text_norm = re.sub(r"\s+", " ", chunk.get("text", "").lower().strip())
    chunk_type = chunk.get("chunk_type", "")
    citation = chunk.get("citation_key", "")
    fingerprint = f"{chunk_type}|{citation}|{text_norm}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks while preserving the first occurrence."""
    seen: set[str] = set()
    result: list[dict] = []

    for chunk in chunks:
        content_hash = _content_hash(chunk)
        if content_hash in seen:
            log.debug(
                f"Dedup: skipping duplicate chunk {chunk.get('chunk_id')} "
                f"(type={chunk.get('chunk_type')})"
            )
            continue
        seen.add(content_hash)
        result.append(chunk)

    return result


def _chunk_table_record(
    tbl: dict,
    doc_meta: dict,
    current_section: str,
    start_index: int,
) -> tuple[list[dict], list[dict]]:
    """Create table-level chunks and optional row/micro-block chunks."""
    retrieval_text = tbl.get("retrieval_text", "").strip()
    if not retrieval_text or len(retrieval_text) < MIN_CHUNK_CHARS:
        return [], []

    page = tbl.get("page", 1)
    block_order = tbl.get("block_order", 0)
    page_is_surrogate = tbl.get("page_is_surrogate", False)
    section = tbl.get("section") or current_section

    table_header = (
        f"{doc_meta['company']} ({doc_meta['ticker']}) | "
        f"{doc_meta['form_type']} FY{doc_meta['fiscal_year']} | "
        f"{section}"
    )

    rows = retrieval_text.split("\n")
    if len(rows) > 1:
        header_line = rows[0]
        data_lines = rows[1:]
    else:
        header_line = ""
        data_lines = rows

    row_groups = _split_rows_by_token_budget(
        data_lines, header_line, table_header, TABLE_CHUNK_MAX_TOKENS
    )
    if not row_groups:
        row_groups = [data_lines]

    tbl_meta = {
        "page": page,
        "page_is_surrogate": page_is_surrogate,
        "citation_key": _citation_key(page, section, block_order, page_is_surrogate),
        "block_order": block_order,
        "section": section,
        "table_title": tbl.get("table_title", ""),
        "statement_type": tbl.get("statement_type", ""),
        "period_type": tbl.get("period_type", ""),
        "period_signals": tbl.get("period_signals", []),
        "periods": tbl.get("periods", []),
        "units": tbl.get("units", []),
        "is_financial_statement": tbl.get("is_financial_statement", False),
    }

    table_chunks: list[dict] = []
    parent_chunk_id: Optional[str] = None

    for group_idx, row_group in enumerate(row_groups):
        group_content = "\n".join(row_group).strip()
        if not group_content or len(group_content) < MIN_CHUNK_CHARS:
            continue

        group_text = f"{header_line}\n{group_content}" if header_line else group_content
        embedded_text = f"{table_header}\n{group_text}"

        chunk_id = f"{doc_meta['doc_id']}_chunk_{start_index + len(table_chunks):04d}"
        if parent_chunk_id is None:
            parent_chunk_id = chunk_id

        table_chunks.append({
            "chunk_id": chunk_id,
            "chunk_index": start_index + len(table_chunks),
            "chunk_type": "table",
            "text": embedded_text,
            "table_group": group_idx,
            "parent_chunk_id": parent_chunk_id,
            "next_chunk_id": None,
            "prev_chunk_id": None,
            "table_json": tbl.get("cell_grid"),
            **tbl_meta,
            **doc_meta,
        })

    row_chunks: list[dict] = []
    cell_grid = tbl.get("cell_grid", [])

    if _should_emit_row_chunks(tbl) and cell_grid:
        period_type_str = tbl.get("period_type", "") or ""
        units_str = ", ".join(tbl.get("units", [])) if tbl.get("units") else ""
        period_end_str = doc_meta.get("period_end_date", "") or ""
        fiscal_quarter = doc_meta.get("fiscal_quarter")

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
            row_label = cell.get("row_label", "")
            col_header = cell.get("col_header", "")
            value = cell.get("value", "")
            key = (row_label, col_header, value)

            if key in seen_cells:
                continue
            seen_cells.add(key)

            row_text_parts = [
                f"{doc_meta['company']} ({doc_meta['ticker']})",
                f"{doc_meta['form_type']} FY{doc_meta['fiscal_year']}",
                tbl.get("statement_type", ""),
            ]
            if period_context:
                row_text_parts.append(period_context)
            row_text_parts.append(f"{row_label} | {col_header}: {value}")

            row_chunks.append({
                "chunk_id": None,
                "chunk_index": None,
                "chunk_type": "row",
                "text": " | ".join(part for part in row_text_parts if part),
                "parent_chunk_id": None,
                "next_chunk_id": None,
                "prev_chunk_id": None,
                "row_label": row_label,
                "col_header": col_header,
                "value": value,
                "table_json": None,
                **tbl_meta,
                **doc_meta,
            })

    if _should_emit_row_chunks(tbl) and len(cell_grid) >= MICRO_BLOCK_MIN_ROWS:
        micro_groups = _group_related_rows(cell_grid)
        seen_mb_texts: set[str] = set()

        for mb_cells in micro_groups:
            lines = [table_header]
            if header_line:
                lines.append(header_line)

            seen_mb: set[tuple] = set()
            for cell in mb_cells:
                row_label = cell.get("row_label", "")
                col_header = cell.get("col_header", "")
                value = cell.get("value", "")
                key = (row_label, col_header, value)

                if key in seen_mb:
                    continue
                seen_mb.add(key)
                lines.append(f"{row_label} | {col_header}: {value}" if col_header else row_label)

            mb_text = "\n".join(lines)
            if len(mb_text) < MIN_CHUNK_CHARS:
                continue

            mb_norm = re.sub(r"\s+", " ", mb_text.lower().strip())
            if mb_norm in seen_mb_texts:
                log.debug(f"micro_block dedup: skipping duplicate at {tbl_meta['citation_key']}")
                continue
            seen_mb_texts.add(mb_norm)

            row_chunks.append({
                "chunk_id": None,
                "chunk_index": None,
                "chunk_type": "micro_block",
                "text": mb_text,
                "parent_chunk_id": parent_chunk_id,
                "next_chunk_id": None,
                "prev_chunk_id": None,
                "row_label": "",
                "col_header": "",
                "value": "",
                "table_json": mb_cells,
                **tbl_meta,
                **doc_meta,
            })

    return table_chunks, row_chunks


def _link_adjacent_chunks(chunks: list[dict]) -> None:
    """Link only linear reading-order chunks with prev/next references."""
    nonlinear_types = frozenset({"row", "micro_block"})
    linear = [chunk for chunk in chunks if chunk.get("chunk_type") not in nonlinear_types]

    for i, chunk in enumerate(linear):
        if i > 0:
            chunk["prev_chunk_id"] = linear[i - 1]["chunk_id"]
        if i < len(linear) - 1:
            chunk["next_chunk_id"] = linear[i + 1]["chunk_id"]


def save_chunks(chunks: list[dict], doc_id: str) -> Path:
    """Write chunk JSONL atomically for one document."""
    out_path = CHUNKS_DIR / f"{doc_id}.jsonl"
    tmp_path = out_path.with_suffix(".tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        tmp_path.replace(out_path)
    except Exception as e:
        log.error(f"save_chunks failed for {doc_id}: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    return out_path


def load_manifest() -> list[dict]:
    """Load manifest.jsonl into memory."""
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
                log.warning(f"Manifest parse error at line {line_no}: {e}")

    return entries


def _write_manifest(entries: list[dict]) -> None:
    """Write the manifest atomically once after processing."""
    tmp = MANIFEST.with_suffix(".tmp")

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        tmp.replace(MANIFEST)
    except Exception as e:
        log.error(f"Manifest write failed: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def run(tickers: list[str] | None = None, rechunk: bool = False) -> None:
    """Chunk eligible parsed documents and update manifest statuses once."""
    all_entries = load_manifest()
    id_to_idx = {entry.get("doc_id"): i for i, entry in enumerate(all_entries)}

    working = all_entries
    if tickers:
        tickers_upper = {ticker.strip().upper() for ticker in tickers}
        working = [
            entry for entry in all_entries
            if entry.get("ticker", "").upper() in tickers_upper
        ]

    to_chunk = [
        entry for entry in working
        if entry.get("parse_status") == "parsed"
        and (rechunk or entry.get("chunk_status") != "chunked")
    ]

    log.info(f"Chunking {len(to_chunk)} documents (rechunk={rechunk})")

    success = 0
    failed = 0
    total_chunks = 0

    for entry in to_chunk:
        doc_id = entry["doc_id"]
        parsed_path = PARSED_DIR / f"{doc_id}.json"

        if not parsed_path.exists():
            log.warning(f"Parsed file missing: {doc_id}")
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["chunk_status"] = "chunk_failed"
            failed += 1
            continue

        chunk_path = CHUNKS_DIR / f"{doc_id}.jsonl"
        if chunk_path.exists() and not rechunk:
            log.info(f"Already chunked: {doc_id}")
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["chunk_status"] = "chunked"
            success += 1
            continue

        with open(parsed_path, encoding="utf-8") as f:
            parsed_doc = json.load(f)

        for field in (
            "sector",
            "industry",
            "fiscal_quarter",
            "period_type",
            "period_end_date",
            "report_priority",
        ):
            if field not in parsed_doc and field in entry:
                parsed_doc[field] = entry[field]

        try:
            chunks = chunk_document(parsed_doc)
            if not chunks:
                log.warning(f"No chunks produced for {doc_id}")
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
            log.info(f"{doc_id:<50} -> {len(chunks):>5} chunks")

        except Exception as e:
            log.error(f"Chunking failed for {doc_id}: {e}", exc_info=True)
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["chunk_status"] = "chunk_failed"
            failed += 1

    _write_manifest(all_entries)

    log.info("Chunking complete.")
    log.info(f"Documents: success={success} failed={failed}")
    log.info(f"Total chunks produced: {total_chunks}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk parsed SEC filings.")
    parser.add_argument("--tickers", nargs="+", help="Subset of tickers to chunk")
    parser.add_argument("--rechunk", action="store_true", help="Re-chunk already chunked docs")
    args = parser.parse_args()

    run(tickers=args.tickers, rechunk=args.rechunk)