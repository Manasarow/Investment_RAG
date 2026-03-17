"""
Docling-based parser for SEC filings (HTM, PDF, DOCX, XLSX).
Extracts prose blocks, tables, and figure captions with page provenance.
Outputs a structured JSON file per document.
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PARSED_DIR = BASE_DIR / "data" / "parsed"
TABLES_DIR = BASE_DIR / "data" / "tables"
MANIFEST = BASE_DIR / "data_manifest" / "manifest.jsonl"

PARSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

_MANIFEST_LOCK = threading.Lock()
_DOCLING_TYPES: dict = {}
_CONVERTER = None
_CONVERTER_LOCK = threading.Lock()


def _get_docling_types() -> dict:
    """Load Docling item classes lazily so import errors stay non-fatal."""
    global _DOCLING_TYPES

    if _DOCLING_TYPES:
        return _DOCLING_TYPES

    try:
        from docling.datamodel.document import (
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
        )
        _DOCLING_TYPES = {
            "SectionHeaderItem": SectionHeaderItem,
            "TextItem": TextItem,
            "TableItem": TableItem,
            "PictureItem": PictureItem,
        }
    except ImportError:
        log.warning("Could not import Docling item types; falling back to class-name checks.")

    return _DOCLING_TYPES


def _safe_text(value) -> str:
    """Normalize text-like inputs into a clean string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_safe_text(v) for v in value).strip()
    if hasattr(value, "text"):
        return _safe_text(value.text)
    return str(value).strip()


XBRL_NAMED_PATTERNS = [
    re.compile(r"http[s]?://\S+"),
    re.compile(r"\d{10}\s+\d{4}-\d{2}-\d{2}"),
    re.compile(r"(iso4217|xbrli|us-gaap|srt):[A-Za-z]"),
    re.compile(r"0{7}\d{3}\s+\w+:\w+Member"),
    re.compile(r"<[a-z]+:[A-Za-z]+\s"),
]
_COLON_NS_RE = re.compile(r"\b[a-z][a-z0-9]*:[A-Za-z]")


def is_xbrl_noise(text: str) -> bool:
    """Filter namespace-heavy or metadata-heavy inline XBRL noise."""
    if len(text) > 300 and text.count("http") > 4:
        return True
    if sum(1 for p in XBRL_NAMED_PATTERNS if p.search(text)) >= 2:
        return True
    if len(text) > 500 and len(_COLON_NS_RE.findall(text)) > 10:
        return True
    return False


_IMAGE_FILENAME_RE = re.compile(
    r"^\s*[\w\-]+\.(jpg|jpeg|png|gif|svg|webp|tiff|bmp)\s*$",
    re.I,
)


def is_image_filename(text: str) -> bool:
    """Detect standalone image filename artifacts emitted as text."""
    return bool(_IMAGE_FILENAME_RE.match(text))


_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"See\s+accompanying\s+Notes\s+to"
    r"|Apple\s+Inc\.\s*\|\s*Q\d\s*\d{4}"
    r"|\bForm\s+10-[QK]\s*\|\s*\d+\s*$"
    r"|^\s*\|\s*\d+\s*$"
    r")",
    re.I | re.MULTILINE,
)

_COVER_FRAGMENT_RE = re.compile(
    r"^(?:"
    r"\(State\s+or\s+other\s+jurisdiction"
    r"|of\s+incorporation(?:\s+or\s+organization)?\)"
    r"|\(Commission"
    r"|File\s+Number\)"
    r"|\(I\.R\.S\.\s+Employer"
    r"|Identification\s+No\.\)"
    r"|\(Address\s+of\s+principal"
    r"|\(Zip\s+Code\)"
    r"|\(Telephone\s+Number"
    r"|Indicate\s+by\s+check\s+mark\s+whether"
    r")\s*$",
    re.I,
)

_PAREN_NUM_SPILL_RE = re.compile(r"^\s*\(\s*[\d,]+(?:\.\d+)?\s*\)\s*$")
_DATE_FRAG_SPILL_RE = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},\s*$",
    re.I,
)

_COLUMN_HEADER_SPILL_RE = re.compile(
    r"^(?:"
    r"Adjusted"
    r"|Unrealized"
    r"|Marketable"
    r"|Non-[Cc]urrent"
    r"|Equivalents?"
    r"|Fair\s+Value\s*$"
    r"|Cash\s+and\s*$"
    r"|Gains?\s*$"
    r"|Losses?\s*$"
    r"|Cost\s*$"
    r")\s*$",
    re.I,
)


def is_boilerplate(text: str) -> bool:
    """Detect recurring footer and boilerplate prose."""
    return bool(_BOILERPLATE_RE.search(text))


def is_cover_fragment(text: str) -> bool:
    """Detect layout fragments from filing cover pages."""
    return bool(_COVER_FRAGMENT_RE.match(text))


def is_table_spill(text: str) -> bool:
    """Detect obvious table fragments that should not be kept as prose."""
    return bool(
        _PAREN_NUM_SPILL_RE.match(text)
        or _DATE_FRAG_SPILL_RE.match(text)
        or _COLUMN_HEADER_SPILL_RE.match(text)
    )


_SEC_HEADING_RE = re.compile(
    r"^(?:"
    r"PART\s+[IVX]+\b"
    r"|Item\s+\d+[A-Za-z]?\."
    r"|Note\s+\d+[\s\-\u2013\u2014]"
    r"|NOTES?\s+TO\s+"
    r"|CONDENSED\s+CONSOLIDATED\s+"
    r")",
    re.I,
)

_NOTE_BODY_TRIGGER_RE = re.compile(
    r"(?<= )"
    r"\b(The|A\b|An\b|During|In\b|As\b|For\b|This|With|Each|All|These|"
    r"Such|Any|Upon|Under|If\b|When|Pursuant|Effective|Following|Including|"
    r"Certain|At\b|By\b|No\b|Our|Its|From)\b",
    re.I,
)


def _extract_note_title(text: str) -> str:
    """Trim long Note headings before they spill into body prose."""
    m = re.match(r"^(Note\s+\d+)\s*[\-\u2013\u2014]\s*", text, re.I)
    if not m:
        return text[:80]

    rest = text[m.end():]
    trigger = _NOTE_BODY_TRIGGER_RE.search(rest)
    title_part = rest[:trigger.start()].strip().rstrip(",").strip() if trigger else rest.strip()
    return (m.group(1) + " - " + title_part).rstrip(" -").strip() or text[:80]


def _extract_short_heading(text: str) -> Optional[str]:
    """Extract a short SEC-style heading from a text block when present."""
    if not _SEC_HEADING_RE.match(text.strip()):
        return None

    stripped = text.strip()
    if len(stripped) <= 120:
        return _extract_note_title(stripped) if re.match(r"^Note\s+\d+", stripped, re.I) else stripped

    if re.match(r"^Note\s+\d+", stripped, re.I):
        return _extract_note_title(stripped)

    m = re.match(
        r"^((?:PART\s+[IVX]+[^.]*?|Item\s+\d+[A-Za-z]?\.\s*[^.]{0,60}|"
        r"CONDENSED\s+CONSOLIDATED\s+[A-Z\s]+))",
        stripped,
        re.I,
    )
    return m.group(1).strip() if m else stripped[:80]


_FIN_STMT_TITLE_RE = re.compile(
    r"(?:Apple\s+Inc\.\s+)?(?:CONDENSED\s+)?CONSOLIDATED\s+(?:CONDENSED\s+)?"
    r"(?:STATEMENTS?\s+OF\s+(?:OPERATIONS|INCOME|EARNINGS|COMPREHENSIVE\s+INCOME|"
    r"CASH\s+FLOWS?|SHAREHOLDERS|STOCKHOLDERS)|BALANCE\s+SHEET)",
    re.I,
)


def _extract_fin_stmt_title(text: str) -> Optional[str]:
    """Return a normalized financial statement title when detected."""
    if not _FIN_STMT_TITLE_RE.search(text):
        return None

    cleaned = re.sub(r"\s*\((?:Unaudited|In millions[^)]*)\).*", "", text, flags=re.I).strip()
    cleaned = re.sub(r"^Apple\s+Inc\.\s+", "", cleaned, flags=re.I).strip()
    return cleaned if len(cleaned) >= 10 else None


_COVER_ADMIN_RE = re.compile(
    r"(?:"
    r"title\s+of\s+each\s+class"
    r"|trading\s+symbol"
    r"|exchange\s+on\s+which\s+registered"
    r"|accelerated\s+filer"
    r"|emerging\s+growth\s+company"
    r"|^\s*Part\s+[IVX]+\s*/\s*Item\s+\d"
    r"|state\s+or\s+other\s+jurisdiction\s+of\s+incorporation"
    r"|i\.r\.s\.\s+employer\s+identification"
    r")",
    re.I,
)


def is_cover_admin_table(header_text: str, retrieval_text: str) -> bool:
    """Detect cover-page, admin, and TOC tables that are not retrieval targets."""
    combined = f"{header_text} {retrieval_text[:300]}"
    return bool(_COVER_ADMIN_RE.search(combined))


_STATEMENT_SCORING: list[tuple[re.Pattern, re.Pattern, str]] = [
    (
        re.compile(
            r"consolidated\s+statements?\s+of\s+"
            r"(operations|income|earnings|profit)",
            re.I,
        ),
        re.compile(r"net\s+(sales|revenue)|gross\s+(profit|margin)|operating\s+income", re.I),
        "income_statement",
    ),
    (
        re.compile(
            r"consolidated\s+statements?\s+of\s+comprehensive\s+income|"
            r"other\s+comprehensive\s+income",
            re.I,
        ),
        re.compile(r"comprehensive\s+income|unrealized\s+(gains?|losses?)", re.I),
        "comprehensive_income_statement",
    ),
    (
        re.compile(
            r"consolidated\s+balance\s+sheet|"
            r"consolidated\s+statements?\s+of\s+financial\s+position",
            re.I,
        ),
        re.compile(r"total\s+assets|total\s+liabilities", re.I),
        "balance_sheet",
    ),
    (
        re.compile(r"consolidated\s+statements?\s+of\s+cash\s+flows?", re.I),
        re.compile(
            r"cash\s+(used|provided|from)\s+(in|by)\s+"
            r"(operating|investing|financing)",
            re.I,
        ),
        "cash_flow_statement",
    ),
    (
        re.compile(
            r"consolidated\s+statements?\s+of\s+(stockholders|shareholders).equity|"
            r"changes\s+in\s+(stockholders|shareholders).equity",
            re.I,
        ),
        re.compile(
            r"common\s+stock\s+and\s+additional|retained\s+earnings|"
            r"accumulated\s+other\s+comprehensive",
            re.I,
        ),
        "equity_statement",
    ),
    (
        re.compile(
            r"segment\s+(information|results|data|revenue)|"
            r"(revenue|income)\s+by\s+(segment|geography|region)",
            re.I,
        ),
        re.compile(r"segment|geographic|region", re.I),
        "segment_table",
    ),
    (
        re.compile(r"earnings\s+per\s+share|per\s+share\s+data", re.I),
        re.compile(r"\beps\b|basic|diluted", re.I),
        "eps_table",
    ),
    (
        re.compile(
            r"(long.term\s+)?debt|borrowings?|"
            r"notes?\s+payable|credit\s+facilit",
            re.I,
        ),
        re.compile(r"maturity|interest\s+rate|principal|coupon", re.I),
        "debt_table",
    ),
    (
        re.compile(
            r"stock.based\s+compensation|share.based|"
            r"stock\s+options?|restricted\s+stock\s+units?",
            re.I,
        ),
        re.compile(r"grant|vest|exercise|fair\s+value", re.I),
        "compensation_table",
    ),
]


def classify_statement_type(title: str, header_text: str, retrieval_text: str = "") -> str:
    """Classify a table using title and header-based scoring."""
    if is_cover_admin_table(header_text, retrieval_text):
        return "cover_admin_table"

    best_label = "other_table"
    best_score = 0

    for title_pat, header_pat, label in _STATEMENT_SCORING:
        score = (3 if title_pat.search(title) else 0) + (1 if header_pat.search(header_text) else 0)
        if score > best_score:
            best_score, best_label = score, label

    return best_label


_PERIOD_ANNUAL_RE = re.compile(r"year\s+ended|annual|twelve\s+months|52[- ]weeks?|53[- ]weeks?", re.I)
_PERIOD_QTRLY_RE = re.compile(r"quarter(?:ly)?|three\s+months|thirteen\s+weeks|nine\s+months|six\s+months", re.I)
_PERIOD_INSTANT_RE = re.compile(r"as\s+of\b|balance\s+at\b", re.I)
_PERIOD_MATURITY_RE = re.compile(
    r"due\s+(after|within|in)|thereafter|remaining\s+(months?|year)|"
    r"maturities|maturity\s+date",
    re.I,
)
_FILING_DATE_RE = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"\s+\d{1,2},[ \t]+\d{4}$",
    re.I,
)


def detect_period_info(header_row: list[str], row_labels: list[str] | None = None) -> dict:
    """Detect broad period semantics from headers and row labels."""
    joined = " ".join(header_row)
    row_joined = " ".join(row_labels or [])

    signals = []
    if _PERIOD_MATURITY_RE.search(joined) or _PERIOD_MATURITY_RE.search(row_joined):
        signals.append("maturity_schedule")
    if _PERIOD_ANNUAL_RE.search(joined):
        signals.append("annual")
    if _PERIOD_QTRLY_RE.search(joined):
        signals.append("quarterly")
    if _PERIOD_INSTANT_RE.search(joined):
        signals.append("instant")

    if "maturity_schedule" in signals:
        period_type = "maturity_schedule"
    elif len(signals) == 0:
        period_type = "unknown"
    elif len(signals) == 1:
        period_type = signals[0]
    else:
        period_type = "mixed"

    return {"period_type": period_type, "period_signals": signals}


_FIN_DOLLAR_RE = re.compile(r"\$\s*[\d,]{4,}")
_FIN_INTEGER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{5,}\b")
_COVER_TEXT_RE = re.compile(r"\$0\.\d{5}|par\s+value|CUSIP|ISIN", re.I)


def is_financial_statement_table(statement_type: str, grid: list[list[str]]) -> bool:
    """Decide whether a table is likely a true financial statement."""
    if statement_type in (
        "income_statement",
        "balance_sheet",
        "cash_flow_statement",
        "equity_statement",
        "comprehensive_income_statement",
    ):
        return True
    if statement_type == "cover_admin_table":
        return False

    all_text = " ".join(cell for row in grid for cell in row)
    if _COVER_TEXT_RE.search(all_text):
        return False

    return (bool(_FIN_DOLLAR_RE.search(all_text)) or bool(_FIN_INTEGER_RE.search(all_text))) and len(grid) >= 3


def is_cover_page_table(grid: list[list[str]]) -> bool:
    """Soft heuristic for tiny non-financial cover-page tables."""
    all_text = " ".join(cell for row in grid for cell in row)
    total_cells = sum(len(row) for row in grid)
    return not _FIN_INTEGER_RE.search(all_text) and total_cells <= 12


def _build_converter():
    """Build the Docling converter with per-format options."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(do_ocr=True, do_table_structure=True)
            ),
            InputFormat.DOCX: WordFormatOption(),
        }
    )


def get_converter():
    """Return the singleton Docling converter instance."""
    global _CONVERTER

    if _CONVERTER is None:
        with _CONVERTER_LOCK:
            if _CONVERTER is None:
                log.info("Initialising Docling converter (one-time load)...")
                _CONVERTER = _build_converter()

    return _CONVERTER


def get_page_range(item) -> tuple[int, int, list[int]]:
    """Extract page provenance from a Docling item when available."""
    try:
        if hasattr(item, "prov") and item.prov:
            pages = sorted({p.page_no for p in item.prov})
            return pages[0], pages[-1], pages
    except Exception:
        pass
    return 1, 1, [1]


def _extract_grid(table) -> list[list[str]]:
    """Convert a Docling table into a compact 2D grid."""
    try:
        td = table.data
        cells = td.table_cells
        num_rows = td.num_rows
        num_cols = td.num_cols

        if not cells or num_rows == 0 or num_cols == 0:
            return []

        grid = [[""] * num_cols for _ in range(num_rows)]
        filled = [[False] * num_cols for _ in range(num_rows)]

        for cell in cells:
            r, c = cell.start_row_offset_idx, cell.start_col_offset_idx
            if r >= num_rows or c >= num_cols or filled[r][c]:
                continue

            text = _safe_text(cell.text)
            if text in ("<!-- rich cell -->", "<!--rich cell-->"):
                text = ""

            grid[r][c] = text
            filled[r][c] = True

        non_empty_cols = [c for c in range(num_cols) if any(grid[r][c] for r in range(num_rows))]
        if not non_empty_cols:
            return []

        grid = [[row[c] for c in non_empty_cols] for row in grid]

        seen, deduped = set(), []
        for row in grid:
            key = "|".join(row)
            if key not in seen and any(cell for cell in row):
                seen.add(key)
                deduped.append(row)

        return deduped

    except Exception as e:
        log.warning(f"    _extract_grid failed: {e}")
        return []


_NUM_CELL_RE = re.compile(r"^\s*[\$\(]?[\d,]+\.?\d*[BKMG%\)]?\s*$")
_PURE_NUM_RE = re.compile(r"^\s*[\(\-]?[\d,]+\.?\d*\)?\s*$")
_UNIT_TOKEN_RE = re.compile(r"^\s*[\$%€£¥]\s*$")


def _row_numeric_density(row: list[str]) -> float:
    """Measure how numeric a row is to separate header rows from data rows."""
    non_empty = [c for c in row if c]
    if not non_empty:
        return 0.0
    return sum(1 for c in non_empty if _NUM_CELL_RE.match(c)) / len(non_empty)


def _split_header_and_data(grid: list[list[str]]) -> tuple[list[str], list[list[str]], list[str]]:
    """Split table rows into merged headers, data rows, and stripped unit tokens."""
    if not grid:
        return [], [], []

    header_row_count = 1
    for i in range(1, min(4, len(grid))):
        if _row_numeric_density(grid[i]) < 0.5:
            header_row_count += 1
        else:
            break

    header_rows = grid[:header_row_count]
    data_rows = list(grid[header_row_count:])
    num_cols = len(header_rows[0])

    merged_header = []
    units = []
    absorbed_row = [""] * num_cols

    for c in range(num_cols):
        parts, col_units, col_absorbed = [], [], ""
        for r in range(header_row_count):
            val = header_rows[r][c] if c < len(header_rows[r]) else ""
            if not val:
                continue
            if _PURE_NUM_RE.match(val):
                col_absorbed = val
            elif _UNIT_TOKEN_RE.match(val):
                col_units.append(val.strip())
            else:
                parts.append(val)

        merged_header.append(" / ".join(parts) if parts else "")
        units.extend(col_units)
        absorbed_row[c] = col_absorbed

    if any(absorbed_row):
        data_rows.insert(0, absorbed_row)

    return merged_header, data_rows, list(dict.fromkeys(units))


def _drop_phantom_columns(header: list[str], data_rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Drop columns that are empty in both header and data."""
    if not header:
        return header, data_rows

    keep = []
    for c in range(len(header)):
        if header[c]:
            keep.append(c)
        elif any(row[c] for row in data_rows if c < len(row)):
            keep.append(c)

    if len(keep) == len(header):
        return header, data_rows

    new_header = [header[c] for c in keep]
    new_data_rows = [[row[c] for c in keep if c < len(row)] for row in data_rows]
    return new_header, new_data_rows


def _compact_value_columns(header: list[str], data_rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Fold empty-header value columns into their preceding labeled column."""
    if not header:
        return header, data_rows

    n = len(header)
    col_owner = list(range(n))

    for c in range(1, n):
        if not header[c] and header[c - 1]:
            col_owner[c] = c - 1

    if col_owner == list(range(n)):
        return header, data_rows

    keep_cols = sorted(set(col_owner))
    new_header = [header[c] for c in keep_cols]
    keep_idx = {c: i for i, c in enumerate(keep_cols)}

    new_data: list[list[str]] = []
    for row in data_rows:
        new_row = [""] * len(keep_cols)
        for orig_c in range(n):
            owner_c = col_owner[orig_c]
            new_c = keep_idx[owner_c]
            val = row[orig_c] if orig_c < len(row) else ""
            if val and not _UNIT_TOKEN_RE.match(val):
                if not new_row[new_c]:
                    new_row[new_c] = val
        new_data.append(new_row)

    return new_header, new_data


_YEAR_HEADER_RE = re.compile(r"\b(19|20)\d{2}\b|Q[1-4]|quarter|ended", re.I)


def _infer_label_col(header: list[str], data_rows: list[list[str]]) -> int:
    """Infer the leftmost mostly-text column that acts as the row label."""
    for col_idx in range(len(header)):
        if _YEAR_HEADER_RE.search(header[col_idx]):
            continue

        values = [row[col_idx] for row in data_rows if col_idx < len(row) and row[col_idx]]
        if not values:
            continue

        if sum(1 for v in values if not _NUM_CELL_RE.match(v)) / len(values) >= 0.6:
            return col_idx

    return 0


def _render_table_text(header: list[str], data_rows: list[list[str]], label_col: int = 0) -> str:
    """Render a table into retrieval-friendly text."""
    lines = []
    header_line = " | ".join(h for h in header if h)
    if header_line:
        lines.append(header_line)

    for row in data_rows:
        parts = []
        for col_idx, val in enumerate(row):
            if not val:
                continue

            if col_idx == label_col:
                parts.append(val)
                continue

            col_name = header[col_idx] if col_idx < len(header) else ""
            parts.append(f"{col_name}: {val}" if col_name and col_name != val else val)

        if parts:
            lines.append(" | ".join(parts))

    return "\n".join(lines)


def build_table_record(
    table,
    page_start: int,
    page_end: int,
    pages: list[int],
    page_is_surrogate: bool,
    section: str,
    table_title: str,
    doc_meta: dict,
) -> Optional[dict]:
    """Build the structured output record for one table."""
    try:
        grid = _extract_grid(table)
        if not grid:
            return None

        header, data_rows, units = _split_header_and_data(grid)
        if not header:
            return None

        header, data_rows = _drop_phantom_columns(header, data_rows)
        if not header:
            return None

        header, data_rows = _compact_value_columns(header, data_rows)
        if not header:
            return None

        header_text = " ".join(h for h in header if h)
        label_col = _infer_label_col(header, data_rows)
        row_labels = [row[label_col] if label_col < len(row) else "" for row in data_rows]

        retrieval_text = _render_table_text(header, data_rows, label_col)
        if not retrieval_text:
            return None

        statement_type = classify_statement_type(table_title, header_text, retrieval_text)
        period_info = detect_period_info(header, row_labels)

        cell_grid = []
        for row in data_rows:
            row_label = row[label_col] if label_col < len(row) else ""
            for col_idx, val in enumerate(row):
                if col_idx == label_col or not val:
                    continue
                if _UNIT_TOKEN_RE.match(val):
                    continue
                if col_idx < len(header):
                    cell_grid.append({
                        "row_label": row_label,
                        "col_header": header[col_idx],
                        "value": val,
                    })

        if statement_type == "cover_admin_table":
            period_cols = []
        else:
            period_cols = [
                h for h in header
                if (re.search(r"\b(19|20)\d{2}\b", h) or re.search(r"Q[1-4]|quarter|months|year|ended|fiscal", h, re.I))
                and not _FILING_DATE_RE.match(h)
            ]

        is_cover = True if statement_type == "cover_admin_table" else is_cover_page_table(grid)
        is_fin_stm = is_financial_statement_table(statement_type, grid)

        return {
            "retrieval_text": retrieval_text,
            "column_headers": header,
            "row_labels": row_labels,
            "label_col_index": label_col,
            "cell_grid": cell_grid,
            "periods": period_cols,
            "units": units,
            "table_title": table_title,
            "statement_type": statement_type,
            "period_type": period_info["period_type"],
            "period_signals": period_info["period_signals"],
            "is_financial_statement": is_fin_stm,
            "is_cover_page": is_cover,
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "pages": pages,
            "page_is_surrogate": page_is_surrogate,
            "section": section,
            **doc_meta,
        }

    except Exception as e:
        log.warning(f"    build_table_record failed: {e}")
        return None


_FIGURE_FALLBACK = "[Figure present -- no caption or metadata available]"


def _describe_figure(item, caption: str) -> str:
    """Build a best-effort textual description for a figure."""
    parts = [caption] if caption else []

    try:
        if hasattr(item, "annotations") and item.annotations:
            for ann in item.annotations:
                ann_text = _safe_text(getattr(ann, "text", None) or ann)
                if ann_text and ann_text not in parts:
                    parts.append(ann_text)
    except Exception:
        pass

    try:
        if hasattr(item, "image") and item.image:
            img = item.image
            if hasattr(img, "uri") and img.uri:
                parts.append(f"[Figure URI: {img.uri}]")
            elif hasattr(img, "pil_image") and img.pil_image is not None:
                w, h = img.pil_image.size
                parts.append(f"[Figure: {w}x{h}px]")
    except Exception:
        pass

    return " | ".join(parts) if parts else _FIGURE_FALLBACK


def parse_excel(raw_path: Path, doc_meta: dict) -> Optional[dict]:
    """Parse .xlsx spreadsheets with openpyxl into table records."""
    if raw_path.suffix.lower() == ".xls":
        log.error(f"  Legacy .xls not supported. Convert {raw_path.name} to .xlsx first.")
        return None

    try:
        import openpyxl
    except ImportError:
        log.error("openpyxl not installed. Run: pip install openpyxl")
        return None

    try:
        wb = openpyxl.load_workbook(str(raw_path), data_only=True)
    except Exception as e:
        log.error(f"  openpyxl failed: {e}")
        return None

    table_records = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if ws.sheet_state != "visible":
            continue

        raw_rows = []
        for row in ws.iter_rows(values_only=True):
            str_row = [str(c) if c is not None else "" for c in row]
            if any(c for c in str_row):
                raw_rows.append(str_row)

        if not raw_rows:
            continue

        header, data_rows, units = _split_header_and_data(raw_rows)
        if not header or not data_rows:
            continue

        header, data_rows = _drop_phantom_columns(header, data_rows)
        header, data_rows = _compact_value_columns(header, data_rows)

        label_col = _infer_label_col(header, data_rows)
        retrieval_text = _render_table_text(header, data_rows, label_col)
        row_labels = [row[label_col] if label_col < len(row) else "" for row in data_rows]
        period_info = detect_period_info(header, row_labels)
        statement_type = classify_statement_type(sheet_name, " ".join(header), retrieval_text)

        cell_grid = []
        for row in data_rows:
            row_label = row[label_col] if label_col < len(row) else ""
            for col_idx, val in enumerate(row):
                if col_idx == label_col or not val or _UNIT_TOKEN_RE.match(val):
                    continue
                if col_idx < len(header):
                    cell_grid.append({
                        "row_label": row_label,
                        "col_header": header[col_idx],
                        "value": val,
                    })

        if statement_type == "cover_admin_table":
            period_cols = []
            is_cover = True
        else:
            period_cols = [
                h for h in header
                if (re.search(r"\b(19|20)\d{2}\b", h) or re.search(r"Q[1-4]|quarter|months|year|ended|fiscal", h, re.I))
                and not _FILING_DATE_RE.match(h)
            ]
            is_cover = is_cover_page_table([[c for c in row] for row in (data_rows or [[]])])

        table_records.append({
            "retrieval_text": retrieval_text,
            "column_headers": header,
            "row_labels": row_labels,
            "label_col_index": label_col,
            "cell_grid": cell_grid,
            "periods": period_cols,
            "units": units,
            "table_title": sheet_name,
            "statement_type": statement_type,
            "period_type": period_info["period_type"],
            "period_signals": period_info["period_signals"],
            "is_financial_statement": bool(_FIN_INTEGER_RE.search(retrieval_text)),
            "is_cover_page": is_cover,
            "page": 1,
            "page_start": 1,
            "page_end": 1,
            "pages": [1],
            "page_is_surrogate": False,
            "section": sheet_name,
            **doc_meta,
        })

    if not table_records:
        log.warning(f"  No usable sheets in {raw_path.name}")
        return None

    output = {
        **doc_meta,
        "raw_path": str(raw_path),
        "prose_blocks": [],
        "table_records": table_records,
        "stats": {"prose_count": 0, "table_count": len(table_records)},
    }
    log.info(f"  Excel parsed: {len(table_records)} sheet(s)")
    return output


def parse_document(raw_path: str, doc_meta: dict) -> Optional[dict]:
    """Parse one filing into prose blocks and structured table records."""
    p = Path(raw_path)
    if not p.is_absolute():
        p = BASE_DIR / p
    raw_path = p

    if not raw_path.exists():
        log.error(f"  File not found: {raw_path}")
        return None

    suffix = raw_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return parse_excel(raw_path, doc_meta)

    log.info(f"  Parsing {raw_path.name} ...")
    try:
        result = get_converter().convert(str(raw_path))
        doc = result.document
    except Exception as e:
        log.error(f"  Docling conversion failed for {raw_path.name}: {e}")
        return None

    is_htm = suffix in (".htm", ".html")
    dtypes = _get_docling_types()

    def _is_type(item, name: str) -> bool:
        """Compare item types safely even when Docling types failed to import."""
        cls = dtypes.get(name)
        return isinstance(item, cls) if cls is not None else type(item).__name__ == name

    prose_blocks = []
    table_records = []
    current_section = "General"
    pending_table_title = "General"
    block_order = 0

    try:
        for item, level in doc.iterate_items():
            block_order += 1

            if _is_type(item, "SectionHeaderItem"):
                text = _safe_text(item.text)
                if text:
                    current_section = text
                    pending_table_title = text

            elif _is_type(item, "TextItem"):
                text = _safe_text(item.text)
                if len(text) < 8:
                    continue
                if is_xbrl_noise(text):
                    continue
                if is_boilerplate(text):
                    continue
                if is_image_filename(text):
                    continue
                if is_cover_fragment(text):
                    continue
                if is_table_spill(text):
                    continue

                fin_title = _extract_fin_stmt_title(text)
                if fin_title:
                    pending_table_title = fin_title
                    current_section = "Item 1 - Financial Statements"
                    continue

                heading = _extract_short_heading(text)
                if heading:
                    current_section = heading
                    pending_table_title = heading
                    if len(text.strip()) <= 120:
                        continue
                    body = text[len(heading):].strip()
                    if len(body) < 8:
                        continue
                    text = body

                ps, pe, pages = get_page_range(item)
                prose_blocks.append({
                    "text": text,
                    "page": ps,
                    "page_start": ps,
                    "page_end": pe,
                    "pages": pages,
                    "page_is_surrogate": is_htm,
                    "section": current_section,
                    "type": "prose",
                    "block_order": block_order,
                })

            elif _is_type(item, "TableItem"):
                ps, pe, pages = get_page_range(item)
                title = pending_table_title
                try:
                    cap = _safe_text(item.caption)
                    if cap:
                        title = cap
                except Exception:
                    pass

                record = build_table_record(
                    item,
                    page_start=ps,
                    page_end=pe,
                    pages=pages,
                    page_is_surrogate=is_htm,
                    section=current_section,
                    table_title=title,
                    doc_meta=doc_meta,
                )
                if record:
                    record["block_order"] = block_order
                    table_records.append(record)
                    pending_table_title = current_section

            elif _is_type(item, "PictureItem"):
                caption = ""
                try:
                    caption = _safe_text(item.caption) or _safe_text(item.text)
                except Exception:
                    pass

                description = _describe_figure(item, caption)
                if description == _FIGURE_FALLBACK:
                    continue

                ps, pe, pages = get_page_range(item)
                prose_blocks.append({
                    "text": f"[Figure] {description}",
                    "page": ps,
                    "page_start": ps,
                    "page_end": pe,
                    "pages": pages,
                    "page_is_surrogate": is_htm,
                    "section": current_section,
                    "type": "figure_caption",
                    "block_order": block_order,
                })

    except Exception as e:
        log.warning(f"  Item iteration partial failure: {e}")

    if not prose_blocks and not table_records:
        log.warning(f"  No content extracted from {raw_path.name}")
        return None

    try:
        file_size_kb = raw_path.stat().st_size / 1024
        min_expected = max(5, int(file_size_kb * 0.02))
        if len(prose_blocks) < min_expected and suffix == ".pdf":
            log.warning(
                f"  Sanity check: only {len(prose_blocks)} prose blocks from "
                f"{file_size_kb:.0f} KB PDF (expected >={min_expected})."
            )
    except Exception:
        pass

    return {
        "doc_id": doc_meta["doc_id"],
        "ticker": doc_meta["ticker"],
        "company": doc_meta["company"],
        "form_type": doc_meta["form_type"],
        "filing_date": doc_meta["filing_date"],
        "fiscal_year": doc_meta["fiscal_year"],
        "source_url": doc_meta["source_url"],
        "raw_path": str(raw_path),
        "prose_blocks": prose_blocks,
        "table_records": table_records,
        "stats": {
            "prose_count": len(prose_blocks),
            "table_count": len(table_records),
        },
    }


def _atomic_json_write(path: Path, data) -> None:
    """Write JSON atomically via a temporary file and replace."""
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception as e:
        log.error(f"  Atomic write failed for {path}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def save_parsed(output: dict) -> Path:
    """Persist the parsed document JSON and optional tables JSON."""
    doc_id = output["doc_id"]
    parsed_path = PARSED_DIR / f"{doc_id}.json"
    _atomic_json_write(parsed_path, output)

    if output["table_records"]:
        _atomic_json_write(TABLES_DIR / f"{doc_id}_tables.json", output["table_records"])

    return parsed_path


def load_manifest() -> list[dict]:
    """Load manifest entries, skipping malformed lines."""
    if not MANIFEST.exists():
        return []

    entries, bad = [], 0
    with open(MANIFEST, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"  Manifest parse error at line {line_no}: {e}")
                bad += 1

    if bad:
        log.warning(f"  {bad} malformed line(s) skipped in manifest.")

    return entries


def _write_manifest(entries: list[dict]) -> None:
    """Rewrite the manifest atomically from the in-memory entry list."""
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


def run(tickers: list[str] | None = None, reparse: bool = False) -> None:
    """Parse pending manifest documents and flush manifest status once per run."""
    with _MANIFEST_LOCK:
        all_entries = load_manifest()

    id_to_idx = {e.get("doc_id"): i for i, e in enumerate(all_entries)}
    working = all_entries

    if tickers:
        tickers_upper = {t.strip().upper() for t in tickers}
        working = [e for e in all_entries if e.get("ticker", "").upper() in tickers_upper]

    pending = [e for e in working if reparse or e.get("parse_status") == "pending"]
    log.info(f"Parsing {len(pending)} documents (reparse={reparse})")

    success = failed = 0
    for entry in pending:
        doc_id = entry["doc_id"]
        rp = Path(entry["raw_path"])
        if not rp.is_absolute():
            rp = BASE_DIR / rp

        parsed_path = PARSED_DIR / f"{doc_id}.json"
        if parsed_path.exists() and not reparse:
            log.info(f"  Already parsed: {doc_id}")
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                all_entries[idx]["parse_status"] = "parsed"
            success += 1
            continue

        doc_meta = {
            "doc_id": doc_id,
            "ticker": entry["ticker"],
            "company": entry.get("company", entry["ticker"]),
            "form_type": entry["form_type"],
            "filing_date": entry["filing_date"],
            "fiscal_year": entry["fiscal_year"],
            "source_url": entry.get("source_url", ""),
        }

        output = parse_document(str(rp), doc_meta)
        new_status = "parsed" if output else "failed"
        idx = id_to_idx.get(doc_id)
        if idx is not None:
            all_entries[idx]["parse_status"] = new_status

        if output:
            save_parsed(output)
            success += 1
        else:
            failed += 1

    with _MANIFEST_LOCK:
        _write_manifest(all_entries)

    log.info(f"\nParsing complete. Success: {success} | Failed: {failed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse SEC filings with Docling.")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--reparse", action="store_true")
    args = parser.parse_args()

    run(tickers=args.tickers, reparse=args.reparse)