"""
SEC EDGAR downloader for 10-K, 10-Q, and 8-K filings.
Downloads raw PDFs/HTML files and writes one manifest entry per document.
"""

import csv
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from tenacity import retry, wait_exponential, stop_after_attempt
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
MANIFEST = BASE_DIR / "data_manifest" / "manifest.jsonl"
TOP50_CSV = BASE_DIR / "data_manifest" / "top50.csv"
LOGS_DIR = BASE_DIR / "logs"

USER_AGENT = os.getenv("SEC_USER_AGENT", "YourName your@email.com")
DATE_FROM = "2023-01-01"
TARGET_FORMS = ["10-K", "10-K405", "10-KSB", "10-Q", "8-K"]

MAX_WORKERS = int(os.getenv("SEC_MAX_WORKERS", "5"))
RATE_LIMIT_RPS = float(os.getenv("SEC_RATE_LIMIT_RPS", "8"))


def _setup_logging() -> logging.Logger:
    """Configure console and rotating-file logging for the downloader."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s — %(message)s"
    )
    console = logging.StreamHandler()
    file_handler = RotatingFileHandler(
        LOGS_DIR / "sec_downloader.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    console.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    logger = logging.getLogger("sec_downloader")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(console)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


log = _setup_logging()


class _RateLimiter:
    """Thread-safe token-bucket limiter shared across all workers."""

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one request token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                delta = now - self._last
                self._last = now
                self._tokens = min(self._rate, self._tokens + delta * self._rate)

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                wait = (1.0 - self._tokens) / self._rate

            time.sleep(wait)


_rate_limiter = _RateLimiter(RATE_LIMIT_RPS)


_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Create one requests.Session per thread so connections can be reused."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(total=0),
            pool_connections=10,
            pool_maxsize=20,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        _thread_local.session = session

    return _thread_local.session


def _req_headers(url: str) -> dict:
    """Return the Host header derived from the target URL."""
    return {"Host": urlparse(url).netloc}


class _ManifestWriter:
    """Buffer manifest entries and flush them to disk in batches."""

    FLUSH_SIZE = 50

    def __init__(self, path: Path) -> None:
        self._path = path
        self._buf: list[dict] = []
        self._lock = threading.Lock()

    def add(self, entry: dict) -> None:
        """Add one manifest row and auto-flush when the buffer is full."""
        with self._lock:
            self._buf.append(entry)
            if len(self._buf) >= self.FLUSH_SIZE:
                self._flush_locked()

    def flush(self) -> None:
        """Write any buffered manifest rows to disk."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Write the current buffer to disk. Caller must hold the lock."""
        if not self._buf:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            for entry in self._buf:
                f.write(json.dumps(entry) + "\n")

        self._buf.clear()


_manifest_writer = _ManifestWriter(MANIFEST)


EARNINGS_8K_ITEMS = {"2.02", "7.01", "8.01"}
EARNINGS_8K_KEYWORDS = [
    "earn", "result", "revenue", "quarter", "fiscal",
    "financial", "guidance", "outlook", "q1", "q2", "q3", "q4",
]

EXHIBIT_TYPES_OF_INTEREST = {
    "EX-99.1", "EX-99.2", "EX-99", "EX-13", "EX-13.1",
}


SIC_TO_SECTOR: dict[int, str] = {
    3510: "Technology", 3523: "Technology", 3524: "Technology", 3530: "Technology",
    3531: "Technology", 3532: "Technology", 3537: "Technology", 3540: "Technology",
    3541: "Technology", 3550: "Technology", 3555: "Technology", 3559: "Technology",
    3560: "Technology", 3561: "Technology", 3562: "Technology", 3564: "Technology",
    3567: "Technology", 3569: "Technology", 3570: "Technology", 3571: "Technology",
    3572: "Technology", 3575: "Technology", 3576: "Technology", 3577: "Technology",
    3578: "Technology", 3579: "Technology", 3580: "Technology", 3585: "Technology",
    3590: "Technology", 3600: "Technology", 3612: "Technology", 3613: "Technology",
    3620: "Technology", 3621: "Technology", 3640: "Technology", 3661: "Technology",
    3663: "Technology", 3669: "Technology", 3670: "Technology", 3672: "Technology",
    3674: "Technology", 3677: "Technology", 3678: "Technology", 3679: "Technology",
    3690: "Technology", 3695: "Technology", 5045: "Technology", 5065: "Technology",
    7370: "Technology", 7371: "Technology", 7372: "Technology", 7373: "Technology",
    7374: "Technology", 7377: "Technology",

    2833: "Healthcare", 2834: "Healthcare", 2835: "Healthcare", 2836: "Healthcare",
    3821: "Healthcare", 3822: "Healthcare", 3823: "Healthcare", 3824: "Healthcare",
    3825: "Healthcare", 3826: "Healthcare", 3827: "Healthcare", 3829: "Healthcare",
    3841: "Healthcare", 3842: "Healthcare", 3843: "Healthcare", 3844: "Healthcare",
    3845: "Healthcare", 3851: "Healthcare", 3861: "Healthcare", 3873: "Healthcare",
    5047: "Healthcare", 5122: "Healthcare", 8000: "Healthcare", 8011: "Healthcare",
    8050: "Healthcare", 8051: "Healthcare", 8060: "Healthcare", 8062: "Healthcare",
    8071: "Healthcare", 8082: "Healthcare", 8090: "Healthcare", 8093: "Healthcare",
    8731: "Healthcare",

    6021: "Financial Services", 6022: "Financial Services", 6029: "Financial Services",
    6035: "Financial Services", 6036: "Financial Services", 6099: "Financial Services",
    6111: "Financial Services", 6141: "Financial Services", 6153: "Financial Services",
    6159: "Financial Services", 6162: "Financial Services", 6163: "Financial Services",
    6172: "Financial Services", 6189: "Financial Services", 6199: "Financial Services",
    6200: "Financial Services", 6211: "Financial Services", 6221: "Financial Services",
    6282: "Financial Services", 6311: "Financial Services", 6321: "Financial Services",
    6324: "Financial Services", 6331: "Financial Services", 6351: "Financial Services",
    6361: "Financial Services", 6399: "Financial Services", 6411: "Financial Services",

    1220: "Energy", 1221: "Energy", 1311: "Energy", 1381: "Energy",
    1382: "Energy", 1389: "Energy", 2911: "Energy", 2950: "Energy",
    2990: "Energy", 3533: "Energy", 5171: "Energy", 5172: "Energy",
    6792: "Energy", 6795: "Energy",

    4900: "Utilities", 4911: "Utilities", 4922: "Utilities", 4923: "Utilities",
    4924: "Utilities", 4931: "Utilities", 4932: "Utilities", 4941: "Utilities",
    4950: "Utilities", 4953: "Utilities", 4955: "Utilities", 4961: "Utilities",
    4991: "Utilities",

    6500: "Real Estate", 6510: "Real Estate", 6512: "Real Estate", 6513: "Real Estate",
    6519: "Real Estate", 6531: "Real Estate", 6532: "Real Estate", 6552: "Real Estate",
    6770: "Real Estate", 6794: "Real Estate", 6798: "Real Estate", 6799: "Real Estate",

    2711: "Communication Services", 2721: "Communication Services",
    2731: "Communication Services", 2732: "Communication Services",
    2741: "Communication Services", 2750: "Communication Services",
    2761: "Communication Services", 2771: "Communication Services",
    2780: "Communication Services", 2790: "Communication Services",
    4812: "Communication Services", 4813: "Communication Services",
    4822: "Communication Services", 4832: "Communication Services",
    4833: "Communication Services", 4841: "Communication Services",
    4899: "Communication Services", 7385: "Communication Services",
    7812: "Communication Services", 7819: "Communication Services",
    7822: "Communication Services", 7829: "Communication Services",
    7830: "Communication Services", 7841: "Communication Services",

    100: "Consumer Staples", 200: "Consumer Staples", 700: "Consumer Staples",
    800: "Consumer Staples", 900: "Consumer Staples",
    2000: "Consumer Staples", 2011: "Consumer Staples", 2013: "Consumer Staples",
    2015: "Consumer Staples", 2020: "Consumer Staples", 2024: "Consumer Staples",
    2030: "Consumer Staples", 2033: "Consumer Staples", 2040: "Consumer Staples",
    2050: "Consumer Staples", 2052: "Consumer Staples", 2060: "Consumer Staples",
    2070: "Consumer Staples", 2080: "Consumer Staples", 2082: "Consumer Staples",
    2086: "Consumer Staples", 2090: "Consumer Staples", 2092: "Consumer Staples",
    2100: "Consumer Staples", 2111: "Consumer Staples",
    5400: "Consumer Staples", 5411: "Consumer Staples", 5412: "Consumer Staples",
    5912: "Consumer Staples",

    2200: "Consumer Discretionary", 2211: "Consumer Discretionary",
    2221: "Consumer Discretionary", 2250: "Consumer Discretionary",
    2253: "Consumer Discretionary", 2273: "Consumer Discretionary",
    2300: "Consumer Discretionary", 2320: "Consumer Discretionary",
    2330: "Consumer Discretionary", 2340: "Consumer Discretionary",
    2390: "Consumer Discretionary", 2510: "Consumer Discretionary",
    2511: "Consumer Discretionary", 2520: "Consumer Discretionary",
    2522: "Consumer Discretionary", 2531: "Consumer Discretionary",
    2540: "Consumer Discretionary", 2590: "Consumer Discretionary",
    3011: "Consumer Discretionary", 3021: "Consumer Discretionary",
    3050: "Consumer Discretionary", 3060: "Consumer Discretionary",
    3100: "Consumer Discretionary", 3140: "Consumer Discretionary",
    3630: "Consumer Discretionary", 3634: "Consumer Discretionary",
    3651: "Consumer Discretionary", 3652: "Consumer Discretionary",
    3711: "Consumer Discretionary", 3713: "Consumer Discretionary",
    3714: "Consumer Discretionary", 3715: "Consumer Discretionary",
    3716: "Consumer Discretionary", 3910: "Consumer Discretionary",
    3911: "Consumer Discretionary", 3931: "Consumer Discretionary",
    3942: "Consumer Discretionary", 3944: "Consumer Discretionary",
    3949: "Consumer Discretionary", 3950: "Consumer Discretionary",
    3960: "Consumer Discretionary", 3990: "Consumer Discretionary",
    5000: "Consumer Discretionary", 5010: "Consumer Discretionary",
    5013: "Consumer Discretionary", 5020: "Consumer Discretionary",
    5030: "Consumer Discretionary", 5031: "Consumer Discretionary",
    5040: "Consumer Discretionary", 5050: "Consumer Discretionary",
    5051: "Consumer Discretionary", 5063: "Consumer Discretionary",
    5064: "Consumer Discretionary", 5070: "Consumer Discretionary",
    5072: "Consumer Discretionary", 5080: "Consumer Discretionary",
    5082: "Consumer Discretionary", 5084: "Consumer Discretionary",
    5090: "Consumer Discretionary", 5094: "Consumer Discretionary",
    5099: "Consumer Discretionary", 5110: "Consumer Discretionary",
    5130: "Consumer Discretionary", 5140: "Consumer Discretionary",
    5141: "Consumer Discretionary", 5150: "Consumer Discretionary",
    5160: "Consumer Discretionary", 5180: "Consumer Discretionary",
    5190: "Consumer Discretionary", 5200: "Consumer Discretionary",
    5211: "Consumer Discretionary", 5271: "Consumer Discretionary",
    5311: "Consumer Discretionary", 5331: "Consumer Discretionary",
    5399: "Consumer Discretionary", 5500: "Consumer Discretionary",
    5531: "Consumer Discretionary", 5600: "Consumer Discretionary",
    5621: "Consumer Discretionary", 5651: "Consumer Discretionary",
    5661: "Consumer Discretionary", 5700: "Consumer Discretionary",
    5712: "Consumer Discretionary", 5731: "Consumer Discretionary",
    5734: "Consumer Discretionary", 5735: "Consumer Discretionary",
    5810: "Consumer Discretionary", 5812: "Consumer Discretionary",
    5900: "Consumer Discretionary", 5940: "Consumer Discretionary",
    5944: "Consumer Discretionary", 5945: "Consumer Discretionary",
    5960: "Consumer Discretionary", 5961: "Consumer Discretionary",
    5990: "Consumer Discretionary", 7000: "Consumer Discretionary",
    7011: "Consumer Discretionary", 7200: "Consumer Discretionary",
    7384: "Consumer Discretionary", 7500: "Consumer Discretionary",
    7510: "Consumer Discretionary", 7900: "Consumer Discretionary",
    7948: "Consumer Discretionary", 7990: "Consumer Discretionary",
    7997: "Consumer Discretionary", 8200: "Consumer Discretionary",
    8300: "Consumer Discretionary", 8351: "Consumer Discretionary",
    8600: "Consumer Discretionary", 8900: "Consumer Discretionary",

    1520: "Industrials", 1531: "Industrials", 1540: "Industrials",
    1600: "Industrials", 1623: "Industrials", 1700: "Industrials", 1731: "Industrials",
    3411: "Industrials", 3412: "Industrials", 3420: "Industrials", 3430: "Industrials",
    3433: "Industrials", 3440: "Industrials", 3442: "Industrials", 3443: "Industrials",
    3444: "Industrials", 3448: "Industrials", 3451: "Industrials", 3452: "Industrials",
    3460: "Industrials", 3470: "Industrials", 3480: "Industrials", 3490: "Industrials",
    3720: "Industrials", 3721: "Industrials", 3724: "Industrials", 3728: "Industrials",
    3730: "Industrials", 3743: "Industrials", 3751: "Industrials", 3760: "Industrials",
    3790: "Industrials", 3812: "Industrials",
    4011: "Industrials", 4013: "Industrials", 4100: "Industrials", 4210: "Industrials",
    4213: "Industrials", 4220: "Industrials", 4231: "Industrials", 4400: "Industrials",
    4412: "Industrials", 4512: "Industrials", 4513: "Industrials", 4522: "Industrials",
    4581: "Industrials", 4610: "Industrials", 4700: "Industrials", 4731: "Industrials",
    7310: "Industrials", 7311: "Industrials", 7320: "Industrials", 7330: "Industrials",
    7331: "Industrials", 7340: "Industrials", 7350: "Industrials", 7359: "Industrials",
    7361: "Industrials", 7363: "Industrials", 7380: "Industrials", 7381: "Industrials",
    7389: "Industrials", 7600: "Industrials", 8111: "Industrials", 8700: "Industrials",
    8711: "Industrials", 8734: "Industrials", 8741: "Industrials", 8742: "Industrials",
    8744: "Industrials",

    1000: "Basic Materials", 1040: "Basic Materials", 1090: "Basic Materials",
    1400: "Basic Materials",
    2400: "Basic Materials", 2421: "Basic Materials", 2430: "Basic Materials",
    2451: "Basic Materials", 2452: "Basic Materials",
    2600: "Basic Materials", 2611: "Basic Materials", 2621: "Basic Materials",
    2631: "Basic Materials", 2650: "Basic Materials", 2670: "Basic Materials",
    2673: "Basic Materials", 2800: "Basic Materials", 2810: "Basic Materials",
    2820: "Basic Materials", 2821: "Basic Materials", 2840: "Basic Materials",
    2842: "Basic Materials", 2844: "Basic Materials", 2851: "Basic Materials",
    2860: "Basic Materials", 2870: "Basic Materials", 2890: "Basic Materials",
    2891: "Basic Materials", 3080: "Basic Materials", 3081: "Basic Materials",
    3086: "Basic Materials", 3089: "Basic Materials", 3211: "Basic Materials",
    3220: "Basic Materials", 3221: "Basic Materials", 3231: "Basic Materials",
    3241: "Basic Materials", 3250: "Basic Materials", 3260: "Basic Materials",
    3270: "Basic Materials", 3272: "Basic Materials", 3281: "Basic Materials",
    3290: "Basic Materials", 3310: "Basic Materials", 3312: "Basic Materials",
    3317: "Basic Materials", 3320: "Basic Materials", 3330: "Basic Materials",
    3334: "Basic Materials", 3341: "Basic Materials", 3350: "Basic Materials",
    3357: "Basic Materials", 3360: "Basic Materials", 3390: "Basic Materials",

    8880: "Other", 8888: "Other", 9721: "Other", 9995: "Other",
}


def sic_to_sector(sic: int | None) -> str:
    """Convert a SIC code into a sector label."""
    if sic is None:
        return "Unknown"
    return SIC_TO_SECTOR.get(sic, "Unknown")


FORM_PRIORITY: dict[str, int] = {
    "10-K": 1,
    "10-K405": 1,
    "10-KSB": 1,
    "10-Q": 2,
    "8-K": 3,
}


def get_report_priority(form: str) -> int:
    """Return the ranking used to compare filing types."""
    return FORM_PRIORITY.get(form.upper(), 9)


def derive_period_type(form: str) -> str:
    """Classify a filing as annual, quarterly, or event-based."""
    f = form.upper()
    if f in ("10-K", "10-K405", "10-KSB", "10-KT"):
        return "annual"
    if f in ("10-Q", "10-QSB", "10-QT"):
        return "quarterly"
    return "event"


def get_fiscal_year_from_period(
    period_end_date: str | None, filing_date: str, form: str
) -> int:
    """Use the report period when available, otherwise fall back to filing date."""
    if period_end_date:
        try:
            from datetime import datetime
            return datetime.strptime(period_end_date[:10], "%Y-%m-%d").year
        except ValueError:
            pass

    from datetime import datetime
    dt = datetime.strptime(filing_date, "%Y-%m-%d")

    if form.upper() in ("10-K", "10-K405", "10-KSB") and dt.month <= 6:
        return dt.year - 1

    return dt.year


def derive_fiscal_quarter(
    period_end_date: str | None, period_type: str
) -> int | None:
    """Infer the fiscal quarter from the report end date for quarterly filings."""
    if period_type != "quarterly" or not period_end_date:
        return None

    try:
        from datetime import datetime
        month = datetime.strptime(period_end_date[:10], "%Y-%m-%d").month
        return (month - 1) // 3 + 1
    except ValueError:
        return None


@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(5))
def get_json(url: str) -> dict:
    """Fetch JSON with rate limiting, retry logic, and pooled connections."""
    _rate_limiter.acquire()
    resp = _get_session().get(url, headers=_req_headers(url), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(5))
def download_file(
    url: str, dest: Path, expected_sha256: str | None = None
) -> bool:
    """
    Download a file safely using a temp file first.
    Returns False on 404, otherwise raises on failure.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    if tmp.exists():
        log.warning(f"Stale .tmp removed: {tmp.name}")
        tmp.unlink()

    if dest.exists():
        if dest.stat().st_size == 0:
            log.warning(f"Zero-byte file, re-downloading: {dest.name}")
            dest.unlink()
        elif expected_sha256:
            if _sha256_file(dest) == expected_sha256:
                log.info(f"Exists (checksum OK): {dest.name}")
                return True
            log.warning(f"Checksum mismatch, re-downloading: {dest.name}")
            dest.unlink()
        else:
            log.info(f"Already exists: {dest.name}")
            return True

    _rate_limiter.acquire()
    resp = _get_session().get(
        url,
        headers=_req_headers(url),
        timeout=(5, 60),
        stream=True,
    )

    if resp.status_code == 404:
        return False

    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)

    content_encoding = resp.headers.get("Content-Encoding", "")
    declared_size: int | None = None

    if not content_encoding:
        cl = resp.headers.get("Content-Length", "")
        if cl.isdigit():
            declared_size = int(cl)

    bytes_written = 0
    try:
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                bytes_written += len(chunk)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if declared_size is not None and bytes_written < declared_size:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Truncated download: expected {declared_size}B got {bytes_written}B — {url}"
        )

    if expected_sha256:
        actual = _sha256_file(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for {url}: "
                f"expected {expected_sha256[:12]}… got {actual[:12]}…"
            )

    tmp.rename(dest)
    log.info(f"Saved {dest.name} ({bytes_written:,}B)")
    return True


def normalize_cik(cik: str) -> str:
    """Pad a CIK to SEC's 10-digit format."""
    return cik.strip().zfill(10)


def _safe_form_label(form: str) -> str:
    """Normalize a form label for file naming."""
    return form.lower().replace("-", "").replace("/", "")


def _acc_to_fmt(acc_clean: str) -> str:
    """Convert an 18-digit accession into SEC's dashed format."""
    return f"{acc_clean[:10]}-{acc_clean[10:12]}-{acc_clean[12:]}"


def is_earnings_8k(items_str: str | None, primary_doc: str) -> bool:
    """Keep only earnings-related 8-K filings."""
    if items_str:
        parsed = {
            tok.strip().strip(",")
            for tok in items_str.replace(",", " ").split()
            if tok.strip()
        }
        return bool(parsed & EARNINGS_8K_ITEMS)

    return any(kw in primary_doc.lower() for kw in EARNINGS_8K_KEYWORDS)


def _parse_filing_rows(
    filings_dict: dict,
    cik_norm: str,
    ticker: str,
    sector: str,
    industry: str,
) -> list[dict]:
    """Convert SEC submission arrays into normalized filing records."""
    keys = ["accessionNumber", "form", "filingDate", "primaryDocument", "reportDate", "items"]
    cols = [filings_dict.get(k, []) for k in keys]
    max_len = max((len(c) for c in cols), default=0)
    padded = [c + [""] * (max_len - len(c)) for c in cols]

    results: list[dict] = []
    for acc, form, date, primary_doc, report_date, items in zip(*padded):
        if form not in TARGET_FORMS:
            continue
        if date < DATE_FROM:
            continue

        period_type = derive_period_type(form)
        results.append({
            "accession": acc.replace("-", ""),
            "accession_fmt": acc,
            "form": form,
            "filing_date": date,
            "period_end_date": report_date or None,
            "period_type": period_type,
            "fiscal_year": get_fiscal_year_from_period(report_date or None, date, form),
            "fiscal_quarter": derive_fiscal_quarter(report_date or None, period_type),
            "report_priority": get_report_priority(form),
            "items": items or None,
            "primary_doc": primary_doc,
            "cik": cik_norm,
            "ticker": ticker,
            "sector": sector,
            "industry": industry,
        })

    return results


def _fetch_filing_index(cik: str, acc_clean: str) -> list[dict]:
    """Fetch the filing index once so it can be reused for primary docs and exhibits."""
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{_acc_to_fmt(acc_clean)}-index.json"
    )
    try:
        return get_json(url).get("documents", [])
    except Exception as e:
        log.debug(f"Index unavailable for {acc_clean}: {e}")
        return []


def get_filings_for_company(cik: str, ticker: str) -> list[dict]:
    """Fetch recent and archived filings for one company."""
    cik_norm = normalize_cik(cik)

    try:
        data = get_json(f"https://data.sec.gov/submissions/CIK{cik_norm}.json")
    except Exception as e:
        log.error(f"Submissions fetch failed for {ticker}: {e}")
        return []

    sic_raw = data.get("sic")
    sector = sic_to_sector(int(sic_raw) if sic_raw else None)
    industry = data.get("sicDescription", "")

    seen_acc: set[str] = set()
    results: list[dict] = []

    def _absorb(rows: list[dict]) -> None:
        """Merge filing rows while avoiding duplicate accessions."""
        for r in rows:
            if r["accession"] not in seen_acc:
                seen_acc.add(r["accession"])
                results.append(r)

    recent = data.get("filings", {}).get("recent", {})
    if recent:
        _absorb(_parse_filing_rows(recent, cik_norm, ticker, sector, industry))

    archive_files = [
        m.get("name", "")
        for m in data.get("filings", {}).get("files", [])
        if m.get("name")
    ]

    if archive_files:
        def _fetch_archive(name: str) -> list[dict]:
            """Fetch and parse one archived submissions page."""
            try:
                d = get_json(f"https://data.sec.gov/submissions/{name}")
                return _parse_filing_rows(d, cik_norm, ticker, sector, industry)
            except Exception as e:
                log.warning(f"Archive {name} failed for {ticker}: {e}")
                return []

        with ThreadPoolExecutor(
            max_workers=min(4, len(archive_files)),
            thread_name_prefix="archive",
        ) as pool:
            for rows in pool.map(_fetch_archive, archive_files):
                _absorb(rows)

    log.info(
        f"{ticker}: {len(results)} qualifying filings since {DATE_FROM} "
        f"(sector={sector})"
    )
    return results


def _download_exhibits_from_index(
    filing: dict,
    doc_id_base: str,
    index_docs: list[dict],
    seen_ids: set,
    new_entries: list,
) -> None:
    """Download selected exhibits listed in the filing index."""
    cik = filing["cik"]
    acc = filing["accession"].replace("-", "")
    form = filing["form"]

    for doc in index_docs:
        ex_type = (doc.get("type") or "").strip().upper()
        if ex_type not in EXHIBIT_TYPES_OF_INTEREST:
            continue

        doc_name = (doc.get("documentUrl") or doc.get("name") or "").strip()
        if not doc_name:
            continue

        ex_label = ex_type.lower().replace("-", "").replace(".", "")
        ex_doc_id = f"{doc_id_base}_{ex_label}"

        if ex_doc_id in seen_ids:
            continue

        file_url = (
            doc_name
            if doc_name.startswith("http")
            else f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc_name}"
        )
        ext = Path(doc_name).suffix.lower() or ".htm"
        dest = RAW_DIR / filing["ticker"] / form / f"{ex_doc_id}{ext}"

        log.info(f"Exhibit {ex_type}: {doc_name}")
        if not download_file(file_url, dest, expected_sha256=doc.get("sha256")):
            log.warning(f"Exhibit 404: {file_url}")
            continue

        new_entries.append({
            "doc_id": ex_doc_id,
            "ticker": filing["ticker"],
            "company": filing.get("company", filing["ticker"]),
            "form_type": form,
            "filing_date": filing["filing_date"],
            "period_end_date": filing.get("period_end_date"),
            "period_type": filing.get("period_type"),
            "fiscal_year": filing.get("fiscal_year"),
            "fiscal_quarter": filing.get("fiscal_quarter"),
            "report_priority": filing.get("report_priority"),
            "sector": filing.get("sector", "Unknown"),
            "industry": filing.get("industry", ""),
            "accession": filing["accession_fmt"],
            "cik": cik,
            "source_url": file_url,
            "raw_path": str(dest.relative_to(BASE_DIR)),
            "file_ext": ext,
            "is_exhibit": True,
            "exhibit_type": ex_type,
            "parse_status": "pending",
            "index_status": "pending",
        })
        seen_ids.add(ex_doc_id)


def load_companies() -> list[dict]:
    """Load the company list from CSV."""
    with open(TOP50_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest() -> set[str]:
    """Load known doc_ids from the manifest to avoid duplicate downloads."""
    seen: set[str] = set()

    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["doc_id"])
                except Exception:
                    pass

    return seen


_seen_ids_lock = threading.Lock()


def download_filing(filing: dict, seen_ids: set) -> list[dict]:
    """Download one filing's primary document and any relevant exhibits."""
    ticker = filing["ticker"]
    form = filing["form"]
    acc = filing["accession"].replace("-", "")
    date = filing["filing_date"]
    cik = filing["cik"]
    primary = filing["primary_doc"]

    if form == "8-K" and not is_earnings_8k(filing.get("items"), primary):
        log.info(f"Skip non-earnings 8-K: {primary} (items={filing.get('items', '—')})")
        return []

    doc_id = f"{ticker.lower()}_{_safe_form_label(form)}_{date}_{acc[-6:]}"

    with _seen_ids_lock:
        if doc_id in seen_ids:
            return []
        seen_ids.add(doc_id)

    file_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{primary}"
    ext = Path(primary).suffix.lower() or ".htm"
    dest = RAW_DIR / ticker / form / f"{doc_id}{ext}"

    log.info(f"→ {ticker} {form} {date}")

    index_docs = _fetch_filing_index(cik, acc)

    primary_lower = primary.lower()
    expected_sha = next(
        (
            d.get("sha256")
            for d in index_docs
            if (d.get("documentUrl") or d.get("name") or "").lower().endswith(primary_lower)
        ),
        None,
    )

    if not download_file(file_url, dest, expected_sha256=expected_sha):
        log.warning(f"404: {file_url}")
        with _seen_ids_lock:
            seen_ids.discard(doc_id)
        return []

    new_entries: list[dict] = [{
        "doc_id": doc_id,
        "ticker": ticker,
        "company": filing.get("company", ticker),
        "form_type": form,
        "filing_date": date,
        "period_end_date": filing.get("period_end_date"),
        "period_type": filing.get("period_type"),
        "fiscal_year": filing.get("fiscal_year"),
        "fiscal_quarter": filing.get("fiscal_quarter"),
        "report_priority": filing.get("report_priority"),
        "sector": filing.get("sector", "Unknown"),
        "industry": filing.get("industry", ""),
        "accession": filing["accession_fmt"],
        "cik": cik,
        "source_url": file_url,
        "raw_path": str(dest.relative_to(BASE_DIR)),
        "file_ext": ext,
        "is_exhibit": False,
        "exhibit_type": None,
        "parse_status": "pending",
        "index_status": "pending",
    }]

    _download_exhibits_from_index(
        filing, doc_id, index_docs, seen_ids, new_entries
    )

    return new_entries


def process_company(company: dict, seen_ids: set) -> int:
    """Fetch metadata and download all qualifying filings for one company."""
    ticker = company["ticker"]
    cik = company["cik"]

    log.info(f"{'=' * 55}")
    log.info(f"Processing {ticker} (CIK: {cik})")

    filings = get_filings_for_company(cik, ticker)
    company_name = company.get("company", ticker)

    total = 0
    for filing in filings:
        filing["company"] = company_name
        entries = download_filing(filing, seen_ids)
        for entry in entries:
            _manifest_writer.add(entry)
        total += len(entries)

    log.info(f"{ticker}: {total} new entries")
    return total


def main(tickers: list[str] | None = None) -> None:
    """Run the downloader for all companies or a selected ticker subset."""
    companies = load_companies()

    if tickers:
        ticker_set = set(tickers)
        companies = [c for c in companies if c["ticker"] in ticker_set]

    seen_ids = load_manifest()

    log.info(
        f"Starting — {len(companies)} companies | "
        f"{len(seen_ids)} docs already in manifest | "
        f"workers={MAX_WORKERS} | rate={RATE_LIMIT_RPS} req/s"
    )
    log.info(f"Log → {LOGS_DIR / 'sec_downloader.log'}")

    total_new = 0
    start = time.monotonic()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
        thread_name_prefix="worker",
    ) as pool:
        futures = {
            pool.submit(process_company, company, seen_ids): company["ticker"]
            for company in companies
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                total_new += future.result()
            except Exception as e:
                log.error(f"Worker failed for {ticker}: {e}", exc_info=True)

    _manifest_writer.flush()

    elapsed = time.monotonic() - start
    log.info(f"{'=' * 55}")
    log.info(
        f"Done. {total_new} new entries | "
        f"{elapsed:.1f}s elapsed | "
        f"{total_new / elapsed:.1f} entries/s"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SEC EDGAR filings in parallel with rate limiting."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional subset of tickers, for example: --tickers AAPL MSFT NVDA",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of worker threads (default {MAX_WORKERS}).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=RATE_LIMIT_RPS,
        help=f"Max aggregate requests per second (default {RATE_LIMIT_RPS}).",
    )
    args = parser.parse_args()

    MAX_WORKERS = args.workers
    RATE_LIMIT_RPS = args.rate
    _rate_limiter = _RateLimiter(RATE_LIMIT_RPS)

    main(tickers=args.tickers)