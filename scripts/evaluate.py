"""
Evaluate the RAG pipeline on a labeled test set.

Supports:
- Retriever metrics
- Generator metrics
- Citation metrics
- Ablation study

Usage:
  python scripts/evaluate.py --output results/
  python scripts/evaluate.py --output results/ --quick
  python scripts/evaluate.py --output results/ --section retriever
  python scripts/evaluate.py --output results/ --section ablation
  python scripts/evaluate.py --output results/ --save-testset-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from dotenv import load_dotenv

    # Load environment variables before importing pipeline modules.
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
    else:
        load_dotenv(override=False)
except ImportError:
    pass


def _bar(done: int, total: int, width: int = 30) -> str:
    """Return a simple progress bar string."""
    filled = int(width * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


class Progress:
    """Single-line progress display for CLI runs."""

    def __init__(self, total: int, label: str = ""):
        self.total = total
        self.label = label
        self.done = 0
        self.start = time.time()
        self._tty = sys.stdout.isatty()
        self._last_line_len = 0

    def _elapsed(self) -> str:
        """Return elapsed time since start."""
        s = int(time.time() - self.start)
        return f"{s // 60}m{s % 60:02d}s"

    def _eta(self) -> str:
        """Estimate remaining time based on current progress."""
        if self.done == 0:
            return "??:??"
        elapsed = time.time() - self.start
        if elapsed < 0.01:
            return "??:??"
        rate = self.done / elapsed
        remaining = (self.total - self.done) / rate
        m, s = divmod(int(remaining), 60)
        return f"{m}m{s:02d}s"

    def update(self, msg: str = "") -> None:
        """Advance progress by one step and refresh the display."""
        self.done += 1
        pct = 100 * self.done // self.total
        line = (
            f"  {self.label}  {_bar(self.done, self.total)}  "
            f"{pct}%  elapsed {self._elapsed()}  eta {self._eta()}"
        )
        if msg:
            line += f"  {msg[:60]}"

        if self._tty:
            clear = " " * self._last_line_len
            sys.stdout.write(f"\r{clear}\r{line}")
            sys.stdout.flush()
            self._last_line_len = len(line)
        else:
            print(line, flush=True)

    def done_msg(self, msg: str = "") -> None:
        """Print a final completion message."""
        line = f"  ✓ {self.label} complete  {self._elapsed()}  {msg}"
        if self._tty:
            clear = " " * self._last_line_len
            sys.stdout.write(f"\r{clear}\r{line}\n")
            sys.stdout.flush()
        else:
            print(line, flush=True)


def _section_header(title: str) -> None:
    """Print a section title for CLI output."""
    print(f"\n{'─' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─' * 60}", flush=True)


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("evaluate")
log.setLevel(logging.INFO)

for _noisy in (
    "httpx",
    "httpcore",
    "qdrant_client",
    "src.generate.query_planner",
    "src.retrieve.hybrid_search",
    "src.index.embedder",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


_FY_CACHE: dict[str, list[int]] = {}


def _patch_fy_cache() -> None:
    """Cache fiscal year lookups to reduce repeated Qdrant calls."""
    try:
        import src.generate.query_planner as qp

        _original = qp._available_fiscal_years

        def _cached(tickers, sector):
            key = str((tuple(sorted(tickers or [])), sector))
            if key not in _FY_CACHE:
                _FY_CACHE[key] = _original(tickers, sector)
            return _FY_CACHE[key]

        qp._available_fiscal_years = _cached
        log.info("Applied fiscal year cache patch")
    except Exception as e:
        log.warning("Could not patch FY cache: %s", e)


# ---------------------------------------------------------------------------
# 10.1  TEST SET
# ---------------------------------------------------------------------------
TEST_SET: list[dict] = [

    # ── BUCKET 1: SINGLE-HOP FACTUAL (50 queries) ──────────────────────────

    # MSFT
    {
        "query_id": "sh_001",
        "query": "What was Microsoft's total revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$245,122 million",
        "gold_alternatives": ["245,122", "$245.1 billion", "245.1B", "$245,122M"],
        "required_docs": ["msft_10k_2024"],
        "required_pages": [41],
        "expected_ticker": "MSFT",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_002",
        "query": "What was Microsoft's net income in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$88,136 million",
        "gold_alternatives": ["88,136", "$88.1 billion", "$88,136M"],
        "required_docs": ["msft_10k_2024"],
        "required_pages": [41],
        "expected_ticker": "MSFT",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_003",
        "query": "What was Microsoft's R&D expense in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$29,510 million",
        "gold_alternatives": ["29,510", "$29.5 billion", "$29,510M"],
        "required_docs": ["msft_10k_2024"],
        "required_pages": [33, 41],
        "expected_ticker": "MSFT",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_004",
        "query": "What was Microsoft's long-term debt as of June 2024?",
        "bucket": "single_hop",
        "gold_answer": "$42,688 million",
        "gold_alternatives": ["42,688", "$42.7 billion", "$42,688M"],
        "required_docs": ["msft_10k_2024"],
        "required_pages": [41, 55],
        "expected_ticker": "MSFT",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_005",
        "query": "What was Microsoft's gross margin percentage in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "70%",
        "gold_alternatives": ["69%", "70.1%", "69.8%"],
        "required_docs": ["msft_10k_2024"],
        "required_pages": [33, 41],
        "expected_ticker": "MSFT",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },

    # AAPL
    {
        "query_id": "sh_006",
        "query": "What was Apple's total revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$391,035 million",
        "gold_alternatives": ["391,035", "$391 billion", "$391B", "391.0B"],
        "required_docs": ["aapl_10k_2024"],
        "required_pages": [20, 32],
        "expected_ticker": "AAPL",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_007",
        "query": "What was Apple's total gross margin in FY2025?",
        "bucket": "single_hop",
        "gold_answer": "$195,201 million",
        "gold_alternatives": ["195,201", "$195.2 billion", "$195,201M"],
        "required_docs": ["aapl_10k_2025"],
        "required_pages": [20, 24],
        "expected_ticker": "AAPL",
        "expected_fy": 2025,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_008",
        "query": "What was Apple's net income in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$93,736 million",
        "gold_alternatives": ["93,736", "$93.7 billion", "$93,736M"],
        "required_docs": ["aapl_10k_2024"],
        "required_pages": [20, 32],
        "expected_ticker": "AAPL",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_009",
        "query": "What were Apple's Services revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$96,169 million",
        "gold_alternatives": ["96,169", "$96.2 billion", "$96,169M"],
        "required_docs": ["aapl_10k_2024"],
        "required_pages": [20, 24, 32],
        "expected_ticker": "AAPL",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_010",
        "query": "What was Apple's R&D expense in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$31,370 million",
        "gold_alternatives": ["31,370", "$31.4 billion", "$31,370M"],
        "required_docs": ["aapl_10k_2024"],
        "required_pages": [20, 32],
        "expected_ticker": "AAPL",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },

    # NVDA
    {
        "query_id": "sh_011",
        "query": "What was NVIDIA's total revenue in its most recent fiscal year 2025?",
        "bucket": "single_hop",
        "gold_answer": "$130,497 million",
        "gold_alternatives": ["130,497", "$130.5 billion", "$130,497M"],
        "required_docs": ["nvda_10k_2025"],
        "required_pages": [25, 41],
        "expected_ticker": "NVDA",
        "expected_fy": 2025,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_012",
        "query": "What was NVIDIA's data center revenue in FY2025?",
        "bucket": "single_hop",
        "gold_answer": "$115,168 million",
        "gold_alternatives": ["115,168", "$115.2 billion", "$115,168M"],
        "required_docs": ["nvda_10k_2025"],
        "required_pages": [25, 41],
        "expected_ticker": "NVDA",
        "expected_fy": 2025,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_013",
        "query": "What was NVIDIA's gross margin percentage in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "72.7%",
        "gold_alternatives": ["72%", "73%", "72.7", "72.8%"],
        "required_docs": ["nvda_10k_2024"],
        "required_pages": [25, 41],
        "expected_ticker": "NVDA",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_014",
        "query": "What was NVIDIA's net income in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$29,760 million",
        "gold_alternatives": ["29,760", "$29.8 billion", "$29,760M"],
        "required_docs": ["nvda_10k_2024"],
        "required_pages": [41],
        "expected_ticker": "NVDA",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_015",
        "query": "What was NVIDIA's R&D spending in FY2025?",
        "bucket": "single_hop",
        "gold_answer": "$12,914 million",
        "gold_alternatives": ["12,914", "$12.9 billion", "$12,914M"],
        "required_docs": ["nvda_10k_2025"],
        "required_pages": [25, 41],
        "expected_ticker": "NVDA",
        "expected_fy": 2025,
        "expected_form": "10-K",
    },

    # GOOGL
    {
        "query_id": "sh_016",
        "query": "What was Alphabet's total revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$350,018 million",
        "gold_alternatives": ["350,018", "$350 billion", "$350,018M"],
        "required_docs": ["googl_10k_2024"],
        "required_pages": [23, 41],
        "expected_ticker": "GOOGL",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_017",
        "query": "What was Google's operating income in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$112,388 million",
        "gold_alternatives": ["112,388", "$112.4 billion", "$112,388M"],
        "required_docs": ["googl_10k_2024"],
        "required_pages": [23, 41],
        "expected_ticker": "GOOGL",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_018",
        "query": "What was Alphabet's YouTube advertising revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$36,149 million",
        "gold_alternatives": ["36,149", "$36.1 billion", "$36,149M"],
        "required_docs": ["googl_10k_2024"],
        "required_pages": [23, 41],
        "expected_ticker": "GOOGL",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_019",
        "query": "What was Alphabet's R&D expense in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$49,074 million",
        "gold_alternatives": ["49,074", "$49.1 billion", "$49,074M"],
        "required_docs": ["googl_10k_2024"],
        "required_pages": [23, 41],
        "expected_ticker": "GOOGL",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_020",
        "query": "What was Google Cloud revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$43,228 million",
        "gold_alternatives": ["43,228", "$43.2 billion", "$43,228M"],
        "required_docs": ["googl_10k_2024"],
        "required_pages": [23, 41],
        "expected_ticker": "GOOGL",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },

    # AMZN
    {
        "query_id": "sh_021",
        "query": "What was Amazon's total net sales in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$637,959 million",
        "gold_alternatives": ["637,959", "$638 billion", "$637,959M"],
        "required_docs": ["amzn_10k_2024"],
        "required_pages": [25, 41],
        "expected_ticker": "AMZN",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_022",
        "query": "What was Amazon Web Services revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$107,576 million",
        "gold_alternatives": ["107,576", "$107.6 billion", "$107,576M"],
        "required_docs": ["amzn_10k_2024"],
        "required_pages": [25, 41],
        "expected_ticker": "AMZN",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_023",
        "query": "What was Amazon's operating income in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$68,593 million",
        "gold_alternatives": ["68,593", "$68.6 billion", "$68,593M"],
        "required_docs": ["amzn_10k_2024"],
        "required_pages": [25, 41],
        "expected_ticker": "AMZN",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_024",
        "query": "What was Amazon's net income in FY2023?",
        "bucket": "single_hop",
        "gold_answer": "$30,425 million",
        "gold_alternatives": ["30,425", "$30.4 billion", "$30,425M"],
        "required_docs": ["amzn_10k_2023"],
        "required_pages": [25, 41],
        "expected_ticker": "AMZN",
        "expected_fy": 2023,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_025",
        "query": "What was Amazon's capital expenditure in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$83,049 million",
        "gold_alternatives": ["83,049", "$83 billion", "$83,049M"],
        "required_docs": ["amzn_10k_2024"],
        "required_pages": [41, 55],
        "expected_ticker": "AMZN",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },

    # TSLA
    {
        "query_id": "sh_026",
        "query": "What was Tesla's total revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$97,690 million",
        "gold_alternatives": ["97,690", "$97.7 billion", "$97,690M"],
        "required_docs": ["tsla_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": "TSLA",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_027",
        "query": "What was Tesla's automotive gross margin in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "17.1%",
        "gold_alternatives": ["17%", "17.1", "17.2%"],
        "required_docs": ["tsla_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": "TSLA",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_028",
        "query": "What was Tesla's long-term debt as reported in its FY2025 10-K?",
        "bucket": "single_hop",
        "gold_answer": "$5,535 million",
        "gold_alternatives": ["5,535", "$5.5 billion", "$5,535M"],
        "required_docs": ["tsla_10k_2025"],
        "required_pages": [52],
        "expected_ticker": "TSLA",
        "expected_fy": 2025,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_029",
        "query": "What was Tesla's net income in FY2023?",
        "bucket": "single_hop",
        "gold_answer": "$14,974 million",
        "gold_alternatives": ["14,974", "$15 billion", "$14,974M"],
        "required_docs": ["tsla_10k_2023"],
        "required_pages": [41, 43],
        "expected_ticker": "TSLA",
        "expected_fy": 2023,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_030",
        "query": "What was Tesla's energy generation and storage revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$10,091 million",
        "gold_alternatives": ["10,091", "$10.1 billion", "$10,091M"],
        "required_docs": ["tsla_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": "TSLA",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },

    # JPM
    {
        "query_id": "sh_031",
        "query": "What was JPMorgan Chase's total net revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$177,564 million",
        "gold_alternatives": ["177,564", "$177.6 billion", "$177,564M"],
        "required_docs": ["jpm_10k_2024"],
        "required_pages": [30, 41],
        "expected_ticker": "JPM",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_032",
        "query": "What was JPMorgan's net income in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$58,471 million",
        "gold_alternatives": ["58,471", "$58.5 billion", "$58,471M"],
        "required_docs": ["jpm_10k_2024"],
        "required_pages": [30, 41],
        "expected_ticker": "JPM",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_033",
        "query": "What was JPMorgan's return on equity in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "17%",
        "gold_alternatives": ["16%", "17", "17.0%", "16.5%"],
        "required_docs": ["jpm_10k_2024"],
        "required_pages": [30],
        "expected_ticker": "JPM",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },

    # WMT
    {
        "query_id": "sh_034",
        "query": "What was Walmart's total net sales in FY2025?",
        "bucket": "single_hop",
        "gold_answer": "$672,754 million",
        "gold_alternatives": ["672,754", "$672.8 billion", "$672,754M"],
        "required_docs": ["wmt_10k_2025"],
        "required_pages": [41, 43],
        "expected_ticker": "WMT",
        "expected_fy": 2025,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_035",
        "query": "What was Walmart's operating income in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$26,381 million",
        "gold_alternatives": ["26,381", "$26.4 billion", "$26,381M"],
        "required_docs": ["wmt_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": "WMT",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_036",
        "query": "What was Walmart's e-commerce sales growth in FY2025?",
        "bucket": "single_hop",
        "gold_answer": "22%",
        "gold_alternatives": ["21%", "22", "22.0%", "21.9%"],
        "required_docs": ["wmt_10k_2025"],
        "required_pages": [30, 43],
        "expected_ticker": "WMT",
        "expected_fy": 2025,
        "expected_form": "10-K",
    },

    # AMD
    {
        "query_id": "sh_037",
        "query": "What was AMD's total revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$25,785 million",
        "gold_alternatives": ["25,785", "$25.8 billion", "$25,785M"],
        "required_docs": ["amd_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": "AMD",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_038",
        "query": "What was AMD's data center segment revenue in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$12,658 million",
        "gold_alternatives": ["12,658", "$12.7 billion", "$12,658M"],
        "required_docs": ["amd_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": "AMD",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_039",
        "query": "What was AMD's gross margin percentage in FY2023?",
        "bucket": "single_hop",
        "gold_answer": "46%",
        "gold_alternatives": ["45%", "46", "46.1%", "45.9%"],
        "required_docs": ["amd_10k_2023"],
        "required_pages": [41, 43],
        "expected_ticker": "AMD",
        "expected_fy": 2023,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_040",
        "query": "What was AMD's R&D expense in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$6,273 million",
        "gold_alternatives": ["6,273", "$6.3 billion", "$6,273M"],
        "required_docs": ["amd_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": "AMD",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },

    # GS
    {
        "query_id": "sh_041",
        "query": "What was Goldman Sachs' total net revenues in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$53,513 million",
        "gold_alternatives": ["53,513", "$53.5 billion", "$53,513M"],
        "required_docs": ["gs_10k_2024"],
        "required_pages": [30, 41],
        "expected_ticker": "GS",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_042",
        "query": "What was Goldman Sachs' net earnings in FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$14,280 million",
        "gold_alternatives": ["14,280", "$14.3 billion", "$14,280M"],
        "required_docs": ["gs_10k_2024"],
        "required_pages": [30, 41],
        "expected_ticker": "GS",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "sh_043",
        "query": "What was Goldman Sachs' return on equity in FY2023?",
        "bucket": "single_hop",
        "gold_answer": "7.5%",
        "gold_alternatives": ["7%", "7.5", "7.6%", "8%"],
        "required_docs": ["gs_10k_2023"],
        "required_pages": [30],
        "expected_ticker": "GS",
        "expected_fy": 2023,
        "expected_form": "10-K",
    },

    # Additional single-hop — 10-Q queries
    {
        "query_id": "sh_044",
        "query": "What was Microsoft's revenue in Q1 FY2025?",
        "bucket": "single_hop",
        "gold_answer": "$65,585 million",
        "gold_alternatives": ["65,585", "$65.6 billion", "$65,585M"],
        "required_docs": ["msft_10q_2024"],
        "required_pages": [20, 26],
        "expected_ticker": "MSFT",
        "expected_fy": 2025,
        "expected_form": "10-Q",
    },
    {
        "query_id": "sh_045",
        "query": "What was NVIDIA's revenue for the quarter ended October 2024?",
        "bucket": "single_hop",
        "gold_answer": "$35,082 million",
        "gold_alternatives": ["35,082", "$35.1 billion", "$35,082M"],
        "required_docs": ["nvda_10q_2025"],
        "required_pages": [20, 26],
        "expected_ticker": "NVDA",
        "expected_fy": 2025,
        "expected_form": "10-Q",
    },
    {
        "query_id": "sh_046",
        "query": "What was Apple's revenue for the quarter ended December 2024?",
        "bucket": "single_hop",
        "gold_answer": "$124,300 million",
        "gold_alternatives": ["124,300", "$124.3 billion", "$124,300M"],
        "required_docs": ["aapl_10q_2025"],
        "required_pages": [9, 20],
        "expected_ticker": "AAPL",
        "expected_fy": 2025,
        "expected_form": "10-Q",
    },
    {
        "query_id": "sh_047",
        "query": "What was Tesla's total revenue in Q3 FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$25,182 million",
        "gold_alternatives": ["25,182", "$25.2 billion", "$25,182M"],
        "required_docs": ["tsla_10q_2024"],
        "required_pages": [9, 20],
        "expected_ticker": "TSLA",
        "expected_fy": 2024,
        "expected_form": "10-Q",
    },
    {
        "query_id": "sh_048",
        "query": "What was AMD's revenue for the quarter ended September 2024?",
        "bucket": "single_hop",
        "gold_answer": "$6,819 million",
        "gold_alternatives": ["6,819", "$6.8 billion", "$6,819M"],
        "required_docs": ["amd_10q_2024"],
        "required_pages": [9, 20],
        "expected_ticker": "AMD",
        "expected_fy": 2024,
        "expected_form": "10-Q",
    },
    {
        "query_id": "sh_049",
        "query": "What was Walmart's total revenues in Q3 FY2025?",
        "bucket": "single_hop",
        "gold_answer": "$169,589 million",
        "gold_alternatives": ["169,589", "$169.6 billion", "$169,589M"],
        "required_docs": ["wmt_10q_2024"],
        "required_pages": [9, 20],
        "expected_ticker": "WMT",
        "expected_fy": 2025,
        "expected_form": "10-Q",
    },
    {
        "query_id": "sh_050",
        "query": "What was JPMorgan's net revenue in Q2 FY2024?",
        "bucket": "single_hop",
        "gold_answer": "$50,985 million",
        "gold_alternatives": ["50,985", "$51 billion", "$50,985M"],
        "required_docs": ["jpm_10q_2024"],
        "required_pages": [9, 20],
        "expected_ticker": "JPM",
        "expected_fy": 2024,
        "expected_form": "10-Q",
    },

    # ── BUCKET 2: MULTI-HOP COMPARATIVE (40 queries) ───────────────────────

    {
        "query_id": "mh_001",
        "query": "Compare the R&D spending as a percentage of revenue between Google and Microsoft in their latest annual filings.",
        "bucket": "multi_hop",
        "gold_answer": "Google 14%; Microsoft 12%",
        "gold_alternatives": ["GOOGL 14%", "MSFT 12%", "Google 14% vs Microsoft 12%"],
        "required_docs": ["googl_10k_2024", "msft_10k_2024"],
        "required_pages": [23, 33],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_002",
        "query": "Compare NVIDIA and AMD's gross margin percentages in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "NVIDIA 74.6%; AMD 47%",
        "gold_alternatives": ["NVDA 74%", "AMD 47%", "NVIDIA higher", "AMD lower"],
        "required_docs": ["nvda_10k_2024", "amd_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_003",
        "query": "Which company had higher revenue in FY2024 — Amazon or Microsoft?",
        "bucket": "multi_hop",
        "gold_answer": "Amazon",
        "gold_alternatives": ["AMZN", "Amazon had higher", "$637 billion vs $245 billion"],
        "required_docs": ["amzn_10k_2024", "msft_10k_2024"],
        "required_pages": [25, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_004",
        "query": "Compare JPMorgan and Goldman Sachs net income in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "JPMorgan $58,471 million; Goldman Sachs $14,280 million",
        "gold_alternatives": ["JPM higher", "JPM $58B", "GS $14B"],
        "required_docs": ["jpm_10k_2024", "gs_10k_2024"],
        "required_pages": [30, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_005",
        "query": "Compare Apple and Microsoft gross margin percentages in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Apple 46.2%; Microsoft 70%",
        "gold_alternatives": ["AAPL 46%", "MSFT 70%", "Microsoft higher"],
        "required_docs": ["aapl_10k_2024", "msft_10k_2024"],
        "required_pages": [20, 33, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_006",
        "query": "Compare Amazon Web Services and Google Cloud revenue in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "AWS $107,576 million; Google Cloud $43,228 million",
        "gold_alternatives": ["AWS larger", "AWS $107B", "GCP $43B"],
        "required_docs": ["amzn_10k_2024", "googl_10k_2024"],
        "required_pages": [25, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_007",
        "query": "How does Tesla's gross margin compare to the overall automotive industry leaders like Amazon in terms of profitability in FY2024?",
        "bucket": "multi_hop",
        "gold_answer": "Tesla gross margin 17.1%; Amazon operating margin 10.8%",
        "gold_alternatives": ["Tesla 17%", "Amazon 10%"],
        "required_docs": ["tsla_10k_2024", "amzn_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_008",
        "query": "Compare NVIDIA and AMD total revenue growth between FY2023 and FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "NVIDIA grew significantly more than AMD",
        "gold_alternatives": ["NVDA revenue increased", "AMD revenue increased"],
        "required_docs": ["nvda_10k_2024", "amd_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_009",
        "query": "Compare Walmart and Amazon operating income in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Amazon $68,593 million; Walmart $26,381 million",
        "gold_alternatives": ["Amazon higher operating income", "AMZN $68B", "WMT $26B"],
        "required_docs": ["amzn_10k_2024", "wmt_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_010",
        "query": "Compare Google and Microsoft capital expenditure in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Google $52,535 million; Microsoft $44,483 million",
        "gold_alternatives": ["Google higher capex", "GOOGL $52B", "MSFT $44B"],
        "required_docs": ["googl_10k_2024", "msft_10k_2024"],
        "required_pages": [23, 41, 55],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_011",
        "query": "How does JPMorgan's return on equity compare to Goldman Sachs in FY2024?",
        "bucket": "multi_hop",
        "gold_answer": "JPMorgan ROE approximately 17%; Goldman Sachs approximately 12%",
        "gold_alternatives": ["JPM higher ROE", "GS lower ROE"],
        "required_docs": ["jpm_10k_2024", "gs_10k_2024"],
        "required_pages": [30],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_012",
        "query": "Compare Apple's Services revenue to Google's advertising revenue in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Apple Services $96,169 million; Google advertising $264,589 million",
        "gold_alternatives": ["Google advertising higher", "AAPL Services", "GOOGL ads"],
        "required_docs": ["aapl_10k_2024", "googl_10k_2024"],
        "required_pages": [20, 23, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_013",
        "query": "Which had higher R&D spending in absolute terms in FY2024 — Google, Apple, or Microsoft?",
        "bucket": "multi_hop",
        "gold_answer": "Google at approximately $49 billion",
        "gold_alternatives": ["Alphabet highest", "GOOGL highest R&D", "Google $49B"],
        "required_docs": ["googl_10k_2024", "aapl_10k_2024", "msft_10k_2024"],
        "required_pages": [20, 23, 33, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_014",
        "query": "Compare Tesla and Amazon's net income margins in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Tesla net income margin approximately 7%; Amazon approximately 9%",
        "gold_alternatives": ["Amazon higher margin", "TSLA lower margin"],
        "required_docs": ["tsla_10k_2024", "amzn_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_015",
        "query": "How do NVIDIA's data center revenues compare to AMD's data center revenues in FY2024?",
        "bucket": "multi_hop",
        "gold_answer": "NVIDIA data center $115,168 million; AMD data center $12,658 million",
        "gold_alternatives": ["NVDA much larger", "AMD $12B", "NVDA $115B"],
        "required_docs": ["nvda_10k_2024", "amd_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_016",
        "query": "Compare Microsoft Azure and Google Cloud revenue growth rates in their most recent annual filings.",
        "bucket": "multi_hop",
        "gold_answer": "Both grew over 20% year-over-year",
        "gold_alternatives": ["Azure grew", "Google Cloud grew", "both grew"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024"],
        "required_pages": [23, 33, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_017",
        "query": "How does Walmart's revenue compare to Amazon's in FY2024?",
        "bucket": "multi_hop",
        "gold_answer": "Amazon $637,959 million; Walmart approximately $665 billion",
        "gold_alternatives": ["Walmart higher total", "WMT larger revenue", "similar scale"],
        "required_docs": ["wmt_10k_2024", "amzn_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_018",
        "query": "Compare Apple and Google's net income in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Apple $93,736 million; Google $100,118 million",
        "gold_alternatives": ["AAPL $93B", "GOOGL $100B", "Google slightly higher"],
        "required_docs": ["aapl_10k_2024", "googl_10k_2024"],
        "required_pages": [20, 23, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_019",
        "query": "Compare NVIDIA and AMD R&D as a percentage of revenue in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "AMD approximately 24%; NVIDIA approximately 10%",
        "gold_alternatives": ["AMD higher R&D%", "NVDA lower R&D%"],
        "required_docs": ["nvda_10k_2024", "amd_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_020",
        "query": "What is the difference in long-term debt between Apple and Microsoft as of their FY2024 annual filings?",
        "bucket": "multi_hop",
        "gold_answer": "Apple approximately $85 billion; Microsoft $42,688 million",
        "gold_alternatives": ["Apple more debt", "AAPL higher long-term debt"],
        "required_docs": ["aapl_10k_2024", "msft_10k_2024"],
        "required_pages": [41, 55],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },

    # Additional multi-hop — trend-based comparisons
    {
        "query_id": "mh_021",
        "query": "How has Microsoft's revenue grown from FY2023 to FY2024?",
        "bucket": "multi_hop",
        "gold_answer": "$211,915 million in FY2023 to $245,122 million in FY2024",
        "gold_alternatives": ["grew 16%", "increased $33B", "FY2023 $211B FY2024 $245B"],
        "required_docs": ["msft_10k_2024"],
        "required_pages": [41],
        "expected_ticker": "MSFT",
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_022",
        "query": "How has NVIDIA's gross margin trended from FY2023 to FY2025?",
        "bucket": "multi_hop",
        "gold_answer": "Gross margin improved significantly year over year",
        "gold_alternatives": ["increased each year", "margins expanded", "grew from 56% to 74%"],
        "required_docs": ["nvda_10k_2023", "nvda_10k_2024", "nvda_10k_2025"],
        "required_pages": [25, 41],
        "expected_ticker": "NVDA",
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_023",
        "query": "How has Tesla's automotive gross margin changed over the past three years?",
        "bucket": "multi_hop",
        "gold_answer": "Declined from approximately 25% in FY2022 to 17% in FY2024",
        "gold_alternatives": ["decreased", "margin compression", "fell from 25% to 17%"],
        "required_docs": ["tsla_10k_2023", "tsla_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": "TSLA",
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_024",
        "query": "How has AMD's revenue grown from FY2023 to FY2024?",
        "bucket": "multi_hop",
        "gold_answer": "Grew from approximately $22,680 million to $25,785 million",
        "gold_alternatives": ["increased 14%", "grew $3B", "FY2023 $22B FY2024 $25B"],
        "required_docs": ["amd_10k_2023", "amd_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": "AMD",
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_025",
        "query": "How has Apple's gross margin trended over the past 3 fiscal years?",
        "bucket": "multi_hop",
        "gold_answer": "$169,148 million in FY2023, $180,683 million in FY2024, $195,201 million in FY2025",
        "gold_alternatives": ["upward trend", "increased each year", "169B 180B 195B"],
        "required_docs": ["aapl_10k_2024", "aapl_10k_2025"],
        "required_pages": [20, 24],
        "expected_ticker": "AAPL",
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_026",
        "query": "Compare NVIDIA's revenue in Q2 FY2025 versus Q2 FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Significantly higher in Q2 FY2025",
        "gold_alternatives": ["grew substantially", "more than doubled", "NVDA Q2 FY2025 higher"],
        "required_docs": ["nvda_10q_2025", "nvda_10q_2024"],
        "required_pages": [9, 20],
        "expected_ticker": "NVDA",
        "expected_fy": None,
        "expected_form": "10-Q",
    },
    {
        "query_id": "mh_027",
        "query": "How do JPMorgan and Goldman Sachs compare in total assets as of their latest 10-K?",
        "bucket": "multi_hop",
        "gold_answer": "JPMorgan has significantly larger total assets",
        "gold_alternatives": ["JPM larger", "JPMorgan bigger balance sheet"],
        "required_docs": ["jpm_10k_2024", "gs_10k_2024"],
        "required_pages": [30, 41, 55],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_028",
        "query": "Compare Google and Microsoft operating margins in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Google approximately 32%; Microsoft approximately 45%",
        "gold_alternatives": ["Microsoft higher operating margin", "MSFT 45%", "GOOGL 32%"],
        "required_docs": ["googl_10k_2024", "msft_10k_2024"],
        "required_pages": [23, 33, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_029",
        "query": "Which company — Walmart or Amazon — reported higher gross profit in FY2024?",
        "bucket": "multi_hop",
        "gold_answer": "Amazon reported higher gross profit",
        "gold_alternatives": ["AMZN higher", "Amazon gross profit larger"],
        "required_docs": ["wmt_10k_2024", "amzn_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_030",
        "query": "Compare Tesla and Walmart's net income in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Walmart significantly higher net income than Tesla",
        "gold_alternatives": ["WMT higher", "Walmart more profitable", "TSLA lower net income"],
        "required_docs": ["tsla_10k_2024", "wmt_10k_2024"],
        "required_pages": [41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_031",
        "query": "How does NVIDIA's R&D spending compare to AMD's in FY2024 in absolute terms?",
        "bucket": "multi_hop",
        "gold_answer": "NVIDIA $12,914 million; AMD $6,273 million",
        "gold_alternatives": ["NVDA higher absolute R&D", "NVIDIA spent more on R&D"],
        "required_docs": ["nvda_10k_2024", "amd_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_032",
        "query": "Compare Apple's iPhone revenue to total Amazon product sales in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Apple iPhone revenue approximately $201 billion; Amazon product sales approximately $259 billion",
        "gold_alternatives": ["Amazon product sales higher", "AAPL iPhone vs AMZN products"],
        "required_docs": ["aapl_10k_2024", "amzn_10k_2024"],
        "required_pages": [20, 25, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_033",
        "query": "How has Walmart's operating income trended over FY2023, FY2024, and FY2025?",
        "bucket": "multi_hop",
        "gold_answer": "Consistently increased each year",
        "gold_alternatives": ["grew year over year", "upward trend", "increased annually"],
        "required_docs": ["wmt_10k_2024", "wmt_10k_2025"],
        "required_pages": [41, 43],
        "expected_ticker": "WMT",
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_034",
        "query": "Compare Goldman Sachs and JPMorgan investment banking revenues in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "JPMorgan investment banking revenues higher",
        "gold_alternatives": ["JPM larger IB revenue", "GS lower IB revenue"],
        "required_docs": ["jpm_10k_2024", "gs_10k_2024"],
        "required_pages": [30, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_035",
        "query": "How does Google's YouTube revenue compare to AMD's total revenue in FY2024?",
        "bucket": "multi_hop",
        "gold_answer": "YouTube $36,149 million; AMD $25,785 million — YouTube larger",
        "gold_alternatives": ["YouTube higher", "GOOGL YouTube > AMD total"],
        "required_docs": ["googl_10k_2024", "amd_10k_2024"],
        "required_pages": [23, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_036",
        "query": "Compare Tesla and NVIDIA net income margins in FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "NVIDIA significantly higher net income margin",
        "gold_alternatives": ["NVDA higher margin", "Tesla lower margin than NVIDIA"],
        "required_docs": ["tsla_10k_2024", "nvda_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_037",
        "query": "How does Apple's cash and equivalents compare to Microsoft's as of FY2024?",
        "bucket": "multi_hop",
        "gold_answer": "Both hold substantial cash; Apple approximately $29 billion, Microsoft approximately $18 billion",
        "gold_alternatives": ["Apple more cash", "AAPL higher cash position"],
        "required_docs": ["aapl_10k_2024", "msft_10k_2024"],
        "required_pages": [41, 55],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_038",
        "query": "Compare Amazon and Walmart's revenue growth rates between FY2023 and FY2024.",
        "bucket": "multi_hop",
        "gold_answer": "Amazon grew faster than Walmart",
        "gold_alternatives": ["AMZN higher growth rate", "Amazon faster growth"],
        "required_docs": ["amzn_10k_2024", "wmt_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_039",
        "query": "Which company had higher operating expenses as a percentage of revenue in FY2024 — Google or Microsoft?",
        "bucket": "multi_hop",
        "gold_answer": "Google had higher operating expenses as a percentage of revenue",
        "gold_alternatives": ["GOOGL higher opex%", "Google more operating expenses relative to revenue"],
        "required_docs": ["googl_10k_2024", "msft_10k_2024"],
        "required_pages": [23, 33, 41],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "mh_040",
        "query": "Compare NVIDIA, AMD, and Intel's (if available) GPU revenue positioning based on their latest 10-K filings.",
        "bucket": "multi_hop",
        "gold_answer": "NVIDIA dominates GPU revenue; AMD is second with data center GPUs growing",
        "gold_alternatives": ["NVDA largest", "AMD growing", "NVIDIA dominant"],
        "required_docs": ["nvda_10k_2025", "amd_10k_2024"],
        "required_pages": [25, 41, 43],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },

    # ── BUCKET 3: THEMATIC SYNTHESIS (30 queries) ──────────────────────────

    {
        "query_id": "th_001",
        "query": "How are semiconductor companies discussing AI-related demand in their latest filings?",
        "bucket": "thematic",
        "gold_answer": "NVIDIA and AMD both highlight strong AI/accelerated computing demand",
        "gold_alternatives": ["GPU demand", "AI accelerators", "data center growth"],
        "required_docs": ["nvda_10k_2025", "amd_10k_2024"],
        "required_pages": [12, 25],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": None,
    },
    {
        "query_id": "th_002",
        "query": "What risk factors related to AI regulation are mentioned across the tech companies in their latest 10-K filings?",
        "bucket": "thematic",
        "gold_answer": "Export controls, data privacy regulation, and AI safety cited across MSFT, GOOGL, NVDA",
        "gold_alternatives": ["regulatory risk", "export controls", "AI governance"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024", "nvda_10k_2025"],
        "required_pages": [10, 12, 15],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_003",
        "query": "How are cloud computing companies discussing capital expenditure increases for AI infrastructure?",
        "bucket": "thematic",
        "gold_answer": "Microsoft, Google, and Amazon all significantly increased capex for AI infrastructure",
        "gold_alternatives": ["increased investment", "data center spending", "AI capex"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024", "amzn_10k_2024"],
        "required_pages": [23, 33, 41, 55],
        "expected_ticker": None,
        "expected_fy": 2024,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_004",
        "query": "What are the main competitive risks that tech companies identify in their 10-K risk factors?",
        "bucket": "thematic",
        "gold_answer": "Competition from hyperscalers, open-source AI, and new market entrants cited broadly",
        "gold_alternatives": ["competitive risk", "market competition", "new entrants"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024", "amzn_10k_2024"],
        "required_pages": [10, 12, 15],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_005",
        "query": "How are financial institutions discussing interest rate risk in their latest filings?",
        "bucket": "thematic",
        "gold_answer": "JPMorgan and Goldman Sachs both discuss net interest income sensitivity to rate changes",
        "gold_alternatives": ["interest rate sensitivity", "NII impact", "rate risk"],
        "required_docs": ["jpm_10k_2024", "gs_10k_2024"],
        "required_pages": [30, 50],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_006",
        "query": "What are technology companies saying about generative AI product launches and monetisation in their most recent filings?",
        "bucket": "thematic",
        "gold_answer": "Microsoft Copilot, Google Gemini, and Amazon Bedrock all cited as new AI products generating revenue",
        "gold_alternatives": ["AI products", "Copilot", "Gemini", "generative AI monetisation"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024", "amzn_10k_2024"],
        "required_pages": [10, 12, 25, 33],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_007",
        "query": "How do companies in your database discuss supply chain risks in their 10-K filings?",
        "bucket": "thematic",
        "gold_answer": "Semiconductor and retail companies cite supplier concentration and geopolitical risks",
        "gold_alternatives": ["supply chain risk", "supplier concentration", "geopolitical risk"],
        "required_docs": ["nvda_10k_2025", "aapl_10k_2024", "wmt_10k_2024"],
        "required_pages": [10, 12],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_008",
        "query": "What are the electric vehicle companies and tech companies saying about sustainability and ESG commitments?",
        "bucket": "thematic",
        "gold_answer": "Tesla focuses on sustainable energy mission; tech companies cite emissions reduction targets",
        "gold_alternatives": ["ESG", "sustainability", "carbon neutral", "emissions"],
        "required_docs": ["tsla_10k_2024", "msft_10k_2024", "googl_10k_2024"],
        "required_pages": [10, 12, 15],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_009",
        "query": "How are companies discussing cybersecurity risks and investments across their latest 10-K filings?",
        "bucket": "thematic",
        "gold_answer": "Multiple companies cite increasing cybersecurity investment and regulatory disclosure requirements",
        "gold_alternatives": ["cybersecurity risk", "data breach", "security investment"],
        "required_docs": ["msft_10k_2024", "jpm_10k_2024", "amzn_10k_2024"],
        "required_pages": [10, 12, 15],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_010",
        "query": "What are semiconductor companies saying about export control regulations and geopolitical risks in their filings?",
        "bucket": "thematic",
        "gold_answer": "NVIDIA and AMD both highlight US export restrictions on advanced chips to China as material risk",
        "gold_alternatives": ["export controls", "China restrictions", "geopolitical semiconductor"],
        "required_docs": ["nvda_10k_2025", "amd_10k_2024"],
        "required_pages": [10, 12, 36],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_011",
        "query": "How are retail companies discussing the impact of inflation on their business in recent filings?",
        "bucket": "thematic",
        "gold_answer": "Walmart discusses cost pressures and consumer trade-down behaviour",
        "gold_alternatives": ["inflation impact", "consumer spending", "cost pressures"],
        "required_docs": ["wmt_10k_2024", "amzn_10k_2024"],
        "required_pages": [10, 12, 30],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_012",
        "query": "How do technology companies describe their cloud segment growth strategies in their MD&A sections?",
        "bucket": "thematic",
        "gold_answer": "Azure, Google Cloud, and AWS all highlight AI integration as key cloud growth driver",
        "gold_alternatives": ["cloud growth", "AI cloud", "Azure growth", "GCP strategy"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024", "amzn_10k_2024"],
        "required_pages": [23, 25, 33],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_013",
        "query": "What are financial companies saying about digital banking and fintech competition?",
        "bucket": "thematic",
        "gold_answer": "JPMorgan and Goldman Sachs both cite fintech and digital banking competition as risk factors",
        "gold_alternatives": ["fintech competition", "digital banking", "neobank"],
        "required_docs": ["jpm_10k_2024", "gs_10k_2024"],
        "required_pages": [10, 12, 30],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_014",
        "query": "How are companies discussing workforce and talent-related risks across sectors?",
        "bucket": "thematic",
        "gold_answer": "Tech and finance companies cite talent competition and retention as key operational risks",
        "gold_alternatives": ["talent risk", "workforce", "employee retention", "human capital"],
        "required_docs": ["msft_10k_2024", "nvda_10k_2025", "jpm_10k_2024"],
        "required_pages": [10, 12],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_015",
        "query": "What are the key growth drivers that technology companies highlight in their most recent annual reports?",
        "bucket": "thematic",
        "gold_answer": "AI, cloud computing, and digital advertising cited as primary growth drivers",
        "gold_alternatives": ["AI growth", "cloud expansion", "digital advertising growth"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024", "amzn_10k_2024", "nvda_10k_2025"],
        "required_pages": [10, 12, 25, 33],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_016",
        "query": "How are companies discussing shareholder returns — buybacks and dividends — in their latest 10-K filings?",
        "bucket": "thematic",
        "gold_answer": "Apple, Microsoft, and Google all have substantial buyback programmes",
        "gold_alternatives": ["share repurchase", "stock buyback", "dividend", "capital return"],
        "required_docs": ["aapl_10k_2024", "msft_10k_2024", "googl_10k_2024"],
        "required_pages": [41, 55, 60],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_017",
        "query": "What are EV and auto-adjacent companies saying about autonomous driving in their risk factors?",
        "bucket": "thematic",
        "gold_answer": "Tesla discusses Autopilot and Full Self-Driving regulatory and liability risks",
        "gold_alternatives": ["autonomous driving risk", "FSD", "self-driving regulation"],
        "required_docs": ["tsla_10k_2024", "tsla_10k_2025"],
        "required_pages": [10, 12],
        "expected_ticker": "TSLA",
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_018",
        "query": "How are companies discussing the impact of AI on their operating costs and workforce productivity?",
        "bucket": "thematic",
        "gold_answer": "Multiple companies cite AI-driven efficiency gains and potential workforce restructuring",
        "gold_alternatives": ["AI efficiency", "cost reduction AI", "productivity gains"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024", "amzn_10k_2024"],
        "required_pages": [10, 23, 33],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_019",
        "query": "What are the stated business segment priorities for semiconductor companies in their latest annual reports?",
        "bucket": "thematic",
        "gold_answer": "NVIDIA prioritises data centre; AMD focuses on data centre and client segments",
        "gold_alternatives": ["data center priority", "segment focus", "GPU segment"],
        "required_docs": ["nvda_10k_2025", "amd_10k_2024"],
        "required_pages": [10, 12, 25],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_020",
        "query": "How are companies discussing macroeconomic uncertainty and recession risk in their latest filings?",
        "bucket": "thematic",
        "gold_answer": "Multiple companies cite macroeconomic headwinds, interest rates, and consumer spending risk",
        "gold_alternatives": ["recession risk", "macroeconomic risk", "economic uncertainty"],
        "required_docs": ["jpm_10k_2024", "wmt_10k_2024", "amzn_10k_2024"],
        "required_pages": [10, 12, 30],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_021",
        "query": "How do tech companies discuss antitrust and regulatory risks in their 10-K filings?",
        "bucket": "thematic",
        "gold_answer": "Google, Amazon, Apple, and Microsoft all face active antitrust scrutiny noted in risk factors",
        "gold_alternatives": ["antitrust risk", "regulatory scrutiny", "competition law"],
        "required_docs": ["googl_10k_2024", "amzn_10k_2024", "aapl_10k_2024", "msft_10k_2024"],
        "required_pages": [10, 12],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_022",
        "query": "How are companies discussing geographic revenue concentration risks?",
        "bucket": "thematic",
        "gold_answer": "Multiple companies note significant US revenue concentration and China exposure risks",
        "gold_alternatives": ["geographic risk", "China exposure", "revenue concentration"],
        "required_docs": ["aapl_10k_2024", "nvda_10k_2025", "amzn_10k_2024"],
        "required_pages": [10, 12, 25],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_023",
        "query": "What are companies saying about AI chip supply constraints in recent filings?",
        "bucket": "thematic",
        "gold_answer": "NVIDIA notes constrained supply; cloud companies note GPU supply as limiting factor",
        "gold_alternatives": ["chip supply", "GPU shortage", "supply constraints", "H100 supply"],
        "required_docs": ["nvda_10k_2025", "msft_10k_2024", "googl_10k_2024"],
        "required_pages": [10, 12, 25, 36],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_024",
        "query": "How are retail companies discussing omnichannel and digital transformation strategies?",
        "bucket": "thematic",
        "gold_answer": "Walmart discusses store-as-fulfilment-centre; Amazon discusses physical-digital integration",
        "gold_alternatives": ["omnichannel", "digital transformation", "e-commerce integration"],
        "required_docs": ["wmt_10k_2024", "amzn_10k_2024"],
        "required_pages": [10, 25, 30],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_025",
        "query": "What are financial institutions saying about credit risk and loan loss provisions in their latest filings?",
        "bucket": "thematic",
        "gold_answer": "JPMorgan and Goldman Sachs both discuss credit risk management and provision for credit losses",
        "gold_alternatives": ["credit risk", "loan loss", "provision", "credit losses"],
        "required_docs": ["jpm_10k_2024", "gs_10k_2024"],
        "required_pages": [30, 50],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_026",
        "query": "How are companies discussing the transition from on-premise to cloud computing in their MD&A?",
        "bucket": "thematic",
        "gold_answer": "Microsoft, Google, and Amazon all discuss ongoing enterprise cloud migration as growth driver",
        "gold_alternatives": ["cloud migration", "on-premise to cloud", "hybrid cloud"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024", "amzn_10k_2024"],
        "required_pages": [23, 25, 33],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_027",
        "query": "What do companies say about intellectual property and patent risks across technology sectors?",
        "bucket": "thematic",
        "gold_answer": "Multiple companies cite IP litigation risk and patent challenges from competitors",
        "gold_alternatives": ["IP risk", "patent risk", "intellectual property litigation"],
        "required_docs": ["aapl_10k_2024", "msft_10k_2024", "nvda_10k_2025"],
        "required_pages": [10, 12],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_028",
        "query": "How are technology companies discussing the role of partnerships and ecosystems in their growth strategy?",
        "bucket": "thematic",
        "gold_answer": "Companies cite developer ecosystems, OEM partnerships, and platform expansion as growth levers",
        "gold_alternatives": ["partnerships", "ecosystem", "developer platform", "OEM"],
        "required_docs": ["msft_10k_2024", "googl_10k_2024", "nvda_10k_2025"],
        "required_pages": [10, 12, 25],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_029",
        "query": "What environmental and climate-related risks are technology and automotive companies disclosing?",
        "bucket": "thematic",
        "gold_answer": "Companies cite physical climate risk, energy costs, and emissions reduction obligations",
        "gold_alternatives": ["climate risk", "environmental risk", "carbon emissions"],
        "required_docs": ["tsla_10k_2024", "msft_10k_2024", "amzn_10k_2024"],
        "required_pages": [10, 12],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
    {
        "query_id": "th_030",
        "query": "How do companies across sectors discuss the impact of AI on their competitive positioning?",
        "bucket": "thematic",
        "gold_answer": "Tech, finance, and retail companies all discuss AI as both opportunity and competitive threat",
        "gold_alternatives": ["AI competitive advantage", "AI positioning", "AI differentiation"],
        "required_docs": ["msft_10k_2024", "jpm_10k_2024", "wmt_10k_2024", "nvda_10k_2025"],
        "required_pages": [10, 12, 25, 33],
        "expected_ticker": None,
        "expected_fy": None,
        "expected_form": "10-K",
    },
]


def recall_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> float:
    """Compute recall@k over retrieved page numbers."""
    if not gold_pages:
        return 1.0
    hits = sum(1 for p in gold_pages if p in retrieved_pages[:k])
    return hits / len(gold_pages)


def precision_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> float:
    """Compute precision@k over retrieved page numbers."""
    top_k = retrieved_pages[:k]
    if not top_k:
        return 0.0
    return sum(1 for p in top_k if p in gold_pages) / len(top_k)


def f1_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> float:
    """Compute F1@k from precision@k and recall@k."""
    r = recall_at_k(retrieved_pages, gold_pages, k)
    p = precision_at_k(retrieved_pages, gold_pages, k)
    return 2 * r * p / (r + p) if (r + p) else 0.0


def mrr(retrieved_pages: list[int], gold_pages: list[int]) -> float:
    """Compute mean reciprocal rank for the first relevant page."""
    for i, p in enumerate(retrieved_pages, 1):
        if p in gold_pages:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> float:
    """Compute nDCG@k for ranked retrieved pages."""
    gold_set = set(gold_pages)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, p in enumerate(retrieved_pages[:k])
        if p in gold_set
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
    return dcg / ideal if ideal else 0.0


def compute_retriever_metrics(results: list[dict]) -> dict:
    """Aggregate retriever metrics across all results."""
    ks = [5, 10, 20]
    agg: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    agg.update({f"precision@{k}": [] for k in ks})
    agg.update({f"f1@{k}": [] for k in ks})
    agg.update({f"ndcg@{k}": [] for k in ks})
    agg["mrr"] = []

    for r in results:
        rp = r.get("retrieved_pages", [])
        gp = r.get("required_pages", [])
        for k in ks:
            agg[f"recall@{k}"].append(recall_at_k(rp, gp, k))
            agg[f"precision@{k}"].append(precision_at_k(rp, gp, k))
            agg[f"f1@{k}"].append(f1_at_k(rp, gp, k))
            agg[f"ndcg@{k}"].append(ndcg_at_k(rp, gp, k))
        agg["mrr"].append(mrr(rp, gp))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}


def _answer_contains_gold(answer: str, gold: str, alts: list[str]) -> bool:
    """Check whether the generated answer contains the gold answer or an allowed variant."""
    al = answer.lower()
    return any(g.lower() in al for g in [gold] + (alts or []))


def compute_generator_metrics(results: list[dict]) -> dict:
    """Aggregate answer correctness, faithfulness proxy, and abstention rate."""
    correctness, faithfulness, abstentions = [], [], []

    for r in results:
        answer = r.get("answer", "")
        hallucinated = r.get("hallucinated_numbers", [])
        is_abstention = answer.strip().startswith("Insufficient evidence")

        abstentions.append(float(is_abstention))
        faithfulness.append(1.0 if (is_abstention or not hallucinated) else 0.0)

        if is_abstention:
            correctness.append(0.0)
        else:
            correctness.append(
                1.0
                if _answer_contains_gold(
                    answer,
                    r.get("gold_answer", ""),
                    r.get("gold_alternatives", []),
                )
                else 0.0
            )

    return {
        "answer_correctness": float(np.mean(correctness)),
        "faithfulness_proxy": float(np.mean(faithfulness)),
        "abstention_rate": float(np.mean(abstentions)),
        "n": len(results),
    }


_CITE_RE = re.compile(
    r"\[(?:[A-Z][A-Z0-9\-]*)\s+(?:[A-Za-z0-9\-\/]+)\s+FY\d{4},\s*pp?\.?\s*(\d+)"
    r"(?:\s*[-–]\s*(\d+))?\]",
    re.IGNORECASE,
)


def _cited_pages(answer: str) -> list[int]:
    """Extract cited page numbers from the generated answer."""
    pages = []
    for m in _CITE_RE.finditer(answer):
        pages.append(int(m.group(1)))
        if m.group(2):
            pages.append(int(m.group(2)))
    return pages


def citation_accuracy(results: list[dict]) -> float:
    """Measure how often the answer cites at least one required page."""
    answerable = [r for r in results if not r.get("answer", "").startswith("Insufficient")]
    if not answerable:
        return 0.0

    hits = sum(
        1
        for r in answerable
        if not r.get("required_pages")
        or any(p in r["required_pages"] for p in _cited_pages(r.get("answer", "")))
    )
    return hits / len(answerable)


def citation_match_rate(results: list[dict]) -> float:
    """Measure how many parsed citations matched retrieved source metadata."""
    all_cites = [c for r in results for c in r.get("citations", [])]
    if not all_cites:
        return 0.0
    return sum(1 for c in all_cites if c.get("matched")) / len(all_cites)


ABLATION_VARIANTS = [
    {"name": "LLM-only (no retrieval)", "dense": False, "sparse": False, "reranker": False, "verifier": False},
    {"name": "Dense-only retrieval", "dense": True, "sparse": False, "reranker": False, "verifier": False},
    {"name": "Sparse-only retrieval", "dense": False, "sparse": True, "reranker": False, "verifier": False},
    {"name": "Hybrid (dense + sparse)", "dense": True, "sparse": True, "reranker": False, "verifier": False},
    {"name": "Hybrid + Reranker", "dense": True, "sparse": True, "reranker": True, "verifier": False},
    {"name": "Hybrid + Reranker + Verifier", "dense": True, "sparse": True, "reranker": True, "verifier": True},
    {"name": "Fixed chunking (512 tokens)", "dense": True, "sparse": True, "reranker": True, "verifier": True, "note": "manual"},
    {"name": "Hierarchical + table-aware", "dense": True, "sparse": True, "reranker": True, "verifier": True, "note": "manual"},
]


def _run_one(query: str, variant: dict) -> dict:
    """Run one query under a specific ablation configuration."""
    from src.generate.generator import generate
    from src.generate.query_planner import plan_retrieval
    from src.retrieve.hybrid_search import hybrid_search, rerank, assemble_context

    dense_on = variant.get("dense", True)
    sparse_on = variant.get("sparse", True)
    rank_on = variant.get("reranker", True)
    verify_on = variant.get("verifier", True)

    if not dense_on and not sparse_on:
        res = generate(query=query, context_chunks=[])
        return {
            "answer": res["answer"],
            "citations": res["citations"],
            "hallucinated_numbers": res["hallucinated_numbers"],
            "retrieved_pages": [],
            "context_chunks": [],
            "verification": {},
        }

    plan = plan_retrieval(query)

    if dense_on and not sparse_on:
        os.environ["RETRIEVAL_SPARSE_WEIGHT"] = "0.0001"
        candidates = hybrid_search(
            query=plan.get("query", query),
            filters=plan.get("filters", {}),
            dense_top_k=plan.get("dense_top_k", 50),
            sparse_top_k=1,
            plan=plan,
        )
        os.environ["RETRIEVAL_SPARSE_WEIGHT"] = "1.1"

    elif sparse_on and not dense_on:
        os.environ["RETRIEVAL_DENSE_WEIGHT"] = "0.0001"
        candidates = hybrid_search(
            query=plan.get("query", query),
            filters=plan.get("filters", {}),
            dense_top_k=1,
            sparse_top_k=plan.get("sparse_top_k", 50),
            plan=plan,
        )
        os.environ["RETRIEVAL_DENSE_WEIGHT"] = "1.0"

    else:
        candidates = hybrid_search(
            query=plan.get("query", query),
            filters=plan.get("filters", {}),
            dense_top_k=plan.get("dense_top_k", 50),
            sparse_top_k=plan.get("sparse_top_k", 50),
            plan=plan,
        )

    ranked = (
        rerank(query, candidates, top_k=plan.get("reranker_k", 12), plan=plan)
        if rank_on
        else candidates[:plan.get("reranker_k", 12)]
    )
    context = assemble_context(ranked, max_chunks=plan.get("final_k", 8), plan=plan)

    if verify_on:
        from src.generate.pipeline import run_query

        full = run_query(query)
        return {
            "answer": full.get("answer", ""),
            "citations": full.get("citations", []),
            "hallucinated_numbers": full.get("hallucinated_numbers", []),
            "retrieved_pages": [
                int(c.get("page", 0))
                for c in full.get("context_used", [])
                if c.get("page")
            ],
            "context_chunks": context,
            "verification": full.get("verification", {}),
        }

    res = generate(query=query, context_chunks=context)
    return {
        "answer": res["answer"],
        "citations": res["citations"],
        "hallucinated_numbers": res["hallucinated_numbers"],
        "retrieved_pages": [int(c.get("page", 0)) for c in context if c.get("page")],
        "context_chunks": context,
        "verification": {},
    }


def _run_batch(
    test_cases: list[dict],
    variant: dict,
    prog: Optional[Progress] = None,
) -> list[dict]:
    """Run all test cases for a single ablation variant."""
    results = []

    for tc in test_cases:
        try:
            r = _run_one(tc["query"], variant)
        except Exception as e:
            log.warning("Query %s failed (%s): %s", tc["query_id"], variant["name"], e)
            r = {
                "answer": "Error",
                "citations": [],
                "hallucinated_numbers": [],
                "retrieved_pages": [],
                "context_chunks": [],
                "verification": {},
            }

        r.update(
            {
                k: tc[k]
                for k in (
                    "query_id",
                    "query",
                    "bucket",
                    "gold_answer",
                    "gold_alternatives",
                    "required_pages",
                    "required_docs",
                )
            }
        )
        results.append(r)

        if prog:
            prog.update(tc["query_id"])

    return results


def _format_ablation_table(rows: list[dict]) -> str:
    """Format ablation metrics as a fixed-width table."""
    h = f"{'Variant':<46} {'Correct':>8} {'Faith':>7} {'R@10':>7} {'nDCG@10':>8} {'CitAcc':>7}"
    sep = "─" * len(h)
    lines = [h, sep]

    def _fmt(val, width, decimals=3):
        if val is None:
            return f"{'N/A':>{width}}"
        return f"{val:>{width}.{decimals}f}"

    for r in rows:
        if "error" in r:
            lines.append(f"{r['variant']:<46} ERROR: {r['error'][:40]}")
            continue

        lines.append(
            f"{r['variant']:<46}"
            f" {_fmt(r.get('answer_correctness'), 8)}"
            f" {_fmt(r.get('faithfulness'), 7)}"
            f" {_fmt(r.get('recall@10'), 7)}"
            f" {_fmt(r.get('ndcg@10'), 8)}"
            f" {_fmt(r.get('citation_accuracy'), 7)}"
        )

    return "\n".join(lines)


def _write_results_table(summary: dict, path: Path) -> None:
    """Write a plain-text summary table to disk."""
    lines = ["=" * 65, "EVALUATION RESULTS", "=" * 65, ""]

    ret = summary.get("retriever_metrics", {}).get("overall", {})
    if ret:
        lines += [
            "10.2 RETRIEVER METRICS",
            "─" * 40,
            f"  Recall@5:      {ret.get('recall@5', 0):.4f}",
            f"  Recall@10:     {ret.get('recall@10', 0):.4f}",
            f"  Recall@20:     {ret.get('recall@20', 0):.4f}",
            f"  Precision@10:  {ret.get('precision@10', 0):.4f}",
            f"  F1@10:         {ret.get('f1@10', 0):.4f}",
            f"  MRR:           {ret.get('mrr', 0):.4f}",
            f"  nDCG@10:       {ret.get('ndcg@10', 0):.4f}",
            "",
        ]

    gen = summary.get("generator_metrics", {}).get("overall", {})
    if gen:
        lines += [
            "10.3 GENERATOR METRICS",
            "─" * 40,
            f"  Answer Correctness:  {gen.get('answer_correctness', 0):.4f}",
            f"  Faithfulness:        {gen.get('faithfulness_proxy', 0):.4f}",
            f"  Abstention Rate:     {gen.get('abstention_rate', 0):.4f}",
        ]
        if "ragas_faithfulness" in gen:
            lines += [
                f"  RAGAS Faithfulness:        {gen['ragas_faithfulness']:.4f}",
                f"  RAGAS Answer Correctness:  {gen['ragas_answer_correctness']:.4f}",
                f"  RAGAS Context Precision:   {gen['ragas_context_precision']:.4f}",
                f"  RAGAS Context Recall:      {gen['ragas_context_recall']:.4f}",
            ]
        lines.append("")

    cit = summary.get("citation_metrics", {})
    if cit:
        lines += [
            "10.4 CITATION METRICS",
            "─" * 40,
            f"  Citation Accuracy:   {cit.get('citation_accuracy', 0):.4f}",
            f"  Citation Match Rate: {cit.get('citation_match_rate', 0):.4f}",
            "",
        ]

    abl = summary.get("ablation_study", [])
    if abl:
        lines += ["10.5 ABLATION STUDY", "─" * 40, _format_ablation_table(abl), ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_failure_analysis(results: list[dict], path: Path) -> dict:
    """Write categorized failure cases to disk for analysis."""
    failures: dict[str, list] = {
        "abstentions": [],
        "wrong_period": [],
        "hallucination_blocked": [],
        "citation_miss": [],
        "errors": [],
    }

    for r in results:
        answer = r.get("answer", "")

        if "error" in r and r.get("answer") == "Error":
            failures["errors"].append(
                {"query_id": r["query_id"], "error": r.get("error", "")}
            )
            continue

        if answer.startswith("Insufficient evidence"):
            failures["abstentions"].append(
                {
                    "query_id": r["query_id"],
                    "query": r.get("query", ""),
                    "bucket": r.get("bucket"),
                    "verification": r.get("verification", {}),
                }
            )

        if not r.get("verification", {}).get("correct_period", True):
            failures["wrong_period"].append(
                {"query_id": r["query_id"], "query": r.get("query", "")}
            )

        if r.get("hallucinated_numbers"):
            failures["hallucination_blocked"].append(
                {
                    "query_id": r["query_id"],
                    "hallucinated": r["hallucinated_numbers"],
                }
            )

        cited = _cited_pages(answer)
        gold = set(r.get("required_pages", []))
        if gold and not any(p in gold for p in cited) and not answer.startswith("Insufficient"):
            failures["citation_miss"].append(
                {
                    "query_id": r["query_id"],
                    "cited": cited,
                    "required": list(gold),
                }
            )

    path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return failures


def run_evaluation(
    test_cases: list[dict],
    output_dir: Path,
    section: Optional[str],
    quick: bool,
) -> dict:
    """Run evaluation, save outputs, and return the summary metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _patch_fy_cache()

    if quick:
        by_bucket: dict[str, list] = {"single_hop": [], "multi_hop": [], "thematic": []}
        for tc in test_cases:
            bucket = tc["bucket"]
            if bucket in by_bucket and len(by_bucket[bucket]) < 5:
                by_bucket[bucket].append(tc)
        test_cases = [tc for group in by_bucket.values() for tc in group]

    total_q = len(test_cases)
    print(f"\n  Evaluation started — {total_q} queries", flush=True)
    if quick:
        print("  Mode: QUICK (5 per bucket)", flush=True)
    print(f"  Output: {output_dir.resolve()}\n", flush=True)

    summary: dict[str, Any] = {}
    raw_results: list[dict] = []

    if section in (None, "retriever", "generator", "citation"):
        _section_header(f"10.2 / 10.3 / 10.4  Running {total_q} queries")
        prog = Progress(total_q, "Queries")

        for tc in test_cases:
            try:
                r = _run_one(
                    tc["query"],
                    {"dense": True, "sparse": True, "reranker": True, "verifier": True},
                )
            except Exception as e:
                log.error("Query %s failed: %s", tc["query_id"], e)
                r = {
                    "answer": "Error",
                    "citations": [],
                    "hallucinated_numbers": [],
                    "retrieved_pages": [],
                    "context_chunks": [],
                    "verification": {},
                    "error": str(e),
                }

            r.update(
                {
                    k: tc[k]
                    for k in (
                        "query_id",
                        "query",
                        "bucket",
                        "gold_answer",
                        "gold_alternatives",
                        "required_pages",
                        "required_docs",
                    )
                }
            )
            raw_results.append(r)
            prog.update(f"{tc['query_id']} {tc['query'][:45]}")

        prog.done_msg(f"{total_q} queries complete")

        raw_path = output_dir / "raw_results.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for r in raw_results:
                f.write(json.dumps(r, default=str) + "\n")

        if section in (None, "retriever"):
            _section_header("10.2  Retriever metrics")
            overall_ret = compute_retriever_metrics(raw_results)
            by_bucket_ret = {}
            for b in ("single_hop", "multi_hop", "thematic"):
                sub = [r for r in raw_results if r.get("bucket") == b]
                if sub:
                    by_bucket_ret[b] = compute_retriever_metrics(sub)

            summary["retriever_metrics"] = {"overall": overall_ret, "by_bucket": by_bucket_ret}
            print(
                f"  Recall@10={overall_ret.get('recall@10', 0):.3f}  "
                f"MRR={overall_ret.get('mrr', 0):.3f}  "
                f"nDCG@10={overall_ret.get('ndcg@10', 0):.3f}",
                flush=True,
            )

        if section in (None, "generator"):
            _section_header("10.3  Generator metrics")
            overall_gen = compute_generator_metrics(raw_results)
            by_bucket_gen = {}
            for b in ("single_hop", "multi_hop", "thematic"):
                sub = [r for r in raw_results if r.get("bucket") == b]
                if sub:
                    by_bucket_gen[b] = compute_generator_metrics(sub)

            summary["generator_metrics"] = {"overall": overall_gen, "by_bucket": by_bucket_gen}
            print(
                f"  Correctness={overall_gen.get('answer_correctness', 0):.3f}  "
                f"Faithfulness={overall_gen.get('faithfulness_proxy', 0):.3f}  "
                f"Abstention={overall_gen.get('abstention_rate', 0):.3f}",
                flush=True,
            )

        if section in (None, "citation"):
            _section_header("10.4  Citation metrics")
            cit_acc = citation_accuracy(raw_results)
            cit_match = citation_match_rate(raw_results)
            summary["citation_metrics"] = {
                "citation_accuracy": cit_acc,
                "citation_match_rate": cit_match,
            }
            print(
                f"  Citation accuracy={cit_acc:.3f}  Match rate={cit_match:.3f}",
                flush=True,
            )

    if section in (None, "ablation"):
        abl_cases: list[dict] = []
        for b in ("single_hop", "multi_hop", "thematic"):
            abl_cases += [tc for tc in test_cases if tc["bucket"] == b][:10]

        n_abl = len(ABLATION_VARIANTS)
        n_q = len(abl_cases)
        total_abl = n_abl * n_q

        _section_header(
            f"10.5  Ablation study — {n_abl} variants × {n_q} queries = {total_abl} calls"
        )
        print("  Manual chunking variants must be run separately", flush=True)

        prog_abl = Progress(total_abl, "Ablation")
        ablation_results: list[dict] = []

        for variant in ABLATION_VARIANTS:
            vname = variant["name"]

            if variant.get("note") == "manual":
                print(f"\n  [{vname}] requires manual chunker swap, skipping", flush=True)
                ablation_results.append(
                    {
                        "variant": vname,
                        "description": "Requires manual chunker configuration",
                        "answer_correctness": None,
                        "faithfulness": None,
                        "recall@10": None,
                        "ndcg@10": None,
                        "citation_accuracy": None,
                    }
                )
                for _ in abl_cases:
                    prog_abl.update(vname[:20])
                continue

            print(f"\n  Running: {vname}", flush=True)
            v_results = []

            for tc in abl_cases:
                try:
                    r = _run_one(tc["query"], variant)
                except Exception as e:
                    log.warning("Ablation %s / %s failed: %s", vname, tc["query_id"], e)
                    r = {
                        "answer": "Error",
                        "citations": [],
                        "hallucinated_numbers": [],
                        "retrieved_pages": [],
                        "context_chunks": [],
                        "verification": {},
                    }

                r.update(
                    {
                        k: tc[k]
                        for k in (
                            "query_id",
                            "query",
                            "bucket",
                            "gold_answer",
                            "gold_alternatives",
                            "required_pages",
                            "required_docs",
                        )
                    }
                )
                v_results.append(r)
                prog_abl.update(f"{vname[:20]} / {tc['query_id']}")

            gen_m = compute_generator_metrics(v_results)
            ret_m = compute_retriever_metrics(v_results)
            cit_a = citation_accuracy(v_results)

            ablation_results.append(
                {
                    "variant": vname,
                    "n_queries": len(v_results),
                    "answer_correctness": gen_m["answer_correctness"],
                    "faithfulness": gen_m["faithfulness_proxy"],
                    "abstention_rate": gen_m["abstention_rate"],
                    "recall@10": ret_m.get("recall@10", 0.0),
                    "precision@10": ret_m.get("precision@10", 0.0),
                    "ndcg@10": ret_m.get("ndcg@10", 0.0),
                    "mrr": ret_m.get("mrr", 0.0),
                    "citation_accuracy": cit_a,
                }
            )

        prog_abl.done_msg("Ablation complete")
        summary["ablation_study"] = ablation_results
        print("\n" + _format_ablation_table(ablation_results), flush=True)

    _section_header("Writing results")

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("  evaluation_summary.json", flush=True)

    if raw_results:
        failures = _write_failure_analysis(raw_results, output_dir / "failure_analysis.json")
        print(
            f"  failure_analysis.json  "
            f"(abstentions={len(failures['abstentions'])}  "
            f"wrong_period={len(failures['wrong_period'])}  "
            f"citation_miss={len(failures['citation_miss'])})",
            flush=True,
        )

    _write_results_table(summary, output_dir / "results_table.txt")
    print("  results_table.txt", flush=True)

    testset_path = output_dir / "testset.jsonl"
    with testset_path.open("w", encoding="utf-8") as f:
        for tc in TEST_SET:
            f.write(json.dumps(tc) + "\n")
    print(f"  testset.jsonl  ({len(TEST_SET)} queries)", flush=True)

    print(f"\n  Done. Results in: {output_dir.resolve()}\n", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation script."""
    p = argparse.ArgumentParser(description="RAG Investment Pipeline Evaluation")
    p.add_argument("--output", default="results/", help="Output directory")
    p.add_argument(
        "--section",
        choices=["retriever", "generator", "citation", "ablation"],
        default=None,
        help="Run only one section",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run 5 queries per bucket for a quick smoke test",
    )
    p.add_argument(
        "--save-testset-only",
        action="store_true",
        help="Write testset.jsonl and exit",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_testset_only:
        path = output_dir / "testset.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for tc in TEST_SET:
                f.write(json.dumps(tc) + "\n")
        print(f"Test set saved to {path}  ({len(TEST_SET)} queries)")
        sys.exit(0)

    run_evaluation(
        test_cases=TEST_SET,
        output_dir=output_dir,
        section=args.section,
        quick=args.quick,
    )