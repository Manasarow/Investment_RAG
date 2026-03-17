"""
CLI interface for querying the RAG system.
Usage: python scripts/query.py --q "What was Microsoft's revenue in FY2024?"
       python scripts/query.py  (interactive mode)
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.generate.pipeline import run_query


def print_result(result: dict):
    print("\n" + "="*60)
    print(f"QUERY  : {result.get('query','')}")
    print(f"INTENT : {result.get('intent','')}")
    print(f"TICKERS: {result.get('tickers','')}")
    print(f"RETRIES: {result.get('retries', 0)}")
    print("="*60)
    print("\nANSWER:\n")
    print(result.get("answer", "No answer generated."))

    citations = result.get("citations", [])
    if citations:
        print(f"\nCITATIONS ({len(citations)}):")
        for c in citations:
            matched = "✓" if c.get("matched") else "✗"
            print(f"  {matched} {c['label']}")

    context_used = result.get("context_used", [])
    if context_used:
        print(f"\nCONTEXT CHUNKS USED ({len(context_used)}):")
        for c in context_used:
            print(f"  [{c['ticker']} {c['form']} FY{c['fy']} p.{c['page']}] "
                  f"{c['type']} — {c['section'][:40]}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, help="Query string")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.q:
        result = run_query(args.q)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_result(result)
    else:
        # Interactive mode
        print("RAG Investment Query System — type 'exit' to quit\n")
        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ("exit", "quit", "q"):
                    break
                if not query:
                    continue
                result = run_query(query)
                print_result(result)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()