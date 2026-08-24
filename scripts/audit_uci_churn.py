#!/usr/bin/env python3
"""Run the deterministic UCI duplicate/proxy-field sensitivity audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge.datasets import run_uci_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/external/uci_iranian_churn/processed/uci_iranian_churn.csv")
    parser.add_argument("--output-root", default="outputs/uci-audit")
    args = parser.parse_args()
    result = run_uci_audit(args.dataset, args.output_root)
    print(f"audit_id: {result['audit_id']}")
    print(f"output_dir: {result['output_dir']}")
    print("llm_calls: 0")
    print("knowledge_persisted: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
