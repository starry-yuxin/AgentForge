#!/usr/bin/env python3
"""Prepare the registered public UCI Iranian Churn dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge.datasets import prepare_uci_iranian_churn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="atomically replace the downloaded archive")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-root", help="exact dataset runtime directory")
    args = parser.parse_args()
    processed, metadata = prepare_uci_iranian_churn(
        root=args.output_root, force=args.force, timeout=args.timeout
    )
    display_root = Path(args.output_root).resolve() if args.output_root else ROOT
    print(f"processed_dataset: {processed.resolve().relative_to(display_root)}")
    print(f"metadata: {metadata.resolve().relative_to(display_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
