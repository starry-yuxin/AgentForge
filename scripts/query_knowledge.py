"""Query the exported AgentForge knowledge graph."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge.knowledge import KnowledgeGraphStore, KnowledgeRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query AgentForge knowledge")
    parser.add_argument("--task", required=True, help="Task type")
    parser.add_argument("--metric", help="Required metric")
    parser.add_argument("--characteristic", action="append", default=[])
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--failure-type")
    return parser.parse_args()


def _print_group(title: str, recommendations: list) -> None:
    print(f"\n{title}:")
    if not recommendations:
        print("  (none)")
        return
    for item in recommendations:
        print(f"  - {item.name} [{item.id}]")
        print("    reasons: " + "; ".join(item.reasons))
        print(f"    source: {item.source_document} / {item.source_section}")


def main() -> None:
    args = parse_args()
    graph_path = ROOT / "knowledge" / "knowledge_graph.graphml"
    if not graph_path.is_file():
        raise FileNotFoundError(
            "knowledge graph not found; run python scripts/build_knowledge_graph.py first"
        )
    store = KnowledgeGraphStore.load_graphml(graph_path)
    result = KnowledgeRetriever(store).query(
        task_type=args.task,
        data_characteristics=args.characteristic,
        required_metric=args.metric,
        constraints=args.constraint,
        failure_type=args.failure_type,
    )
    print("AgentForge knowledge retrieval")
    _print_group("Recommended algorithms", result.algorithms)
    _print_group("Recommended preprocessing", result.preprocessors)
    _print_group("Recommended metrics", result.metrics)
    _print_group("Relevant failure experiences", result.failure_experiences)


if __name__ == "__main__":
    main()

