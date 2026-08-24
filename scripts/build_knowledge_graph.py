"""Build and export the AgentForge knowledge graph."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge.knowledge import KnowledgeGraphStore, load_capabilities
from agentforge.knowledge.importer import validate_sources


def main() -> None:
    knowledge_dir = ROOT / "knowledge"
    capabilities = load_capabilities(knowledge_dir / "capabilities.json")
    validate_sources(capabilities, knowledge_dir / "documents")

    store = KnowledgeGraphStore()
    store.build_from_capabilities(capabilities)

    benchmark_path = ROOT / "outputs" / "benchmark_result.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            "outputs/benchmark_result.json not found; run python scripts/run_demo.py first"
        )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    run_id = store.add_validation_run(benchmark)

    graphml_path = knowledge_dir / "knowledge_graph.graphml"
    json_path = knowledge_dir / "knowledge_graph.json"
    store.export_graphml(graphml_path)
    store.export_json(json_path)

    relation_types = {data["relation"] for _, _, data in store.graph.edges(data=True)}
    print(f"Loaded {len(capabilities)} validated capabilities")
    print(f"Wrote ValidationRun: {run_id}")
    print(f"Graph: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print("Relations: " + ", ".join(sorted(relation_types)))
    print(f"GraphML: {graphml_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

