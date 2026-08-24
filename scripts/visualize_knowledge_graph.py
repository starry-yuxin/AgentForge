"""Render the committed knowledge graph as a deterministic SVG."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "Task": "#0f766e", "Algorithm": "#2563eb", "Preprocessor": "#7c3aed",
    "Metric": "#db2777", "FailureExperience": "#dc2626", "ValidationRun": "#ea580c",
    "Dataset": "#0891b2", "Dependency": "#4b5563", "Constraint": "#64748b",
    "SourceDocument": "#16a34a",
}
LABELED_TYPES = {"Task", "Algorithm", "Preprocessor", "Metric", "FailureExperience", "Dataset"}
REPRESENTATIVE_NODES = {
    "binary_classification", "logistic_regression", "random_forest",
    "numerical_imputation", "categorical_imputation", "one_hot_encoding",
    "standard_scaling", "class_weight_balancing", "validation_threshold_optimization",
    "f1", "roc_auc", "missing_value_error", "categorical_encoding_error",
    "data_leakage_risk", "stage1_validation", "customer_churn_sample",
}


def _load_graph(path: Path) -> nx.MultiDiGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    edge_key = "links" if "links" in payload else "edges"
    return nx.node_link_graph(payload, edges=edge_key)


def render(graph_path: Path, output_path: Path) -> None:
    complete_graph = _load_graph(graph_path)
    selected = REPRESENTATIVE_NODES & set(complete_graph)
    graph = complete_graph.subgraph(selected).copy()
    positions = nx.spring_layout(nx.Graph(graph), seed=42, k=0.5, iterations=150)
    width, height, margin = 1600, 1050, 100
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    def project(node: str) -> tuple[float, float]:
        x, y = positions[node]
        px = margin + (x - x_min) / max(x_max - x_min, 1e-9) * (width - 2 * margin)
        py = margin + (y - y_min) / max(y_max - y_min, 1e-9) * (height - 2 * margin)
        return px, py

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">AgentForge knowledge graph</title>',
        '<desc id="desc">Representative subgraph generated from the committed AgentForge knowledge graph.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g stroke="#cbd5e1" stroke-width="1" stroke-opacity="0.55">',
    ]
    for source, target in graph.edges():
        x1, y1 = project(source)
        x2, y2 = project(target)
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    lines.append('</g>')
    for node, attributes in graph.nodes(data=True):
        node_type = attributes.get("node_type", "Unknown")
        x, y = project(node)
        radius = 11 if node_type in LABELED_TYPES else 6
        color = COLORS.get(node_type, "#94a3b8")
        label = html.escape(str(attributes.get("name", node)))
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
                     f'stroke="#ffffff" stroke-width="2"><title>{label} ({node_type})</title></circle>')
        if node_type in LABELED_TYPES:
            lines.append(f'<text x="{x + radius + 4:.1f}" y="{y + 4:.1f}" '
                         f'font-family="system-ui,sans-serif" font-size="12" fill="#0f172a">{label}</text>')
    counts = Counter(attributes.get("node_type", "Unknown") for _, attributes in graph.nodes(data=True))
    legend_x, legend_y = 30, 30
    lines.append('<g font-family="system-ui,sans-serif" font-size="13" fill="#0f172a">')
    lines.append(f'<text x="{legend_x}" y="{legend_y}" font-size="18" font-weight="600">AgentForge Knowledge Graph</text>')
    lines.append(f'<text x="{legend_x}" y="{legend_y + 24}">Representative: {graph.number_of_nodes()} of '
                 f'{complete_graph.number_of_nodes()} nodes · {graph.number_of_edges()} edges</text>')
    for index, (node_type, count) in enumerate(sorted(counts.items())):
        y = legend_y + 50 + index * 20
        lines.append(f'<circle cx="{legend_x + 6}" cy="{y - 4}" r="6" fill="{COLORS.get(node_type, "#94a3b8")}"/>')
        lines.append(f'<text x="{legend_x + 18}" y="{y}">{html.escape(node_type)} ({count})</text>')
    lines.append('</g></svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "knowledge" / "knowledge_graph.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets" / "knowledge-graph.svg")
    args = parser.parse_args()
    render(args.graph, args.output)
    print(f"Rendered {args.output}")


if __name__ == "__main__":
    main()
