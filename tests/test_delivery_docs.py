"""Offline checks for final delivery documentation and visualization."""

from pathlib import Path
import ast
import json
import re
import subprocess
import sys

from agentforge.cli import _parser

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_SUFFIXES = {".md", ".json", ".py", ".txt", ".svg", ".diff"}


def test_readme_documents_real_entry_points_and_safety_boundaries() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = [
        "python -m agentforge.cli demo", "selection_metric_on_validation",
        "final_metrics_on_test", "deterministic", "hybrid", "llm",
        "不是容器、虚拟机或生产级安全沙箱", "```mermaid",
        "docs/assets/knowledge-graph.svg",
    ]
    assert all(item in readme for item in required)
    assert "/" + "Users/" not in readme
    assert "Authorization" + ": Bearer" not in readme


def test_documented_delivery_files_exist() -> None:
    for relative in [
        "docs/reproducibility.md", "examples/README.md", "knowledge/schema.md",
        "knowledge/capabilities.json", "knowledge/knowledge_graph.json",
        "knowledge/knowledge_graph.graphml", "scripts/visualize_knowledge_graph.py",
        "docs/assets/knowledge-graph.svg",
    ]:
        assert (ROOT / relative).is_file(), relative


def test_public_delivery_artifacts_do_not_contain_home_paths() -> None:
    paths = [ROOT / "README.md", *(ROOT / "docs").rglob("*"), *(ROOT / "examples").rglob("*")]
    for path in paths:
        if path.is_file() and path.suffix in PUBLIC_TEXT_SUFFIXES:
            source = path.read_text(encoding="utf-8")
            assert "/" + "Users/" not in source, path
            assert "Authorization" + ": Bearer" not in source, path


def test_readme_local_links_and_mermaid_blocks_are_valid() -> None:
    document = ROOT / "README.md"
    source = document.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", source):
        if target.startswith(("https://", "http://", "#")):
            continue
        assert (document.parent / target.split("#", 1)[0]).exists(), target
    assert source.count("```mermaid") == 2
    assert source.count("```") % 2 == 0


def test_documented_cli_arguments_exist() -> None:
    parser = _parser()
    demo = parser.parse_args(["demo", "--inject-failure", "missing_imputer"])
    assert demo.command == "demo" and demo.inject_failure == "missing_imputer"
    run = parser.parse_args([
        "run", "--request", "customer churn", "--dataset", "data/churn_sample.csv",
        "--metric", "f1", "--minimum-score", "0.60", "--no-persist",
    ])
    assert run.command == "run" and run.metric == "f1" and run.no_persist


def test_examples_are_parseable_and_sanitized() -> None:
    for path in (ROOT / "examples").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in (ROOT / "examples").rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    forbidden = ["/" + "Users/yuxin/", "Authorization" + ": Bearer", "sk-" + "live"]
    for path in (ROOT / "examples").rglob("*"):
        if path.is_file() and path.suffix in PUBLIC_TEXT_SUFFIXES:
            source = path.read_text(encoding="utf-8")
            assert not any(value in source for value in forbidden), path


def test_dotenv_is_ignored_and_example_has_no_key_value() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True,
        check=False,
    )
    assert ignored.returncode == 0 and ignored.stdout.strip() == ".env"
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    key_lines = [line for line in example.splitlines() if line.startswith("OPENAI_API_KEY=")]
    assert key_lines == ["OPENAI_API_KEY="]


def test_visualization_script_writes_sanitized_svg(tmp_path: Path) -> None:
    output = tmp_path / "graph.svg"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "visualize_knowledge_graph.py"),
         "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    source = output.read_text(encoding="utf-8")
    assert source.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Representative: 16 of 62 nodes" in source
    for label in ("LogisticRegression", "RandomForest", "F1", "ROCAUC", "MissingValueError"):
        assert label in source
    assert "/" + "Users/" not in source
