"""Run the deterministic stage-one benchmark."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge.pipeline import run_benchmark


def main() -> None:
    data_path = ROOT / "data" / "churn_sample.csv"
    output_path = ROOT / "outputs" / "benchmark_result.json"
    result = run_benchmark(data_path, output_path)
    print("AgentForge churn benchmark")
    for candidate in result["candidates"]:
        metrics = candidate["metrics"]
        print(
            f"- {candidate['algorithm']}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"runtime_seconds={metrics['runtime_seconds']:.4f}"
        )
    print(f"Best algorithm by {result['primary_metric']}: {result['best_algorithm']}")
    print(f"JSON result: {output_path}")


if __name__ == "__main__":
    main()

