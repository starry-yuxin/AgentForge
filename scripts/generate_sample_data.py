"""Create the default customer churn dataset."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge.data import generate_churn_data


def main() -> None:
    output = ROOT / "data" / "churn_sample.csv"
    frame = generate_churn_data(output)
    print(f"Generated {len(frame)} rows at {output}")
    print(f"Churn rate: {frame['churn'].mean():.3f}")
    print(f"Missing values: {int(frame.isna().sum().sum())}")


if __name__ == "__main__":
    main()

