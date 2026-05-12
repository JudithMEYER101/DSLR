import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_csv import load


def is_index_like(column: pd.Series) -> bool:
    """Return True if the column looks like a simple row index."""
    if not pd.api.types.is_numeric_dtype(column):
        return False

    values = column.dropna().to_numpy()
    if len(values) == 0:
        return False

    return (
        all(values[i] == i for i in range(len(values)))
        or all(values[i] == i + 1 for i in range(len(values)))
    )


def get_numeric_columns(dataset: pd.DataFrame, min_non_null: int = 10) -> list:
    """Return usable numeric columns for plotting."""

    numeric_df = dataset.select_dtypes(include=["number"])
    valid_columns = []

    for col in numeric_df.columns:
        series = dataset[col]

        if str(col).startswith("Unnamed"):
            continue
        if is_index_like(series):
            continue
        if series.notna().sum() < min_non_null:
            continue
        if series.nunique(dropna=True) <= 1:
            continue

        valid_columns.append(col)

    return valid_columns


def select_features(dataset: pd.DataFrame, max_features: int = 5) -> list:
    """Select the first valid numeric features."""
    cols = get_numeric_columns(dataset, min_non_null=10)
    return cols[:max_features]


def plot_pair_plot(dataset: pd.DataFrame, features: list) -> None:
    """Display and save a pair plot without category groups."""

    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))

    if n == 1:
        axes = [[axes]]

    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            ax = axes[i][j]

            if i == j:
                values = dataset[f1].dropna()
                if len(values) > 0:
                    ax.hist(values, bins=30, alpha=0.7, edgecolor="black")
            else:
                sub_df = dataset[[f2, f1]].dropna()
                if len(sub_df) > 0:
                    ax.scatter(sub_df[f2], sub_df[f1], alpha=0.5, s=10)

            if i == n - 1:
                ax.set_xlabel(str(f2)[:10], fontsize=8)
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(str(f1)[:10], fontsize=8)
            else:
                ax.set_yticks([])

    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/pair_plot_no_houses.png", dpi=300, bbox_inches="tight")
    print("Saved pair plot to output/pair_plot_no_houses.png")

    plt.show()


def main():
    """Load dataset, select valid numeric features, and display a pair plot."""

    if len(sys.argv) != 2:
        print("Usage: python3 pair_plot.py <dataset.csv>")
        return

    dataset = load(sys.argv[1])
    if dataset is None:
        return

    features = select_features(dataset, max_features=14)

    if not features:
        print("Error: no valid numeric features found.")
        return

    print("Selected features:")
    for f in features:
        print(
            f"- {f} | non-null: {dataset[f].notna().sum()} | "
            f"unique: {dataset[f].nunique(dropna=True)}"
        )

    plot_pair_plot(dataset, features)


if __name__ == "__main__":
    main()