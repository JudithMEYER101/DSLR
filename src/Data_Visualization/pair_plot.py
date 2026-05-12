import sys
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_csv import load


HOUSE_COLORS = {
    "Gryffindor": "red",
    "Hufflepuff": "yellow",
    "Ravenclaw": "blue",
    "Slytherin": "green",
}


def get_numeric_columns(dataset: pd.DataFrame) -> list:
    """Return all numeric columns."""

    numeric_df = dataset.select_dtypes(include=["number"])
    return numeric_df.columns.tolist()


def select_features(dataset: pd.DataFrame, max_features: int = 5) -> list:
    """
    Select a subset of features to keep the plot readable.
    Here we simply take the first N numeric features.
    """

    cols = get_numeric_columns(dataset)
    return cols[:max_features]


def plot_pair_plot(dataset: pd.DataFrame, features: list) -> None:
    """Display pair plot (scatter matrix)."""

    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

    n = len(features)
    # Define grid size
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))

    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            
            # Current subplot
            ax = axes[i][j]

            # Create the histograms (ex: Astronomy vs Astronomy)
            if i == j:
                for house in houses:
                    values = dataset[dataset["Hogwarts House"] == house][f1].dropna()
                    if len(values) == 0:
                        continue

                    ax.hist(
                        values,
                        bins=30,
                        alpha=0.5,
                        color=HOUSE_COLORS[house],
                        edgecolor="black",
                    )
            # create the scatter plots (ex: Astronomy vs Herbology)
            else:
                for house in houses:
                    sub_df = dataset[dataset["Hogwarts House"] == house][[f2, f1]].dropna()

                    if len(sub_df) == 0:
                        continue

                    ax.scatter(
                        sub_df[f2],
                        sub_df[f1],
                        color=HOUSE_COLORS[house],
                        alpha=0.5,
                        s=10,
                    )

            # Create labels only on bottom row
            if i == n - 1:
                ax.set_xlabel(f2[:10], fontsize=8)
            else:
                ax.set_xticks([])

            # Create labels only on left row
            if j == 0:
                ax.set_ylabel(f1[:10], fontsize=8)
            else:
                ax.set_yticks([])

    # Create legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color, label=house, markersize=6)
        for house, color in HOUSE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.9, 0.97])

    # Save output
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/pair_plot.png", dpi=300, bbox_inches="tight")
    print("Saved pair plot to output/pair_plot.png")

    plt.show()


def main():
    """Take an amount of features and display a pair plot
    saving it in output"""

    if len(sys.argv) != 2:
        print("Usage: python3 pair_plot.py <dataset.csv>")
        return

    dataset = load(sys.argv[1])
    if dataset is None:
        return

    if "Hogwarts House" not in dataset.columns:
        print("Error: 'Hogwarts House' column not found.")
        return

    features = select_features(dataset, max_features=13)

    if not features:
        print("Error: no numeric features found.")
        return

    print("Selected features for pair plot:")
    for f in features:
        print("-", f)

    plot_pair_plot(dataset, features)


if __name__ == "__main__":
    main()