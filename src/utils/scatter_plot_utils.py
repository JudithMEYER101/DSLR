import os
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from utils import statistics as stat

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

def ft_correlation(x, y):
    """Calculate pear correlation between two lists.
        the closer to 1 the more positive correlation
        the closer to -1 the more negative corelation
        the closer to 0 the less correlation"""

    if not x or not y:
        return None
    if len(x) != len(y) or len(x) < 2:
        return None

    mean_x = stat.ft_mean(x)
    mean_y = stat.ft_mean(y)

    # Check distance of x and y from their mean 
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / (len(x) - 1)

    # Check how spread x and y are
    std_x = stat.ft_std(x)
    std_y = stat.ft_std(y)

    if std_x is None or std_y is None or std_x == 0 or std_y == 0:
        return None

    # Normalize covarience by removing scale effect with std
    return covariance / (std_x * std_y)


def compute_similarity_scores(dataset: pd.DataFrame, courses: list) -> list:
    """
    Compute similarity score for each pair of numeric features.
    Higher absolute correlation = more similar.
    """

    results = []

    # Generate all pairs
    for course1, course2 in itertools.combinations(courses, 2):
        pair_df = dataset[[course1, course2]].dropna()

        if len(pair_df) < 2:
            continue

        # Compute correlation
        x = pair_df[course1].tolist()
        y = pair_df[course2].tolist()

        corr = ft_correlation(x, y)

        if pd.isna(corr):
            continue

        # Store result (correlation is -1 to 1 and score is absolute)
        results.append({
            "course1": course1,
            "course2": course2,
            "correlation": corr,
            "score": abs(corr),
        })

    # Sort result (Highest to lowest correlation)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def print_similarity_ranking(results: list, top_n: int = 20) -> None:
    """Print highest correlation to top_n correlation"""

    # Print header
    print("\nMost similar feature pairs:\n")
    print(f"{'Rank':<6}{'Feature 1':<30}{'Feature 2':<30}{'Corr':>12}")
    print("-" * 78)

    # Print result pair by pair
    for i, result in enumerate(results[:top_n], start=1):
        print(
            f"{i:<6}"
            f"{result['course1']:<30}"
            f"{result['course2']:<30}"
            f"{result['correlation']:>12.6f}"
        )

    # Print best pair
    if results:
        best = results[0]
        print(f"\nMost similar features: {best['course1']} and {best['course2']}")


def plot_scatter_pairs(dataset: pd.DataFrame, ranked_results: list, top_n: int = 4) -> None:
    """Display scatter plots for the top similar pairs up to top_n"""

    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    top_results = ranked_results[:top_n]

    # Define grid size
    n_plots = len(top_results)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    # Scatter value in each plot
    for i, result in enumerate(top_results):
        course1 = result["course1"]
        course2 = result["course2"]
        corr = result["correlation"]
        ax = axes[i]

        for house in houses:
            house_df = dataset[dataset["Hogwarts House"] == house][[course1, course2]].dropna()

            if len(house_df) == 0:
                continue

            # house_df[course1] = x, house_df[course2] = y, s = size of points
            ax.scatter(
                house_df[course1],
                house_df[course2],
                alpha=0.6,
                label=house,
                color=HOUSE_COLORS[house],
                s=15,
            )

        # Set labels
        ax.set_title(f"{course1[:12]} / {course2[:12]} [corr={corr:.3f}]", fontsize=10)
        ax.set_xlabel(course1[:12], fontsize=9)
        ax.set_ylabel(course2[:12], fontsize=9)

    # Remove empty plots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    # Get and place legend
    handles = []
    labels = []
    for ax in axes[:n_plots]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break

    if handles:
        fig.legend(handles, labels, loc="center right")

    # Adjust spacing
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])

    # Output result to output/scatter_plot
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/scatter_plot.png", dpi=300, bbox_inches="tight")
    print("Saved scatter plot to output/scatter_plot.png")

    plt.show()
