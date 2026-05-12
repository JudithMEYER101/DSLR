import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def compute_overlap_scores(dataset: pd.DataFrame, courses: list) -> list:
    """Get homogeneity score based on overlap of pair
    of all houses after normalization"""

    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    results = []

    for course in courses:
        house_values = {}

        for house in houses:
            values = dataset[dataset["Hogwarts House"] == house][course].dropna().values
            if len(values) > 0:
                house_values[house] = values

        if len(house_values) < 2:
            continue

        all_values = dataset[course].dropna().values

        # bins defines the number of slice we evaluate
        bins = np.histogram_bin_edges(all_values, bins=30)

        if len(bins) < 2:
            continue

        bin_width = bins[1] - bins[0]
        histograms = {}

        # Histograms per houses
        for house, values in house_values.items():
            hist, _ = np.histogram(values, bins=bins, density=True)
            histograms[house] = hist

        overlaps = []

        # Generate pairs (G, H) (G, R) (...) and check overlap
        for house1, house2 in itertools.combinations(histograms.keys(), 2):
            hist1 = histograms[house1]
            hist2 = histograms[house2]
            overlap = np.sum(np.minimum(hist1, hist2)) * bin_width
            overlaps.append(overlap)

        # Average overlap score per course
        avg_overlap = sum(overlaps) / len(overlaps)

        results.append({
            "course": course,
            "score": avg_overlap,
        })

    # Sort overlap (highest to lowest)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def print_homogeneity_ranking(results: list) -> None:
    """Print courses sorted from most to least homogeneous."""

    print("\nCourses sorted from most to least homogeneous:\n")
    print(f"{'Rank':<6}{'Course':<30}{'Overlap':>12}")
    print("-" * 48)

    for i, result in enumerate(results, start=1):
        print(
            f"{i:<6}"
            f"{result['course']:<30}"
            f"{result['score']:>12.6f}"
        )

    if results:
        print(f"\nMost homogeneous course: {results[0]['course']}")


def plot_histograms(dataset: pd.DataFrame, ranked_results: list) -> None:
    """Display all course histograms ordered by homogeneity score."""

    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

    n_courses = len(ranked_results)

    # Define placements and size of histograms
    n_cols = 4
    n_rows = math.ceil(n_courses / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten()

    # Create subplot
    for i, result in enumerate(ranked_results):
        course = result["course"]
        score = result["score"]
        ax = axes[i]

        all_values = dataset[course].dropna().values
        # bins defines the number of slice we evaluate (number of column in histogram)
        bins = np.histogram_bin_edges(all_values, bins=30)

        # Draw one histogram per house on each subplot
        for house in houses:
            values = dataset[dataset["Hogwarts House"] == house][course].dropna()

            if len(values) == 0:
                continue

            # alpha = transparency
            ax.hist(
                values,
                bins=bins,
                alpha=0.5,
                label=house,
                color=HOUSE_COLORS[house],
                edgecolor="black",
            )

        ax.set_title(f"{course[:14]} [overlap={score:.3f}]", fontsize=10)
        # Kept empty for clarity (x is notes and y is students number)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Delete empty subplot
    for i in range(n_courses, len(axes)):
        fig.delaxes(axes[i])

    # Build and place legend
    handles = []
    labels = []
    for ax in axes[:n_courses]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break

    if handles:
        fig.legend(handles, labels, loc="center right")

    # Adjust spacing between subplot
    fig.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.tight_layout(rect=[0.1, 0, 0.85, 0.97])

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/histogram.png", dpi=300, bbox_inches="tight")
    print("Saved histogram to output/histogram.png")

    plt.show()

