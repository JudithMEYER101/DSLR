import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_csv import load
import utils.histogram_utils as hu

def main() -> None:
    """Display histogram of all course and evaluate
    homogeneity base on overlap of pairs"""

    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset.csv>")
        return

    dataset = load(sys.argv[1])
    if dataset is None:
        return

    if "Hogwarts House" not in dataset.columns:
        print("Error: 'Hogwarts House' column not found.")
        return

    courses = hu.get_numeric_columns(dataset)
    if not courses:
        print("Error: no numeric columns found.")
        return

    ranked_results = hu.compute_overlap_scores(dataset, courses)
    if not ranked_results:
        print("Error: could not compute overlap scores.")
        return

    hu.print_homogeneity_ranking(ranked_results)
    hu.plot_histograms(dataset, ranked_results)


if __name__ == "__main__":
    main()


# Tried a statistical approach using mean and std but 
# result where better with overlap of pair