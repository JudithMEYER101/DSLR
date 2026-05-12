import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_csv import load
import utils.scatter_plot_utils as us




def main() -> None:
    """Compare correlation for each pair of course for all
    students, print the most correlated pair and plot them
    in a scatter plot, show the plot and save it in output"""

    if len(sys.argv) != 2:
        print("Usage: python3 scatter_plot.py <dataset.csv>")
        return

    dataset = load(sys.argv[1])
    if dataset is None:
        return

    if "Hogwarts House" not in dataset.columns:
        print("Error: 'Hogwarts House' column not found.")
        return

    courses = us.get_numeric_columns(dataset)
    if not courses:
        print("Error: no numeric columns found.")
        return

    ranked_results = us.compute_similarity_scores(dataset, courses)
    if not ranked_results:
        print("Error: could not compute similarity scores.")
        return

    us.print_similarity_ranking(ranked_results)
    us.plot_scatter_pairs(dataset, ranked_results)


if __name__ == "__main__":
    main()