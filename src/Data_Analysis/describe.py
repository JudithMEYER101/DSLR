import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_csv import load
from utils.statistics import ft_count, ft_mean, ft_std, ft_min, ft_percentile, ft_max

RED = "\033[38;5;203m"
GREEN = "\033[38;5;84m"
YELLOW = "\033[38;5;220m"
BLUE = "\033[38;5;39m"
RESET = "\033[0m"

COLORS = [RED, GREEN, YELLOW, BLUE]

def main():
    """Describe numerical value of a dataser"""

    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        return

    path = sys.argv[1]
    dataset = load(path)
    if dataset is None:
        return

    numeric_df = dataset.select_dtypes(include=["number"])

    stats = {
        "Count": [],
        "Mean": [],
        "Std": [],
        "Min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "Max": [],
    }

    columns = numeric_df.columns.tolist()

    for column in columns:
        # Get rid of incomplete value
        values = numeric_df[column].dropna().tolist()
        values.sort()

        stats["Count"].append(ft_count(values))
        stats["Mean"].append(ft_mean(values))
        stats["Std"].append(ft_std(values))
        stats["Min"].append(ft_min(values))
        stats["25%"].append(ft_percentile(values, 0.25))
        stats["50%"].append(ft_percentile(values, 0.50))
        stats["75%"].append(ft_percentile(values, 0.75))
        stats["Max"].append(ft_max(values))

    print_stats(columns, stats)


def print_stats(columns, stats):
    label_width = 12
    value_width = 12

    # Shorten title columns for lisibility
    short_cols = [col[:10] for col in columns]

    print(f"{'':<{label_width}}", end="")
    for i, col in enumerate(short_cols):
        color = COLORS[i % 4]
        print(f"{color}{col:>{value_width}}{RESET}", end="")
    print()

    for stat_name, values in stats.items():
        print(f"{stat_name:<{label_width}}", end="")

        # Add color
        for i, value in enumerate(values):
            color = COLORS[i % 4]

            if value is None:
                formatted = ""
            else:
                formatted = f"{value:.6f}".rstrip('0').rstrip('.')

            print(f"{color}{formatted:>{value_width}}{RESET}", end="")

        print()


if __name__ == "__main__":
    main()