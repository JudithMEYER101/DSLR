import sys
import os
import csv
from datetime import date, datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.statistics import (
    ft_count,
    ft_mean,
    ft_std,
    ft_min,
    ft_percentile,
    ft_max,
)

RED = "\033[38;5;203m"
GREEN = "\033[38;5;84m"
YELLOW = "\033[38;5;220m"
BLUE = "\033[38;5;39m"
RESET = "\033[0m"

COLORS = [RED, GREEN, YELLOW, BLUE]
MAX_COL_WIDTH = 13


def load_csv(path):
    """Load a CSV file without pandas."""
    try:
        with open(path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)
    except Exception as error:
        print(f"{RED}Error: {error}{RESET}")
        return None


def is_float(value):
    """Check if a value can be converted to float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def truncate(name):
    """Limit displayed column names to 13 characters."""
    return name[:MAX_COL_WIDTH]


def format_value(value):
    """Format numbers without useless trailing zeros."""
    if value is None:
        return "NaN"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")

    return str(value)


def compute_age(birthday_str):
    """Compute age from a YYYY-MM-DD birthday string."""
    try:
        birthday = datetime.strptime(birthday_str, "%Y-%m-%d").date()
        today = date.today()
        age = today.year - birthday.year
        if (today.month, today.day) < (birthday.month, birthday.day):
            age -= 1
        return age
    except (TypeError, ValueError):
        return None


def print_stats(columns, stats):
    """Print the statistics table."""
    first_col_width = 10
    col_width = MAX_COL_WIDTH + 2

    header = " " * first_col_width
    for i, column in enumerate(columns):
        color = COLORS[i % len(COLORS)]
        header += f"{color}{truncate(column):>{col_width}}{RESET}"
    print(header)

    for stat_name, values in stats.items():
        line = f"{BLUE}{stat_name:<{first_col_width}}{RESET}"
        for value in values:
            formatted = format_value(value)
            line += f"{formatted:>{col_width}}"
        print(line)


def describe_numeric(data):
    """Describe all numeric columns except index-like columns."""
    columns = list(data[0].keys())
    numeric_columns = []

    for column in columns:
        clean_column = column.strip().lower()

        if clean_column in ("", "index", "id") or "index" in clean_column:
            continue

        values = [row[column] for row in data if row[column] != ""]
        if values and all(is_float(value) for value in values):
            numeric_columns.append(column)

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

    for column in numeric_columns:
        values = [float(row[column]) for row in data if row[column] != ""]
        values.sort()

        stats["Count"].append(ft_count(values))
        stats["Mean"].append(ft_mean(values))
        stats["Std"].append(ft_std(values))
        stats["Min"].append(ft_min(values))
        stats["25%"].append(ft_percentile(values, 0.25))
        stats["50%"].append(ft_percentile(values, 0.50))
        stats["75%"].append(ft_percentile(values, 0.75))
        stats["Max"].append(ft_max(values))

    print_stats(numeric_columns, stats)


def extra_stats(data):
    """Print the extra requested student statistics."""
    print(f"\n{YELLOW}Extra stats{RESET}")

    total_students = len(data)

    hands = [
        row["Best Hand"].strip().lower()
        for row in data
        if row.get("Best Hand") and row["Best Hand"].strip()
    ]
    right_handed_count = sum(1 for hand in hands if hand == "right")
    right_handed_percentage = (
        (right_handed_count / len(hands)) * 100 if hands else 0
    )

    first_name_lengths = [
        len(row["First Name"].strip())
        for row in data
        if row.get("First Name") and row["First Name"].strip()
    ]
    last_name_lengths = [
        len(row["Last Name"].strip())
        for row in data
        if row.get("Last Name") and row["Last Name"].strip()
    ]

    birthdays = [
        row["Birthday"]
        for row in data
        if row.get("Birthday") and row["Birthday"].strip()
    ]

    birthday_counts = {}
    for birthday in birthdays:
        try:
            parsed = datetime.strptime(birthday, "%Y-%m-%d")
            key = (parsed.month, parsed.day)
            birthday_counts[key] = birthday_counts.get(key, 0) + 1
        except ValueError:
            continue

    students_sharing_with_at_least_3_others = sum(
        count for count in birthday_counts.values() if count >= 4
    )

    ages = []
    for birthday in birthdays:
        age = compute_age(birthday)
        if age is not None:
            ages.append(age)

    age_counts = {}
    for age in ages:
        age_counts[age] = age_counts.get(age, 0) + 1

    mean_first_name_length = ft_mean(first_name_lengths) if first_name_lengths else None
    mean_last_name_length = ft_mean(last_name_lengths) if last_name_lengths else None
    mean_age = ft_mean(ages) if ages else None

    print(f"Total students: {GREEN}{total_students}{RESET}")

    print(
        f"Right-handed students: "
        f"{GREEN}{format_value(right_handed_percentage)}%{RESET}"
    )

    if mean_first_name_length is not None:
        print(
            f"Mean first name length: "
            f"{GREEN}{format_value(mean_first_name_length)}{RESET}"
        )
    else:
        print("Mean first name length: NaN")

    if mean_last_name_length is not None:
        print(
            f"Mean last name length: "
            f"{GREEN}{format_value(mean_last_name_length)}{RESET}"
        )
    else:
        print("Mean last name length: NaN")

    print(
        "Students sharing their birthday with at least 3 students: "
        f"{GREEN}{students_sharing_with_at_least_3_others}{RESET}"
    )

    if mean_age is not None:
        print(f"Mean age: {GREEN}{format_value(mean_age)}{RESET}")
    else:
        print("Mean age: NaN")

    print(f"\n{YELLOW}Students per age{RESET}")
    for age in sorted(age_counts):
        print(f"{BLUE}Age {age:<3}{RESET}: {GREEN}{age_counts[age]}{RESET}")


def main():
    """Describe numeric columns and print extra student statistics."""
    if len(sys.argv) != 2:
        print(f"{RED}Usage: python describe.py dataset_train.csv{RESET}")
        return

    path = sys.argv[1]
    data = load_csv(path)
    if not data:
        return

    describe_numeric(data)
    extra_stats(data)


if __name__ == "__main__":
    main()