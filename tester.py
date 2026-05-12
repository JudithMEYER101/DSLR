import sys
import pandas as pd


def compute_accuracy(train_path, pred_path):
    # Load datasets
    try:
        train = pd.read_csv(train_path)
        pred = pd.read_csv(pred_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Check required columns
    required_cols = ["Index", "Hogwarts House"]
    for col in required_cols:
        if col not in train.columns or col not in pred.columns:
            print(f"Missing required column: {col}")
            return

    # Merge on Index
    merged = pd.merge(
        train[["Index", "Hogwarts House"]],
        pred[["Index", "Hogwarts House"]],
        on="Index",
        suffixes=("_true", "_pred")
    )

    if len(merged) == 0:
        print("No matching rows found between datasets.")
        return

    # Compute accuracy
    correct = (merged["Hogwarts House_true"] == merged["Hogwarts House_pred"]).sum()
    total = len(merged)
    accuracy = (correct / total) * 100

    print(f"Accuracy: {accuracy:.4f}% ({correct}/{total})")


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compute_accuracy.py dataset_train.csv houses.csv")
        return

    train_path = sys.argv[1]
    pred_path = sys.argv[2]

    compute_accuracy(train_path, pred_path)


if __name__ == "__main__":
    main()