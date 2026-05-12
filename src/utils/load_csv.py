import pandas as pd


def load(path: str):
    """Load a CSV dataset, print its dimensions, and return it."""
    try:
        # Use first column as dataframe index
        dataset = pd.read_csv(path, index_col=0)
        print(f"Loading dataset of dimensions {dataset.shape}")
        return dataset
    except Exception:
        print("Error: could not load dataset.")
        return None
