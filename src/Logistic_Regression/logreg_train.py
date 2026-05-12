import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_csv import load


# Feature selection using the most 
# relevant feature identified previously
# Astronomy act as global separator (isolate all houses)
# Herbology and Ancient runes as complementary global separator
# Divination and flying as class specific separator

FEATURES = [
    "Astronomy",
    "Herbology",
    "Ancient Runes",
    "Divination",
    "Flying",

]

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


def sigmoid(z):
    """Apply sigmoid function to turn value into a probability
    between 1 and 0 (closer to 1 = closer to true)"""

    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def normalize_features(df: pd.DataFrame, means=None, stds=None):
    """Normalize features with z-score to even features weigths"""

    if means is None:
        means = df.mean()
    if stds is None:
        stds = df.std(ddof=1)

    stds = stds.replace(0, 1)
    normalized = (df - means) / stds
    return normalized, means, stds


def prepare_data(dataset: pd.DataFrame):
    """Extract features and labels from training dataset."""

    missing = [col for col in FEATURES + ["Hogwarts House"] if col not in dataset.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x_df = dataset[FEATURES].copy()
    y = dataset["Hogwarts House"].copy()

    # Remove rows with missing values in selected features or label
    valid_mask = x_df.notna().all(axis=1) & y.notna()
    x_df = x_df[valid_mask]
    y = y[valid_mask]

    x_df, means, stds = normalize_features(x_df)
    x = x_df.to_numpy(dtype=float)

    return x, y.to_numpy(), means, stds


def train_one_vs_rest(x, y_binary, learning_rate=0.1, iterations=5000):
    """Train one binary logistic regression model with gradient descent.
    (One vs rest)"""

    m, n = x.shape
    weights = np.zeros(n)
    bias = 0.0

    for _ in range(iterations):

        # Linear regression
        z = np.dot(x, weights) + bias

        # Convert to probability
        y_hat = sigmoid(z)

        # Derivatife of the cost (how to ajust : dw = weight, db = bias)
        dw = np.dot(x.T, (y_hat - y_binary)) / m
        db = np.sum(y_hat - y_binary) / m

        # Gradient descent (adjustment with db and dw)
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias


def train_models(x, y):
    """Train one-vs-rest models for all houses."""

    models = {}

    for house in HOUSES:
        y_binary = np.array([1 if label == house else 0 for label in y], dtype=float)
        weights, bias = train_one_vs_rest(x, y_binary)

        # Save the model
        models[house] = {
            "weights": weights.tolist(),
            "bias": float(bias),
        }

    # Return all models
    return models


def save_model(filepath, models, means, stds):
    """Save trained models and normalization stats."""

    payload = {
        "features": FEATURES,
        "houses": HOUSES,
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "models": models,
    }

    # Write JSON file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 logreg_train.py dataset_train.csv")
        return

    dataset = load(sys.argv[1])
    if dataset is None:
        return

    # Prepare data
    try:
        x, y, means, stds = prepare_data(dataset)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Train the models
    models = train_models(x, y)

    # Save weights in output/weights.json
    os.makedirs("output", exist_ok=True)
    save_model("output/weights.json", models, means, stds)
    print("Model trained successfully.")
    print("Weights saved to output/weights.json")


if __name__ == "__main__":
    main()