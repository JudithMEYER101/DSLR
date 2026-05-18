import sys
import os
import json
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_csv import load
from bonus.stochastic_gradient_descent import train_one_vs_rest_sgd
from bonus.mini_batch_gradient_descent import train_one_vs_rest_minibatch


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

# SGD BONUS
BATCH_LEARNING_RATE = 0.1
BATCH_ITERATIONS = 5000
SGD_LEARNING_RATE = 0.01
SGD_EPOCHS = 60
# MINIBATCH BONUS
MINIBATCH_LEARNING_RATE = 0.05
MINIBATCH_EPOCHS = 80
MINIBATCH_SIZE = 32


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


def train_one_vs_rest(x, y_binary, learning_rate=BATCH_LEARNING_RATE, iterations=BATCH_ITERATIONS):
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


def train_models(x, y, use_sgd=False, use_minibatch=False, minibatch_size=MINIBATCH_SIZE):
    """Train one-vs-rest models for all houses."""

    models = {}

    for house in HOUSES:
        y_binary = np.array([1 if label == house else 0 for label in y], dtype=float)
        if use_sgd:
            weights, bias = train_one_vs_rest_sgd(
                x,
                y_binary,
                learning_rate=SGD_LEARNING_RATE,
                epochs=SGD_EPOCHS,
            )
        elif use_minibatch:
            weights, bias = train_one_vs_rest_minibatch(
                x,
                y_binary,
                learning_rate=MINIBATCH_LEARNING_RATE,
                epochs=MINIBATCH_EPOCHS,
                batch_size=minibatch_size,
            )
        else:
            weights, bias = train_one_vs_rest(x, y_binary)

        # SAVE
        models[house] = {
            "weights": weights.tolist(),
            "bias": float(bias),
        }

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

    # WRITE SAVE
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a one-vs-rest logistic regression model."
    )
    parser.add_argument("dataset", help="Path to dataset_train.csv")
    parser.add_argument(
        "--sgd",
        action="store_true",
        help="Use stochastic gradient descent (bonus).",
    )
    parser.add_argument(
        "--minibatch",
        action="store_true",
        help="Use mini-batch gradient descent (bonus).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MINIBATCH_SIZE,
        help="Mini-batch size (used with --minibatch).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sgd and args.minibatch:
        print("Error: choose only one bonus flag (--sgd or --minibatch).")
        return

    dataset = load(args.dataset)
    if dataset is None:
        return

    # Prepare data
    try:
        x, y, means, stds = prepare_data(dataset)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Train the models
    models = train_models(
        x,
        y,
        use_sgd=args.sgd,
        use_minibatch=args.minibatch,
        minibatch_size=args.batch_size,
    )

    # SAVE JSON
    os.makedirs("output", exist_ok=True)
    save_model("output/weights.json", models, means, stds)
    print("Model trained successfully.")
    print("Weights saved to output/weights.json")


if __name__ == "__main__":
    main()