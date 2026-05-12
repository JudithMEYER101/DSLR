import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_csv import load


def sigmoid(z):
    """Apply sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def load_model(filepath):
    """Load trained weights file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_features(dataset: pd.DataFrame, model_data):
    """Prepare and normalize features using training means/stds."""

    features = model_data["features"]
    means = pd.Series(model_data["means"])
    stds = pd.Series(model_data["stds"]).replace(0, 1)

    missing = [col for col in features if col not in dataset.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x_df = dataset[features].copy()

    # Fill missing values with training means
    x_df = x_df.fillna(means)

    x_df = (x_df - means) / stds
    return x_df.to_numpy(dtype=float)


def predict_classes(x, model_data):
    """Predict house for each sample."""

    houses = model_data["houses"]
    models = model_data["models"]

    probabilities = []

    for house in houses:

        # Load weights and bias by features
        weights = np.array(models[house]["weights"], dtype=float)
        bias = float(models[house]["bias"])

        # Compute linear score
        z = np.dot(x, weights) + bias

        # Convert to probability
        probs = sigmoid(z)
        probabilities.append(probs)

    probabilities = np.array(probabilities).T
    # Select highest probability
    best_indices = np.argmax(probabilities, axis=1)
    predictions = [houses[i] for i in best_indices]

    return predictions


def save_predictions(dataset: pd.DataFrame, predictions):
    """Save predictions to houses.csv in required format."""

    output = pd.DataFrame(
        {"Hogwarts House": predictions},
        index=dataset.index,
    )
    output.index.name = "Index"
    output.to_csv("houses.csv")
    print("Predictions saved to houses.csv")


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 logreg_predict.py dataset_test.csv weights.json")
        return

    dataset = load(sys.argv[1])
    if dataset is None:
        return

    try:
        model_data = load_model(sys.argv[2])
        x = prepare_features(dataset, model_data)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return

    predictions = predict_classes(x, model_data)
    save_predictions(dataset, predictions)


if __name__ == "__main__":
    main()