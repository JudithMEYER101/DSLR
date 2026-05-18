import numpy as np


# OVERFLOW MANAGEMENT

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# STOCHASTIC

def train_one_vs_rest_sgd(x, y_binary, learning_rate=0.01, epochs=60):
    # x IS normalize matrix course scores from dataset_train WHERE each row is one student, each column is one selected feature
    # y_binary IS labels for this one-vs-rest model WHERE 1 if the student is from current house, else 0

		# NEUTRAL MODEL
    m, n = x.shape
    weights = np.zeros(n)
    bias = 0.0

		# FOR EACH EPOCH
    for _ in range(epochs):
        indices = np.random.permutation(m)
        for i in indices:
            # STOCHASTIC UPDATE (1/examples)
            xi = x[i]
            yi = y_binary[i]

            # STATS STORE
            z = np.dot(xi, weights) + bias
            y_hat = sigmoid(z)
            # Prediction error (direction and magnitude of correction).
            error = y_hat - yi

            # ERROR MANAGEMENT
            weights -= learning_rate * error * xi
            bias -= learning_rate * error

    return weights, bias
