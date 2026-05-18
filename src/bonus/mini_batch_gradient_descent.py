import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def train_one_vs_rest_minibatch(x, y_binary, learning_rate=0.05, epochs=80, batch_size=32):

		# NEUTRAL
    m, n = x.shape
    weights = np.zeros(n)
    bias = 0.0

    for _ in range(epochs):
        # SHUFFLE
        indices = np.random.permutation(m)
        for start in range(0, m, batch_size):
            # DEFAULT BATCH == 32
            batch_idx = indices[start:start + batch_size]
            xb = x[batch_idx]
            yb = y_binary[batch_idx]

            # STATS
            z = np.dot(xb, weights) + bias
            y_hat = sigmoid(z)

            # ERROR
            error = y_hat - yb
            size = len(batch_idx)
            if size == 0:
                continue

            # AVERAGE GRADIENT
            dw = np.dot(xb.T, error) / size
            db = np.sum(error) / size

            # UPDATE PARAMS
            weights -= learning_rate * dw
            bias -= learning_rate * db

    return weights, bias
