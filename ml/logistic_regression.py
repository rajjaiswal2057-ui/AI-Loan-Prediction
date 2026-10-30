"""Logistic Regression + Batch Gradient Descent implemented from scratch with NumPy.

Flow:  z = XW + b  ->  p = sigmoid(z)  ->  BCE loss (+ L2)  ->  gradients  ->  update W, b
"""
import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)          # numerical stability
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, epochs=1000, l2=0.0, threshold=0.5):
        self.lr, self.epochs, self.l2, self.threshold = learning_rate, epochs, l2, threshold
        self.weights, self.bias = None, 0.0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _loss(self, X, y):
        p = sigmoid(X @ self.weights + self.bias)
        # Reported loss is pure BCE so train/validation curves are comparable.
        # (The optimiser minimises BCE + (lambda/2)*||W||^2; the L2 term enters via the gradient.)
        return binary_cross_entropy(y, p)

    def fit(self, X, y, X_val=None, y_val=None):
        n, d = X.shape
        self.weights, self.bias = np.zeros(d), 0.0
        self.history = {k: [] for k in self.history}
        for _ in range(self.epochs):
            p = sigmoid(X @ self.weights + self.bias)          # forward pass
            error = p - y                                      # dL/dz
            grad_w = (X.T @ error) / n + self.l2 * self.weights  # gradient of loss wrt W
            grad_b = float(np.mean(error))                     # gradient wrt bias
            self.weights -= self.lr * grad_w                   # gradient descent update
            self.bias -= self.lr * grad_b
            self.history["train_loss"].append(self._loss(X, y))
            self.history["train_acc"].append(float(np.mean(self.predict(X) == y)))
            if X_val is not None:
                self.history["val_loss"].append(self._loss(X_val, y_val))
                self.history["val_acc"].append(float(np.mean(self.predict(X_val) == y_val)))
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def to_dict(self):
        return {"weights": self.weights.tolist(), "bias": self.bias, "lr": self.lr,
                "epochs": self.epochs, "l2": self.l2, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["lr"], d["epochs"], d["l2"], d["threshold"])
        m.weights, m.bias = np.array(d["weights"]), d["bias"]
        return m
