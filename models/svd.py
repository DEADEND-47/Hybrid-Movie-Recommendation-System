"""
models/svd.py - Classical Matrix Factorization using Regularized Biased SVD.
Implements r_hat_ui = mu + b_u + b_i + P_u . Q_i^T with SGD optimization,
regularization, and clipping to the valid rating domain [1.0, 5.0].
"""

import pickle
import numpy as np
import pandas as pd
from typing import Optional


class SVDRecommender:
    """
    Biased Matrix Factorization model (Funk SVD):
    Predicts: r_hat(u, i) = mu + b_u + b_i + P_u @ Q_i.T
    """

    def __init__(
        self,
        n_factors: int = 32,
        lr: float = 0.005,
        reg: float = 0.02,
        n_epochs: int = 30,
        random_state: int = 42
    ):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.mu: float = 0.0
        self.b_u: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None  # (n_users, n_factors)
        self.Q: Optional[np.ndarray] = None  # (n_items, n_factors)

        self.num_users: int = 0
        self.num_items: int = 0

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
        verbose: bool = True
    ) -> "SVDRecommender":
        """
        Train latent factors and user/item bias terms using regularized SGD.
        """
        self.num_users = num_users if num_users is not None else int(train_df["user_idx"].max()) + 1
        self.num_items = num_items if num_items is not None else int(train_df["item_idx"].max()) + 1

        users = train_df["user_idx"].values.astype(np.int32)
        items = train_df["item_idx"].values.astype(np.int32)
        ratings = train_df["rating"].values.astype(np.float32)
        n_samples = len(ratings)

        rng = np.random.default_rng(self.random_state)

        self.mu = float(ratings.mean())
        self.b_u = np.zeros(self.num_users, dtype=np.float32)
        self.b_i = np.zeros(self.num_items, dtype=np.float32)

        # Small normal initialization
        self.P = rng.normal(0, 0.05, size=(self.num_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.05, size=(self.num_items, self.n_factors)).astype(np.float32)

        if verbose:
            print(f"[*] Training SVD (Factors={self.n_factors}, Epochs={self.n_epochs}, LR={self.lr}, Reg={self.reg})...")

        for epoch in range(1, self.n_epochs + 1):
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            loss = 0.0
            for idx in indices:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]

                pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred
                loss += err * err

                # SGD updates with L2 regularization
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                p_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_old - self.reg * self.Q[i])

            rmse = np.sqrt(loss / n_samples)
            if verbose and (epoch % 10 == 0 or epoch == self.n_epochs):
                print(f"    Epoch {epoch:02d}/{self.n_epochs:02d} - Train RMSE: {rmse:.4f}")

        if verbose:
            print(f"[OK] SVD training complete. Final Train RMSE: {rmse:.4f}")
        return self

    def predict_pair(self, u_idx: int, i_idx: int) -> float:
        """Predict rating for a single (user, item) pair, clipped to [1.0, 5.0]."""
        u_valid = (0 <= u_idx < self.num_users)
        i_valid = (0 <= i_idx < self.num_items)

        if u_valid and i_valid:
            pred = self.mu + self.b_u[u_idx] + self.b_i[i_idx] + float(np.dot(self.P[u_idx], self.Q[i_idx]))
        elif u_valid:
            pred = self.mu + self.b_u[u_idx]
        elif i_valid:
            pred = self.mu + self.b_i[i_idx]
        else:
            pred = self.mu

        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, u_idx: int, item_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict ratings for an array of items for user u_idx.
        Returns array of ratings in [1.0, 5.0].
        """
        if item_indices is None:
            item_indices = np.arange(self.num_items)

        valid_u = (0 <= u_idx < self.num_users)

        if not valid_u:
            # Fallback to global mean + item biases if known
            preds = np.zeros(len(item_indices), dtype=np.float32)
            for k, i_idx in enumerate(item_indices):
                if 0 <= i_idx < self.num_items:
                    preds[k] = self.mu + self.b_i[i_idx]
                else:
                    preds[k] = self.mu
            return np.clip(preds, 1.0, 5.0)

        user_vec = self.P[u_idx]  # (K,)
        user_bias = self.b_u[u_idx]

        preds = np.zeros(len(item_indices), dtype=np.float32)
        for k, i_idx in enumerate(item_indices):
            if 0 <= i_idx < self.num_items:
                preds[k] = self.mu + user_bias + self.b_i[i_idx] + np.dot(user_vec, self.Q[i_idx])
            else:
                preds[k] = self.mu + user_bias

        return np.clip(preds, 1.0, 5.0)

    def save(self, filepath: str) -> None:
        """Serialize SVD model to disk."""
        data = {
            "n_factors": self.n_factors,
            "mu": self.mu,
            "b_u": self.b_u,
            "b_i": self.b_i,
            "P": self.P,
            "Q": self.Q,
            "num_users": self.num_users,
            "num_items": self.num_items
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OK] SVD model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SVDRecommender":
        """Load serialized SVD model from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        model = cls(n_factors=data["n_factors"])
        model.mu = data["mu"]
        model.b_u = data["b_u"]
        model.b_i = data["b_i"]
        model.P = data["P"]
        model.Q = data["Q"]
        model.num_users = data["num_users"]
        model.num_items = data["num_items"]
        print(f"[OK] SVD model loaded from {filepath} ({model.num_users} users, {model.num_items} items)")
        return model
