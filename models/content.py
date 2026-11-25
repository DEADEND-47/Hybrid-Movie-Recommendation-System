"""
models/content.py - Content-based recommendation engine using TF-IDF and Cosine Similarity.
Builds similarity profiles across movie synopsis, genres, and metadata.
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class ContentRecommender:
    """TF-IDF and Cosine Similarity content-based recommendation engine."""

    def __init__(self):
        self.movies_df: Optional[pd.DataFrame] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.num_items: int = 0

    def fit(
        self,
        movies_df: pd.DataFrame,
        tfidf_matrix: Optional[Any] = None,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None
    ) -> "ContentRecommender":
        """
        Fit the content engine using movies metadata and TF-IDF representations.
        """
        self.movies_df = movies_df.copy().sort_values("item_idx").reset_index(drop=True)
        self.num_items = len(self.movies_df)

        if tfidf_matrix is not None and tfidf_vectorizer is not None:
            self.tfidf_matrix = tfidf_matrix
            self.tfidf_vectorizer = tfidf_vectorizer
        else:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df["content_soup"])

        # Compute full cosine similarity matrix (dense M x M float32)
        print(f"[*] Computing ({self.num_items}x{self.num_items}) Cosine Similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix).astype(np.float32)
        print(f"[OK] Content engine fitted successfully. Similarity matrix shape: {self.similarity_matrix.shape}")
        return self

    def predict_user_scores(
        self,
        rated_item_indices: List[int],
        ratings: List[float],
        candidate_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute predicted content preference scores for candidate movies given a user's ratings.
        Returns normalized scores in [0, 1].
        """
        if candidate_indices is None:
            candidate_indices = np.arange(self.num_items)

        if len(rated_item_indices) == 0:
            # Baseline neutral score if no rating history provided
            return np.full(len(candidate_indices), 0.5, dtype=np.float32)

        valid_pairs = [
            (idx, r) for idx, r in zip(rated_item_indices, ratings)
            if 0 <= idx < self.num_items
        ]

        if not valid_pairs:
            return np.full(len(candidate_indices), 0.5, dtype=np.float32)

        rated_idxs = [p[0] for p in valid_pairs]
        # Center ratings around 3.0 so positive ratings boost and negative ratings penalize
        weights = np.array([p[1] - 3.0 for p in valid_pairs], dtype=np.float32)

        # Shape: (len(rated_idxs), len(candidate_indices))
        sim_sub = self.similarity_matrix[rated_idxs, :][:, candidate_indices]

        # Weighted dot product: sum_i (w_i * sim(i, j)) / (sum_i sim(i, j) + eps)
        pos_sim = np.maximum(sim_sub, 0.0)
        numerator = np.dot(weights, pos_sim)
        denominator = np.sum(pos_sim, axis=0) + 1e-6

        raw_scores = numerator / denominator

        # Map centered score from approximately [-2, +2] to [0, 1]
        normalized_scores = 1.0 / (1.0 + np.exp(-1.5 * raw_scores))
        return normalized_scores.astype(np.float32)

    def recommend_similar(
        self,
        item_idx: int,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find top-K most similar movies to a given movie item index (for item-item & cold-start).
        """
        if item_idx < 0 or item_idx >= self.num_items:
            return []

        sim_scores = self.similarity_matrix[item_idx].copy()
        if exclude_self:
            sim_scores[item_idx] = -1.0

        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            row = self.movies_df.iloc[idx]
            results.append({
                "item_idx": int(idx),
                "raw_item_id": int(row["raw_item_id"]),
                "title": row["title"],
                "genres": row["genres"],
                "similarity_score": float(sim_scores[idx]),
                "poster_url": row["poster_url"],
                "plot": row["plot"]
            })
        return results

    def save(self, filepath: str) -> None:
        """Serialize content engine to disk."""
        data = {
            "movies_df": self.movies_df,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "similarity_matrix": self.similarity_matrix,
            "num_items": self.num_items
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OK] Content engine saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ContentRecommender":
        """Load serialized content engine from disk."""
        engine = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        engine.movies_df = data["movies_df"]
        engine.tfidf_vectorizer = data["tfidf_vectorizer"]
        engine.tfidf_matrix = data["tfidf_matrix"]
        engine.similarity_matrix = data["similarity_matrix"]
        engine.num_items = data["num_items"]
        print(f"[OK] Content engine loaded from {filepath} ({engine.num_items} items)")
        return engine
