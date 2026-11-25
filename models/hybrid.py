"""
models/hybrid.py - Weighted Hybrid Fusion Engine.
Integrates Neural Collaborative Filtering (NCF), Biased SVD, and Content-Based Filtering.
Features dynamic weight calibration, cold-start fallback routing, and explanation breakdown.
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from models.content import ContentRecommender
from models.svd import SVDRecommender
from models.ncf import NCFRecommender


def _load_hybrid_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load hybrid configuration block from config.yaml if present."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            return cfg.get("models", {}).get("hybrid", {})
    return {}


class HybridRecommender:
    """
    Weighted Hybrid Recommendation Engine.
    Fuses predictions from NCF, SVD, and Content-based models with cold-start routing.
    """

    DEFAULT_WEIGHTS = {
        "ncf": 0.45,
        "svd": 0.35,
        "content": 0.20
    }
    DEFAULT_COLD_START_ALPHA = 0.5

    def __init__(
        self,
        content_model: ContentRecommender,
        svd_model: SVDRecommender,
        ncf_model: NCFRecommender,
        default_weights: Optional[Dict[str, float]] = None,
        cold_start_alpha: Optional[float] = None,
        config_path: str = "config.yaml"
    ):
        hybrid_cfg = _load_hybrid_config(config_path)
        self.content_model = content_model
        self.svd_model = svd_model
        self.ncf_model = ncf_model
        self.default_weights = default_weights or hybrid_cfg.get("default_weights", self.DEFAULT_WEIGHTS.copy())
        cfg_alpha = hybrid_cfg.get("cold_start_alpha", self.DEFAULT_COLD_START_ALPHA)
        self.cold_start_alpha = float(cold_start_alpha if cold_start_alpha is not None else cfg_alpha)
        self.movies_df = content_model.movies_df
        self.num_items = content_model.num_items

    def recommend(
        self,
        u_idx: Optional[int] = None,
        user_ratings: Optional[Dict[int, float]] = None,
        top_k: int = 10,
        exclude_rated: bool = True,
        weights: Optional[Dict[str, float]] = None,
        candidate_item_indices: Optional[np.ndarray] = None,
        cold_start_alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate Top-K recommendations for a user.

        Args:
            u_idx: Contiguous user index in training matrix, or None for cold-start users.
            user_ratings: Dict mapping {item_idx: rating} for current user session.
            top_k: Number of recommendations to return.
            exclude_rated: Whether to filter out already rated movies.
            weights: Dict specifying {'ncf': float, 'svd': float, 'content': float}.
            candidate_item_indices: Optional subset of candidate item indices.
            cold_start_alpha: Optional weight alpha in [0, 1] for blended cold start:
                              Score = alpha * CosineSim(content) + (1 - alpha) * ItemBias(svd).
                              alpha=1.0 gives pure content fallback, alpha=0.0 gives pure popularity prior.

        Returns:
            List of recommendation dicts sorted by final hybrid score.
        """
        active_weights = (weights or self.default_weights).copy()

        # Normalize weights to sum to 1.0
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}
        else:
            active_weights = self.DEFAULT_WEIGHTS.copy()

        user_ratings = user_ratings or {}
        rated_indices = list(user_ratings.keys())
        rated_scores = list(user_ratings.values())

        # Determine Cold-Start Condition
        is_user_cold_start = (u_idx is None or u_idx < 0 or u_idx >= self.svd_model.num_users)

        if is_user_cold_start:
            alpha = cold_start_alpha if cold_start_alpha is not None else self.cold_start_alpha
            alpha = max(0.0, min(1.0, float(alpha)))
            if alpha >= 1.0:
                effective_weights = {"ncf": 0.0, "svd": 0.0, "content": 1.0}
                routing_mode = "COLD_START_CONTENT"
            elif alpha <= 0.0:
                effective_weights = {"ncf": 0.0, "svd": 1.0, "content": 0.0}
                routing_mode = "COLD_START_POPULARITY"
            else:
                effective_weights = {"ncf": 0.0, "svd": 1.0 - alpha, "content": alpha}
                routing_mode = f"COLD_START_BLENDED (alpha={alpha:.2f})"
        else:
            effective_weights = active_weights
            routing_mode = "HYBRID_FUSION"

        if candidate_item_indices is None:
            candidate_item_indices = np.arange(self.num_items)

        # 1. Content Scores (in [0, 1])
        if effective_weights["content"] > 0:
            content_scores = self.content_model.predict_user_scores(
                rated_item_indices=rated_indices,
                ratings=rated_scores,
                candidate_indices=candidate_item_indices
            )
        else:
            content_scores = np.zeros(len(candidate_item_indices), dtype=np.float32)

        # 2. SVD Scores (predicted rating in [1, 5] -> normalized to [0, 1])
        if effective_weights["svd"] > 0:
            if is_user_cold_start:
                # Use global item bias / popularity prior for unrated users: r_hat = mu + b_i
                svd_preds = self.svd_model.predict(u_idx=-1, item_indices=candidate_item_indices)
            else:
                svd_preds = self.svd_model.predict(u_idx=u_idx, item_indices=candidate_item_indices)
            svd_scores = (svd_preds - 1.0) / 4.0
        else:
            svd_preds = np.full(len(candidate_item_indices), 3.0, dtype=np.float32)
            svd_scores = np.full(len(candidate_item_indices), 0.5, dtype=np.float32)

        # 3. NCF Scores (predicted rating in [1, 5] -> normalized to [0, 1])
        if effective_weights["ncf"] > 0 and not is_user_cold_start:
            ncf_preds = self.ncf_model.predict(u_idx=u_idx, item_indices=candidate_item_indices)
            ncf_scores = (ncf_preds - 1.0) / 4.0
        else:
            ncf_preds = np.full(len(candidate_item_indices), 3.0, dtype=np.float32)
            ncf_scores = np.full(len(candidate_item_indices), 0.5, dtype=np.float32)

        # 4. Weighted Linear Combination
        hybrid_scores = (
            effective_weights["ncf"] * ncf_scores +
            effective_weights["svd"] * svd_scores +
            effective_weights["content"] * content_scores
        )

        # Map back to calibrated [1.0, 5.0] predicted rating
        predicted_ratings = 1.0 + 4.0 * hybrid_scores

        # Rank candidates
        sorted_order = np.argsort(hybrid_scores)[::-1]

        recommendations = []
        rated_set = set(rated_indices)

        for pos in sorted_order:
            item_idx = int(candidate_item_indices[pos])
            if exclude_rated and item_idx in rated_set:
                continue

            movie_row = self.movies_df.iloc[item_idx]
            recommendations.append({
                "item_idx": item_idx,
                "raw_item_id": int(movie_row["raw_item_id"]),
                "title": movie_row["title"],
                "genres": movie_row["genres"],
                "plot": movie_row["plot"],
                "poster_url": movie_row["poster_url"],
                "predicted_rating": round(float(predicted_ratings[pos]), 3),
                "hybrid_score": round(float(hybrid_scores[pos]), 4),
                "routing_mode": routing_mode,
                "component_scores": {
                    "ncf_rating": round(float(ncf_preds[pos]), 3),
                    "svd_rating": round(float(svd_preds[pos]), 3),
                    "content_score": round(float(content_scores[pos]), 4)
                },
                "active_weights": effective_weights
            })

            if len(recommendations) >= top_k:
                break

        return recommendations
