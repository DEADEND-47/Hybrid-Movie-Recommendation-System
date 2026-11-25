"""
tests/test_hybrid.py - Unit tests for Hybrid Recommender fusion mathematics and routing.
Validates weight normalization, weighted linear combination logic, and cold-start fallback.
"""

import pytest
import numpy as np
import pandas as pd
from models.content import ContentRecommender
from models.svd import SVDRecommender
from models.ncf import NCFRecommender
from models.hybrid import HybridRecommender


@pytest.fixture(scope="module")
def hybrid_system():
    """Load system models for testing hybrid logic."""
    content = ContentRecommender.load("artifacts/content_engine.pkl")
    svd = SVDRecommender.load("artifacts/svd_model.pkl")
    ncf = NCFRecommender.load("artifacts/ncf_model.pt")
    return HybridRecommender(content, svd, ncf)


def test_weight_normalization(hybrid_system):
    """Verify that unnormalized weights (e.g., 45, 35, 20) sum to 1.0."""
    unnorm_weights = {"ncf": 45.0, "svd": 35.0, "content": 20.0}
    recs = hybrid_system.recommend(
        u_idx=0,
        user_ratings={1: 5.0},
        top_k=5,
        weights=unnorm_weights
    )

    assert len(recs) == 5
    first = recs[0]
    weights_used = first["active_weights"]
    total = sum(weights_used.values())
    assert pytest.approx(total, abs=1e-5) == 1.0
    assert pytest.approx(weights_used["ncf"], abs=1e-3) == 0.45
    assert pytest.approx(weights_used["svd"], abs=1e-3) == 0.35
    assert pytest.approx(weights_used["content"], abs=1e-3) == 0.20


def test_cold_start_routing_unknown_user(hybrid_system):
    """Verify cold-start routing respects config.yaml default alpha and explicit alpha overrides."""
    # Test default cold-start routing (reads cold_start_alpha from config.yaml = 0.5)
    recs_none = hybrid_system.recommend(
        u_idx=None,
        user_ratings={1: 5.0, 50: 4.5},
        top_k=5
    )
    assert len(recs_none) == 5
    expected_alpha = hybrid_system.cold_start_alpha
    for r in recs_none:
        assert r["routing_mode"] == f"COLD_START_BLENDED (alpha={expected_alpha:.2f})"
        assert pytest.approx(r["active_weights"]["content"], abs=1e-3) == expected_alpha
        assert pytest.approx(r["active_weights"]["svd"], abs=1e-3) == 1.0 - expected_alpha
        assert r["active_weights"]["ncf"] == 0.0

    # Test explicit pure content mode (alpha=1.0) with out-of-bounds user index
    recs_oob = hybrid_system.recommend(
        u_idx=999999,
        user_ratings={10: 4.0},
        top_k=5,
        cold_start_alpha=1.0
    )
    for r in recs_oob:
        assert r["routing_mode"] == "COLD_START_CONTENT"
        assert r["active_weights"]["content"] == 1.0
        assert r["active_weights"]["svd"] == 0.0

    # Test explicit pure popularity prior (alpha=0.0)
    recs_pop = hybrid_system.recommend(
        u_idx=None,
        user_ratings={10: 4.0},
        top_k=5,
        cold_start_alpha=0.0
    )
    for r in recs_pop:
        assert r["routing_mode"] == "COLD_START_POPULARITY"
        assert r["active_weights"]["svd"] == 1.0
        assert r["active_weights"]["content"] == 0.0


def test_hybrid_prediction_bounds(hybrid_system):
    """Verify that hybrid predicted ratings strictly stay within valid bounds [1.0, 5.0]."""
    recs = hybrid_system.recommend(
        u_idx=5,
        user_ratings={0: 5.0, 1: 1.0},
        top_k=20
    )
    for r in recs:
        assert 1.0 <= r["predicted_rating"] <= 5.0, f"Predicted rating {r['predicted_rating']} out of bounds!"
        assert 0.0 <= r["hybrid_score"] <= 1.0, f"Hybrid score {r['hybrid_score']} out of [0, 1]!"


def test_exclude_rated_items(hybrid_system):
    """Verify that rated items are filtered out of recommendations when exclude_rated=True."""
    rated_set = {10, 20, 30}
    ratings_dict = {i: 5.0 for i in rated_set}
    recs = hybrid_system.recommend(
        u_idx=0,
        user_ratings=ratings_dict,
        top_k=10,
        exclude_rated=True
    )
    rec_indices = {r["item_idx"] for r in recs}
    assert len(rec_indices & rated_set) == 0, "Rated items appeared in recommendations!"


def test_blended_cold_start_routing(hybrid_system):
    """Verify that cold_start_alpha blends content similarity and SVD item bias."""
    recs_blended = hybrid_system.recommend(
        u_idx=None,
        user_ratings={1: 5.0, 50: 4.5},
        top_k=5,
        cold_start_alpha=0.5
    )
    assert len(recs_blended) == 5
    for r in recs_blended:
        assert r["routing_mode"] == "COLD_START_BLENDED (alpha=0.50)"
        assert pytest.approx(r["active_weights"]["content"], abs=1e-3) == 0.5
        assert pytest.approx(r["active_weights"]["svd"], abs=1e-3) == 0.5
        assert pytest.approx(r["active_weights"]["ncf"], abs=1e-3) == 0.0

