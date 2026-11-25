"""
api.py - FastAPI Production Serving Endpoint for Hybrid Movie Recommendation System.
Provides RESTful inference APIs:
  - GET  /health              -> System health check and model loading status
  - POST /recommend           -> Personalized Top-K recommendations with component score breakdown
  - POST /cold-start/simulate -> Real-time semantic similarity retrieval for brand-new unrated movies
"""

import os
import yaml
import pickle
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from models.content import ContentRecommender
from models.svd import SVDRecommender
from models.ncf import NCFRecommender
from models.hybrid import HybridRecommender

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RecommenderAPI")

# Global System State
system_state: Dict[str, Any] = {}


def _load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load centralized configuration from config.yaml."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load system models, config defaults, and artifacts on startup."""
    cfg = _load_config("config.yaml")
    artifacts_dir = cfg.get("dataset", {}).get("artifacts_dir", "artifacts")
    hybrid_cfg = cfg.get("models", {}).get("hybrid", {})
    default_weights = hybrid_cfg.get("default_weights", {"ncf": 0.45, "svd": 0.35, "content": 0.20})
    cold_start_alpha = float(hybrid_cfg.get("cold_start_alpha", 0.5))

    maps_path = os.path.join(artifacts_dir, "user_item_maps.pkl")
    content_path = os.path.join(artifacts_dir, "content_engine.pkl")
    svd_path = os.path.join(artifacts_dir, "svd_model.pkl")
    ncf_path = os.path.join(artifacts_dir, "ncf_model.pt")

    if not all(os.path.exists(p) for p in [maps_path, content_path, svd_path, ncf_path]):
        raise RuntimeError("Model artifacts missing in /artifacts. Please run train.py first.")

    logger.info("[*] Loading production recommendation artifacts (seed-42) and config.yaml defaults...")
    with open(maps_path, "rb") as f:
        maps = pickle.load(f)

    content_model = ContentRecommender.load(content_path)
    svd_model = SVDRecommender.load(svd_path)
    ncf_model = NCFRecommender.load(ncf_path)
    hybrid_model = HybridRecommender(
        content_model,
        svd_model,
        ncf_model,
        default_weights=default_weights,
        cold_start_alpha=cold_start_alpha,
        config_path="config.yaml"
    )

    system_state["config"] = cfg
    system_state["maps"] = maps
    system_state["content"] = content_model
    system_state["svd"] = svd_model
    system_state["ncf"] = ncf_model
    system_state["hybrid"] = hybrid_model
    system_state["movies_df"] = content_model.movies_df
    logger.info(f"[OK] All models loaded (cold_start_alpha={hybrid_model.cold_start_alpha}, default_weights={hybrid_model.default_weights}).")

    yield
    system_state.clear()


app = FastAPI(
    title="CineMatch AI - Recommendation API",
    description="Production-grade RESTful API serving PyTorch NCF + SVD + Content Hybrid Recommender.",
    version="1.0.0",
    lifespan=lifespan
)


# --- Request & Response Schemas ---

class UserRatingInput(BaseModel):
    movie_id: int = Field(..., description="Raw MovieLens movie ID or indexed ID")
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating given by user (1.0 - 5.0)")


class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="Optional raw user ID (None triggers dynamic cold-start routing)")
    ratings: Optional[List[UserRatingInput]] = Field(default=[], description="List of user ratings to seed interactive profile")
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom linear fusion weights for components (defaults to config.yaml)"
    )
    cold_start_alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional cold-start blend alpha (defaults to config.yaml)"
    )
    exclude_rated: bool = Field(True, description="Filter out movies the user has already rated")


class MovieRecommendation(BaseModel):
    item_idx: int
    raw_item_id: int
    title: str
    genres: str
    predicted_rating: float
    hybrid_score: float
    routing_mode: str
    poster_url: str
    plot: str
    component_scores: Dict[str, float]


class RecommendResponse(BaseModel):
    user_id: Optional[int]
    recommendations_count: int
    routing_mode: str
    recommendations: List[MovieRecommendation]


class ColdStartInput(BaseModel):
    title: str = Field(..., description="Title of brand-new, unseen movie")
    genres: List[str] = Field(..., description="List of genre labels")
    plot: str = Field(..., description="Synopsis or plot overview text")
    top_k: int = Field(5, ge=1, le=20, description="Number of nearest neighbors to retrieve")


class ColdStartResponse(BaseModel):
    query_title: str
    similar_movies: List[Dict[str, Any]]


# --- API Routes ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint exposing system hardware and loaded model stats."""
    if "hybrid" not in system_state:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models not loaded yet.")

    return {
        "status": "healthy",
        "service": "CineMatch AI Hybrid Recommendation Service",
        "device": "CUDA (RTX 3050)" if torch.cuda.is_available() else "CPU",
        "indexed_movies": system_state["hybrid"].num_items,
        "indexed_users": system_state["maps"]["num_users"],
        "default_weights": system_state["hybrid"].default_weights,
        "cold_start_alpha": system_state["hybrid"].cold_start_alpha
    }


@app.post("/recommend", response_model=RecommendResponse, status_code=status.HTTP_200_OK)
async def recommend_movies(payload: RecommendRequest):
    """
    Generate Top-K movie recommendations using Weighted Hybrid Fusion.
    Automatically handles cold-start users via config-driven blended content + item-bias fallback.
    """
    if "hybrid" not in system_state:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    hybrid: HybridRecommender = system_state["hybrid"]
    maps = system_state["maps"]

    # Resolve User Index
    u_idx = None
    if payload.user_id is not None:
        u_idx = maps["user2idx"].get(payload.user_id, None)

    # Resolve Movie IDs in ratings
    ratings_dict = {}
    for r in payload.ratings:
        if r.movie_id in maps["item2idx"]:
            idx = maps["item2idx"][r.movie_id]
            ratings_dict[idx] = r.rating
        elif 0 <= r.movie_id < hybrid.num_items:
            ratings_dict[r.movie_id] = r.rating

    recs = hybrid.recommend(
        u_idx=u_idx,
        user_ratings=ratings_dict,
        top_k=payload.top_k,
        exclude_rated=payload.exclude_rated,
        weights=payload.weights,
        cold_start_alpha=payload.cold_start_alpha
    )

    routing_mode = recs[0]["routing_mode"] if recs else "UNKNOWN"

    return RecommendResponse(
        user_id=payload.user_id,
        recommendations_count=len(recs),
        routing_mode=routing_mode,
        recommendations=recs
    )


@app.post("/cold-start/simulate", response_model=ColdStartResponse, status_code=status.HTTP_200_OK)
async def simulate_cold_start_movie(payload: ColdStartInput):
    """
    Compute real-time semantic nearest-neighbors for an unrated, brand-new movie.
    Resolves the cold-start problem by vectorizing text into the fitted TF-IDF matrix.
    """
    if "content" not in system_state:
        raise HTTPException(status_code=503, detail="Content model not loaded.")

    content: ContentRecommender = system_state["content"]
    movies_df = system_state["movies_df"]

    # Build content soup on the fly
    soup = f"{payload.title} {' '.join(payload.genres)} {payload.plot}"
    query_vec = content.tfidf_vectorizer.transform([soup])

    sims = cosine_similarity(query_vec, content.tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:payload.top_k]

    results = []
    for idx in top_indices:
        row = movies_df.iloc[idx]
        results.append({
            "item_idx": int(idx),
            "raw_item_id": int(row["raw_item_id"]),
            "title": row["title"],
            "genres": row["genres"],
            "similarity_score": round(float(sims[idx]), 4),
            "poster_url": row["poster_url"],
            "plot": row["plot"]
        })

    return ColdStartResponse(
        query_title=payload.title,
        similar_movies=results
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
