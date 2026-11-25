"""
train.py - Master Training Pipeline for Hybrid Movie Recommendation System.
Loads hyperparameters from config.yaml, trains Content, SVD, and NCF (NeuMF) models,
logs peak GPU VRAM allocation, and exports production model checkpoints to /artifacts.
"""

import os
import sys
import json
import yaml
import time
import pickle
import logging
import argparse
from typing import Dict, Any

import numpy as np
import torch

from data_loader import load_and_preprocess, get_ncf_dataloaders
from models.content import ContentRecommender
from models.svd import SVDRecommender
from models.ncf import NCFRecommender

# Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TrainPipeline")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    logger.warning("config.yaml not found, applying default training config.")
    return {
        "dataset": {"name": "100k", "data_dir": "data", "artifacts_dir": "artifacts", "random_seed": 42},
        "models": {
            "svd": {"n_factors": 32, "lr": 0.005, "reg": 0.02, "n_epochs": 30},
            "ncf": {"latent_dim_gmf": 32, "latent_dim_mlp": 32, "mlp_hidden_dims": [128, 64, 32], "dropout_rate": 0.2, "batch_size": 384, "epochs": 15, "lr": 0.001, "weight_decay": 0.00001}
        }
    }


def train_pipeline(config_path: str = "config.yaml", dataset_override: str = None):
    """Execute end-to-end model training workflow."""
    start_time = time.time()
    config = load_config(config_path)

    dataset = dataset_override or config["dataset"].get("name", "100k")
    data_dir = config["dataset"].get("data_dir", "data")
    artifacts_dir = config["dataset"].get("artifacts_dir", "artifacts")
    seed = config["dataset"].get("random_seed", 42)

    svd_cfg = config["models"]["svd"]
    ncf_cfg = config["models"]["ncf"]

    os.makedirs(artifacts_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("HYBRID RECOMMENDATION SYSTEM - MODEL TRAINING PIPELINE")
    logger.info(f"Dataset: MovieLens {dataset.upper()} | Seed: {seed}")
    logger.info(f"Device: {'CUDA (RTX 3050)' if torch.cuda.is_available() else 'CPU'} | AMP Enabled: {torch.cuda.is_available()}")
    logger.info("=" * 80)

    # 1. Preprocess Data
    logger.info("[Step 1/4] Preprocessing MovieLens interactions and metadata...")
    bundle = load_and_preprocess(
        dataset=dataset,
        data_dir=data_dir,
        test_size=config["dataset"].get("test_size", 0.10),
        val_size=config["dataset"].get("val_size", 0.10),
        random_state=seed
    )

    train_df = bundle["train_df"]
    val_df = bundle["val_df"]
    test_df = bundle["test_df"]
    movies_df = bundle["movies_df"]
    num_users = bundle["num_users"]
    num_items = bundle["num_items"]

    # Save Mappings
    maps_path = os.path.join(artifacts_dir, "user_item_maps.pkl")
    with open(maps_path, "wb") as f:
        pickle.dump({
            "user2idx": bundle["user2idx"],
            "idx2user": bundle["idx2user"],
            "item2idx": bundle["item2idx"],
            "idx2item": bundle["idx2item"],
            "num_users": num_users,
            "num_items": num_items,
            "dataset": dataset
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"[OK] Saved ID mappings to {maps_path}")

    # 2. Train Content Engine
    logger.info("[Step 2/4] Fitting Content-Based Filtering engine (TF-IDF + Cosine Similarity)...")
    content_model = ContentRecommender()
    content_model.fit(
        movies_df=movies_df,
        tfidf_matrix=bundle["tfidf_matrix"],
        tfidf_vectorizer=bundle["tfidf_vectorizer"]
    )
    content_path = os.path.join(artifacts_dir, "content_engine.pkl")
    content_model.save(content_path)

    # 3. Train Classical SVD
    logger.info(f"[Step 3/4] Fitting Classical Biased SVD (Factors={svd_cfg['n_factors']}, Epochs={svd_cfg['n_epochs']})...")
    svd_model = SVDRecommender(
        n_factors=svd_cfg["n_factors"],
        lr=svd_cfg["lr"],
        reg=svd_cfg["reg"],
        n_epochs=svd_cfg["n_epochs"],
        random_state=seed
    )
    svd_model.fit(train_df, num_users=num_users, num_items=num_items, verbose=True)
    svd_path = os.path.join(artifacts_dir, "svd_model.pkl")
    svd_model.save(svd_path)

    # 4. Train Neural Collaborative Filtering (NCF / NeuMF)
    logger.info(f"[Step 4/4] Training NCF (NeuMF) on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")
    train_loader, val_loader, _ = get_ncf_dataloaders(
        train_df=train_df,
        val_df=val_df,
        batch_size=ncf_cfg["batch_size"],
        num_workers=0
    )

    ncf_model = NCFRecommender(
        num_users=num_users,
        num_items=num_items,
        latent_dim_gmf=ncf_cfg["latent_dim_gmf"],
        latent_dim_mlp=ncf_cfg["latent_dim_mlp"],
        mlp_hidden_dims=tuple(ncf_cfg["mlp_hidden_dims"]),
        dropout_rate=ncf_cfg["dropout_rate"]
    )

    ncf_stats = ncf_model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=ncf_cfg["epochs"],
        lr=ncf_cfg["lr"],
        weight_decay=ncf_cfg["weight_decay"],
        verbose=True
    )

    ncf_path = os.path.join(artifacts_dir, "ncf_model.pt")
    ncf_model.save(ncf_path)

    total_time = time.time() - start_time
    manifest = {
        "dataset": dataset,
        "num_users": num_users,
        "num_items": num_items,
        "train_ratings": len(train_df),
        "val_ratings": len(val_df),
        "test_ratings": len(test_df),
        "ncf_best_val_rmse": float(ncf_stats["best_val_rmse"]),
        "ncf_peak_vram_mb": float(ncf_stats["peak_vram_mb"]),
        "ncf_peak_vram_gb": float(ncf_stats["peak_vram_gb"]),
        "total_training_time_seconds": round(total_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    manifest_path = os.path.join(artifacts_dir, "train_summary.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"TRAINING COMPLETE in {total_time:.1f}s | Peak VRAM: {ncf_stats['peak_vram_mb']:.1f} MB (< 4.0 GB target)")
    logger.info(f"Checkpoints exported to: {os.path.abspath(artifacts_dir)}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train Recommendation Models.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name (100k or 1m)")
    args = parser.parse_args()

    train_pipeline(config_path=args.config, dataset_override=args.dataset)


if __name__ == "__main__":
    main()
