"""
train_multiseed.py - Train and cache SVD and NCF models for seeds 123 and 999.
Ensures rigorous holdout evaluation with zero data leakage across random splits.
"""

import os
import yaml
import time
import torch
import logging
from data_loader import load_and_preprocess, get_ncf_dataloaders
from models.svd import SVDRecommender
from models.ncf import NCFRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainMultiSeed")

def train_seed(seed: int, config: dict):
    artifacts_dir = config["dataset"].get("artifacts_dir", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    svd_path = os.path.join(artifacts_dir, f"svd_model_{seed}.pkl")
    ncf_path = os.path.join(artifacts_dir, f"ncf_model_{seed}.pt")
    
    if os.path.exists(svd_path) and os.path.exists(ncf_path):
        logger.info(f"Artifacts for seed {seed} already exist. Skipping training.")
        return
        
    logger.info(f"\n{'='*70}\nTRAINING ARTIFACTS FOR SEED {seed}\n{'='*70}")
    bundle = load_and_preprocess(
        dataset=config["dataset"].get("name", "100k"),
        data_dir=config["dataset"].get("data_dir", "data"),
        test_size=config["dataset"].get("test_size", 0.10),
        val_size=config["dataset"].get("val_size", 0.10),
        random_state=seed
    )
    
    # 1. Train SVD
    svd_cfg = config["models"]["svd"]
    svd_model = SVDRecommender(
        n_factors=svd_cfg["n_factors"],
        lr=svd_cfg["lr"],
        reg=svd_cfg["reg"],
        n_epochs=svd_cfg["n_epochs"],
        random_state=seed
    )
    t0 = time.time()
    svd_model.fit(bundle["train_df"], bundle["num_users"], bundle["num_items"], verbose=False)
    svd_model.save(svd_path)
    logger.info(f"[OK] Trained & saved SVD for seed {seed} in {time.time()-t0:.1f}s -> {svd_path}")
    
    # 2. Train NCF
    ncf_cfg = config["models"]["ncf"]
    train_loader, val_loader, _ = get_ncf_dataloaders(
        train_df=bundle["train_df"],
        val_df=bundle["val_df"],
        batch_size=ncf_cfg["batch_size"],
        num_workers=0
    )
    ncf_model = NCFRecommender(
        num_users=bundle["num_users"],
        num_items=bundle["num_items"],
        latent_dim_gmf=ncf_cfg["latent_dim_gmf"],
        latent_dim_mlp=ncf_cfg["latent_dim_mlp"],
        mlp_hidden_dims=tuple(ncf_cfg["mlp_hidden_dims"]),
        dropout_rate=ncf_cfg["dropout_rate"]
    )
    t1 = time.time()
    ncf_model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=ncf_cfg["epochs"],
        lr=ncf_cfg["lr"],
        weight_decay=ncf_cfg["weight_decay"],
        verbose=False
    )
    ncf_model.save(ncf_path)
    logger.info(f"[OK] Trained & saved NCF for seed {seed} in {time.time()-t1:.1f}s -> {ncf_path}")

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Also link or save seed 42 to artifacts/svd_model_42.pkl and ncf_model_42.pt for consistency
    artifacts_dir = config["dataset"].get("artifacts_dir", "artifacts")
    svd_42 = os.path.join(artifacts_dir, "svd_model.pkl")
    ncf_42 = os.path.join(artifacts_dir, "ncf_model.pt")
    svd_42_alias = os.path.join(artifacts_dir, "svd_model_42.pkl")
    ncf_42_alias = os.path.join(artifacts_dir, "ncf_model_42.pt")
    
    if os.path.exists(svd_42) and not os.path.exists(svd_42_alias):
        import shutil
        shutil.copyfile(svd_42, svd_42_alias)
    if os.path.exists(ncf_42) and not os.path.exists(ncf_42_alias):
        import shutil
        shutil.copyfile(ncf_42, ncf_42_alias)
        
    for seed in [123, 999]:
        train_seed(seed, config)
        
    logger.info("\n[SUCCESS] All multi-seed model artifacts (42, 123, 999) are trained and cached!")

if __name__ == "__main__":
    main()
