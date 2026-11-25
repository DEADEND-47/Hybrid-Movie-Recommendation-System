"""
data_loader.py - Preprocessing, vectorization, dataset splitting, and PyTorch DataLoaders.
Handles MovieLens 100K and 1M, builds contiguous 0-indexed mappings,
computes TF-IDF content soups, and creates 80/10/10 train/val/test splits.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


class MovieRatingDataset(Dataset):
    """PyTorch Dataset for explicit collaborative filtering rating prediction."""

    def __init__(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]


def load_raw_interactions(dataset: str = "100k", data_dir: str = "data") -> pd.DataFrame:
    """Load raw interaction ratings dataframe from MovieLens files."""
    ml_dir = os.path.join(data_dir, f"ml-{dataset}")
    if dataset == "100k":
        path = os.path.join(ml_dir, "u.data")
        df = pd.read_csv(
            path,
            sep="\t",
            names=["raw_user_id", "raw_item_id", "rating", "timestamp"],
            engine="python"
        )
    elif dataset == "1m":
        path = os.path.join(ml_dir, "ratings.dat")
        df = pd.read_csv(
            path,
            sep="::",
            names=["raw_user_id", "raw_item_id", "rating", "timestamp"],
            engine="python"
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}. Choose '100k' or '1m'.")

    df["rating"] = df["rating"].astype(np.float32)
    return df


def load_metadata_cache(dataset: str = "100k", data_dir: str = "data") -> Dict[str, Dict[str, Any]]:
    """Load enriched metadata cache json."""
    cache_path = os.path.join(data_dir, "cache", f"metadata_cache_{dataset}.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Metadata cache {cache_path} not found. Please run 'python download_data.py --dataset {dataset}' first."
        )
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mappings(ratings_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Construct contiguous 0-indexed bidirectional mappings for users and items."""
    unique_users = sorted(ratings_df["raw_user_id"].unique())
    unique_items = sorted(ratings_df["raw_item_id"].unique())

    user2idx = {u: i for i, u in enumerate(unique_users)}
    idx2user = {i: u for i, u in enumerate(unique_users)}
    item2idx = {m: i for i, m in enumerate(unique_items)}
    idx2item = {i: m for i, m in enumerate(unique_items)}

    return user2idx, idx2user, item2idx, idx2item


def prepare_movie_features(
    item2idx: Dict[int, int],
    metadata_cache: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Prepare structured DataFrame of movie metadata and content soup aligned with item_idx."""
    rows = []
    for raw_id, idx in item2idx.items():
        m_info = metadata_cache.get(str(raw_id), {})
        title = m_info.get("title", f"Movie #{raw_id}")
        genres = m_info.get("genres", ["Drama"])
        plot = m_info.get("plot", "")
        poster_url = m_info.get("poster_url", "")

        genre_str = " ".join([g.replace("-", "").replace("'", "") for g in genres])
        content_soup = f"{title} {genre_str} {plot}"

        rows.append({
            "item_idx": idx,
            "raw_item_id": raw_id,
            "title": title,
            "genres": ", ".join(genres),
            "plot": plot,
            "poster_url": poster_url,
            "content_soup": content_soup
        })

    movies_df = pd.DataFrame(rows).sort_values("item_idx").reset_index(drop=True)
    return movies_df


def split_data(
    ratings_df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform train/val/test split (80/10/10 by default) reproducibly."""
    # First split off test set (10%)
    train_val_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Then split val from train (val_size / (1.0 - test_size) = 0.1 / 0.9 ~ 0.1111)
    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        shuffle=True
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def load_and_preprocess(
    dataset: str = "100k",
    data_dir: str = "data",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    End-to-end data pipeline:
    1. Loads interactions and metadata cache.
    2. Builds contiguous 0-indexed ID mappings.
    3. Builds movie metadata DataFrame with content soups.
    4. Splits into train (80%), validation (10%), and test (10%).
    5. Fits TF-IDF vectorizer on movie content soups.
    """
    raw_ratings = load_raw_interactions(dataset=dataset, data_dir=data_dir)
    metadata_cache = load_metadata_cache(dataset=dataset, data_dir=data_dir)

    user2idx, idx2user, item2idx, idx2item = build_mappings(raw_ratings)

    # Map raw IDs to contiguous indexes
    raw_ratings["user_idx"] = raw_ratings["raw_user_id"].map(user2idx)
    raw_ratings["item_idx"] = raw_ratings["raw_item_id"].map(item2idx)

    clean_ratings = raw_ratings[["user_idx", "item_idx", "rating", "timestamp", "raw_user_id", "raw_item_id"]]

    # Split data (80/10/10)
    train_df, val_df, test_df = split_data(
        clean_ratings,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    # Prepare movie features aligned to item_idx
    movies_df = prepare_movie_features(item2idx, metadata_cache)

    # Fit TF-IDF Vectorizer on content soup
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(movies_df["content_soup"])

    print(f"[OK] Preprocessing summary ({dataset.upper()}):")
    print(f"     Users: {len(user2idx)}, Movies: {len(item2idx)}, Total Ratings: {len(clean_ratings)}")
    print(f"     Train: {len(train_df)} ({len(train_df)/len(clean_ratings):.1%}), "
          f"Val: {len(val_df)} ({len(val_df)/len(clean_ratings):.1%}), "
          f"Test: {len(test_df)} ({len(test_df)/len(clean_ratings):.1%})")
    print(f"     TF-IDF Matrix Shape: {tfidf_matrix.shape}")

    return {
        "dataset_name": dataset,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "movies_df": movies_df,
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "tfidf_vectorizer": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "num_users": len(user2idx),
        "num_items": len(item2idx)
    }


def get_ncf_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    batch_size: int = 384,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create PyTorch DataLoaders for NCF training, validation, and testing."""
    train_dataset = MovieRatingDataset(
        train_df["user_idx"].values,
        train_df["item_idx"].values,
        train_df["rating"].values
    )
    val_dataset = MovieRatingDataset(
        val_df["user_idx"].values,
        val_df["item_idx"].values,
        val_df["rating"].values
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = None
    if test_df is not None:
        test_dataset = MovieRatingDataset(
            test_df["user_idx"].values,
            test_df["item_idx"].values,
            test_df["rating"].values
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_bundle = load_and_preprocess(dataset="100k")
    train_loader, val_loader, test_loader = get_ncf_dataloaders(
        data_bundle["train_df"],
        data_bundle["val_df"],
        data_bundle["test_df"],
        batch_size=384
    )
    sample_users, sample_items, sample_ratings = next(iter(train_loader))
    print(f"[OK] Verified DataLoader batch: Users: {sample_users.shape}, Items: {sample_items.shape}, Ratings: {sample_ratings.shape}")
