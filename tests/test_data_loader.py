"""
tests/test_data_loader.py - Unit tests for data loading and preprocessing.
Validates split ratios, lack of data leakage, contiguous mappings, and DataLoader batches.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from data_loader import load_and_preprocess, split_data, build_mappings, get_ncf_dataloaders


@pytest.fixture(scope="module")
def data_bundle():
    """Load cached preprocessed data bundle once for testing."""
    return load_and_preprocess(dataset="100k", test_size=0.1, val_size=0.1, random_state=42)


def test_split_proportions(data_bundle):
    """Verify that split proportions adhere strictly to 80% train, 10% val, 10% test."""
    train_len = len(data_bundle["train_df"])
    val_len = len(data_bundle["val_df"])
    test_len = len(data_bundle["test_df"])
    total = train_len + val_len + test_len

    assert total == 100000, f"Expected 100,000 ratings, got {total}"
    assert train_len == 80000, f"Expected 80,000 train samples (80%), got {train_len}"
    assert val_len == 10000, f"Expected 10,000 val samples (10%), got {val_len}"
    assert test_len == 10000, f"Expected 10,000 test samples (10%), got {test_len}"


def test_no_data_leakage(data_bundle):
    """Verify that train, val, and test splits have zero mutual interaction leakage."""
    train_df = data_bundle["train_df"]
    val_df = data_bundle["val_df"]
    test_df = data_bundle["test_df"]

    # Combine user and item into a unique interaction key
    train_keys = set(zip(train_df["user_idx"], train_df["item_idx"], train_df["timestamp"]))
    val_keys = set(zip(val_df["user_idx"], val_df["item_idx"], val_df["timestamp"]))
    test_keys = set(zip(test_df["user_idx"], test_df["item_idx"], test_df["timestamp"]))

    assert len(train_keys & test_keys) == 0, "Train and Test splits leak identical interaction entries!"
    assert len(train_keys & val_keys) == 0, "Train and Val splits leak identical interaction entries!"
    assert len(val_keys & test_keys) == 0, "Val and Test splits leak identical interaction entries!"


def test_contiguous_id_mappings(data_bundle):
    """Verify that user and item IDs are 0-indexed, contiguous, and non-empty."""
    user2idx = data_bundle["user2idx"]
    item2idx = data_bundle["item2idx"]

    num_users = len(user2idx)
    num_items = len(item2idx)

    user_indices = sorted(user2idx.values())
    item_indices = sorted(item2idx.values())

    assert user_indices == list(range(num_users)), "User indices must be contiguous 0..N-1"
    assert item_indices == list(range(num_items)), "Item indices must be contiguous 0..M-1"
    assert min(user_indices) == 0 and max(user_indices) == num_users - 1
    assert min(item_indices) == 0 and max(item_indices) == num_items - 1


def test_ncf_dataloader_batches(data_bundle):
    """Verify that PyTorch DataLoader yields correctly shaped batches with right dtypes."""
    train_loader, val_loader, _ = get_ncf_dataloaders(
        data_bundle["train_df"],
        data_bundle["val_df"],
        batch_size=256
    )

    batch_u, batch_i, batch_r = next(iter(train_loader))

    assert batch_u.shape == torch.Size([256])
    assert batch_i.shape == torch.Size([256])
    assert batch_r.shape == torch.Size([256])

    assert batch_u.dtype == torch.long
    assert batch_i.dtype == torch.long
    assert batch_r.dtype == torch.float32

    # Ratings should all be in valid rating range [1.0, 5.0]
    assert torch.all(batch_r >= 1.0) and torch.all(batch_r <= 5.0)
