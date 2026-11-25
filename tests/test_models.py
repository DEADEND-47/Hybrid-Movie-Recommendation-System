"""
tests/test_models.py - Unit tests for Content, SVD, and NCF individual model architectures.
"""

import pytest
import numpy as np
import torch
from models.content import ContentRecommender
from models.svd import SVDRecommender
from models.ncf import NCFRecommender, NeuMF


def test_svd_prediction_bounds():
    """Verify that SVD model predictions are bounded to [1.0, 5.0]."""
    svd = SVDRecommender.load("artifacts/svd_model.pkl")
    preds = svd.predict(u_idx=0, item_indices=np.arange(100))
    assert np.all(preds >= 1.0) and np.all(preds <= 5.0)

    # Test unknown user fallback
    unknown_preds = svd.predict(u_idx=999999, item_indices=np.arange(50))
    assert np.all(unknown_preds >= 1.0) and np.all(unknown_preds <= 5.0)


def test_content_engine_similarity():
    """Verify that Content engine similarity matrix is symmetric and bounded [0, 1]."""
    content = ContentRecommender.load("artifacts/content_engine.pkl")
    sim = content.similarity_matrix

    assert sim.shape == (content.num_items, content.num_items)
    # Check diagonal is ~1.0
    diag = np.diag(sim)
    assert np.allclose(diag, 1.0, atol=1e-3)
    # Check symmetry
    assert np.allclose(sim[:50, :50], sim[:50, :50].T, atol=1e-4)


def test_ncf_forward_pass():
    """Verify that NCF NeuMF loads in eval mode and outputs valid rating tensors."""
    ncf = NCFRecommender.load("artifacts/ncf_model.pt")
    assert not ncf.model.training, "NCFRecommender.load() must leave model in eval() mode!"
    u_tensor = torch.tensor([0, 1, 2], dtype=torch.long, device=ncf.device)
    i_tensor = torch.tensor([10, 20, 30], dtype=torch.long, device=ncf.device)

    with torch.no_grad():
        preds = ncf.model(u_tensor, i_tensor)

    assert preds.shape == torch.Size([3])
    assert torch.all(preds >= 1.0) and torch.all(preds <= 5.0)
