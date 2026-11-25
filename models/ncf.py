"""
models/ncf.py - Neural Collaborative Filtering (NeuMF) in PyTorch.
Fuses Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) branches.
Supports Automatic Mixed Precision (torch.cuda.amp), GPU memory tracking, and CPU fallback.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class NeuMF(nn.Module):
    """
    NeuMF: Neural Collaborative Filtering combining GMF and MLP.
    - GMF branch: dot product of latent user & item embeddings
    - MLP branch: multi-layer non-linear representation of user & item embeddings
    - Fusion layer: dense projection to calibrated rating output
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim_gmf: int = 32,
        latent_dim_mlp: int = 32,
        mlp_hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.2
    ):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_gmf = latent_dim_gmf
        self.latent_dim_mlp = latent_dim_mlp

        # 1. GMF Embeddings
        self.user_embed_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.item_embed_gmf = nn.Embedding(num_items, latent_dim_gmf)

        # 2. MLP Embeddings
        self.user_embed_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, latent_dim_mlp)

        # 3. MLP Layers
        mlp_layers = []
        in_dim = latent_dim_mlp * 2
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout_rate))
            in_dim = hidden_dim
        self.mlp_network = nn.Sequential(*mlp_layers)

        # 4. Final Fusion Layer (GMF dim + last MLP dim -> 1 output)
        final_in_dim = latent_dim_gmf + mlp_hidden_dims[-1]
        self.fusion_layer = nn.Linear(final_in_dim, 1)

        # Initialize weights with standard Gaussian
        self._init_weights()

    def _init_weights(self):
        """Xavier and normal weight initialization for stable convergence."""
        nn.init.normal_(self.user_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embed_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embed_mlp.weight, std=0.01)

        for m in self.mlp_network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns predicted ratings calibrated in [1.0, 5.0].
        """
        # GMF Branch
        gmf_u = self.user_embed_gmf(user_indices)
        gmf_i = self.item_embed_gmf(item_indices)
        phi_gmf = gmf_u * gmf_i

        # MLP Branch
        mlp_u = self.user_embed_mlp(user_indices)
        mlp_i = self.item_embed_mlp(item_indices)
        phi_mlp = torch.cat([mlp_u, mlp_i], dim=-1)
        phi_mlp = self.mlp_network(phi_mlp)

        # Fusion
        phi_fusion = torch.cat([phi_gmf, phi_mlp], dim=-1)
        out = self.fusion_layer(phi_fusion)

        # Map through sigmoid to [0, 1] then scale to [1.0, 5.0]
        prediction = 1.0 + 4.0 * torch.sigmoid(out.squeeze(-1))
        return prediction


class NCFRecommender:
    """Wrapper managing NeuMF model training, evaluation, AMP mixed precision, and inference."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim_gmf: int = 32,
        latent_dim_mlp: int = 32,
        mlp_hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.2,
        device: Optional[str] = None
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_gmf = latent_dim_gmf
        self.latent_dim_mlp = latent_dim_mlp
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout_rate = dropout_rate

        # Auto-detect CUDA with CPU fallback
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = NeuMF(
            num_users=num_users,
            num_items=num_items,
            latent_dim_gmf=latent_dim_gmf,
            latent_dim_mlp=latent_dim_mlp,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout_rate=dropout_rate
        ).to(self.device)

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.training_history: List[Dict[str, float]] = []

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train NeuMF model using MSE loss, Adam optimizer, and torch.amp mixed precision.
        Explicitly monitors and prints peak VRAM allocation after each epoch.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        best_state = None
        peak_vram_mb = 0.0

        if verbose:
            print(f"[*] Training NCF (NeuMF) on {self.device.type.upper()}:")
            print(f"    Users: {self.num_users}, Items: {self.num_items}, Embedding Dim: {self.latent_dim_gmf} (GMF)+{self.latent_dim_mlp} (MLP)")
            print(f"    Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            # Training Phase
            self.model.train()
            train_loss_total = 0.0
            n_train_samples = 0

            for user_batch, item_batch, rating_batch in train_loader:
                user_batch = user_batch.to(self.device, non_blocking=True)
                item_batch = item_batch.to(self.device, non_blocking=True)
                rating_batch = rating_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    predictions = self.model(user_batch, item_batch)
                    loss = criterion(predictions, rating_batch)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                train_loss_total += loss.item() * len(rating_batch)
                n_train_samples += len(rating_batch)

            train_rmse = np.sqrt(train_loss_total / n_train_samples)

            # Validation Phase
            self.model.eval()
            val_loss_total = 0.0
            n_val_samples = 0

            with torch.no_grad():
                for user_batch, item_batch, rating_batch in val_loader:
                    user_batch = user_batch.to(self.device, non_blocking=True)
                    item_batch = item_batch.to(self.device, non_blocking=True)
                    rating_batch = rating_batch.to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                        preds = self.model(user_batch, item_batch)
                        v_loss = criterion(preds, rating_batch)

                    val_loss_total += v_loss.item() * len(rating_batch)
                    n_val_samples += len(rating_batch)

            val_rmse = np.sqrt(val_loss_total / n_val_samples)
            epoch_time = time.time() - epoch_start

            # Measure peak VRAM
            if self.device.type == "cuda":
                current_peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                peak_vram_mb = max(peak_vram_mb, current_peak)
                vram_str = f"Peak VRAM: {current_peak:.1f} MB"
            else:
                vram_str = "VRAM: N/A (CPU)"

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            history_entry = {
                "epoch": epoch,
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "epoch_time_sec": float(epoch_time),
                "peak_vram_mb": float(peak_vram_mb)
            }
            self.training_history.append(history_entry)

            if verbose:
                flag = " [BEST]" if val_rmse == best_val_rmse else ""
                print(f"    Epoch {epoch:02d}/{epochs:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Time: {epoch_time:.2f}s | {vram_str}{flag}")

        # Restore best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Flag peak VRAM if it approaches 4GB limit
        peak_gb = peak_vram_mb / 1024.0
        print(f"[OK] NCF Training Complete. Best Val RMSE: {best_val_rmse:.4f} | Peak GPU VRAM: {peak_gb:.2f} GB ({peak_vram_mb:.1f} MB)")
        if peak_gb > 4.0:
            print(f"[WARNING] Peak VRAM ({peak_gb:.2f} GB) exceeded 4GB target ceiling!")
        else:
            print(f"[OK] VRAM usage is safely within RTX 3050 target (< 4.0 GB). Headroom: {6.0 - peak_gb:.2f} GB.")

        return {
            "best_val_rmse": best_val_rmse,
            "peak_vram_mb": peak_vram_mb,
            "peak_vram_gb": peak_gb,
            "history": self.training_history
        }

    def predict_pair(self, u_idx: int, i_idx: int) -> float:
        """Predict rating for a single user-item pair."""
        if not (0 <= u_idx < self.num_users and 0 <= i_idx < self.num_items):
            return 3.0  # Neutral fallback

        self.model.eval()
        with torch.no_grad():
            u_t = torch.tensor([u_idx], dtype=torch.long, device=self.device)
            i_t = torch.tensor([i_idx], dtype=torch.long, device=self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                pred = self.model(u_t, i_t).item()
        return float(pred)

    def predict(
        self,
        u_idx: int,
        item_indices: Optional[np.ndarray] = None,
        batch_size: int = 1024
    ) -> np.ndarray:
        """
        Predict ratings for an array of items for a given user.
        Uses batched forward passes for high throughput.
        """
        if item_indices is None:
            item_indices = np.arange(self.num_items)

        if not (0 <= u_idx < self.num_users):
            return np.full(len(item_indices), 3.0, dtype=np.float32)

        self.model.eval()
        predictions = np.zeros(len(item_indices), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(item_indices), batch_size):
                end = min(start + batch_size, len(item_indices))
                batch_items = item_indices[start:end]

                # Filter valid items
                valid_mask = (batch_items >= 0) & (batch_items < self.num_items)
                if not np.any(valid_mask):
                    predictions[start:end] = 3.0
                    continue

                u_batch = torch.full((len(batch_items),), u_idx, dtype=torch.long, device=self.device)
                i_batch = torch.tensor(batch_items, dtype=torch.long, device=self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    preds = self.model(u_batch, i_batch)

                predictions[start:end] = preds.cpu().numpy()

        return predictions

    def save(self, filepath: str) -> None:
        """Save NCF model weights and metadata to disk."""
        checkpoint = {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "latent_dim_gmf": self.latent_dim_gmf,
            "latent_dim_mlp": self.latent_dim_mlp,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "dropout_rate": self.dropout_rate,
            "state_dict": self.model.state_dict(),
            "training_history": self.training_history
        }
        torch.save(checkpoint, filepath)
        print(f"[OK] NCF model checkpoint saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "NCFRecommender":
        """Load NCF model from checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu")
        recommender = cls(
            num_users=checkpoint["num_users"],
            num_items=checkpoint["num_items"],
            latent_dim_gmf=checkpoint["latent_dim_gmf"],
            latent_dim_mlp=checkpoint["latent_dim_mlp"],
            mlp_hidden_dims=checkpoint["mlp_hidden_dims"],
            dropout_rate=checkpoint["dropout_rate"],
            device=device
        )
        recommender.model.load_state_dict(checkpoint["state_dict"])
        recommender.model.eval()
        recommender.training_history = checkpoint.get("training_history", [])
        print(f"[OK] NCF model loaded from {filepath} on {recommender.device.type.upper()}")
        return recommender
