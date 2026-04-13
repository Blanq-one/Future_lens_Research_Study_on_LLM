"""
Linear Model Approximation (Section 2.2 of the paper).

Trains a linear model f_theta that maps:
    h_T^l  ->  h_{T+N}^L

i.e., from a hidden state at layer l, position T,
to the final-layer hidden state at position T+N.

The vocabulary can then be read by applying the pretrained decoder head.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np


class LinearProbe(nn.Module):
    """
    Linear probe: h_T^l -> h_{T+N}^L

    One probe is trained per (layer l, future offset N) pair.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, hidden_dim) hidden state at layer l, position T
        Returns:
            (batch, hidden_dim) predicted hidden state at layer L, position T+N
        """
        return self.linear(h)


class LinearProbeTrainer:
    """
    Trains and evaluates LinearProbe models for all (layer, N) combinations.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        max_n: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 20,
        batch_size: int = 256,
        patience: int = 5,
        device: str = "cuda",
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_n = max_n
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device

        # Create probes: probes[l][n] = LinearProbe
        # Layer indices 0..num_layers (0 = embedding layer)
        self.probes: Dict[int, Dict[int, LinearProbe]] = {}
        for l in range(num_layers + 1):
            self.probes[l] = {}
            for n in range(max_n + 1):
                self.probes[l][n] = LinearProbe(hidden_dim).to(device)

    def train(
        self,
        hidden_states: List[torch.Tensor],
        final_hs_future: List[torch.Tensor],
        val_fraction: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train all probes.

        Args:
            hidden_states: list of (num_layers+1, hidden_dim) tensors (one per sample)
            final_hs_future: list of (max_n+1, hidden_dim) tensors (one per sample)

        Returns:
            Training history dict
        """
        # Stack into tensors
        # all_hs: (num_samples, num_layers+1, hidden_dim)
        all_hs = torch.stack(hidden_states).float()
        # all_future: (num_samples, max_n+1, hidden_dim)
        all_future = torch.stack(final_hs_future).float()

        num_samples = all_hs.size(0)
        val_size = int(num_samples * val_fraction)
        train_size = num_samples - val_size

        indices = torch.randperm(num_samples)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

        history = {"train_loss": [], "val_loss": []}

        total_probes = (self.num_layers + 1) * (self.max_n + 1)
        pbar = tqdm(total=total_probes, desc="Training linear probes")

        for l in range(self.num_layers + 1):
            for n in range(self.max_n + 1):
                probe = self.probes[l][n]
                optimizer = torch.optim.AdamW(
                    probe.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )

                # Training data for this (l, n) pair
                X_train = all_hs[train_idx, l].to(self.device)  # (train_size, hidden_dim)
                Y_train = all_future[train_idx, n].to(self.device)
                X_val = all_hs[val_idx, l].to(self.device)
                Y_val = all_future[val_idx, n].to(self.device)

                train_ds = TensorDataset(X_train, Y_train)
                train_loader = DataLoader(
                    train_ds, batch_size=self.batch_size, shuffle=True
                )

                best_val_loss = float("inf")
                patience_counter = 0

                for epoch in range(self.num_epochs):
                    # Train
                    probe.train()
                    epoch_loss = 0.0
                    for X_batch, Y_batch in train_loader:
                        pred = probe(X_batch)
                        loss = nn.functional.mse_loss(pred, Y_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    # Validate
                    probe.eval()
                    with torch.no_grad():
                        val_pred = probe(X_val)
                        val_loss = nn.functional.mse_loss(val_pred, Y_val).item()

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            break

                pbar.update(1)
                pbar.set_postfix({"layer": l, "n": n, "val_loss": f"{best_val_loss:.4f}"})

        pbar.close()
        return history

    @torch.no_grad()
    def predict(
        self,
        h_l: torch.Tensor,
        layer: int,
        n: int,
    ) -> torch.Tensor:
        """
        Predict future hidden state.

        Args:
            h_l: (batch, hidden_dim) hidden state at layer l
            layer: which layer l
            n: how many tokens ahead (0 = immediate next)

        Returns:
            (batch, hidden_dim) predicted final-layer hidden state
        """
        probe = self.probes[layer][n]
        probe.eval()
        return probe(h_l.to(self.device))

    def save(self, path: str):
        """Save all probe weights."""
        state = {}
        for l in self.probes:
            for n in self.probes[l]:
                key = f"layer_{l}_n_{n}"
                state[key] = self.probes[l][n].state_dict()
        torch.save(state, path)
        print(f"Saved linear probes to {path}")

    def load(self, path: str):
        """Load probe weights."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        for l in self.probes:
            for n in self.probes[l]:
                key = f"layer_{l}_n_{n}"
                if key in state:
                    self.probes[l][n].load_state_dict(state[key])
        print(f"Loaded linear probes from {path}")
