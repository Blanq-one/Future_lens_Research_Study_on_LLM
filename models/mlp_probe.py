"""
MLP Probe (Extension beyond original paper).

A nonlinear probe that maps h_T^l -> h_{T+N}^L using a small MLP.
Tests whether additional nonlinear signal about future tokens
exists in hidden states beyond what a linear model can capture.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List
from tqdm import tqdm


class MLPProbe(nn.Module):
    """
    Nonlinear MLP probe: h_T^l -> h_{T+N}^L

    Architecture: Linear -> GELU -> Dropout -> Linear (with residual)
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 4096,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = hidden_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = mlp_hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

        # Residual connection: output = mlp(h) + h
        self.use_residual = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.mlp(h)
        if self.use_residual:
            out = out + h
        return out


class MLPProbeTrainer:
    """Train MLP probes for all (layer, N) combinations."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        max_n: int = 5,
        mlp_hidden_dim: int = 4096,
        mlp_num_layers: int = 2,
        dropout: float = 0.1,
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
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

        self.probes: Dict[int, Dict[int, MLPProbe]] = {}
        for l in range(num_layers + 1):
            self.probes[l] = {}
            for n in range(max_n + 1):
                self.probes[l][n] = MLPProbe(
                    hidden_dim, mlp_hidden_dim, mlp_num_layers, dropout
                ).to(device)

    def train(
        self,
        hidden_states: List[torch.Tensor],
        final_hs_future: List[torch.Tensor],
        val_fraction: float = 0.1,
    ):
        all_hs = torch.stack(hidden_states).float()
        all_future = torch.stack(final_hs_future).float()

        num_samples = all_hs.size(0)
        val_size = int(num_samples * val_fraction)
        indices = torch.randperm(num_samples)
        train_idx = indices[:num_samples - val_size]
        val_idx = indices[num_samples - val_size:]

        total = (self.num_layers + 1) * (self.max_n + 1)
        pbar = tqdm(total=total, desc="Training MLP probes")

        for l in range(self.num_layers + 1):
            for n in range(self.max_n + 1):
                probe = self.probes[l][n]
                optimizer = torch.optim.AdamW(
                    probe.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.num_epochs
                )

                X_train = all_hs[train_idx, l].to(self.device)
                Y_train = all_future[train_idx, n].to(self.device)
                X_val = all_hs[val_idx, l].to(self.device)
                Y_val = all_future[val_idx, n].to(self.device)

                train_ds = TensorDataset(X_train, Y_train)
                train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

                best_val_loss = float("inf")
                patience_counter = 0

                for epoch in range(self.num_epochs):
                    probe.train()
                    for X_batch, Y_batch in train_loader:
                        pred = probe(X_batch)
                        loss = nn.functional.mse_loss(pred, Y_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    scheduler.step()

                    probe.eval()
                    with torch.no_grad():
                        val_pred = probe(X_val)
                        val_loss = nn.functional.mse_loss(val_pred, Y_val).item()

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

    @torch.no_grad()
    def predict(self, h_l: torch.Tensor, layer: int, n: int) -> torch.Tensor:
        probe = self.probes[layer][n]
        probe.eval()
        return probe(h_l.to(self.device))

    def save(self, path: str):
        state = {}
        for l in self.probes:
            for n in self.probes[l]:
                state[f"layer_{l}_n_{n}"] = self.probes[l][n].state_dict()
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        for l in self.probes:
            for n in self.probes[l]:
                key = f"layer_{l}_n_{n}"
                if key in state:
                    self.probes[l][n].load_state_dict(state[key])
