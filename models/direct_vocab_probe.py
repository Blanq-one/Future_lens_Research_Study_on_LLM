"""
Direct Vocabulary Prediction (Section 2.1 of the paper).

Trains a linear model g_theta that maps:
    h_T^l  ->  z_{T+N}  (logits over vocabulary)

Unlike the linear probe in linear_probe.py, this directly predicts
the output distribution without using the pretrained decoder head.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List
from tqdm import tqdm


class DirectVocabProbe(nn.Module):
    """
    Direct vocabulary probe: h_T^l -> logits over V

    One probe per (layer l, future offset N).
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, hidden_dim) hidden state at layer l, position T
        Returns:
            (batch, vocab_size) logits over vocabulary
        """
        return self.linear(h)


class DirectVocabProbeTrainer:
    """Train DirectVocabProbe for all (layer, N) combinations."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
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
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_n = max_n
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device

        self.probes: Dict[int, Dict[int, DirectVocabProbe]] = {}
        for l in range(num_layers + 1):
            self.probes[l] = {}
            for n in range(max_n + 1):
                self.probes[l][n] = DirectVocabProbe(hidden_dim, vocab_size).to(device)

    def train(
        self,
        hidden_states: List[torch.Tensor],
        target_token_ids: List[List[int]],
        val_fraction: float = 0.1,
    ) -> Dict:
        """
        Train all probes using cross-entropy loss.

        Args:
            hidden_states: list of (num_layers+1, hidden_dim)
            target_token_ids: list of lists, each inner list has max_n+1 token IDs
                (the greedy-decoded tokens at positions T+1 through T+N+1)
        """
        all_hs = torch.stack(hidden_states).float()  # (S, L+1, D)
        # Build target tensor: (S, max_n+1)
        all_targets = torch.tensor(
            [ids[:self.max_n + 1] for ids in target_token_ids],
            dtype=torch.long,
        )

        num_samples = all_hs.size(0)
        val_size = int(num_samples * val_fraction)
        indices = torch.randperm(num_samples)
        train_idx = indices[:num_samples - val_size]
        val_idx = indices[num_samples - val_size:]

        total = (self.num_layers + 1) * (self.max_n + 1)
        pbar = tqdm(total=total, desc="Training vocab probes")

        for l in range(self.num_layers + 1):
            for n in range(self.max_n + 1):
                probe = self.probes[l][n]
                optimizer = torch.optim.AdamW(
                    probe.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
                criterion = nn.CrossEntropyLoss()

                X_train = all_hs[train_idx, l].to(self.device)
                Y_train = all_targets[train_idx, n].to(self.device)
                X_val = all_hs[val_idx, l].to(self.device)
                Y_val = all_targets[val_idx, n].to(self.device)

                train_ds = TensorDataset(X_train, Y_train)
                train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

                best_val_loss = float("inf")
                patience_counter = 0

                for epoch in range(self.num_epochs):
                    probe.train()
                    for X_batch, Y_batch in train_loader:
                        logits = probe(X_batch)
                        loss = criterion(logits, Y_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    probe.eval()
                    with torch.no_grad():
                        val_logits = probe(X_val)
                        val_loss = criterion(val_logits, Y_val).item()

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
        """Predict vocabulary logits from hidden state."""
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
