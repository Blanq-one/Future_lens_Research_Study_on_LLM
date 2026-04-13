"""
Extract hidden states from all layers of a transformer model.
Provides batched, memory-efficient extraction with caching.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
import json


class HiddenStateExtractor:
    """
    Extracts hidden states from all layers at specified token positions.
    
    Given input text, runs the model and collects:
    - Hidden states h_t^l for every layer l at every token position t
    - Model output logits / predictions
    - Ground truth next tokens
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.num_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size

    @torch.no_grad()
    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run model forward pass and extract all hidden states.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) attention mask

        Returns:
            Dict with:
                'hidden_states': tuple of (batch, seq_len, hidden_dim) for each layer (L+1 total, including embedding)
                'logits': (batch, seq_len, vocab_size) output logits
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        return {
            "hidden_states": outputs.hidden_states,  # tuple of (batch, seq, hidden_dim)
            "logits": outputs.logits,                 # (batch, seq, vocab)
        }

    @torch.no_grad()
    def extract_at_positions(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hidden states at specific token positions only (memory efficient).

        Args:
            input_ids: (batch_size, seq_len)
            positions: (batch_size,) position index T for each sample
            attention_mask: (batch_size, seq_len)

        Returns:
            Dict with:
                'hidden_states_at_T': (num_layers+1, batch_size, hidden_dim) states at position T
                'logits_at_T': (batch_size, vocab_size) logits at position T
                'predictions': (batch_size, max_n) greedy predictions for tokens T+1 through T+max_n
        """
        out = self.extract_hidden_states(input_ids, attention_mask)

        batch_size = input_ids.size(0)
        batch_idx = torch.arange(batch_size, device=self.device)

        # Collect hidden states at position T from each layer
        hs_at_T = torch.stack([
            layer_hs[batch_idx, positions]  # (batch, hidden_dim)
            for layer_hs in out["hidden_states"]
        ])  # (num_layers+1, batch, hidden_dim)

        logits_at_T = out["logits"][batch_idx, positions]  # (batch, vocab)

        return {
            "hidden_states_at_T": hs_at_T,
            "logits_at_T": logits_at_T,
        }

    @torch.no_grad()
    def generate_and_extract(
        self,
        input_ids: torch.Tensor,
        position_T: int,
        max_n: int = 5,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        For a single sample: run model, get hidden states at position T,
        then greedily generate N tokens and get their final-layer hidden states.

        Args:
            input_ids: (1, seq_len) single sample
            position_T: the token position T
            max_n: how many tokens ahead to generate and track

        Returns:
            Dict with hidden states at T for all layers,
            plus final-layer hidden states at T+1 through T+N,
            plus the generated token IDs.
        """
        assert input_ids.size(0) == 1, "Single sample only"
        input_ids = input_ids.to(self.device)

        # First pass: get hidden states at position T
        out = self.extract_hidden_states(input_ids, attention_mask)

        hs_at_T = torch.stack([
            layer_hs[0, position_T]  # (hidden_dim,)
            for layer_hs in out["hidden_states"]
        ])  # (num_layers+1, hidden_dim)

        # Greedy decode next max_n tokens and collect final-layer hidden states
        generated_ids = []
        final_layer_hs = []

        current_ids = input_ids[:, :position_T + 1]
        if attention_mask is not None:
            current_mask = attention_mask[:, :position_T + 1]
        else:
            current_mask = None

        for step in range(max_n + 1):
            step_out = self.extract_hidden_states(current_ids, current_mask)
            last_logits = step_out["logits"][0, -1]  # (vocab,)
            next_token = last_logits.argmax(dim=-1)  # scalar

            # Final layer hidden state at last position
            final_hs = step_out["hidden_states"][-1][0, -1]  # (hidden_dim,)
            final_layer_hs.append(final_hs)
            generated_ids.append(next_token.item())

            # Extend sequence
            current_ids = torch.cat([
                current_ids,
                next_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)
            if current_mask is not None:
                current_mask = torch.cat([
                    current_mask,
                    torch.ones(1, 1, device=self.device, dtype=current_mask.dtype)
                ], dim=1)

        return {
            "hidden_states_at_T": hs_at_T,          # (L+1, hidden_dim)
            "final_layer_hs_future": torch.stack(final_layer_hs),  # (max_n+1, hidden_dim)
            "generated_ids": generated_ids,           # list of ints
            "logits_at_T": out["logits"][0, position_T],
        }


def extract_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_n: int = 5,
    max_context_length: int = 512,
    filter_correct: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    max_samples: int = 10000,
) -> Dict[str, List]:
    """
    Extract training/test data from a list of text samples.

    For each text:
    1. Tokenize
    2. Pick a random position T (with enough room for N future tokens)
    3. Run model, get hidden states at T for all layers
    4. Get final-layer hidden states at T+1 through T+N
    5. Record ground truth tokens

    Returns a dict of lists ready for probe training.
    """
    extractor = HiddenStateExtractor(model, tokenizer, device, dtype)
    rng = np.random.RandomState(42)

    data = {
        "hidden_states": [],       # List of (num_layers+1, hidden_dim) tensors
        "final_hs_future": [],     # List of (max_n+1, hidden_dim) tensors
        "generated_ids": [],       # List of token ID lists
        "positions": [],           # List of position T
        "token_at_T": [],          # Token ID at position T
        "context_text": [],        # Original text up to T
    }

    collected = 0
    for text in tqdm(texts, desc="Extracting hidden states"):
        if collected >= max_samples:
            break

        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_context_length,
        )
        input_ids = tokens["input_ids"]
        seq_len = input_ids.size(1)

        # Need at least max_n+2 tokens (position T + max_n+1 future tokens)
        if seq_len < max_n + 3:
            continue

        # Pick random position T, leaving room for future tokens
        # We want positions where the model can generate max_n tokens after
        T = rng.randint(1, seq_len - max_n - 1)

        # Check if model predicts correctly at position T
        if filter_correct:
            with torch.no_grad():
                logits = model(
                    input_ids.to(device),
                    output_hidden_states=False,
                ).logits
                pred_token = logits[0, T].argmax().item()
                actual_next = input_ids[0, T + 1].item()
                if pred_token != actual_next:
                    continue

        # Full extraction
        result = extractor.generate_and_extract(
            input_ids, position_T=T, max_n=max_n
        )

        data["hidden_states"].append(result["hidden_states_at_T"].cpu())
        data["final_hs_future"].append(result["final_layer_hs_future"].cpu())
        data["generated_ids"].append(result["generated_ids"])
        data["positions"].append(T)
        data["token_at_T"].append(input_ids[0, T].item())
        data["context_text"].append(
            tokenizer.decode(input_ids[0, :T + 1])
        )
        collected += 1

    print(f"Collected {collected} samples")
    return data


def save_extracted_data(data: Dict, path: str):
    """Save extracted data to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save tensors
    torch.save(
        {
            "hidden_states": data["hidden_states"],
            "final_hs_future": data["final_hs_future"],
        },
        path / "tensors.pt",
    )

    # Save metadata
    meta = {
        "generated_ids": data["generated_ids"],
        "positions": data["positions"],
        "token_at_T": data["token_at_T"],
        "context_text": data["context_text"],
    }
    torch.save(meta, path / "metadata.pt")
    print(f"Saved extracted data to {path}")


def load_extracted_data(path: str) -> Dict:
    """Load previously extracted data."""
    path = Path(path)
    tensors = torch.load(path / "tensors.pt", map_location="cpu", weights_only=False)
    meta = torch.load(path / "metadata.pt", map_location="cpu", weights_only=False)
    return {**tensors, **meta}
