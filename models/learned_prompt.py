"""
Learned Prompt Causal Intervention (Section 2.4 of the paper).

This is the paper's BEST performing method.

Instead of using a fixed generic prompt, we optimize a soft prompt
(continuous embeddings) that maximally surfaces future-token information
from a transplanted hidden state.

For each layer l, we train a soft prompt c_opt = [c_1, ..., c_M] that
minimizes KL divergence between:
  - The output distribution when h_T^l is transplanted into the soft prompt context
  - The original model's output distribution

This is essentially prefix tuning, but for probing rather than task adaptation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import copy


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot find layers for {type(model).__name__}")


def _get_embed_layer(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
        return model.gpt_neox.embed_in
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    raise ValueError(f"Cannot find embedding for {type(model).__name__}")


class LearnedPrompt(nn.Module):
    """
    A trainable soft prompt (continuous embeddings) for a specific layer l.

    The prompt consists of M continuous vectors in embedding space.
    During forward pass, these are used as if they were token embeddings,
    with the hidden state h_T^l transplanted at the last position.
    """

    def __init__(self, hidden_dim: int, prompt_length: int = 10):
        super().__init__()
        self.prompt_length = prompt_length
        # Initialize soft prompt embeddings
        self.prompt_embeds = nn.Parameter(
            torch.randn(1, prompt_length, hidden_dim) * 0.02
        )

    def forward(self) -> torch.Tensor:
        """Returns: (1, prompt_length, hidden_dim) soft prompt embeddings."""
        return self.prompt_embeds


class LearnedPromptIntervention:
    """
    Trains and evaluates learned prompt interventions for each layer.

    For each layer l:
    1. Train a soft prompt that, when combined with a transplanted hidden state,
       best reproduces the original model's future token predictions.
    2. At test time, transplant h_T^l and generate using the learned prompt.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt_length: int = 10,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.device = device
        self.dtype = dtype
        self.num_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

        # One learned prompt per layer
        self.prompts: Dict[int, LearnedPrompt] = {}
        for l in range(self.num_layers):
            self.prompts[l] = LearnedPrompt(
                self.hidden_dim, prompt_length
            ).to(device)

        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def _create_intervention_input(
        self,
        soft_prompt: torch.Tensor,
        layer_idx: int,
        h_l_T: torch.Tensor,
        continuation_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Create the input for the model with:
        1. Soft prompt embeddings (bypassing token embedding)
        2. Hidden state h_l_T injected at the last soft prompt position at layer l
        3. Optional continuation tokens appended after

        This requires hooking into the model's forward pass.

        Returns: (input_ids_for_shape, position_M)
        """
        # We use a dummy input_ids for shape but override via hooks
        M = soft_prompt.size(1) - 1
        total_len = soft_prompt.size(1)
        if continuation_ids is not None:
            total_len += continuation_ids.size(1)

        return total_len, M

    def train_layer(
        self,
        layer_idx: int,
        hidden_states: List[torch.Tensor],
        generated_ids: List[List[int]],
        max_n: int = 3,
        lr: float = 1e-2,
        num_epochs: int = 50,
        batch_size: int = 64,
    ):
        """
        Train the learned prompt for a specific layer.

        For each training sample:
        1. Get h_T^l from the stored hidden states
        2. Create input: [soft_prompt, <inject h_T^l at position M>]
        3. For each N=0..max_n, append ground truth tokens and compute
           KL divergence between intervention output and original output

        Args:
            layer_idx: which layer to train for
            hidden_states: list of (num_layers+1, hidden_dim) from data extraction
            generated_ids: list of token ID sequences (ground truth)
            max_n: maximum future tokens to predict
            lr: learning rate for the soft prompt
            num_epochs: training epochs
            batch_size: batch size
        """
        prompt = self.prompts[layer_idx]
        optimizer = torch.optim.Adam(prompt.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        num_samples = len(hidden_states)
        layers = _get_layers(self.model)
        embed_layer = _get_embed_layer(self.model)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Shuffle
            indices = torch.randperm(num_samples)

            for batch_start in range(0, num_samples, batch_size):
                batch_idx = indices[batch_start:batch_start + batch_size]
                batch_loss = torch.tensor(0.0, device=self.device)

                for i in batch_idx:
                    i = i.item()
                    # h_T^l from original context (layer_idx + 1 because index 0 is embedding)
                    h_l_T = hidden_states[i][layer_idx + 1].to(self.device).float()
                    gt_ids = generated_ids[i][:max_n + 1]

                    # Get soft prompt embeddings
                    soft_embeds = prompt()  # (1, M+1, hidden_dim)
                    M = soft_embeds.size(1) - 1

                    # We need to:
                    # 1. Feed soft embeddings as input (bypass token embedding)
                    # 2. At layer layer_idx, replace hidden state at position M with h_l_T
                    # 3. Run the rest of the model
                    # 4. Compare output distribution with ground truth

                    # Hook to replace embedding output with soft prompt
                    embed_hook = None
                    layer_hook = None

                    def make_embed_hook(embeds):
                        def hook_fn(module, input, output):
                            batch_size = output.size(0)
                            new_output = embeds.expand(batch_size, -1, -1)
                            return new_output
                        return hook_fn

                    def make_layer_hook(pos, state):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                hs = output[0].clone()
                                hs[:, pos, :] = state
                                return (hs,) + output[1:]
                            return output
                        return hook_fn

                    # Create dummy input_ids matching soft prompt length
                    dummy_ids = torch.zeros(
                        1, self.prompt_length,
                        dtype=torch.long, device=self.device
                    )

                    # Register hooks
                    embed_hook = embed_layer.register_forward_hook(
                        make_embed_hook(soft_embeds)
                    )
                    layer_hook = layers[layer_idx].register_forward_hook(
                        make_layer_hook(M, h_l_T)
                    )

                    try:
                        out = self.model(dummy_ids, output_hidden_states=False)
                        pred_logits = out.logits[0, M]  # N=0 prediction
                        pred_probs = torch.log_softmax(pred_logits, dim=-1)

                        # Loss for N=0: cross-entropy with ground truth token
                        if len(gt_ids) > 0:
                            target = torch.tensor([gt_ids[0]], device=self.device)
                            batch_loss += nn.functional.nll_loss(
                                pred_probs.unsqueeze(0), target
                            )
                    finally:
                        embed_hook.remove()
                        layer_hook.remove()

                    # For N=1..max_n: append ground truth tokens and predict next
                    for n in range(1, min(max_n + 1, len(gt_ids))):
                        # Append ground truth continuation tokens to soft prompt
                        continuation = torch.tensor(
                            [gt_ids[:n]], dtype=torch.long, device=self.device
                        )
                        cont_embeds = embed_layer(continuation)  # (1, n, hidden_dim)

                        combined_embeds = torch.cat([soft_embeds, cont_embeds], dim=1)
                        dummy_ids_ext = torch.zeros(
                            1, self.prompt_length + n,
                            dtype=torch.long, device=self.device
                        )

                        def make_embed_hook_ext(embeds):
                            def hook_fn(module, input, output):
                                return embeds.expand(output.size(0), -1, -1)
                            return hook_fn

                        embed_hook = embed_layer.register_forward_hook(
                            make_embed_hook_ext(combined_embeds)
                        )
                        layer_hook = layers[layer_idx].register_forward_hook(
                            make_layer_hook(M, h_l_T)
                        )

                        try:
                            out = self.model(dummy_ids_ext, output_hidden_states=False)
                            pred_logits = out.logits[0, -1]
                            pred_probs = torch.log_softmax(pred_logits, dim=-1)

                            target = torch.tensor([gt_ids[n]], device=self.device)
                            batch_loss += nn.functional.nll_loss(
                                pred_probs.unsqueeze(0), target
                            )
                        finally:
                            embed_hook.remove()
                            layer_hook.remove()

                batch_loss = batch_loss / len(batch_idx)
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(prompt.parameters(), 1.0)
                optimizer.step()

                epoch_loss += batch_loss.item()
                num_batches += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"  Layer {layer_idx}, Epoch {epoch+1}/{num_epochs}, "
                      f"Loss: {avg_loss:.4f}")

    def train_all_layers(
        self,
        hidden_states: List[torch.Tensor],
        generated_ids: List[List[int]],
        max_n: int = 3,
        lr: float = 1e-2,
        num_epochs: int = 50,
        batch_size: int = 64,
    ):
        """Train learned prompts for all layers."""
        for l in tqdm(range(self.num_layers), desc="Training learned prompts"):
            self.train_layer(
                l, hidden_states, generated_ids,
                max_n=max_n, lr=lr, num_epochs=num_epochs,
                batch_size=batch_size,
            )

    @torch.no_grad()
    def predict(
        self,
        h_l_T: torch.Tensor,
        layer_idx: int,
        continuation_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Use the learned prompt to predict the next token given h_T^l.

        Args:
            h_l_T: (hidden_dim,) hidden state at layer l, position T
            layer_idx: which layer
            continuation_ids: optional list of token IDs to append (for N>0)

        Returns:
            (vocab_size,) logits for the next token prediction
        """
        prompt = self.prompts[layer_idx]
        prompt.eval()

        soft_embeds = prompt()  # (1, M+1, hidden_dim)
        M = soft_embeds.size(1) - 1

        layers = _get_layers(self.model)
        embed_layer = _get_embed_layer(self.model)

        if continuation_ids is not None and len(continuation_ids) > 0:
            cont = torch.tensor([continuation_ids], dtype=torch.long, device=self.device)
            cont_embeds = embed_layer(cont)
            combined = torch.cat([soft_embeds, cont_embeds], dim=1)
        else:
            combined = soft_embeds

        total_len = combined.size(1)
        dummy_ids = torch.zeros(1, total_len, dtype=torch.long, device=self.device)

        def embed_hook(module, input, output):
            return combined.expand(output.size(0), -1, -1)

        def layer_hook(module, input, output):
            if isinstance(output, tuple):
                hs = output[0].clone()
                hs[:, M, :] = h_l_T.to(hs.device)
                return (hs,) + output[1:]
            return output

        h1 = embed_layer.register_forward_hook(embed_hook)
        h2 = layers[layer_idx].register_forward_hook(layer_hook)

        try:
            out = self.model(dummy_ids, output_hidden_states=False)
            logits = out.logits[0, -1]
        finally:
            h1.remove()
            h2.remove()

        return logits

    @torch.no_grad()
    def evaluate(
        self,
        hidden_states: List[torch.Tensor],
        generated_ids: List[List[int]],
        max_n: int = 3,
    ) -> Dict[int, Dict[int, float]]:
        """
        Evaluate learned prompt intervention accuracy for all layers.

        Returns: results[layer][n] = accuracy
        """
        results = {}
        num_samples = len(hidden_states)

        for l in tqdm(range(self.num_layers), desc="Evaluating learned prompts"):
            results[l] = {n: 0 for n in range(max_n + 1)}

            for i in range(num_samples):
                h_l_T = hidden_states[i][l + 1].to(self.device).float()
                gt_ids = generated_ids[i]

                # N=0
                logits = self.predict(h_l_T, l)
                pred = logits.argmax().item()
                if len(gt_ids) > 0 and pred == gt_ids[0]:
                    results[l][0] += 1

                # N=1..max_n
                for n in range(1, min(max_n + 1, len(gt_ids))):
                    continuation = gt_ids[:n]
                    logits = self.predict(h_l_T, l, continuation)
                    pred = logits.argmax().item()
                    if pred == gt_ids[n]:
                        results[l][n] += 1

            for n in results[l]:
                results[l][n] /= num_samples

        return results

    def save(self, path: str):
        state = {l: self.prompts[l].state_dict() for l in self.prompts}
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        for l in state:
            if l in self.prompts:
                self.prompts[l].load_state_dict(state[l])
