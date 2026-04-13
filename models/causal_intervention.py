"""
Fixed Prompt Causal Intervention (Section 2.3 of the paper).

Transplants a hidden state h_T^l from an original context into a completely
different (fixed, generic) context at the same layer l. Then measures whether
the transplanted state steers the model to generate the same tokens as the
original context would have produced.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import copy


class HookManager:
    """Manages forward hooks for injecting hidden states into transformer layers."""

    def __init__(self):
        self.hooks = []
        self.stored_states = {}

    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.stored_states = {}

    def register_injection_hook(
        self,
        model: AutoModelForCausalLM,
        layer_idx: int,
        position: int,
        replacement_state: torch.Tensor,
    ):
        """
        Register a hook that replaces the hidden state at a specific
        layer and position with a given replacement state.
        """
        layers = _get_layers(model)
        layer = layers[layer_idx]

        def hook_fn(module, input, output):
            # output is typically a tuple; hidden states are the first element
            if isinstance(output, tuple):
                hidden_states = output[0]
                hidden_states[:, position, :] = replacement_state.to(hidden_states.device)
                return (hidden_states,) + output[1:]
            else:
                output[:, position, :] = replacement_state.to(output.device)
                return output

        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)


def _get_layers(model):
    """Get the list of transformer layers from model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    else:
        raise ValueError(f"Cannot find layers for {type(model).__name__}")


class FixedPromptIntervention:
    """
    Performs causal intervention by transplanting hidden states
    into fixed generic prompts.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        fixed_prompts: List[str],
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.fixed_prompts = fixed_prompts
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.hook_manager = HookManager()

    @torch.no_grad()
    def _get_hidden_states_from_original(
        self,
        input_ids: torch.Tensor,
        position_T: int,
    ) -> torch.Tensor:
        """
        Get hidden states at position T from all layers of the original context.

        Returns: (num_layers+1, hidden_dim) - hidden states from embedding + all layers
        """
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids, output_hidden_states=True)

        hs = torch.stack([
            layer_hs[0, position_T]
            for layer_hs in outputs.hidden_states
        ])  # (num_layers+1, hidden_dim)

        return hs

    @torch.no_grad()
    def intervene_and_generate(
        self,
        original_input_ids: torch.Tensor,
        position_T: int,
        layer: int,
        fixed_prompt_idx: int = 0,
        max_new_tokens: int = 5,
    ) -> Dict:
        """
        Transplant h_T^l from original context into fixed prompt,
        then generate tokens.

        Args:
            original_input_ids: (1, seq_len) the original input
            position_T: position T in the original input
            layer: which layer l to transplant from/to
            fixed_prompt_idx: which fixed prompt to use
            max_new_tokens: how many tokens to generate after intervention

        Returns:
            Dict with generated tokens and probabilities
        """
        # Step 1: Get hidden state from original context at layer l, position T
        all_hs = self._get_hidden_states_from_original(original_input_ids, position_T)
        h_l_T = all_hs[layer + 1]  # +1 because index 0 is embedding layer

        # Step 2: Tokenize fixed prompt
        fixed_prompt = self.fixed_prompts[fixed_prompt_idx]
        fixed_tokens = self.tokenizer(
            fixed_prompt, return_tensors="pt"
        )["input_ids"].to(self.device)
        M = fixed_tokens.size(1) - 1  # Last position in fixed prompt

        # Step 3: Set up intervention hook at layer l, position M
        self.hook_manager.clear()
        self.hook_manager.register_injection_hook(
            self.model, layer, M, h_l_T
        )

        # Step 4: Run fixed prompt through model with intervention
        outputs = self.model(fixed_tokens, output_hidden_states=False)
        first_logits = outputs.logits[0, M]  # (vocab,)
        first_probs = torch.softmax(first_logits, dim=-1)
        first_token = first_logits.argmax().item()

        self.hook_manager.clear()

        # Step 5: Generate subsequent tokens (without intervention, but WITH
        # the generated tokens appended)
        generated_tokens = [first_token]
        generated_probs = [first_probs.cpu()]
        current_ids = torch.cat([
            fixed_tokens,
            torch.tensor([[first_token]], device=self.device)
        ], dim=1)

        for step in range(max_new_tokens):
            # For subsequent tokens, we re-do the intervention so the hidden
            # state influence persists through the layers
            self.hook_manager.clear()
            self.hook_manager.register_injection_hook(
                self.model, layer, M, h_l_T
            )

            out = self.model(current_ids, output_hidden_states=False)
            next_logits = out.logits[0, -1]
            next_probs = torch.softmax(next_logits, dim=-1)
            next_token = next_logits.argmax().item()

            generated_tokens.append(next_token)
            generated_probs.append(next_probs.cpu())

            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]], device=self.device)
            ], dim=1)

            self.hook_manager.clear()

        return {
            "generated_token_ids": generated_tokens,
            "generated_text": self.tokenizer.decode(generated_tokens),
            "token_probs": generated_probs,
        }

    @torch.no_grad()
    def evaluate(
        self,
        hidden_states: List[torch.Tensor],
        generated_ids: List[List[int]],
        max_n: int = 5,
    ) -> Dict[str, Dict]:
        """
        Evaluate fixed prompt intervention across all layers and prompts.

        For each sample, for each layer, transplant the hidden state into
        each fixed prompt and check if the generated tokens match.

        Args:
            hidden_states: list of (num_layers+1, hidden_dim) from original contexts
            generated_ids: list of lists of ground truth generated token IDs

        Returns:
            Results dict with accuracy per layer, per N, per prompt
        """
        results = {}
        num_samples = len(hidden_states)

        for prompt_idx, prompt in enumerate(self.fixed_prompts):
            prompt_key = f"prompt_{prompt_idx}"
            results[prompt_key] = {}

            # Tokenize fixed prompt once
            fixed_tokens = self.tokenizer(
                prompt, return_tensors="pt"
            )["input_ids"].to(self.device)
            M = fixed_tokens.size(1) - 1

            for l in tqdm(range(self.num_layers), desc=f"Evaluating prompt {prompt_idx}"):
                correct = {n: 0 for n in range(max_n + 1)}

                for i in range(num_samples):
                    h_l_T = hidden_states[i][l + 1].to(self.device)  # +1 for embedding offset

                    # Intervention
                    self.hook_manager.clear()
                    self.hook_manager.register_injection_hook(
                        self.model, l, M, h_l_T
                    )

                    out = self.model(fixed_tokens, output_hidden_states=False)
                    pred_token = out.logits[0, M].argmax().item()
                    self.hook_manager.clear()

                    # Check N=0
                    if pred_token == generated_ids[i][0]:
                        correct[0] += 1

                    # Generate more and check N=1..max_n
                    current_ids = torch.cat([
                        fixed_tokens,
                        torch.tensor([[generated_ids[i][0]]], device=self.device)
                    ], dim=1)

                    for n in range(1, min(max_n + 1, len(generated_ids[i]))):
                        self.hook_manager.clear()
                        self.hook_manager.register_injection_hook(
                            self.model, l, M, h_l_T
                        )
                        out = self.model(current_ids, output_hidden_states=False)
                        pred = out.logits[0, -1].argmax().item()
                        self.hook_manager.clear()

                        if pred == generated_ids[i][n]:
                            correct[n] += 1

                        current_ids = torch.cat([
                            current_ids,
                            torch.tensor([[generated_ids[i][n]]], device=self.device)
                        ], dim=1)

                results[prompt_key][l] = {
                    n: correct[n] / num_samples for n in correct
                }

        # Average across prompts
        results["average"] = {}
        for l in range(self.num_layers):
            results["average"][l] = {}
            for n in range(max_n + 1):
                vals = [
                    results[f"prompt_{p}"][l][n]
                    for p in range(len(self.fixed_prompts))
                    if n in results[f"prompt_{p}"][l]
                ]
                results["average"][l][n] = sum(vals) / len(vals) if vals else 0.0

        return results
