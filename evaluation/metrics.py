"""
Evaluation metrics for Future Lens experiments.

Implements:
- Precision@k: Does the top-predicted token appear in the model's top-k?
- Surprisal: -log P(predicted_token) under the model's distribution
- Confidence calibration: Accuracy binned by model confidence
- Bigram baseline: Simple bigram frequency baseline
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from transformers import AutoModelForCausalLM


def precision_at_k(
    predicted_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[int, float]:
    """
    Compute Precision@k: fraction of samples where the target token
    appears in the top-k predictions.

    Args:
        predicted_logits: (num_samples, vocab_size) predicted logits
        target_token_ids: (num_samples,) ground truth token IDs
        k_values: list of k values to compute

    Returns:
        Dict mapping k -> precision value
    """
    results = {}
    num_samples = predicted_logits.size(0)

    for k in k_values:
        top_k = predicted_logits.topk(k, dim=-1).indices  # (num_samples, k)
        target_expanded = target_token_ids.unsqueeze(1)  # (num_samples, 1)
        matches = (top_k == target_expanded).any(dim=1)  # (num_samples,)
        results[k] = matches.float().mean().item()

    return results


def surprisal(
    predicted_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> float:
    """
    Compute average surprisal: -log P(target) where P is softmax of predicted logits.

    Args:
        predicted_logits: (num_samples, vocab_size)
        target_token_ids: (num_samples,)

    Returns:
        Average surprisal (lower is better)
    """
    log_probs = torch.log_softmax(predicted_logits, dim=-1)
    target_log_probs = log_probs.gather(
        1, target_token_ids.unsqueeze(1)
    ).squeeze(1)
    return -target_log_probs.mean().item()


def confidence_calibration(
    predicted_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    original_model_probs: torch.Tensor,
    bins: List[float] = [0.0, 0.3, 0.6, 0.9, 1.0],
) -> Dict[str, float]:
    """
    Compute accuracy of predictions binned by model confidence.

    The key insight from the paper: predictions are more accurate
    when the model is more confident in its next-token prediction.

    Args:
        predicted_logits: (num_samples, vocab_size) from the probe
        target_token_ids: (num_samples,) ground truth
        original_model_probs: (num_samples,) model's max probability for next token
        bins: confidence bin edges

    Returns:
        Dict mapping bin label -> accuracy
    """
    predictions = predicted_logits.argmax(dim=-1)
    correct = (predictions == target_token_ids)

    results = {}
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (original_model_probs >= low) & (original_model_probs < high)
        if mask.sum() > 0:
            acc = correct[mask].float().mean().item()
            count = mask.sum().item()
            label = f"{int(low*100)}-{int(high*100)}%"
            results[label] = {"accuracy": acc, "count": count}

    return results


def compute_bigram_baseline(
    texts: List[str],
    tokenizer,
    test_token_ids: List[int],
    test_next_ids: List[int],
) -> float:
    """
    Compute bigram baseline accuracy.

    Collects bigram frequencies from texts, then for each test sample,
    predicts the most likely next token given the current token.

    Args:
        texts: training texts for collecting bigram statistics
        tokenizer: the model's tokenizer
        test_token_ids: list of token IDs at position T
        test_next_ids: list of ground truth token IDs at position T+1

    Returns:
        Accuracy of bigram predictions
    """
    # Collect bigram counts
    bigram_counts = defaultdict(Counter)

    for text in texts:
        tokens = tokenizer.encode(text)
        for i in range(len(tokens) - 1):
            bigram_counts[tokens[i]][tokens[i + 1]] += 1

    # Predict using most common next token
    correct = 0
    for tok, next_tok in zip(test_token_ids, test_next_ids):
        if tok in bigram_counts and bigram_counts[tok]:
            pred = bigram_counts[tok].most_common(1)[0][0]
            if pred == next_tok:
                correct += 1

    return correct / len(test_token_ids) if test_token_ids else 0.0


def compute_all_metrics(
    predicted_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute all standard metrics for a set of predictions.

    Returns:
        Dict with precision@k for each k, plus surprisal
    """
    results = {}

    # Precision@k
    prec = precision_at_k(predicted_logits, target_token_ids, k_values)
    for k, v in prec.items():
        results[f"precision@{k}"] = v

    # Surprisal
    results["surprisal"] = surprisal(predicted_logits, target_token_ids)

    return results


def evaluate_probe_all_layers(
    probe_trainer,
    decoder_head: torch.nn.Module,
    hidden_states: List[torch.Tensor],
    generated_ids: List[List[int]],
    num_layers: int,
    max_n: int,
    k_values: List[int] = [1, 5, 10],
    use_decoder_head: bool = True,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate a probe (linear or MLP) across all layers and N values.

    Args:
        probe_trainer: LinearProbeTrainer or MLPProbeTrainer
        decoder_head: the model's lm_head for decoding hidden states to vocab
        hidden_states: list of (num_layers+1, hidden_dim)
        generated_ids: list of generated token ID lists
        num_layers: number of transformer layers
        max_n: maximum future token offset
        k_values: precision@k values
        use_decoder_head: if True, decode predicted hidden states through lm_head;
                         if False, probe directly predicts vocab logits
        device: computation device

    Returns:
        Nested dict: results[layer][n] = {precision@1, precision@5, ..., surprisal}
    """
    all_hs = torch.stack(hidden_states).float()  # (S, L+1, D)
    num_samples = all_hs.size(0)

    results = {}

    for l in range(num_layers + 1):
        results[l] = {}
        for n in range(max_n + 1):
            # Get predictions
            h_l = all_hs[:, l].to(device)
            pred_hs = probe_trainer.predict(h_l, l, n)  # (S, D) or (S, V)

            if use_decoder_head:
                # Decode through pretrained head
                with torch.no_grad():
                    pred_logits = decoder_head(pred_hs.to(device))  # (S, V)
            else:
                pred_logits = pred_hs  # Already vocab logits

            # Get targets
            targets = torch.tensor(
                [ids[n] if n < len(ids) else 0 for ids in generated_ids],
                dtype=torch.long,
            ).to(device)

            # Compute metrics
            metrics = compute_all_metrics(pred_logits, targets, k_values)
            results[l][n] = metrics

    return results
