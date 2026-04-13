"""
Evaluate all methods on test data and generate comparison results.

Usage:
    python -m scripts.evaluate_all \
        --model EleutherAI/pythia-2.8b \
        --max_n 5
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from config import Config
from models.model_loader import load_model, get_decoder_head
from models.hidden_states import load_extracted_data
from models.linear_probe import LinearProbeTrainer
from models.direct_vocab_probe import DirectVocabProbeTrainer
from models.mlp_probe import MLPProbeTrainer
from models.causal_intervention import FixedPromptIntervention
from models.learned_prompt import LearnedPromptIntervention
from evaluation.metrics import (
    evaluate_probe_all_layers,
    compute_all_metrics,
    confidence_calibration,
)
from visualization.future_lens_viz import generate_accuracy_plots_html


def main():
    parser = argparse.ArgumentParser(description="Evaluate all methods")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--probes_dir", type=str, default="./outputs")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_n", type=int, default=5)
    parser.add_argument("--eval_fixed_prompt", action="store_true",
                        help="Also evaluate fixed prompt intervention (slow)")
    parser.add_argument("--eval_learned_prompt", action="store_true",
                        help="Also evaluate learned prompt intervention (slow)")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test data
    test_path = os.path.join(args.data_dir, model_short, "test")
    print(f"Loading test data from {test_path}")
    test_data = load_extracted_data(test_path)

    hidden_states = test_data["hidden_states"]
    final_hs_future = test_data["final_hs_future"]
    generated_ids = test_data["generated_ids"]

    num_samples = len(hidden_states)
    num_layers = hidden_states[0].size(0) - 1
    hidden_dim = hidden_states[0].size(1)

    print(f"Test samples: {num_samples}, Layers: {num_layers}, Hidden dim: {hidden_dim}")

    # Load model for decoder head and interventions
    cfg = Config()
    cfg.model.model_name = args.model
    cfg.model.load_in_8bit = args.load_in_8bit
    cfg.model.load_in_4bit = args.load_in_4bit
    model, tokenizer = load_model(cfg.model)

    decoder_head = get_decoder_head(model)
    vocab_size = model.config.vocab_size

    probes_dir = os.path.join(args.probes_dir, model_short, "probes")
    all_results = {}

    # === Evaluate Linear Probes ===
    linear_path = os.path.join(probes_dir, "linear_hidden.pt")
    if os.path.exists(linear_path):
        print("\n" + "="*60)
        print("Evaluating LINEAR HIDDEN STATE PROBES")
        print("="*60)

        linear_trainer = LinearProbeTrainer(
            hidden_dim=hidden_dim, num_layers=num_layers,
            max_n=args.max_n, device=device,
        )
        linear_trainer.load(linear_path)

        linear_results = evaluate_probe_all_layers(
            linear_trainer, decoder_head,
            hidden_states, generated_ids,
            num_layers, args.max_n,
            use_decoder_head=True, device=device,
        )
        all_results["linear_hidden"] = linear_results
        _print_summary("Linear Hidden", linear_results, num_layers, args.max_n)

    # === Evaluate Direct Vocab Probes ===
    vocab_path = os.path.join(probes_dir, "direct_vocab.pt")
    if os.path.exists(vocab_path):
        print("\n" + "="*60)
        print("Evaluating DIRECT VOCAB PROBES")
        print("="*60)

        vocab_trainer = DirectVocabProbeTrainer(
            hidden_dim=hidden_dim, vocab_size=vocab_size,
            num_layers=num_layers, max_n=args.max_n, device=device,
        )
        vocab_trainer.load(vocab_path)

        vocab_results = evaluate_probe_all_layers(
            vocab_trainer, decoder_head,
            hidden_states, generated_ids,
            num_layers, args.max_n,
            use_decoder_head=False, device=device,
        )
        all_results["linear_vocab"] = vocab_results
        _print_summary("Linear Vocab", vocab_results, num_layers, args.max_n)

    # === Evaluate MLP Probes ===
    mlp_path = os.path.join(probes_dir, "mlp_hidden.pt")
    if os.path.exists(mlp_path):
        print("\n" + "="*60)
        print("Evaluating MLP PROBES")
        print("="*60)

        mlp_trainer = MLPProbeTrainer(
            hidden_dim=hidden_dim, num_layers=num_layers,
            max_n=args.max_n, device=device,
        )
        mlp_trainer.load(mlp_path)

        mlp_results = evaluate_probe_all_layers(
            mlp_trainer, decoder_head,
            hidden_states, generated_ids,
            num_layers, args.max_n,
            use_decoder_head=True, device=device,
        )
        all_results["mlp_hidden"] = mlp_results
        _print_summary("MLP Hidden", mlp_results, num_layers, args.max_n)

    # === Evaluate Fixed Prompt Intervention ===
    if args.eval_fixed_prompt:
        print("\n" + "="*60)
        print("Evaluating FIXED PROMPT INTERVENTION")
        print("="*60)

        fixed_intervention = FixedPromptIntervention(
            model=model, tokenizer=tokenizer,
            fixed_prompts=cfg.intervention.fixed_prompts,
            device=device,
        )
        fixed_results = fixed_intervention.evaluate(
            hidden_states, generated_ids, max_n=args.max_n,
        )
        all_results["fixed_prompt"] = fixed_results.get("average", {})

    # === Evaluate Learned Prompt Intervention ===
    learned_path = os.path.join(probes_dir, "learned_prompts.pt")
    if args.eval_learned_prompt and os.path.exists(learned_path):
        print("\n" + "="*60)
        print("Evaluating LEARNED PROMPT INTERVENTION")
        print("="*60)

        learned_intervention = LearnedPromptIntervention(
            model=model, tokenizer=tokenizer,
            prompt_length=cfg.intervention.learned_prompt_length,
            device=device,
        )
        learned_intervention.load(learned_path)

        learned_results = learned_intervention.evaluate(
            hidden_states, generated_ids,
            max_n=min(args.max_n, cfg.intervention.learned_prompt_max_n),
        )
        all_results["learned_prompt"] = _format_intervention_results(learned_results)

    # === Save results ===
    results_dir = os.path.join(args.output_dir, model_short, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save raw results
    results_path = os.path.join(results_dir, "all_results.json")
    _save_results_json(all_results, results_path)
    print(f"\nResults saved to {results_path}")

    # Generate comparison plots
    plots_path = os.path.join(results_dir, "accuracy_plots.html")
    generate_accuracy_plots_html(
        all_results,
        method_names=list(all_results.keys()),
        max_n=args.max_n,
        num_layers=num_layers,
        output_path=plots_path,
    )
    print(f"Plots saved to {plots_path}")

    # Print best results table (like Table 2 in the paper)
    print("\n" + "="*60)
    print("BEST RESULTS SUMMARY (Table 2 equivalent)")
    print("="*60)
    _print_best_results_table(all_results, args.max_n)


def _print_summary(name, results, num_layers, max_n):
    """Print a quick summary of results for a method."""
    for n in range(min(max_n + 1, 4)):
        best_layer = max(
            range(num_layers + 1),
            key=lambda l: results.get(l, {}).get(n, {}).get("precision@1", 0)
        )
        best_acc = results.get(best_layer, {}).get(n, {}).get("precision@1", 0)
        print(f"  N={n}: Best layer={best_layer}, Precision@1={best_acc:.3f}")


def _format_intervention_results(results):
    """Convert intervention results to same format as probe results."""
    formatted = {}
    for layer in results:
        formatted[layer] = {}
        for n in results[layer]:
            formatted[layer][n] = {"precision@1": results[layer][n]}
    return formatted


def _save_results_json(results, path):
    """Save results dict (convert any non-serializable keys)."""
    serializable = {}
    for method, method_results in results.items():
        serializable[method] = {}
        for layer, layer_results in method_results.items():
            serializable[method][str(layer)] = {}
            for n, metrics in layer_results.items():
                serializable[method][str(layer)][str(n)] = metrics
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def _print_best_results_table(results, max_n):
    """Print best results table like Table 2 in the paper."""
    header = f"{'Method':<20} {'Lens':>8}"
    for n in range(1, min(max_n + 1, 4)):
        header += f" {'N='+str(n):>8}"
    print(header)
    print("-" * len(header))

    for method in results:
        row = f"{method:<20}"
        # Lens = N=0 best accuracy
        for n in range(min(max_n + 1, 4)):
            best_acc = 0
            for layer in results[method]:
                acc = results[method][layer].get(n, {})
                if isinstance(acc, dict):
                    acc = acc.get("precision@1", 0)
                best_acc = max(best_acc, acc)
            row += f" {best_acc:>7.1%}"
        print(row)


if __name__ == "__main__":
    main()
