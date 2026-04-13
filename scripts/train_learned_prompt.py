"""
Train learned soft prompts for causal intervention.

This trains the paper's BEST method: optimizing a continuous prefix
that maximally surfaces future-token information from transplanted hidden states.

Usage:
    python -m scripts.train_learned_prompt \
        --model EleutherAI/pythia-2.8b \
        --max_n 3 \
        --epochs 50
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import Config
from models.model_loader import load_model
from models.hidden_states import load_extracted_data
from models.learned_prompt import LearnedPromptIntervention


def main():
    parser = argparse.ArgumentParser(description="Train learned prompts")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_n", type=int, default=3)
    parser.add_argument("--prompt_length", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_train_samples", type=int, default=10000,
                        help="Max samples to use for training (learned prompts are expensive)")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    # Load model (needed for forward passes during training)
    cfg = Config()
    cfg.model.model_name = args.model
    cfg.model.load_in_8bit = args.load_in_8bit
    cfg.model.load_in_4bit = args.load_in_4bit
    # Use float32 for learned prompt training (needs gradients)
    cfg.model.dtype = torch.float32

    model, tokenizer = load_model(cfg.model)

    # Load extracted data
    data_path = os.path.join(args.data_dir, model_short, "train")
    print(f"Loading training data from {data_path}")
    data = load_extracted_data(data_path)

    hidden_states = data["hidden_states"][:args.max_train_samples]
    generated_ids = data["generated_ids"][:args.max_train_samples]

    print(f"Using {len(hidden_states)} samples for learned prompt training")

    # Create intervention
    device = cfg.model.device
    intervention = LearnedPromptIntervention(
        model=model,
        tokenizer=tokenizer,
        prompt_length=args.prompt_length,
        device=device,
        dtype=torch.float32,
    )

    # Train for all layers
    print("\n" + "="*60)
    print("Training LEARNED PROMPTS (best method)")
    print("="*60)

    intervention.train_all_layers(
        hidden_states=hidden_states,
        generated_ids=generated_ids,
        max_n=args.max_n,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save
    save_dir = os.path.join(args.output_dir, model_short, "probes")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "learned_prompts.pt")
    intervention.save(save_path)
    print(f"\nSaved learned prompts to {save_path}")


if __name__ == "__main__":
    main()
