"""
Train linear and MLP probes on extracted hidden state data.

Usage:
    python -m scripts.train_probes \
        --model EleutherAI/pythia-2.8b \
        --max_n 5 \
        --epochs 20
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import Config
from models.model_loader import load_model, get_model_info
from models.hidden_states import load_extracted_data
from models.linear_probe import LinearProbeTrainer
from models.direct_vocab_probe import DirectVocabProbeTrainer
from models.mlp_probe import MLPProbeTrainer


def main():
    parser = argparse.ArgumentParser(description="Train probes")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_n", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no_linear", action="store_true")
    parser.add_argument("--no_vocab", action="store_true")
    parser.add_argument("--no_mlp", action="store_true")
    parser.add_argument("--mlp_hidden", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    # Load extracted data
    data_path = os.path.join(args.data_dir, model_short, "train")
    print(f"Loading training data from {data_path}")
    data = load_extracted_data(data_path)

    hidden_states = data["hidden_states"]
    final_hs_future = data["final_hs_future"]
    generated_ids = data["generated_ids"]

    num_samples = len(hidden_states)
    num_layers_plus_one = hidden_states[0].size(0)  # includes embedding
    num_layers = num_layers_plus_one - 1
    hidden_dim = hidden_states[0].size(1)

    print(f"Loaded {num_samples} samples")
    print(f"Layers: {num_layers}, Hidden dim: {hidden_dim}")

    # Need vocab size for direct vocab probe - load model config
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.model)
    vocab_size = model_config.vocab_size

    save_dir = os.path.join(args.output_dir, model_short, "probes")
    os.makedirs(save_dir, exist_ok=True)

    # === Train Linear Probes (hidden state prediction) ===
    if not args.no_linear:
        print("\n" + "="*60)
        print("Training LINEAR PROBES (h_T^l -> h_{T+N}^L)")
        print("="*60)

        linear_trainer = LinearProbeTrainer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_n=args.max_n,
            lr=args.lr,
            weight_decay=1e-4,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
        )
        linear_trainer.train(hidden_states, final_hs_future)
        linear_trainer.save(os.path.join(save_dir, "linear_hidden.pt"))

    # === Train Direct Vocabulary Probes ===
    if not args.no_vocab:
        print("\n" + "="*60)
        print("Training DIRECT VOCAB PROBES (h_T^l -> logits)")
        print("="*60)

        vocab_trainer = DirectVocabProbeTrainer(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            max_n=args.max_n,
            lr=args.lr,
            weight_decay=1e-4,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
        )
        vocab_trainer.train(hidden_states, generated_ids)
        vocab_trainer.save(os.path.join(save_dir, "direct_vocab.pt"))

    # === Train MLP Probes (extension) ===
    if not args.no_mlp:
        print("\n" + "="*60)
        print("Training MLP PROBES (h_T^l -> h_{T+N}^L, nonlinear)")
        print("="*60)

        mlp_trainer = MLPProbeTrainer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_n=args.max_n,
            mlp_hidden_dim=min(args.mlp_hidden, hidden_dim * 2),
            mlp_num_layers=2,
            dropout=0.1,
            lr=args.lr,
            weight_decay=1e-4,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
        )
        mlp_trainer.train(hidden_states, final_hs_future)
        mlp_trainer.save(os.path.join(save_dir, "mlp_hidden.pt"))

    print(f"\nAll probes saved to {save_dir}")


if __name__ == "__main__":
    main()
