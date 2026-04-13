"""
Extract hidden states and future token data from a model + dataset.

Usage:
    python -m scripts.extract_data \
        --model EleutherAI/pythia-2.8b \
        --num_samples 10000 \
        --max_n 5
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from config import Config, ModelConfig, DataConfig
from models.model_loader import load_model
from models.hidden_states import extract_dataset, save_extracted_data


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from model")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--dataset", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=1000)
    parser.add_argument("--max_n", type=int, default=5)
    parser.add_argument("--max_context_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./data/cache")
    parser.add_argument("--no_filter", action="store_true",
                        help="Don't filter for correct predictions")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup config
    cfg = Config()
    cfg.model.model_name = args.model
    cfg.model.load_in_8bit = args.load_in_8bit
    cfg.model.load_in_4bit = args.load_in_4bit
    cfg.data.num_train_samples = args.num_samples
    cfg.data.num_test_samples = args.num_test
    cfg.data.max_context_length = args.max_context_length
    cfg.data.seed = args.seed

    # Load model
    model, tokenizer = load_model(cfg.model)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset, split="train", streaming=True)
        texts = []
        for i, sample in enumerate(dataset):
            if i >= args.num_samples + args.num_test + 5000:  # Extra buffer
                break
            text = sample.get("text", sample.get("content", ""))
            if len(text) > 100:
                texts.append(text)
    except Exception as e:
        print(f"Could not load dataset {args.dataset}: {e}")
        print("Falling back to synthetic data for testing...")
        texts = _generate_fallback_texts(args.num_samples + args.num_test)

    print(f"Loaded {len(texts)} texts")

    # Split into train and test texts
    train_texts = texts[:len(texts) - args.num_test]
    test_texts = texts[len(texts) - args.num_test:]

    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    device = cfg.model.device
    dtype = cfg.model.dtype

    # Extract training data
    print("\n=== Extracting TRAINING data ===")
    train_data = extract_dataset(
        model, tokenizer, train_texts,
        max_n=args.max_n,
        max_context_length=args.max_context_length,
        filter_correct=not args.no_filter,
        device=device, dtype=dtype,
        max_samples=args.num_samples,
    )
    train_path = os.path.join(args.output_dir, model_short, "train")
    save_extracted_data(train_data, train_path)

    # Extract test data
    print("\n=== Extracting TEST data ===")
    test_data = extract_dataset(
        model, tokenizer, test_texts,
        max_n=args.max_n,
        max_context_length=args.max_context_length,
        filter_correct=not args.no_filter,
        device=device, dtype=dtype,
        max_samples=args.num_test,
    )
    test_path = os.path.join(args.output_dir, model_short, "test")
    save_extracted_data(test_data, test_path)

    print(f"\nDone! Train: {len(train_data['hidden_states'])} samples, "
          f"Test: {len(test_data['hidden_states'])} samples")
    print(f"Saved to {os.path.join(args.output_dir, model_short)}")


def _generate_fallback_texts(n: int):
    """Generate simple fallback texts for testing when dataset isn't available."""
    templates = [
        "The capital of France is Paris, which is known for the Eiffel Tower and its beautiful architecture.",
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
        "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 60 million square miles.",
        "Albert Einstein developed the theory of relativity, which fundamentally changed our understanding of physics.",
        "Python is a popular programming language known for its simplicity and readability in software development.",
        "The Great Wall of China is one of the most impressive architectural feats in human history.",
        "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
        "The human genome contains approximately three billion base pairs of DNA in each cell.",
        "Shakespeare wrote many famous plays including Hamlet, Romeo and Juliet, and A Midsummer Night's Dream.",
        "Climate change is one of the most pressing challenges facing humanity in the twenty-first century.",
    ]
    import random
    random.seed(42)
    return [random.choice(templates) for _ in range(n)]


if __name__ == "__main__":
    main()
