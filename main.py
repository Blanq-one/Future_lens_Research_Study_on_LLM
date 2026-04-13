"""
Future Lens v2 — Full Pipeline Orchestrator.

Runs the complete pipeline:
1. Extract hidden states from model on dataset
2. Train all probes (linear, vocab, MLP)
3. Train learned prompts (best method)
4. Evaluate all methods
5. Generate visualizations

Usage:
    python main.py --model EleutherAI/pythia-2.8b --num_train 10000 --num_test 1000

For large models, consider running steps individually:
    python -m scripts.extract_data --model meta-llama/Llama-3.1-8B --load_in_4bit
    python -m scripts.train_probes --model meta-llama/Llama-3.1-8B
    python -m scripts.train_learned_prompt --model meta-llama/Llama-3.1-8B --load_in_4bit
    python -m scripts.evaluate_all --model meta-llama/Llama-3.1-8B --load_in_4bit
    python -m scripts.visualize --model meta-llama/Llama-3.1-8B --prompt "The president of"
"""

import argparse
import os
import sys
import time
import torch

from config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Future Lens v2: Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small model
  python main.py --model EleutherAI/pythia-410m --num_train 1000 --num_test 100 --max_n 3

  # Full run with Pythia 2.8B
  python main.py --model EleutherAI/pythia-2.8b --num_train 10000 --num_test 1000

  # Large model with quantization
  python main.py --model meta-llama/Llama-3.1-8B --load_in_4bit --num_train 5000

  # Skip expensive steps
  python main.py --model EleutherAI/pythia-2.8b --skip_learned_prompt --skip_intervention
        """,
    )

    # Model
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b",
                        help="HuggingFace model name")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")

    # Data
    parser.add_argument("--num_train", type=int, default=10000,
                        help="Number of training samples to extract")
    parser.add_argument("--num_test", type=int, default=1000,
                        help="Number of test samples")
    parser.add_argument("--dataset", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--max_context_length", type=int, default=512)

    # Probes
    parser.add_argument("--max_n", type=int, default=5,
                        help="Maximum future token offset to predict")
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--probe_lr", type=float, default=1e-3)

    # Learned prompts
    parser.add_argument("--learned_prompt_epochs", type=int, default=50)
    parser.add_argument("--learned_prompt_lr", type=float, default=1e-2)
    parser.add_argument("--prompt_length", type=int, default=10)
    parser.add_argument("--learned_prompt_max_n", type=int, default=5)
    parser.add_argument("--learned_prompt_samples", type=int, default=10000)

    # Skip flags
    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument("--skip_probes", action="store_true")
    parser.add_argument("--skip_learned_prompt", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--skip_visualization", action="store_true")
    parser.add_argument("--skip_intervention", action="store_true",
                        help="Skip fixed prompt intervention evaluation")

    # Visualization
    parser.add_argument("--viz_prompt", type=str, default="Marty McFly from",
                        help="Prompt for Future Lens visualization")
    parser.add_argument("--viz_max_future", type=int, default=5)

    # Paths
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    args = parser.parse_args()
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    print("="*70)
    print("  FUTURE LENS v2 — Full Pipeline")
    print(f"  Model: {args.model}")
    print(f"  Train: {args.num_train}, Test: {args.num_test}, Max N: {args.max_n}")
    print("="*70)

    start_time = time.time()

    # ================================================================
    # STEP 1: Extract hidden states
    # ================================================================
    if not args.skip_extraction:
        print("\n\n" + "█"*70)
        print("  STEP 1: Extracting hidden states from model")
        print("█"*70)

        cmd = (
            f"python -m scripts.extract_data "
            f"--model {args.model} "
            f"--dataset {args.dataset} "
            f"--num_samples {args.num_train} "
            f"--num_test {args.num_test} "
            f"--max_n {args.max_n} "
            f"--max_context_length {args.max_context_length} "
            f"--output_dir {args.data_dir} "
        )
        if args.load_in_8bit:
            cmd += " --load_in_8bit"
        if args.load_in_4bit:
            cmd += " --load_in_4bit"

        ret = os.system(cmd)
        if ret != 0:
            print("ERROR: Data extraction failed!")
            sys.exit(1)
    else:
        print("\n  [Skipping extraction]")

    # ================================================================
    # STEP 2: Train probes
    # ================================================================
    if not args.skip_probes:
        print("\n\n" + "█"*70)
        print("  STEP 2: Training probes (linear, vocab, MLP)")
        print("█"*70)

        cmd = (
            f"python -m scripts.train_probes "
            f"--model {args.model} "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--max_n {args.max_n} "
            f"--epochs {args.probe_epochs} "
            f"--batch_size {args.probe_batch_size} "
            f"--lr {args.probe_lr} "
        )
        ret = os.system(cmd)
        if ret != 0:
            print("WARNING: Probe training had issues")
    else:
        print("\n  [Skipping probe training]")

    # ================================================================
    # STEP 3: Train learned prompts
    # ================================================================
    if not args.skip_learned_prompt:
        print("\n\n" + "█"*70)
        print("  STEP 3: Training learned prompts (best method)")
        print("█"*70)

        cmd = (
            f"python -m scripts.train_learned_prompt "
            f"--model {args.model} "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--max_n {args.learned_prompt_max_n} "
            f"--epochs {args.learned_prompt_epochs} "
            f"--lr {args.learned_prompt_lr} "
            f"--prompt_length {args.prompt_length} "
            f"--max_train_samples {args.learned_prompt_samples} "
        )
        if args.load_in_8bit:
            cmd += " --load_in_8bit"
        if args.load_in_4bit:
            cmd += " --load_in_4bit"

        ret = os.system(cmd)
        if ret != 0:
            print("WARNING: Learned prompt training had issues")
    else:
        print("\n  [Skipping learned prompt training]")

    # ================================================================
    # STEP 4: Evaluate
    # ================================================================
    if not args.skip_evaluation:
        print("\n\n" + "█"*70)
        print("  STEP 4: Evaluating all methods")
        print("█"*70)

        cmd = (
            f"python -m scripts.evaluate_all "
            f"--model {args.model} "
            f"--data_dir {args.data_dir} "
            f"--probes_dir {args.output_dir} "
            f"--output_dir {args.output_dir} "
            f"--max_n {args.max_n} "
        )
        if not args.skip_intervention:
            cmd += " --eval_fixed_prompt"
        if not args.skip_learned_prompt:
            cmd += " --eval_learned_prompt"
        if args.load_in_8bit:
            cmd += " --load_in_8bit"
        if args.load_in_4bit:
            cmd += " --load_in_4bit"

        ret = os.system(cmd)
        if ret != 0:
            print("WARNING: Evaluation had issues")
    else:
        print("\n  [Skipping evaluation]")

    # ================================================================
    # STEP 5: Visualize
    # ================================================================
    if not args.skip_visualization and not args.skip_learned_prompt:
        print("\n\n" + "█"*70)
        print("  STEP 5: Generating Future Lens visualization")
        print("█"*70)

        cmd = (
            f"python -m scripts.visualize "
            f"--model {args.model} "
            f"--prompt \"{args.viz_prompt}\" "
            f"--probes_dir {args.output_dir} "
            f"--output_dir {args.output_dir} "
            f"--max_future {args.viz_max_future} "
            f"--prompt_length {args.prompt_length} "
        )
        if args.load_in_8bit:
            cmd += " --load_in_8bit"
        if args.load_in_4bit:
            cmd += " --load_in_4bit"

        ret = os.system(cmd)
    else:
        print("\n  [Skipping visualization]")

    # Done!
    elapsed = time.time() - start_time
    print("\n\n" + "="*70)
    print(f"  PIPELINE COMPLETE — {elapsed/60:.1f} minutes")
    print(f"  Results: {os.path.join(args.output_dir, model_short)}")
    print("="*70)


if __name__ == "__main__":
    main()
