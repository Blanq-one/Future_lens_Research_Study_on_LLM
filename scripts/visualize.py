"""
Generate Future Lens visualizations for specific prompts.

Usage:
    python -m scripts.visualize \
        --model EleutherAI/pythia-2.8b \
        --prompt "Marty McFly from" \
        --max_future 4
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import Config
from models.model_loader import load_model
from models.learned_prompt import LearnedPromptIntervention
from visualization.future_lens_viz import FutureLensVisualizer


def main():
    parser = argparse.ArgumentParser(description="Generate Future Lens visualization")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--prompt", type=str, default="Marty McFly from",
                        help="Input prompt to visualize")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="File with one prompt per line")
    parser.add_argument("--probes_dir", type=str, default="./outputs")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_future", type=int, default=4,
                        help="Number of future tokens to predict per cell")
    parser.add_argument("--prompt_length", type=int, default=10)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of layers to show (default: all)")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    # Load model
    cfg = Config()
    cfg.model.model_name = args.model
    cfg.model.load_in_8bit = args.load_in_8bit
    cfg.model.load_in_4bit = args.load_in_4bit
    cfg.model.dtype = torch.float32  # Need float32 for soft prompt hooks
    model, tokenizer = load_model(cfg.model)

    device = cfg.model.device

    # Load learned prompts
    learned_path = os.path.join(args.probes_dir, model_short, "probes", "learned_prompts.pt")
    if not os.path.exists(learned_path):
        print(f"ERROR: Learned prompts not found at {learned_path}")
        print("Please run: python -m scripts.train_learned_prompt first")
        sys.exit(1)

    intervention = LearnedPromptIntervention(
        model=model, tokenizer=tokenizer,
        prompt_length=args.prompt_length,
        device=device,
    )
    intervention.load(learned_path)

    # Create visualizer
    visualizer = FutureLensVisualizer(
        model=model, tokenizer=tokenizer,
        learned_prompts=intervention,
        device=device,
    )

    # Parse layers
    layers_to_show = None
    if args.layers:
        layers_to_show = [int(l) for l in args.layers.split(",")]

    # Get prompts
    prompts = [args.prompt]
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    # Generate visualizations
    viz_dir = os.path.join(args.output_dir, model_short, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\nGenerating Future Lens for: '{prompt}'")

        lens_data = visualizer.generate_lens_data(
            prompt,
            max_future_tokens=args.max_future,
            layers_to_show=layers_to_show,
        )

        # Sanitize prompt for filename
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:50]
        output_path = os.path.join(viz_dir, f"future_lens_{safe_name}.html")

        visualizer.render_html(lens_data, output_path=output_path)
        print(f"  Saved to {output_path}")

    print(f"\nAll visualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()
