# Future Lens v2: Anticipating Subsequent Tokens from Hidden States

A modernized reimplementation and extension of "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State" (Pal et al., 2023).

## Improvements Over Original
- **Modern Models**: Llama 3.1, Gemma 2, Mistral, Mixtral, Pythia (original: GPT-J-6B only)
- **Extended Horizon**: Up to N=10 tokens ahead (original: N=3)
- **Nonlinear Probes**: MLP probes alongside linear probes
- **Base vs Instruct**: Compare base and instruction-tuned variants
- **Interactive Visualization**: HTML-based Future Lens heatmap

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline
python main.py --model EleutherAI/pythia-2.8b --num_train 10000 --num_test 1000 --max_n 5

# Or step-by-step:
python -m scripts.extract_data --model EleutherAI/pythia-2.8b --num_samples 10000
python -m scripts.train_probes --model EleutherAI/pythia-2.8b --max_n 5
python -m scripts.train_learned_prompt --model EleutherAI/pythia-2.8b --max_n 3
python -m scripts.evaluate_all --model EleutherAI/pythia-2.8b
python -m scripts.visualize --model EleutherAI/pythia-2.8b --prompt "Marty McFly from"
```
