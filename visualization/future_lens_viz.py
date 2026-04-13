"""
Future Lens Visualization.

Creates an interactive HTML visualization showing, for each hidden state
(layer × token position), the most likely sequence of future tokens
and their confidence levels. Inspired by Figure 6 of the paper.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json

from models.learned_prompt import LearnedPromptIntervention


class FutureLensVisualizer:
    """
    Generates the Future Lens visualization for a given prompt.

    For each (layer, token_position) pair, uses the learned prompt
    to predict the next several tokens and their confidence levels.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        learned_prompts: LearnedPromptIntervention,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learned_prompts = learned_prompts
        self.device = device
        self.num_layers = model.config.num_hidden_layers

    @torch.no_grad()
    def generate_lens_data(
        self,
        prompt: str,
        max_future_tokens: int = 4,
        layers_to_show: Optional[List[int]] = None,
    ) -> Dict:
        """
        Generate Future Lens data for a prompt.

        For each layer and token position, predict the next max_future_tokens
        using the learned prompt intervention.

        Args:
            prompt: input text
            max_future_tokens: how many future tokens to show per cell
            layers_to_show: which layers to include (None = all)

        Returns:
            Dict with visualization data
        """
        # Tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        seq_len = input_ids.size(1)

        # Get token strings
        token_strings = [
            self.tokenizer.decode([input_ids[0, i].item()])
            for i in range(seq_len)
        ]

        # Get hidden states at all layers for all positions
        outputs = self.model(input_ids, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)

        if layers_to_show is None:
            layers_to_show = list(range(self.num_layers))

        # For each (layer, position), predict future tokens
        lens_data = []

        for l in layers_to_show:
            layer_data = []
            for t in range(seq_len):
                h_l_t = all_hidden_states[l + 1][0, t].float()  # (hidden_dim,)

                future_tokens = []
                future_probs = []
                continuation = []

                for n in range(max_future_tokens):
                    logits = self.learned_prompts.predict(
                        h_l_t, l,
                        continuation if continuation else None,
                    )
                    probs = torch.softmax(logits, dim=-1)
                    top_prob, top_idx = probs.max(dim=-1)

                    token_str = self.tokenizer.decode([top_idx.item()])
                    future_tokens.append(token_str)
                    future_probs.append(top_prob.item())
                    continuation.append(top_idx.item())

                layer_data.append({
                    "tokens": future_tokens,
                    "probs": future_probs,
                    "avg_confidence": np.mean(future_probs),
                })

            lens_data.append({
                "layer": l,
                "predictions": layer_data,
            })

        return {
            "prompt": prompt,
            "token_strings": token_strings,
            "layers_shown": layers_to_show,
            "num_future_tokens": max_future_tokens,
            "lens_data": lens_data,
        }

    def render_html(
        self,
        lens_data: Dict,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Render the Future Lens as an interactive HTML page.

        Args:
            lens_data: output from generate_lens_data
            output_path: if provided, save HTML to this path

        Returns:
            HTML string
        """
        token_strings = lens_data["token_strings"]
        layers_shown = lens_data["layers_shown"]
        data = lens_data["lens_data"]
        prompt = lens_data["prompt"]

        # Build the HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Future Lens: {prompt}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        background: #0a0a0a;
        color: #e0e0e0;
        padding: 24px;
    }}
    h1 {{
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #ffffff;
    }}
    .subtitle {{
        font-size: 13px;
        color: #888;
        margin-bottom: 24px;
    }}
    .prompt-display {{
        font-size: 14px;
        color: #4fc3f7;
        margin-bottom: 20px;
        padding: 12px 16px;
        background: #1a1a2e;
        border-radius: 8px;
        border: 1px solid #2a2a4e;
    }}
    .grid-container {{
        overflow-x: auto;
        padding-bottom: 16px;
    }}
    table {{
        border-collapse: separate;
        border-spacing: 2px;
        font-size: 11px;
    }}
    th {{
        padding: 6px 10px;
        text-align: center;
        background: #1a1a2e;
        color: #aaa;
        font-weight: 500;
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    th.layer-header {{
        text-align: right;
        padding-right: 12px;
        position: sticky;
        left: 0;
        z-index: 20;
        background: #1a1a2e;
    }}
    td {{
        padding: 4px 6px;
        text-align: center;
        cursor: pointer;
        border-radius: 3px;
        transition: all 0.15s;
        position: relative;
        min-width: 80px;
    }}
    td:hover {{
        transform: scale(1.05);
        z-index: 5;
        box-shadow: 0 0 12px rgba(255,255,255,0.15);
    }}
    .cell-text {{
        font-size: 10px;
        line-height: 1.3;
        max-width: 90px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .tooltip {{
        display: none;
        position: absolute;
        bottom: 105%;
        left: 50%;
        transform: translateX(-50%);
        background: #222;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 100;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        text-align: left;
        line-height: 1.6;
    }}
    td:hover .tooltip {{ display: block; }}
    .tooltip-token {{
        font-weight: 600;
    }}
    .tooltip-prob {{
        color: #888;
        margin-left: 8px;
    }}
    .legend {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 20px;
        font-size: 12px;
        color: #888;
    }}
    .legend-bar {{
        width: 200px;
        height: 14px;
        border-radius: 3px;
        background: linear-gradient(to right, #1a1a2e, #e65100);
    }}
</style>
</head>
<body>
<h1>Future Lens Visualization</h1>
<div class="subtitle">Predicted future tokens from each hidden state (layer × position)</div>
<div class="prompt-display">Prompt: "{prompt}"</div>
<div class="grid-container">
<table>
<thead>
<tr>
<th class="layer-header">Layer</th>
"""

        # Column headers: token strings
        for i, tok in enumerate(token_strings):
            display = tok.replace("<", "&lt;").replace(">", "&gt;").strip() or "·"
            html += f'<th>{display}</th>\n'
        html += '</tr></thead><tbody>\n'

        # Rows: one per layer
        for layer_entry in data:
            l = layer_entry["layer"]
            preds = layer_entry["predictions"]
            html += f'<tr><th class="layer-header">L{l+1}</th>\n'

            for t, cell in enumerate(preds):
                conf = cell["avg_confidence"]
                # Color: interpolate from dark blue to orange based on confidence
                r = int(30 + 200 * conf)
                g = int(30 + 60 * conf)
                b = int(50 - 20 * conf)
                bg_color = f"rgb({min(r,255)},{min(g,255)},{max(b,0)})"
                text_color = "#fff" if conf > 0.3 else "#aaa"

                # Cell text: show first predicted token
                cell_text = cell["tokens"][0].replace("<", "&lt;").replace(">", "&gt;").strip() or "·"

                # Tooltip: show all future tokens with probabilities
                tooltip_lines = ""
                for n, (tok, prob) in enumerate(zip(cell["tokens"], cell["probs"])):
                    tok_clean = tok.replace("<", "&lt;").replace(">", "&gt;").strip() or "·"
                    tooltip_lines += (
                        f'<div><span class="tooltip-token">N={n}:</span> '
                        f'{tok_clean}'
                        f'<span class="tooltip-prob">({prob:.1%})</span></div>'
                    )

                html += (
                    f'<td style="background:{bg_color};color:{text_color}">'
                    f'<div class="cell-text">{cell_text}</div>'
                    f'<div class="tooltip">{tooltip_lines}</div>'
                    f'</td>\n'
                )

            html += '</tr>\n'

        html += """</tbody></table></div>
<div class="legend">
    <span>Low confidence</span>
    <div class="legend-bar"></div>
    <span>High confidence</span>
</div>
</body></html>"""

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
            print(f"Saved visualization to {output_path}")

        return html


def generate_accuracy_plots_html(
    results: Dict,
    method_names: List[str],
    max_n: int,
    num_layers: int,
    bigram_baselines: Optional[Dict[int, float]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate an HTML page with accuracy plots (Precision@1 vs layer)
    for each N value, similar to Figure 4 of the paper.

    Args:
        results: dict mapping method_name -> {layer -> {n -> {metrics}}}
        method_names: list of method names to plot
        max_n: maximum N value
        num_layers: number of layers
        bigram_baselines: optional bigram accuracy per N
        output_path: save path

    Returns:
        HTML string with embedded Chart.js plots
    """
    # Prepare data for charts
    charts_data = {}
    for n in range(max_n + 1):
        charts_data[n] = {}
        for method in method_names:
            if method in results:
                layers = sorted(results[method].keys())
                accs = [
                    results[method][l].get(n, {}).get("precision@1", 0)
                    for l in layers
                ]
                charts_data[n][method] = {
                    "layers": [int(l) for l in layers],
                    "accuracies": accs,
                }

    charts_json = json.dumps(charts_data)
    bigram_json = json.dumps(bigram_baselines or {})

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>Future Lens Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    body {{ font-family: sans-serif; background: #fafafa; padding: 24px; }}
    h1 {{ margin-bottom: 24px; }}
    .charts-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
        gap: 24px;
    }}
    .chart-container {{
        background: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
</style>
</head><body>
<h1>Future Lens: Precision@1 by Layer</h1>
<div class="charts-grid" id="charts"></div>
<script>
const data = {charts_json};
const bigrams = {bigram_json};
const colors = {{
    'learned_prompt': '#d32f2f',
    'fixed_prompt': '#1976d2',
    'linear_hidden': '#388e3c',
    'linear_vocab': '#f57c00',
    'mlp_hidden': '#7b1fa2',
}};

const container = document.getElementById('charts');

for (const [n, methods] of Object.entries(data)) {{
    const div = document.createElement('div');
    div.className = 'chart-container';
    div.innerHTML = `<canvas id="chart_${{n}}"></canvas>`;
    container.appendChild(div);

    const datasets = [];
    for (const [method, vals] of Object.entries(methods)) {{
        datasets.push({{
            label: method,
            data: vals.accuracies,
            borderColor: colors[method] || '#999',
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 2,
        }});
    }}

    // Bigram baseline as horizontal line
    if (bigrams[n] !== undefined) {{
        const numPoints = Object.values(methods)[0]?.layers.length || 28;
        datasets.push({{
            label: 'Bigram baseline',
            data: Array(numPoints).fill(bigrams[n]),
            borderColor: '#999',
            borderDash: [5, 5],
            backgroundColor: 'transparent',
            pointRadius: 0,
        }});
    }}

    const layers = Object.values(methods)[0]?.layers || [];
    new Chart(document.getElementById(`chart_${{n}}`), {{
        type: 'line',
        data: {{ labels: layers, datasets }},
        options: {{
            responsive: true,
            plugins: {{
                title: {{ display: true, text: `N = ${{n}}` }},
                legend: {{ position: 'bottom' }},
            }},
            scales: {{
                x: {{ title: {{ display: true, text: 'Layer' }} }},
                y: {{ title: {{ display: true, text: 'Precision@1' }}, min: 0, max: 1 }},
            }},
        }},
    }});
}}
</script>
</body></html>"""

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

    return html
