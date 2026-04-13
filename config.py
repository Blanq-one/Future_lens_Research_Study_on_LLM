"""
Central configuration for Future Lens v2.
All hyperparameters, model configs, and paths are defined here.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """Configuration for the target LLM."""
    model_name: str = "EleutherAI/pythia-2.8b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    use_flash_attention: bool = False


@dataclass
class DataConfig:
    """Configuration for data extraction and processing."""
    dataset_name: str = "monology/pile-uncopyrighted"
    dataset_split: str = "train"
    num_train_samples: int = 10000
    num_test_samples: int = 1000
    max_context_length: int = 512
    min_context_length: int = 64
    seed: int = 42
    # Only sample from positions where model predicts correctly
    filter_correct_predictions: bool = True


@dataclass
class ProbeConfig:
    """Configuration for probe training."""
    # How many tokens ahead to predict (N=0 is next-token, N=1 is two ahead, etc.)
    max_n: int = 5
    # Probe types
    train_linear_hidden: bool = True    # Linear: h_T^l -> h_{T+N}^L
    train_linear_vocab: bool = True     # Linear: h_T^l -> vocabulary logits
    train_mlp_hidden: bool = True       # MLP: h_T^l -> h_{T+N}^L (extension)
    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 10
    patience: int = 3  # Early stopping patience
    # MLP probe architecture
    mlp_hidden_dim: int = 4096
    mlp_num_layers: int = 2
    mlp_dropout: float = 0.1


@dataclass
class InterventionConfig:
    """Configuration for causal intervention experiments."""
    # Fixed prompts for causal intervention
    fixed_prompts: List[str] = field(default_factory=lambda: [
        'Hello! Could you please tell me more about "',
        'The multi-tokens present here are "',
        'The concepts in this hidden state listed are: (',
        'This state is describing about the following concept:',
    ])
    # Learned prompt
    learned_prompt_length: int = 10
    learned_prompt_lr: float = 1e-2
    learned_prompt_epochs: int = 50
    learned_prompt_batch_size: int = 64
    learned_prompt_train_samples: int = 10000
    # How many tokens ahead to optimize for
    learned_prompt_max_n: int = 5


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    precision_k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    # Confidence calibration bins
    confidence_bins: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 0.9, 1.0])


@dataclass
class VisualizationConfig:
    """Configuration for Future Lens visualization."""
    max_tokens_display: int = 10
    max_future_tokens: int = 5
    colormap: str = "YlOrRd"
    output_format: str = "html"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./data/cache"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def model_short_name(self) -> str:
        return self.model.model_name.split("/")[-1].lower().replace("-", "_")


# MODEL REGISTRY: known configurations for supported models
MODEL_REGISTRY = {
    # Llama 3.1
    "meta-llama/Llama-3.1-8B": {"num_layers": 32, "hidden_dim": 4096, "vocab_size": 128256},
    "meta-llama/Llama-3.1-70B": {"num_layers": 80, "hidden_dim": 8192, "vocab_size": 128256},
    # Gemma 2
    "google/gemma-2-9b": {"num_layers": 42, "hidden_dim": 3584, "vocab_size": 256000},
    "google/gemma-2-27b": {"num_layers": 46, "hidden_dim": 4608, "vocab_size": 256000},
    # Mistral
    "mistralai/Mistral-7B-v0.3": {"num_layers": 32, "hidden_dim": 4096, "vocab_size": 32768},
    "mistralai/Mixtral-8x7B-v0.1": {"num_layers": 32, "hidden_dim": 4096, "vocab_size": 32000},
    # Pythia
    "EleutherAI/pythia-6.9b": {"num_layers": 32, "hidden_dim": 4096, "vocab_size": 50304},
    "EleutherAI/pythia-2.8b": {"num_layers": 32, "hidden_dim": 2560, "vocab_size": 50304},
    "EleutherAI/pythia-1.4b": {"num_layers": 24, "hidden_dim": 2048, "vocab_size": 50304},
    "EleutherAI/pythia-410m": {"num_layers": 24, "hidden_dim": 1024, "vocab_size": 50304},
    # GPT-J (original paper)
    "EleutherAI/gpt-j-6B": {"num_layers": 28, "hidden_dim": 4096, "vocab_size": 50400},
}
