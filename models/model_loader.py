"""
Universal model loader for any HuggingFace causal LM.
Handles Llama, Gemma, Mistral, Mixtral, Pythia, GPT-J, etc.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from typing import Tuple, Dict, Any
from config import ModelConfig, MODEL_REGISTRY


def load_model(
    cfg: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal LM and its tokenizer from HuggingFace.

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {cfg.model_name}")
    print(f"  Device: {cfg.device}, Dtype: {cfg.dtype}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=cfg.dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif cfg.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Model kwargs
    model_kwargs = dict(
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=cfg.dtype,
        device_map="auto" if cfg.device == "cuda" else None,
        output_hidden_states=True,  # Critical: we need all hidden states
    )

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if cfg.use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            print("  Flash attention not available, using default.")

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    if cfg.device != "cuda" or (not cfg.load_in_4bit and not cfg.load_in_8bit):
        if hasattr(model, "to"):
            model = model.to(cfg.device)

    model.eval()
    print(f"  Model loaded. Layers: {model.config.num_hidden_layers}, "
          f"Hidden dim: {model.config.hidden_size}, "
          f"Vocab: {model.config.vocab_size}")

    return model, tokenizer


def get_model_info(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """Extract key architecture info from a loaded model."""
    config = model.config
    return {
        "num_layers": config.num_hidden_layers,
        "hidden_dim": config.hidden_size,
        "vocab_size": config.vocab_size,
        "num_heads": getattr(config, "num_attention_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "model_type": getattr(config, "model_type", "unknown"),
    }


def get_decoder_head(model: AutoModelForCausalLM) -> torch.nn.Module:
    """
    Extract the final decoder head (lm_head) from the model.
    This maps hidden states -> vocabulary logits.
    """
    if hasattr(model, "lm_head"):
        return model.lm_head
    elif hasattr(model, "embed_out"):
        return model.embed_out
    else:
        raise ValueError(
            f"Cannot find decoder head for model type: {type(model).__name__}. "
            "Please add support in model_loader.py"
        )


def get_embedding_layer(model: AutoModelForCausalLM) -> torch.nn.Module:
    """Extract the token embedding layer."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens  # Llama, Gemma, Mistral
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
        return model.gpt_neox.embed_in  # Pythia
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte  # GPT-J, GPT-2
    else:
        raise ValueError(f"Cannot find embedding layer for {type(model).__name__}")


def get_layers(model: AutoModelForCausalLM):
    """Get the list of transformer layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, Gemma, Mistral
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # Pythia
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-J, GPT-2
    else:
        raise ValueError(f"Cannot find layers for {type(model).__name__}")
