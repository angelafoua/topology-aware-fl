"""RoBERTa-Large + LoRA model builder."""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .lora import LoRAConfig, inject_lora


MODEL_NAME = "roberta-large"


def build_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def build_model(num_labels: int, lora_cfg: LoRAConfig):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    # Freeze backbone + classification head (paper freezes the head).
    for p in model.parameters():
        p.requires_grad = False
    inject_lora(model, lora_cfg)
    return model
