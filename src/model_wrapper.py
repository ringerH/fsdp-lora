# src/model_wrapper.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model
