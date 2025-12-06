# src/lora_utils.py
from peft import LoraConfig, get_peft_model, TaskType


def print_trainable_params(model, rank: int = 0):
    if rank != 0:
        return
    trainable, total = 0, 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    pct = 100 * trainable / total if total > 0 else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def add_lora_to_model(
    base_model,
    rank: int,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules=None,
):
    if target_modules is None:
        target_modules = ["c_attn", "c_proj"]  # GPT-2 style modules

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    lora_model = get_peft_model(base_model, config)
    print_trainable_params(lora_model, rank=rank)
    return lora_model
