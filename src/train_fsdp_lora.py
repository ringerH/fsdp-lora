# src/train_fsdp_lora.py
import torch
import torch.optim as optim

from src.fsdp_utils import setup_distributed, cleanup_distributed, wrap_with_fsdp
from src.model_wrapper import load_tokenizer, load_base_model
from src.lora_utils import add_lora_to_model
from src.dataset import build_dataloader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


MODEL_NAME = "distilgpt2"
MAX_LENGTH = 128
BATCH_SIZE = 4
LR = 5e-4
EPOCHS = 2
DATA_FILE = "data/sample.txt"


def generate_demo(model, tokenizer, device, rank: int):
    if rank != 0:
        return

    if isinstance(model, FSDP):
        model_to_use = model.module
    else:
        model_to_use = model

    model_to_use.eval()

    prompt_instr = "Explain the difference between supervised and unsupervised learning."
    prompt = f"Instruction: {prompt_instr}\nResponse:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model_to_use.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== DEMO GENERATION ===")
    print(text)
    print("=======================\n")


def main():
    rank, local_rank, world_size, device = setup_distributed()

    if rank == 0:
        print(f"World size: {world_size}, device: {device}, model: {MODEL_NAME}")
        print("Using LoRA + FSDP (world_size may be 1 on Kaggle).")

    tokenizer = load_tokenizer(MODEL_NAME)
    base_model = load_base_model(MODEL_NAME, device)

    lora_model = add_lora_to_model(base_model, rank=rank)

    model = wrap_with_fsdp(lora_model, use_orig_params=True)

    train_loader, train_sampler = build_dataloader(
        tokenizer=tokenizer,
        file_path=DATA_FILE,
        max_length=MAX_LENGTH,
        world_size=world_size,
        rank=rank,
        batch_size=BATCH_SIZE,
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    model.train()
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if rank == 0 and step % 5 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    if rank == 0:
        print("Training finished (LoRA + FSDP).")
        generate_demo(model, tokenizer, device, rank)

    cleanup_distributed()


if __name__ == "__main__":
    main()
