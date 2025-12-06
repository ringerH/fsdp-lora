import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "distilgpt2"  # small, safe for T4
MAX_LENGTH = 128
BATCH_SIZE = 4
LR = 5e-5
EPOCHS = 2
DATA_FILE = "data/sample.txt"


def setup_distributed():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def cleanup_distributed():
    dist.destroy_process_group()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=128):
        self.examples = []

        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        # Very simple split: each block separated by blank line or pattern
        blocks = [b.strip() for b in raw.split("### Instruction:") if b.strip()]
        for block in blocks:
            # We expect "instruction ... ### Response: ..."
            if "### Response:" not in block:
                continue
            instr_part, resp_part = block.split("### Response:", 1)
            instr = instr_part.strip()
            resp = resp_part.strip()

            text = f"Instruction: {instr}\nResponse: {resp}"
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append(
                {
                    "input_ids": enc["input_ids"][0],
                    "attention_mask": enc["attention_mask"][0],
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # For simple language modeling, labels = input_ids
        return {
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["input_ids"],
        }


def build_model(device):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model = FSDP(model)
    return model


def build_dataloader(tokenizer, world_size, rank):
    dataset = InstructionDataset(tokenizer, DATA_FILE, max_length=MAX_LENGTH)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=True,
    )
    return loader, sampler


def main():
    rank, local_rank, world_size, device = setup_distributed()

    if rank == 0:
        print(f"World size: {world_size}, device: {device}, model: {MODEL_NAME}")

    tokenizer = load_tokenizer()
    model = build_model(device)

    train_loader, train_sampler = build_dataloader(tokenizer, world_size, rank)

    optimizer = optim.AdamW(model.parameters(), lr=LR)

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
        print("Training finished.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
