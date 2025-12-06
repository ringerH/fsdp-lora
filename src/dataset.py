# src/dataset.py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length: int = 128):
        self.examples = []

        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        blocks = [b.strip() for b in raw.split("### Instruction:") if b.strip()]
        for block in blocks:
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
        return {
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["input_ids"],
        }


def build_dataloader(
    tokenizer,
    file_path: str,
    max_length: int,
    world_size: int,
    rank: int,
    batch_size: int,
):
    dataset = InstructionDataset(tokenizer, file_path, max_length=max_length)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
    )
    return loader, sampler
