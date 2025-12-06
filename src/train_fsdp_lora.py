
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


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


def build_model(device):
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    model = FSDP(model)
    return model


def build_dataloader(world_size, rank, batch_size=32):
    num_samples = 1024
    x = torch.randn(num_samples, 128)
    y = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader, sampler


def main():
    rank, local_rank, world_size, device = setup_distributed()

    model = build_model(device)
    train_loader, train_sampler = build_dataloader(world_size, rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    num_epochs = 3

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    if rank == 0:
        print("Training finished.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
