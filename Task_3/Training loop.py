import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from fairscale.nn.data_parallel import FullyShardedDataParallel
from Task_1.Main_model import GPTLanguageModel, estimate_loss, get_batch
from hyperparameters import device , learning_rate , max_iters , eval_interval


# Model instantiation
model = GPTLanguageModel().to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e3} K parameters in the training loop model")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# DDP/FSDP initialization
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        # DDP initialization
        model = DistributedDataParallel(model)
    elif torch.cuda.device_count() == 1:
        # Single GPU
        pass

# FSDP initialization
# Uncomment the following lines for FSDP
# from fairscale.nn.data_parallel import FullyShardedDataParallel
# model = FullyShardedDataParallel(model)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    # DDP: All-reduce gradient and backward pass
    if torch.cuda.device_count() > 1:
        loss = loss.mean()  # DDP requires averaging the loss across all GPUs
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # FSDP: All-reduce gradient and backward pass
    # Uncomment the following lines for FSDP
    # model.all_reduce_grads()
    # optimizer.step()

    optimizer.step()

