import time
import torch
import torch.distributed as dist
import math
import wandb


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
# 10 billion tokens, and we have 524288 tokens per batch
# max_steps = 10b/524288 = 19073
max_steps = 19073


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def train(step, model, train_loader, optimizer, grad_accum_steps, device, device_type, t0, log_file, master_process, ddp_world_size, ddp=False):
    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # amp
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. scale th loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        # manually toggle this require_backward_grad_sync to perform the all reduce at while doing backprop, 
        # to sync the gradient updates across gpus avoid using 
        # instead we could use the kinda ugly and long ddp sync context manager
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    # avergae accumulated loss across gpus
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # get new lr
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # wait for gpu to finish all scheduled work
    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0) # time in s

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"step: {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        wandb.log({"step": step, "loss": loss_accum.item(), "lr": lr, "norm": norm, "dt": dt*1000, "tok/sec": tokens_per_sec})
        
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
