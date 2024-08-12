import os
import wandb
import torch
import torch.distributed as dist


def evaluate(step, last_step, model, raw_model, val_loader, optimizer, device, device_type, log_file, log_dir, master_process, ddp=False):
    model.eval()
    val_loader.reset()

    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20

        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()

    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        wandb.log({"validation loss": val_loss_accum.item()})

        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        if step > 0 and (step % 5000 == 0 or last_step):
        # if step > 0 and (step % 1 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': val_loss_accum.item(),
                'optim': optimizer.state_dict(),
                'seed': 1319
            }
            torch.save(checkpoint, checkpoint_path)
