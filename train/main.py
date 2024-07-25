import time
import os
import wandb
import math
import torch
from torch.nn import functional as F
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoTokenizer
import sentencepiece as sp
from hellaswag import render_example, iterate_examples, get_most_likely_row
from data import DataLoaderLite
from model import Config, LLM


wandb.init(
    # set the wandb project where this run will be logged
    project="smallLM",
    # track hyperparameters and run metadata
    config={
    "seq_len": 1024,
    "vocab_size": 50304,
    "n_layer": 32,
    "n_head": 24,
    "embed_size": 768,
    "learning_rate": 6e-4,
    }
)


# setup DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for no i think we need CUDA for DDP"
   
    init_process_group(backend='nccl')
   
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])

    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
   
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# to have 'cuda' as type instead of 'cuda:0' and 'cuda:1' etc
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288  # 2**19, 0.5M, in number of tokens
B = 8  # micro batch size
T = 1024  # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print(f"i'm GPU {ddp_rank}")

# init dataloader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', master_process=master_process)

torch.set_float32_matmul_precision('high')

# create model
model = LLM(Config(vocab_size=50304))
model.to(device)
print('raw model')
print(model)
# using torch complie might take some time while training the model for first step
# but it makes your training much faster
# this is like the GCC compiler of neutral nets
# speedup comes from reducing python overhead + gpu read/writes
use_compile = True
if use_compile:
    model = torch.compile(model)
    print('compiled model')
    print(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
print('ddp model')
print(model)
raw_model = model.module if ddp else model
# print(f"raw model device: {raw_model.device}")

# to make sure we are not being biased towards any element at the beginning of the training
# and also make sure the weights init was legit, we have a formula to calcuate what's the reasonable starting loss (with no training at all)
# hopefully every vocab element (each one of our classes) is getting a uniform probability, which means we are not favoring any tokens -> we are not too confident about any element at the beginning of the training
# having vocab_size=number_of_classes=50257, and the loss is cross entropy, which is the negative likelood, then a reasonable range for the starting loss should be around -ln(1/50527) ~= 10.82 in this case
# and we are gettin 11.06 as an inital loss, which is still in the range!
# print(loss)

max_lr = 6e-4
min_lr = max_lr * 0.1
# 10 billion tokens, and we have 524288 tokens per batch
# max_steps = 10b/524288 = 19073
warmup_steps = 715
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


optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)
# enc = tiktoken.get_encoding('gpt2')
enc = sp.SentencePieceProcessor()
enc.load('new-tokenizer/tokenizer.model')

# create the log directory we will write checkpoints to and log to
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # evalaute every 100 steps
    if step % 100 == 0 or last_step:
    # if step % 10 == 0 or last_step:
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
                    'seed': 1337
                }
                torch.save(checkpoint, checkpoint_path)

    # evaluate hellaswag
    # if (step % 250 == 0 or last_step) and (not use_compile):
    # if (step % 250 == 0 or last_step):
    if step % 10 == 0:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue

            # render the example
            _, tokens, mask, label = render_example(example)
            tokens, mask = tokens.to(device), mask.to(device)
            
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # reduce the stats across all gpus
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            wandb.log({"HellaSwag accuracy": num_correct_norm/num_total})

            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # generate from the model excpet step 0, which is noise
    # disabled torch.compile throws an error i can't solve rn
    # TODO: FIX THE TORCH COMPILE ISSUE
    # if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
    # if ((step > 0 and step % 250 == 0) or last_step):
    if step % 10 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)

                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)

                # get the probs
                probs = F.softmax(logits, dim=-1)

                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)

                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)

        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

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

if ddp:
    destroy_process_group()

wandb.finish()
