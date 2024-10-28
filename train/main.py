import os
import time
import wandb
import torch
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
import sentencepiece as sp
from hellaswag import evaluate_trained_model
from data import DataLoaderLite
from model import Config, LLM
from training_loop import train
from evaluation_loop import evaluate
from generation_loop import generate


torch.manual_seed(1319)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1319)

torch.set_float32_matmul_precision('high')

wandb.init(
    project="smallLM",
    # track hyperparameters and run metadata
    config={
    "seq_len": 1024,
    # set the wandb project where this run will be logged
    "vocab_size": 50304,
    "n_layer": 32,
    "n_head": 32,
    "embed_size": 1024,
    "learning_rate": 6e-4,
    "tokenizer": 'gpt-2'
    }
)

# 10 billion tokens, and we have 524288 tokens per batch
# max_steps = 10b/524288 = 19073
max_steps = 19073
total_batch_size = 524288  # 2**19, 0.5M, in number of tokens
B = 4  # micro batch size
T = 1024  # sequence length

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

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print(f"i'm GPU {ddp_rank}")

# init dataloader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', master_process=master_process)

# create model
model = LLM(Config(vocab_size=50304))
model.to(device)
print('raw model')
print(model)

# using torch complie might take some time while training the model for first step
# but it makes your training much faster
# this is like the GCC compiler of neutral nets
# speedup comes from reducing python overhead + gpu read/writes
use_compile = False
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

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)
enc = tiktoken.get_encoding('gpt2')
# enc = sp.SentencePieceProcessor()
# enc.load('new-tokenizer/tokenizer.model')

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
        evaluate(step, last_step, model, raw_model, val_loader, optimizer, device, device_type, log_file, log_dir, master_process, ddp=ddp)

    # evaluate hellaswag
    if (step % 500 == 0 or last_step):
        evaluate_trained_model(step, model, device, device_type, log_file, master_process, ddp_world_size, ddp_rank, ddp=ddp)

    # generate from the model excpet step 0, which is noise
    # disabled torch.compile throws an error i can't solve rn
    # TODO: FIX THE TORCH COMPILE ISSUE
    if ((step > 0 and step % 500 == 0) or last_step):
        generate(model, enc, device, device_type, ddp_rank)

    # training loop
    train(step, model, train_loader, optimizer, grad_accum_steps, device, device_type, t0, log_file, master_process, ddp_world_size, ddp)


if ddp:
    destroy_process_group()

wandb.finish()
