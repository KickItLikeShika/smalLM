import sys

import tiktoken
import torch
from torch.nn import functional as F

from model import LLM, Config

def generate(text, model, enc, device):
    model.eval()
    max_length = 128
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
    tokens = xgen[0, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(decoded)

if len(sys.argv) < 2:
    print("Please provide text input as argument")
    print("Usage: python infer.py \"your text here\"")
    sys.exit(1)

device = torch.device('cuda')
enc = tiktoken.get_encoding('gpt2')
model = LLM(Config(vocab_size=50304))

ckpt = torch.hub.load_state_dict_from_url(
    'https://huggingface.co/KickItLikeShika/smalLM-v0.1/resolve/main/pytorch_model.bin',
    map_location='cpu'
)

model.load_state_dict(ckpt)
model.to(device)

text = sys.argv[1]

generate(text, model, enc, device)