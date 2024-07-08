import time
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


# B -> batch size
# T -> seq length


@dataclass
class Config:
    block_size: int = 1024  # max seq length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 265 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of attention heads
    embed_size: int = 768  # embeddings dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.n_head == 0
        
        # key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size)
        
        # output projection
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        
        self.n_head = config.n_head
        self.embed_size = config.embed_size
        
        # not a 'bias', but more of a mask, but just following HF/OPAI naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()  # batch_size, sequence length, embedding dimension
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        
        q, k, v = qkv.split(self.embed_size, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # attention (matrializes the large (T, T) matrix for all the queries and keys)
        # traditional attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)        
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)

        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.c_proj(y)
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_size, 4 * config.embed_size)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.embed_size, config.embed_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_size)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # doing additions (+) for the residual connections
        # don't do += to avoid inplace operations, it's not good in torch
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x))
        return x


class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.embed_size),  # weights of token embeddings
            wpe = nn.Embedding(self.config.block_size, self.config.embed_size),  # weights of position embeddings (position encoding)
            h = nn.ModuleList([TransformerDecoderBlock(self.config) for _ in range(self.config.n_layer)]),  # number of layers of the transformer decoder block
            ln_f = nn.LayerNorm(self.config.embed_size)  # layer normalization
        ))
        
        # head to choose the next token to generate (think of it like doing classifying over the whole vocab)
        self.lm_head = nn.Linear(self.config.embed_size, self.config.vocab_size, bias=False)  
        
        # weight sharing scheme (recommened by attention is all you need paper)
        self.transformer.wte.weight = self.lm_head.weight
        # another benifit from this is also saving memory, since this will be now the same matrix in memory, and this is very big amount of params
        # embed_size*vocab_size=768*50257=38,597,376

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # this scaling is done to control the growth of activation inside a resdiual connection path
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 2 * self.config.n_layer = the number of resdiaul connections in our model
                # as each layer has 2 residual connections
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape T, embed_size
        tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, embed_size)
        x = tok_emb + pos_emb
        
        # forward the blocks of the transforme
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layer nor + classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # calculate loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        # at init load tokens from disk and store them in memory
        with open('../input.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (self.B * self.T)} batches")
        
        # state 
        self.current_position = 0
        
    def next_batch(self):
        buf = self.tokens[self.current_position : self.current_position + self.B * self.T + 1]
        x = (buf[:-1]).view(self.B, self.T)  # inputs
        y = (buf[1:]).view(self.B, self.T)  # targets

        # advance the position in the tensor
        self.current_position += self.B * self.T

        # if loading the next batch would be out of bounds, reset
        if self.current_position + (self.B * self.T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

# device = "cpu"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# init dataloader
train_loader = DataLoaderLite(B=8, T=1024)

torch.set_float32_matmul_precision('high')

model = LLM(Config())
model.to(device)

# using torch complie might take some time while training the model for first step
# but it makes your training much faster
# this is like the GCC compiler of neutral nets
# speedup comes from reducing python overhead + gpu read/writes
model = torch.compile(model)

print(model)


# to make sure we are not being biased towards any element at the beginning of the training
# and also make sure the weights init was legit, we have a formula to calcuate what's the reasonable starting loss (with no training at all)
# hopefully every vocab element (each one of our classes) is getting a uniform probability, which means we are not favoring any tokens -> we are not too confident about any element at the beginning of the training
# having vocab_size=number_of_classes=50257, and the loss is cross entropy, which is the negative likelood, then a reasonable range for the starting loss should be around -ln(1/50527) ~= 10.82 in this case
# and we are gettin 11.06 as an inital loss, which is still in the range!
# print(loss)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # amp
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()
    optimizer.step()

    # wait for gpu to finish all scheduled work
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000  # time in ms
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step: {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")


import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model, ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# generate! right now x = (B, T), where B=5 and T=8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        
        # get the probs
        probs = F.softmax(logits, dim=-1)
        
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)


# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
