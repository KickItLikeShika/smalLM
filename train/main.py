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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)
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
        x += self.attn(self.ln_1(x)) 
        x += self.mlp(self.ln_2(x))
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
    
    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
        
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape T, embed_size
        tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, embed_size)
        x = tok_emb + pos_emb
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layer nor + classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # B, T, vocab_size
        return logits


# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")


model = LLM(Config())
model.to(device)
print(model)

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