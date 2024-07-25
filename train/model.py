from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding



@dataclass
class Config:
    block_size: int = 1024  # max seq length
    vocab_size: int = 50304  # number of tokens: 50,000 BPE merges + 265 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 32  # number of layers
    n_head: int = 24  # number of attention heads
    embed_size: int = 768  # embeddings dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.n_head == 0
        self.rotary_emb = RotaryEmbedding(dim=config.n_head)
        # key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size)

        # output projection
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

        self.n_head = config.n_head
        self.embed_size = config.embed_size

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
        
        # rotaty positional encoding is applied to query and key, instead of the embeddings directly.
        # cos, sin = rope
        # apply rope in fp32 significanly stabalize training
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
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
            # wpe = nn.Embedding(self.config.block_size, self.config.embed_size),  # weights of position embeddings (position encoding)
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
        # pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # pos_emb = self.transformer.wpe(pos)  # position embeddings of shape T, embed_size
        # tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, embed_size)
        x = self.transformer.wte(idx)  # token embeddings (B, T, embed_size)
        # x = tok_emb + pos_emb
        
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer
