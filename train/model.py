from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _compute_default_rope_parameters(
    config=None,
    device=torch.device('cuda'),
    seq_len=None,
    **rope_kwargs,
):
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        dim = int((config.embed_size // config.n_head) * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


@dataclass
class Config:
    block_size: int = 1024  # max seq length
    vocab_size: int = 50304  # number of tokens: 50,000 BPE merges + 265 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of attention heads
    embed_size: int = 768  # embeddings dimension
    rope_theta: float = 500000
    rope_scaling = None
    max_position_embeddings = 1024


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=1024,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.n_head == 0
        # self.rotary_emb = RotaryEmbedding(dim=config.n_head)
        # key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size)

        # output projection
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

        self.n_head = config.n_head
        self.embed_size = config.embed_size

    def forward(self, x, position_embeddings):
        B, T, C = x.size()  # batch_size, sequence length, embedding dimension
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.embed_size, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        # k = k.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        # v = v.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # q = q.view(B, T, self.n_head, self.embed_size)
        # k = k.view(B, T, self.n_head, self.embed_size)
        # v = v.view(B, T, self.n_head, self.embed_size)
        # # rotaty positional encoding is applied to query and key, instead of the embeddings directly.
        # cos, sin = rope
        # apply rope in fp32 significanly stabalize training
        # q = self.rotary_emb.rotate_queries_or_keys(q)
        # k = self.rotary_emb.rotate_queries_or_keys(k)
        
        # q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

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
    
    def forward(self, x, position_embeddings):
        # doing additions (+) for the residual connections
        # don't do += to avoid inplace operations, it's not good in torch
        x = x + self.attn(self.ln_1(x), position_embeddings)
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

        self.rotary_emb = RotaryEmbedding(config=config)

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
        
        x = self.transformer.wte(idx)  # token embeddings (B, T, embed_size)
        position_ids = torch.arange(
            0, x.shape[1], device=x.device
        ).unsqueeze(0)
        # forward the token and position embeddings
        # pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # pos_emb = self.transformer.wpe(pos)  # position embeddings of shape T, embed_size
        # tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, embed_size)
        hidden_states = x
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # x = tok_emb + pos_emb
        
        # forward the blocks of the transforme
        for block in self.transformer.h:
            x = block(x, position_embeddings)
        
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
