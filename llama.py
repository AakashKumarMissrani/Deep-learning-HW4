from contextlib import nullcontext
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_llama import LlamaPreTrainedModel, LlamaConfig
from rope import apply_rotary_emb
from utils import *


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return x_normalized

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight + self.bias


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.compute_query = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.compute_key = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_value = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_output = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def compute_query_key_value_scores(self,
                                       query: torch.Tensor,
                                       key: torch.Tensor,
                                       value: torch.Tensor) -> torch.Tensor:
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        batch_size, n_heads, seqlen, _ = query.shape
        mask = torch.triu(torch.ones(seqlen, seqlen, device=query.device), diagonal=1)
        attn_scores = attn_scores + mask[None, None, :, :] * float('-inf')

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        return output

    def forward(self, x: torch.Tensor):
        batch_size, seqlen, _ = x.shape

        query = self.compute_query(x)
        key = self.compute_key(x)
        value = self.compute_value(x)

        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        query, key = apply_rotary_emb(query, key, self.head_dim, self.max_seq_len)

        key = torch.repeat_interleave(key, dim=2, repeats=self.n_rep)
        value = torch.repeat_interleave(value, dim=2, repeats=self.n_rep)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        output = self.compute_query_key_value_scores(query, key, value)

        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)
        output = self.resid_dropout(self.compute_output(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x):
        return self.dropout(self.w2(self.SwiGLU(x)))


class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.ffn_norm = LayerNorm(config.dim, eps=config.layer_norm_eps)

    def forward(self, x):
        # Attention block
        input_norm = self.attention_norm(x)
        attn_output = self.attention(input_norm)
        x = x + attn_output

        # Feed-forward block
        ffn_input = self.ffn_norm(x)
        ffn_output = self.feed_forward(ffn_input)
        x = x + ffn_output

        return x


class Llama(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.params = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([LlamaLayer(layer_id, config) for layer_id in range(config.n_layers)])
        self.norm = LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('compute_output.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
        else:
            logits = self.output(h[:, [-1], :])

        return logits, h

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Numerical stability
                logits = logits - logits.max(dim=-1, keepdim=True).values

                probs = F.softmax(logits, dim=-1)

                # Replace NaN/inf and clamp negatives
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs = probs.clamp(min=0.0)

                # ---- Check for all-zero rows ----
                probs_sum = probs.sum(dim=-1, keepdim=True)
                zero_rows = probs_sum == 0
                if zero_rows.any():
                    # fallback to uniform distribution over vocab
                    probs[zero_rows.expand_as(probs)] = 1.0 / probs.size(-1)
                    probs_sum = probs.sum(dim=-1, keepdim=True)

                # normalize to sum=1
                probs = probs / probs_sum

                idx_next = torch.multinomial(probs, num_samples=1)


            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def load_pretrained(checkpoint):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = "float32"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    checkpoint_dict = torch.load(checkpoint, map_location=device)
    config = LlamaConfig(**checkpoint_dict['model_args'])
    model = Llama(config)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    return model