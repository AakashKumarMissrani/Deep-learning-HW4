from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    """
    query = query.float()
    key = key.float()
    _, seqlen, _, _ = query.shape

    device = query.device

    # Frequency bases: 1 / theta^(2i/head_dim) for i=0 to head_dim//2 - 1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # Position indices
    t = torch.arange(seqlen, device=device, dtype=torch.float32)

    # Compute angles: (seqlen, head_dim//2)
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    # Cos and sin for full dimension (duplicate for real/imag)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seqlen, head_dim)
    cos = emb.cos()
    sin = emb.sin()

    # Add batch and head dims → (1, seqlen, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Split query/key into real/imag parts (shape: ..., head_dim//2)
    query_real = query[..., ::2]
    query_imag = query[..., 1::2]
    key_real   = key[..., ::2]
    key_imag   = key[..., 1::2]

    # Apply rotation (cos/sin broadcast to match head_dim//2)
    q_real = query_real * cos - query_imag * sin
    q_imag = query_real * sin + query_imag * cos
    k_real = key_real * cos - key_imag * sin
    k_imag = key_real * sin + key_imag * cos

    # Recombine
    query_out = torch.cat((q_real, q_imag), dim=-1)
    key_out   = torch.cat((k_real, k_imag), dim=-1)

    return query_out.type_as(query), key_out.type_as(key)