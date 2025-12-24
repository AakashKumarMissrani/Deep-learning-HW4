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
    Apply rotary positional embeddings to query and key.
    Shapes:
      query, key: (batch, seqlen, n_heads, head_dim)
    """

    query = query.float()
    key = key.float()

    _, seqlen, _, _ = query.shape
    device = query.device

    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Inverse frequency for each pair
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    # Position indices
    t = torch.arange(seqlen, device=device, dtype=torch.float32)

    # (seqlen, head_dim//2)
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    # Cosine and sine
    cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, head_dim//2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(2)

    # Split into even / odd dimensions
    query_real = query[..., ::2]
    query_imag = query[..., 1::2]
    key_real   = key[..., ::2]
    key_imag   = key[..., 1::2]

    # Apply rotation
    query_out_real = query_real * cos - query_imag * sin
    query_out_imag = query_real * sin + query_imag * cos
    key_out_real   = key_real * cos - key_imag * sin
    key_out_imag   = key_real * sin + key_imag * cos

    # Interleave back to original shape
    query_out = torch.stack((query_out_real, query_out_imag), dim=-1).flatten(-2)
    key_out   = torch.stack((key_out_real, key_out_imag), dim=-1).flatten(-2)

    return query_out.type_as(query), key_out.type_as(key)
