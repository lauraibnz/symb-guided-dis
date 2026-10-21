"""Adapter for the external Music2Latent audio codec."""

from __future__ import annotations

from typing import Optional

import torch


def build(device: Optional[str] = None):
    """Construct Music2Latent's external encoder/decoder wrapper."""
    from music2latent import EncoderDecoder as Music2LatentEncoderDecoder

    kwargs = {"device": device} if device is not None else {}
    return Music2LatentEncoderDecoder(**kwargs)


def _ensure_model_latent_batch(
    latent_like,
    *,
    latent_dim: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    latent = torch.as_tensor(latent_like, dtype=torch.float32, device=device)
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if latent.ndim != 3:
        raise ValueError(f"Expected latent with shape [B, C, T] or [C, T], got {tuple(latent.shape)}")
    if latent.shape[1] != int(latent_dim):
        raise ValueError(
            f"Expected latent channels {latent_dim}, got {latent.shape[1]} for shape {tuple(latent.shape)}"
        )
    return latent


def encode_batch(codec, chunks_tensor: torch.Tensor) -> torch.Tensor:
    """Encode a batch of waveform chunks with Music2Latent."""
    return codec.encode(chunks_tensor)


def encode_audio(
    codec,
    waveform,
    *,
    latent_dim: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Encode one waveform and normalize its latent shape to ``[B, C, T]``."""
    return _ensure_model_latent_batch(codec.encode(waveform), latent_dim=latent_dim, device=device)


def encode_audio_batch(
    codec,
    waveforms,
    *,
    latent_dim: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Encode a batch of mono waveforms and normalize its latent shape."""
    waveforms = torch.as_tensor(waveforms, dtype=torch.float32)
    if waveforms.ndim == 1:
        return encode_audio(codec, waveforms, latent_dim=latent_dim, device=device)
    if waveforms.ndim == 3 and waveforms.shape[1] == 1:
        waveforms = waveforms[:, 0, :]
    elif waveforms.ndim == 3 and waveforms.shape[2] == 1:
        waveforms = waveforms[:, :, 0]
    if waveforms.ndim != 2:
        raise ValueError(
            "Expected batched waveforms with shape [B, T], [B, 1, T], or [B, T, 1], "
            f"got {tuple(waveforms.shape)}"
        )
    raw_latent = codec.encode(waveforms.to(device=device) if device is not None else waveforms)
    return _ensure_model_latent_batch(raw_latent, latent_dim=latent_dim, device=device)


def decode_audio(codec, latent_batch, *, latent_dim: int):
    """Decode a normalized model latent with Music2Latent."""
    latent_batch = _ensure_model_latent_batch(latent_batch, latent_dim=latent_dim).detach().cpu().numpy()
    return codec.decode(latent_batch)
