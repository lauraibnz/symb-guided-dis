"""Waveform loading, chunking, and source-level normalization."""

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio


# These define the representation consumed by the released training configs.
# They are intentionally module defaults rather than preparation-script knobs.
DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_CHUNK_SIZE_SECONDS = 10.0
DEFAULT_SILENCE_THRESHOLD = 0.05
DEFAULT_LOUDNESS_TARGET_DBFS = -30.0
DEFAULT_LOUDNESS_ACTIVE_THRESHOLD_DBFS = -50.0
DEFAULT_LOUDNESS_MAX_GAIN_DB = 12.0
DEFAULT_LOUDNESS_PEAK_DBFS = -1.0


def load_audio_chunks(
    audio_path: Union[str, Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_size_seconds: float = DEFAULT_CHUNK_SIZE_SECONDS,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    """Load mono audio, resample it, and return its complete fixed-size chunks.

    The returned waveform is not truncated; callers that construct stem-level
    augmentations need the complete resampled stem.  ``chunks`` contains only
    full chunks and is ``None`` if the waveform is shorter than one chunk.
    """
    waveform, source_sample_rate = torchaudio.load(str(audio_path))
    if int(source_sample_rate) != int(sample_rate):
        waveform = torchaudio.transforms.Resample(int(source_sample_rate), int(sample_rate))(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.contiguous()

    chunk_samples = int(float(sample_rate) * float(chunk_size_seconds))
    total_samples = int(waveform.shape[-1])
    num_full_chunks = total_samples // chunk_samples
    if num_full_chunks == 0:
        return waveform, None, chunk_samples, 0

    truncated_samples = num_full_chunks * chunk_samples
    chunks = waveform[:truncated_samples].reshape(-1, chunk_samples).contiguous()
    return waveform, chunks, chunk_samples, truncated_samples


def _dbfs_from_amplitude(amplitude: float) -> float:
    return 20.0 * math.log10(max(float(amplitude), 1e-12))


def normalize_active_rms_loudness(
    waveform: torch.Tensor,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    target_dbfs: float = DEFAULT_LOUDNESS_TARGET_DBFS,
    active_threshold_dbfs: float = DEFAULT_LOUDNESS_ACTIVE_THRESHOLD_DBFS,
    max_gain_db: float = DEFAULT_LOUDNESS_MAX_GAIN_DB,
    peak_dbfs: float = DEFAULT_LOUDNESS_PEAK_DBFS,
) -> tuple[torch.Tensor, dict]:
    """Apply one safe, robust gain to an entire mono source waveform.

    Loudness is the median RMS of one-second windows above the activity
    threshold. It is measured once per source recording, never per chunk, so
    musical dynamics within a performance remain intact.
    """
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform [T], got {tuple(waveform.shape)}.")
    if int(sample_rate) <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if float(max_gain_db) < 0.0:
        raise ValueError(f"max_gain_db must be non-negative, got {max_gain_db}.")

    rms_values = torch.stack(
        [window.square().mean().sqrt() for window in waveform.split(int(sample_rate)) if window.numel() > 0]
    )
    rms_dbfs = 20.0 * torch.log10(rms_values.clamp_min(1e-12))
    active_rms_dbfs = rms_dbfs[rms_dbfs > float(active_threshold_dbfs)]
    peak_amplitude = float(waveform.abs().max()) if waveform.numel() else 0.0
    peak_before_dbfs = _dbfs_from_amplitude(peak_amplitude)

    if active_rms_dbfs.numel() == 0:
        return waveform, {
            "mode": "active_rms", "applied": False, "reason": "no_active_windows",
            "target_dbfs": float(target_dbfs), "active_threshold_dbfs": float(active_threshold_dbfs),
            "max_gain_db": float(max_gain_db), "peak_dbfs": float(peak_dbfs),
            "peak_before_dbfs": peak_before_dbfs,
        }

    source_active_dbfs = float(active_rms_dbfs.median())
    desired_gain_db = float(target_dbfs) - source_active_dbfs
    bounded_gain_db = max(-float(max_gain_db), min(float(max_gain_db), desired_gain_db))
    applied_gain_db = min(bounded_gain_db, float(peak_dbfs) - peak_before_dbfs)
    normalized = (waveform * float(10.0 ** (applied_gain_db / 20.0))).contiguous()
    return normalized, {
        "mode": "active_rms", "applied": True, "target_dbfs": float(target_dbfs),
        "active_threshold_dbfs": float(active_threshold_dbfs), "max_gain_db": float(max_gain_db),
        "peak_dbfs": float(peak_dbfs), "source_active_rms_dbfs": source_active_dbfs,
        "desired_gain_db": desired_gain_db, "applied_gain_db": applied_gain_db,
        "peak_before_dbfs": peak_before_dbfs,
        "peak_after_dbfs": _dbfs_from_amplitude(float(normalized.abs().max())),
    }
