import math
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import librosa
except ImportError:
    librosa = None

_SOX_EFFECTS_AVAILABLE = None


def _get_sox_effects():
    global _SOX_EFFECTS_AVAILABLE
    try:
        import torchaudio.sox_effects as sox_effects
    except (ImportError, OSError) as exc:
        raise ImportError(
            "torchaudio.sox_effects is required for timbre-preserving augmentations. "
            "Install a torchaudio build with SoX support."
        ) from exc
    if _SOX_EFFECTS_AVAILABLE is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sox_effects.apply_effects_tensor(
                    torch.zeros(1, 16, dtype=torch.float32),
                    16000,
                    [],
                )
            _SOX_EFFECTS_AVAILABLE = True
        except (ImportError, OSError):
            _SOX_EFFECTS_AVAILABLE = False
    if not _SOX_EFFECTS_AVAILABLE:
        raise ImportError(
            "torchaudio.sox_effects is installed but libsox is unavailable on this system."
        )
    return sox_effects


def _apply_librosa_pitch_tempo(
    waveform: torch.Tensor,
    sample_rate: int,
    stretch: float,
    pitch_semitones: int,
) -> torch.Tensor:
    if librosa is None:
        raise ImportError(
            "librosa is required as a fallback when torchaudio SoX effects are unavailable."
        )

    wav_np = _ensure_mono_waveform(waveform).detach().cpu().numpy()
    if abs(float(stretch) - 1.0) > 1e-4:
        wav_np = librosa.effects.time_stretch(wav_np, rate=float(stretch))
    if int(pitch_semitones) != 0:
        wav_np = librosa.effects.pitch_shift(
            wav_np,
            sr=int(sample_rate),
            n_steps=float(pitch_semitones),
        )
    return torch.as_tensor(wav_np, dtype=torch.float32)


def _ensure_mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform [T] or stereo [C, T], got {tuple(waveform.shape)}")
    return waveform.contiguous()


def _align_waveform_length(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    waveform = _ensure_mono_waveform(waveform)
    cur_len = int(waveform.shape[-1])
    tgt_len = int(target_num_samples)
    if cur_len < tgt_len:
        waveform = torch.nn.functional.pad(waveform, (0, tgt_len - cur_len))
    elif cur_len > tgt_len:
        waveform = waveform[:tgt_len]
    return waveform.contiguous()


class PitchTimeStretchTransform:

    def __init__(
        self,
        sample_rate: int,
        ts_min: float = 0.9,
        ts_max: float = 1.1,
        pitch_min: int = -3,
        pitch_max: int = 3,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.ts_min = float(ts_min)
        self.ts_max = float(ts_max)
        self.pitch_min = int(pitch_min)
        self.pitch_max = int(pitch_max)
    def _sample_params(self, rng: random.Random) -> Tuple[float, int]:
        stretch = rng.uniform(self.ts_min, self.ts_max)
        pitch = rng.randint(self.pitch_min, self.pitch_max) if self.pitch_min != self.pitch_max else int(self.pitch_min)
        return float(stretch), int(pitch)

    def _apply_pitch_tempo(self, waveform: torch.Tensor, stretch: float, pitch_semitones: int) -> torch.Tensor:
        waveform = _ensure_mono_waveform(waveform)
        if waveform.numel() == 0:
            return waveform

        effects: List[List[str]] = []
        if abs(float(stretch) - 1.0) > 1e-4:
            effects.append(["tempo", f"{float(stretch):.6f}"])
        if int(pitch_semitones) != 0:
            effects.append(["pitch", f"{float(pitch_semitones) * 100.0:.2f}"])
            effects.append(["rate", str(self.sample_rate)])

        if not effects:
            return waveform

        try:
            sox_effects = _get_sox_effects()
            augmented, _ = sox_effects.apply_effects_tensor(
                waveform.unsqueeze(0),
                self.sample_rate,
                effects,
            )
            return augmented.squeeze(0).to(dtype=torch.float32)
        except (ImportError, OSError):
            return _apply_librosa_pitch_tempo(
                waveform,
                sample_rate=self.sample_rate,
                stretch=stretch,
                pitch_semitones=pitch_semitones,
            )

    def apply_global(
        self,
        waveform: torch.Tensor,
        rng: Optional[random.Random] = None,
        return_params: bool = False,
        align_output_length: bool = True,
    ):
        rng = rng or random.Random()
        waveform = _ensure_mono_waveform(waveform)
        target_num_samples = int(waveform.shape[-1])

        stretch, pitch = self._sample_params(rng)
        augmented = self._apply_pitch_tempo(waveform, stretch=stretch, pitch_semitones=pitch)

        if align_output_length:
            augmented = _align_waveform_length(augmented, target_num_samples)
        params = {
            "mode": "global_stem" if not align_output_length else "global_chunk",
            "stretch": float(stretch),
            "pitch_semitones": int(pitch),
            "output_num_samples": int(augmented.shape[-1]),
        }
        if return_params:
            return augmented, params
        return augmented


def build_timbre_augmented_chunk_bank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    chunk_samples: int,
    num_augments: int = 4,
    ts_min: float = 0.9,
    ts_max: float = 1.1,
    pitch_min: int = -3,
    pitch_max: int = 3,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]]]:
    waveform = _ensure_mono_waveform(waveform)
    rng = rng or random.Random()
    num_augments = int(num_augments)
    if num_augments <= 0:
        raise ValueError(f"num_augments must be positive, got {num_augments}")

    transform = PitchTimeStretchTransform(
        sample_rate=sample_rate,
        ts_min=ts_min,
        ts_max=ts_max,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
    )

    total_samples = int(waveform.shape[-1])
    num_full_chunks = total_samples // int(chunk_samples)
    if num_full_chunks <= 0:
        raise ValueError("Waveform is shorter than one full chunk.")

    bank_chunks: List[torch.Tensor] = []
    bank_params: List[List[Dict[str, Any]]] = []
    for _ in range(num_augments):
        augmented_stem, stem_params = transform.apply_global(
            waveform,
            rng=rng,
            return_params=True,
            align_output_length=False,
        )
        stretch = float(stem_params["stretch"])
        augmented_chunks: List[torch.Tensor] = []
        augment_params: List[Dict[str, Any]] = []
        for chunk_index in range(num_full_chunks):
            original_start = int(chunk_index * int(chunk_samples))
            augmented_start = int(round(float(original_start) / max(stretch, 1e-8)))
            augmented_end = augmented_start + int(chunk_samples)
            chunk_aug = augmented_stem[augmented_start:augmented_end]
            chunk_aug = _align_waveform_length(chunk_aug, int(chunk_samples))
            augmented_chunks.append(chunk_aug)
            params = dict(stem_params)
            params.update(
                {
                    "chunk_index": int(chunk_index),
                    "original_start_sample": int(original_start),
                    "augmented_start_sample": int(augmented_start),
                }
            )
            augment_params.append(params)
        bank_chunks.append(torch.stack(augmented_chunks, dim=0))
        bank_params.append(augment_params)

    return torch.stack(bank_chunks, dim=0), bank_params


def replay_timbre_augmented_stem(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    params: Dict[str, Any],
) -> torch.Tensor:
    """Recreate one stem-level timbre augmentation from stored parameters.

    This is the counterpart to :func:`build_timbre_augmented_chunk_bank` for
    derived-dataset tools.  It supports the current ``global_stem`` format and
    the explicit original-audio fallback written when augmentation fails.
    """
    waveform = _ensure_mono_waveform(waveform)
    if params.get("fallback") == "original":
        return waveform
    if params.get("mode") != "global_stem":
        raise ValueError(
            "Only recorded global_stem timbre augmentations can be replayed; "
            f"got mode={params.get('mode')!r}."
        )

    transform = PitchTimeStretchTransform(sample_rate=sample_rate)
    augmented = transform._apply_pitch_tempo(
        waveform,
        stretch=float(params["stretch"]),
        pitch_semitones=int(params["pitch_semitones"]),
    )
    return augmented.contiguous()


def chunk_recorded_timbre_augmentation(
    augmented_stem: torch.Tensor,
    *,
    chunk_samples: int,
    params: Dict[str, Any],
) -> torch.Tensor:
    """Select a fixed-size chunk from a replayed timbre-augmented stem."""
    start = int(params.get("augmented_start_sample", params.get("original_start_sample", 0)))
    chunk = augmented_stem[start : start + int(chunk_samples)]
    return _align_waveform_length(chunk, int(chunk_samples))
