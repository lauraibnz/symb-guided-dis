"""Frozen music-CLAP audio features used for timbre conditioning."""

from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn
import torchaudio


# LAION's released music checkpoint is intentionally the only default here:
# this project uses CLAP as a music-timbre representation, not a general-audio
# replacement for the latent codec.
DEFAULT_CLAP_MUSIC_CHECKPOINT = "pretrained/music_audioset_epoch_15_esc_90.14.pt"
CLAP_SAMPLE_RATE = 48_000


class CLAPMusicTimbreExtractor(nn.Module):
    """Extract frozen audio embeddings from the LAION music-CLAP checkpoint.

    Input audio must be mono floating-point waveforms sampled at 48 kHz, with
    shape ``[batch, samples]`` (or a single ``[samples]`` waveform).  The
    checkpoint is deliberately local and explicit so a prepared dataset
    records exactly which music representation it was built with.
    """

    sample_rate = CLAP_SAMPLE_RATE

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        amodel: str = "HTSAT-base",
        enable_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amodel = str(amodel)
        self.enable_fusion = bool(enable_fusion)

        checkpoint = Path(checkpoint_path or DEFAULT_CLAP_MUSIC_CHECKPOINT).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[3] / checkpoint
        self.checkpoint_path = checkpoint.resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "The LAION music-CLAP checkpoint was not found at "
                f"'{self.checkpoint_path}'. Place it at "
                f"'{DEFAULT_CLAP_MUSIC_CHECKPOINT}' relative to the repository, "
                "or pass --clap-checkpoint explicitly."
            )

        try:
            import laion_clap
        except ImportError as exc:
            raise ImportError(
                "CLAP timbre extraction requires the optional 'laion-clap' package. "
                "Install it in the active environment before preparing a dataset."
            ) from exc

        self.model = laion_clap.CLAP_Module(
            enable_fusion=self.enable_fusion,
            amodel=self.amodel,
            device=str(self.device),
        )
        self.model.load_ckpt(ckpt=str(self.checkpoint_path))
        self.model.eval()

    @torch.inference_mode()
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = torch.as_tensor(waveforms, dtype=torch.float32)
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError(
                "CLAPMusicTimbreExtractor expects mono waveforms shaped [B, T], "
                f"got {tuple(waveforms.shape)}."
            )

        embeddings = self.model.get_audio_embedding_from_data(
            waveforms.to(self.device),
            use_tensor=True,
        )
        # Preserve the checkpoint output exactly. LAION music-CLAP currently
        # returns normalized vectors itself, but this extractor deliberately
        # does not add any post-processing.
        return embeddings


def extract_clap_timbre_embeddings(
    extractor: CLAPMusicTimbreExtractor,
    waveforms: torch.Tensor,
    source_sample_rate: int,
    batch_size: int,
) -> torch.Tensor:
    """Resample waveform chunks and extract CPU CLAP vectors in bounded batches."""
    if waveforms.ndim != 2:
        raise ValueError(f"Expected waveform chunks shaped [B, T], got {tuple(waveforms.shape)}.")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    embeddings = []
    for start in range(0, int(waveforms.shape[0]), int(batch_size)):
        batch = waveforms[start : start + int(batch_size)].to(device="cpu", dtype=torch.float32)
        if int(source_sample_rate) != CLAP_SAMPLE_RATE:
            batch = torchaudio.functional.resample(batch, int(source_sample_rate), CLAP_SAMPLE_RATE)
        embeddings.append(extractor(batch).cpu())
    return torch.cat(embeddings, dim=0)
