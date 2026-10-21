import hashlib
import os
from pathlib import Path
import numpy as np
import torch


def load_midi_file_cached(midi_path, midi_cache, missing_midi_paths, pretty_midi_module):
    """
    Load a MIDI file with caching.

    Returns:
        pretty_midi.PrettyMIDI or None if unavailable/invalid.
    """
    if midi_path in midi_cache:
        return midi_cache[midi_path]

    if midi_path is None or not os.path.exists(midi_path):
        if midi_path not in missing_midi_paths:
            print(f"Warning: MIDI not found, using empty roll: {midi_path}")
            missing_midi_paths.add(midi_path)
        midi_cache[midi_path] = None
        return None

    try:
        midi_obj = pretty_midi_module.PrettyMIDI(midi_path)
    except Exception as exc:
        if midi_path not in missing_midi_paths:
            print(f"Warning: Failed to parse MIDI '{midi_path}', using empty roll: {exc}")
            missing_midi_paths.add(midi_path)
        midi_obj = None

    midi_cache[midi_path] = midi_obj
    return midi_obj


def build_midi_roll_chunk(midi_obj, start_sec, end_sec, num_frames):
    """
    Build chunk-aligned binary MIDI roll [128, T] from PrettyMIDI.

    Args:
        midi_obj: pretty_midi.PrettyMIDI or None
        start_sec (float): chunk start time
        end_sec (float): chunk end time
        num_frames (int): number of temporal frames in output roll

    Returns:
        torch.Tensor of shape [128, num_frames] with dtype uint8
    """
    if num_frames <= 0:
        return torch.zeros((128, 0), dtype=torch.uint8)

    if midi_obj is None:
        return torch.zeros((128, num_frames), dtype=torch.uint8)

    duration = max(1e-6, float(end_sec) - float(start_sec))

    times = float(start_sec) + (
        (np.arange(num_frames, dtype=np.float32) + 0.5)
        * (duration / float(num_frames))
    )

    roll = midi_obj.get_piano_roll(times=times)
    roll_bin = (roll > 0).astype(np.uint8)

    return torch.from_numpy(roll_bin)


def chunk_has_midi_activity(midi_obj, start_sec, end_sec):
    """
    Return True when any MIDI note overlaps the requested time window.

    Args:
        midi_obj: pretty_midi.PrettyMIDI or None
        start_sec (float): chunk start time
        end_sec (float): chunk end time
    """
    if midi_obj is None:
        return False

    start_sec = float(start_sec)
    end_sec = float(end_sec)
    if end_sec <= start_sec:
        return False

    for instrument in midi_obj.instruments:
        for note in instrument.notes:
            if float(note.end) > start_sec and float(note.start) < end_sec:
                return True
    return False


def _cache_key_for_audio(audio_path):
    return hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()


def require_pretty_midi():
    """Import pretty_midi only for preparation paths that need it."""
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            "This dataset-preparation path requires pretty_midi. Install the project dependencies first."
        ) from exc
    return pretty_midi


def transcribe_audio_to_midi(
    *, audio_path, output_dir, backend="basic_pitch", force=False,
):
    """Create and cache a MIDI transcription for an audio file."""
    backend = str(backend).strip().lower()
    if backend == "none":
        return None
    if backend != "basic_pitch":
        raise ValueError(f"Unsupported transcription backend '{backend}'. Expected one of: none, basic_pitch.")

    audio_path = Path(audio_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path = output_dir / f"{audio_path.stem}_{_cache_key_for_audio(audio_path)[:10]}.mid"
    if midi_path.exists() and not force:
        return str(midi_path)

    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise ImportError(
            "Automatic transcription with backend='basic_pitch' requires the project's Python 3.11 environment. "
            "Install the project dependencies before preparing automatic-transcription data."
        ) from exc
    _, midi_data, _ = predict(str(audio_path))
    midi_data.write(str(midi_path))
    return str(midi_path)


def resolve_example_midi_path(
    *, audio_path, midi_path, transcription_backend="none", transcription_cache_dir=None,
    transcription_force=False,
):
    """Prefer dataset MIDI; otherwise optionally create a cached transcription."""
    if midi_path is not None and Path(midi_path).exists():
        return str(Path(midi_path).expanduser().resolve()), "dataset"
    if str(transcription_backend).strip().lower() == "none":
        return None, "missing"
    if transcription_cache_dir is None:
        transcription_cache_dir = Path(audio_path).expanduser().resolve().parent / "_transcribed_midi"
    resolved = transcribe_audio_to_midi(
        audio_path=audio_path, output_dir=transcription_cache_dir, backend=transcription_backend,
        force=transcription_force,
    )
    return resolved, "transcription"
