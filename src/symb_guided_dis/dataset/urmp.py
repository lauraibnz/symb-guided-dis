"""URMP-specific annotation helpers.

URMP's ``Sco_*.mid`` files are score-time MIDI and are not aligned to the
recorded performances. The per-stem ``Notes_*.txt`` files are the aligned
symbolic annotations, so they are the source of truth for training labels and
structure-augmentation rendering.
"""

import math
from pathlib import Path
from typing import Union

DEFAULT_URMP_REFERENCE_MIDI_VELOCITY = 100


URMP_INSTRUMENT_PROGRAMS = {
    "vn": 40,  # Violin
    "va": 41,  # Viola
    "vc": 42,  # Cello
    "db": 43,  # Contrabass
    "tpt": 56,  # Trumpet
    "tbn": 57,  # Trombone
    "tba": 58,  # Tuba
    "hn": 60,  # French horn
    "sax": 65,  # Alto saxophone
    "ob": 68,  # Oboe
    "bn": 70,  # Bassoon
    "cl": 71,  # Clarinet
    "fl": 73,  # Flute
}


def urmp_instrument_program(instrument_code: str) -> int:
    """Return a sensible General MIDI program for a URMP instrument code."""
    return int(URMP_INSTRUMENT_PROGRAMS.get(str(instrument_code).lower(), 0))


def _hz_to_midi(frequency_hz: float) -> int:
    midi_pitch = round(69.0 + 12.0 * math.log2(float(frequency_hz) / 440.0))
    return max(0, min(127, int(midi_pitch)))


def write_urmp_notes_midi(
    notes_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    velocity: int = 100,
    program: int = 0,
    force: bool = False,
) -> Path:
    """Convert an audio-aligned URMP ``Notes`` file to a standard MIDI file.

    Each annotation row is ``onset_seconds, frequency_hz, duration_seconds``.
    URMP does not provide note velocity, so callers choose one fixed MIDI
    velocity for the generated reference transcription.
    """
    notes_path = Path(notes_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not notes_path.is_file():
        raise FileNotFoundError(f"URMP Notes file not found: {notes_path}")
    if not 1 <= int(velocity) <= 127:
        raise ValueError(f"URMP MIDI velocity must be in [1, 127], got {velocity}.")
    if not 0 <= int(program) <= 127:
        raise ValueError(f"MIDI program must be in [0, 127], got {program}.")
    if output_path.is_file() and not force:
        return output_path

    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            "Converting URMP Notes annotations to MIDI requires pretty_midi. "
            "Install it with: pip install pretty_midi"
        ) from exc

    instrument = pretty_midi.Instrument(program=int(program))
    for line_number, line in enumerate(notes_path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 3:
            raise ValueError(
                f"Invalid URMP Notes row in {notes_path}:{line_number}; "
                "expected onset_seconds, frequency_hz, duration_seconds."
            )
        onset, frequency_hz, duration = (float(value) for value in fields[:3])
        if frequency_hz <= 0.0 or duration <= 0.0:
            continue
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(velocity),
                pitch=_hz_to_midi(frequency_hz),
                start=max(0.0, onset),
                end=max(0.0, onset + duration),
            )
        )

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    midi.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return output_path


def cached_urmp_notes_midi_path(
    notes_path: Union[str, Path],
    cache_dir: Union[str, Path],
    *,
    velocity: int,
    program: int,
    force: bool = False,
) -> Path:
    """Materialize a versioned per-stem MIDI cache entry for a URMP Notes file."""
    notes_path = Path(notes_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_name = f"{notes_path.stem}_velocity{int(velocity):03d}_program{int(program):03d}.mid"
    return write_urmp_notes_midi(
        notes_path,
        cache_dir / output_name,
        velocity=velocity,
        program=program,
        force=force,
    )


def resolve_urmp_reference_midi(metadata, midi_cache_dir):
    """Return a temporary reference MIDI path for one URMP parser record.

    Preparation owns the lifetime of ``midi_cache_dir``. The converted MIDI is
    only an intermediate for rendering structure augmentations and building
    the stored MIDI roll; it is never retained beside the finished LMDB.
    """
    notes_path = metadata.get("urmp_notes_path")
    if not notes_path:
        return None, None
    cache_dir = Path(midi_cache_dir).expanduser().resolve()
    program = int(metadata.get("program_num", urmp_instrument_program(metadata.get("instrument_code", ""))))
    midi_path = cached_urmp_notes_midi_path(
        notes_path,
        cache_dir,
        velocity=DEFAULT_URMP_REFERENCE_MIDI_VELOCITY,
        program=program,
    )
    return str(midi_path), "urmp_notes"
