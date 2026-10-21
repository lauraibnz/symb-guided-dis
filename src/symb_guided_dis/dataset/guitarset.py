"""GuitarSet parsing and its annotation-to-MIDI conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Union


GUITARSET_PROGRAM = 25  # zero-indexed General MIDI acoustic steel guitar
GUITARSET_REFERENCE_VELOCITY = 100


def iter_guitarset_note_annotations(jams_path: Union[str, Path]) -> Iterable[list[dict[str, Any]]]:
    """Yield ``note_midi`` observations from a GuitarSet JAMS JSON file.

    GuitarSet annotations are ordinary JAMS JSON.  Reading the small JSON
    structure directly avoids making the full ``jams`` package a preparation
    dependency.  ``pitch_midi`` is supported as a compatibility fallback.
    """
    jams_path = Path(jams_path).expanduser().resolve()
    payload = json.loads(jams_path.read_text())
    annotations = payload.get("annotations") or []
    selected = [annotation for annotation in annotations if annotation.get("namespace") == "note_midi"]
    if not selected:
        selected = [annotation for annotation in annotations if annotation.get("namespace") == "pitch_midi"]
    if not selected:
        raise ValueError(f"No note_midi or pitch_midi annotations in GuitarSet file: {jams_path}")

    for annotation in selected:
        data = annotation.get("data") or []
        if isinstance(data, dict):
            data = data.values()
        observations = [observation for observation in data if isinstance(observation, dict)]
        if observations:
            yield observations


def write_guitarset_jams_midi(
    jams_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    velocity: int = GUITARSET_REFERENCE_VELOCITY,
    program: int = GUITARSET_PROGRAM,
    preserve_pitch_bends: bool = True,
    force: bool = False,
) -> Path:
    """Convert GuitarSet's note annotations into a standard MIDI file.

    GuitarSet note annotations have onset, duration, and fractional MIDI pitch,
    but no note velocity.  We therefore use a documented constant velocity and
    retain the fractional component as an onset pitch bend, following the
    GuitarSet reference converter.  Each annotation track becomes one MIDI
    instrument, preserving the string-level organization of the source.
    """
    jams_path = Path(jams_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not jams_path.is_file():
        raise FileNotFoundError(f"GuitarSet JAMS file not found: {jams_path}")
    if not 1 <= int(velocity) <= 127:
        raise ValueError(f"GuitarSet MIDI velocity must be in [1, 127], got {velocity}.")
    if not 0 <= int(program) <= 127:
        raise ValueError(f"MIDI program must be in [0, 127], got {program}.")
    if output_path.is_file() and not force:
        return output_path

    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            "Converting GuitarSet annotations requires pretty_midi. "
            "Install it with: pip install pretty_midi"
        ) from exc

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for observations in iter_guitarset_note_annotations(jams_path):
        instrument = pretty_midi.Instrument(program=int(program))
        for observation in observations:
            try:
                onset = float(observation["time"])
                duration = float(observation["duration"])
                pitch_value = float(observation["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if duration <= 0.0:
                continue
            pitch = max(0, min(127, int(round(pitch_value))))
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=pitch,
                    start=max(0.0, onset),
                    end=max(0.0, onset + duration),
                )
            )
            if preserve_pitch_bends:
                # PrettyMIDI uses the conventional default +/- two-semitone
                # bend range, hence 4096 units per semitone.
                bend = int(round((pitch_value - pitch) * 4096.0))
                instrument.pitch_bends.append(
                    pretty_midi.PitchBend(pitch=max(-8192, min(8191, bend)), time=max(0.0, onset))
                )
        if instrument.notes:
            midi.instruments.append(instrument)

    if not midi.instruments:
        raise ValueError(f"No valid GuitarSet notes were found in {jams_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return output_path


def cached_guitarset_midi_path(
    jams_path: Union[str, Path],
    cache_dir: Union[str, Path],
    *,
    velocity: int = GUITARSET_REFERENCE_VELOCITY,
    program: int = GUITARSET_PROGRAM,
    preserve_pitch_bends: bool = True,
) -> Path:
    jams_path = Path(jams_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_suffix = "guitarset" if preserve_pitch_bends else "guitarset_rounded"
    output_name = f"{jams_path.stem}_velocity{int(velocity):03d}_program{int(program):03d}_{output_suffix}.mid"
    return write_guitarset_jams_midi(
        jams_path,
        cache_dir / output_name,
        velocity=velocity,
        program=program,
        preserve_pitch_bends=preserve_pitch_bends,
    )


def resolve_guitarset_reference_midi(metadata, midi_cache_dir):
    jams_path = metadata.get("guitarset_jams_path")
    if not jams_path:
        return None, None
    midi_path = cached_guitarset_midi_path(
        jams_path,
        midi_cache_dir,
        velocity=GUITARSET_REFERENCE_VELOCITY,
        program=int(metadata.get("program_num", GUITARSET_PROGRAM)),
    )
    return str(midi_path), "guitarset_jams"


def resolve_guitarset_discrete_midi(metadata, midi_cache_dir):
    """Return rounded no-bend MIDI for GuitarSet's binary pitch target.

    The bend-preserving counterpart remains the source for structure
    augmentation rendering. This file is used only to construct ``midi_roll``;
    otherwise a fractional bend would activate both adjacent integer pitches
    when the PrettyMIDI piano roll is thresholded to binary values.
    """
    jams_path = metadata.get("guitarset_jams_path")
    if not jams_path:
        return None, None
    midi_path = cached_guitarset_midi_path(
        jams_path,
        midi_cache_dir,
        velocity=GUITARSET_REFERENCE_VELOCITY,
        program=int(metadata.get("program_num", GUITARSET_PROGRAM)),
        preserve_pitch_bends=False,
    )
    return str(midi_path), "guitarset_jams_rounded"
