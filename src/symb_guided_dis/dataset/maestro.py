"""MAESTRO dataset manifest helpers.

MAESTRO v3 stores its WAV/MIDI pairs under year directories and supplies a
column-oriented JSON manifest.  The manifest's explicit paths and native
split are authoritative; its ``duration`` field is retained as provenance but
is deliberately not used to determine audio/chunk duration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MAESTRO_PIANO_PROGRAM = 0  # zero-based General MIDI acoustic grand piano
MAESTRO_REQUIRED_FIELDS = (
    "audio_filename",
    "midi_filename",
    "split",
    "year",
    "canonical_composer",
    "canonical_title",
    "duration",
)
MAESTRO_SPLITS = {"train", "validation", "test"}


def _manifest_path(root: Path) -> Path:
    manifests = sorted(root.glob("maestro-v*.json"))
    if len(manifests) != 1:
        found = ", ".join(str(path.name) for path in manifests) or "none"
        raise FileNotFoundError(
            "Expected exactly one MAESTRO JSON manifest named 'maestro-v*.json' "
            f"under {root}; found {found}."
        )
    return manifests[0]


def _resolve_manifest_file(root: Path, relative_path: str, *, kind: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"MAESTRO {kind} path must be a safe relative path, got {relative_path!r}.")
    resolved = (root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"MAESTRO manifest references a missing {kind} file: {resolved}")
    return resolved


def load_maestro_records(input_dir: str | Path) -> list[dict[str, Any]]:
    """Load and validate MAESTRO's manifest into one record per performance."""
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MAESTRO root directory does not exist: {root}")

    manifest_path = _manifest_path(root)
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MAESTRO JSON manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"MAESTRO manifest must be a JSON object: {manifest_path}")

    missing_fields = [field for field in MAESTRO_REQUIRED_FIELDS if field not in manifest]
    if missing_fields:
        raise ValueError(f"MAESTRO manifest is missing required fields: {missing_fields}")
    if any(not isinstance(manifest[field], dict) for field in MAESTRO_REQUIRED_FIELDS):
        raise ValueError("MAESTRO manifest fields must be column dictionaries keyed by record ID.")

    record_ids = set(manifest["audio_filename"])
    for field in MAESTRO_REQUIRED_FIELDS:
        if set(manifest[field]) != record_ids:
            raise ValueError(f"MAESTRO manifest field {field!r} does not have the same record IDs.")

    try:
        ordered_ids = sorted(record_ids, key=int)
    except ValueError:
        ordered_ids = sorted(record_ids)

    records = []
    for record_id in ordered_ids:
        split = str(manifest["split"][record_id])
        if split not in MAESTRO_SPLITS:
            raise ValueError(f"Unknown MAESTRO split {split!r} for record {record_id}.")
        audio_relative = str(manifest["audio_filename"][record_id])
        midi_relative = str(manifest["midi_filename"][record_id])
        audio_path = _resolve_manifest_file(root, audio_relative, kind="audio")
        midi_path = _resolve_manifest_file(root, midi_relative, kind="MIDI")
        if audio_path.suffix.lower() != ".wav":
            raise ValueError(f"MAESTRO audio must be WAV, got {audio_path}")
        if midi_path.suffix.lower() not in {".mid", ".midi"}:
            raise ValueError(f"MAESTRO annotation must be MIDI, got {midi_path}")

        # Include the year directory in the ID: filename stems alone are not
        # guaranteed to remain unique across recording years.
        track_id = str(Path(audio_relative).with_suffix(""))
        records.append(
            {
                "audio_path": str(audio_path),
                "metadata": {
                    "dataset_name": "maestro",
                    "track": track_id,
                    "stem": "piano",
                    "program_num": MAESTRO_PIANO_PROGRAM,
                    "midi_path": str(midi_path),
                    "split": split,
                    "split_group": track_id,
                    "maestro_record_id": str(record_id),
                    "maestro_audio_filename": audio_relative,
                    "maestro_midi_filename": midi_relative,
                    "maestro_year": int(manifest["year"][record_id]),
                    "maestro_composer": str(manifest["canonical_composer"][record_id]),
                    "maestro_title": str(manifest["canonical_title"][record_id]),
                    # Provenance only. Preparation reads the WAV header when
                    # it loads audio, which avoids trusting this field for
                    # chunk timing.
                    "maestro_manifest_duration": float(manifest["duration"][record_id]),
                },
            }
        )
    return records
