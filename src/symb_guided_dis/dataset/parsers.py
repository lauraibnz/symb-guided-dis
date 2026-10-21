import os
import re
from pathlib import Path
from typing import Callable

from tqdm import tqdm
import yaml

from symb_guided_dis.dataset.urmp import resolve_urmp_reference_midi, urmp_instrument_program
from symb_guided_dis.dataset.guitarset import (
    GUITARSET_PROGRAM,
    resolve_guitarset_discrete_midi,
    resolve_guitarset_reference_midi,
)
from symb_guided_dis.dataset.maestro import load_maestro_records
from symb_guided_dis.dataset.splits import (
    DEFAULT_SPLIT_FRACTIONS,
    DEFAULT_SPLIT_SEED,
    assign_group_splits,
    normalize_split_fractions,
)


DATASET_PARSERS = {}
DATASET_MIDI_RESOLVERS = {}
DATASET_MIDI_ROLL_RESOLVERS = {}
SLAKH_AUDIO_EXTENSIONS = (".flac", ".wav")


def register_dataset_parser(name: str) -> Callable:
    parser_name = str(name).strip().lower()
    if not parser_name:
        raise ValueError("Dataset parser name must be non-empty.")

    def decorator(fn: Callable) -> Callable:
        if parser_name in DATASET_PARSERS:
            raise ValueError(f"Dataset parser '{parser_name}' is already registered.")
        DATASET_PARSERS[parser_name] = fn
        return fn

    return decorator


def register_dataset_midi_resolver(name: str) -> Callable:
    """Register a parser-owned symbolic-MIDI resolver for preparation."""
    parser_name = str(name).strip().lower()
    if not parser_name:
        raise ValueError("Dataset parser name must be non-empty.")

    def decorator(fn: Callable) -> Callable:
        if parser_name in DATASET_MIDI_RESOLVERS:
            raise ValueError(f"Dataset MIDI resolver '{parser_name}' is already registered.")
        DATASET_MIDI_RESOLVERS[parser_name] = fn
        return fn

    return decorator


def resolve_dataset_midi_path(parser_name, metadata, midi_cache_dir):
    """Resolve parser-specific symbolic MIDI, if the parser supplies one."""
    resolver = DATASET_MIDI_RESOLVERS.get(str(parser_name).strip().lower())
    if resolver is None:
        return None, None
    return resolver(metadata=metadata, midi_cache_dir=midi_cache_dir)


def register_dataset_midi_roll_resolver(name: str) -> Callable:
    """Register a parser-owned resolver for the MIDI used to build midi_roll."""
    parser_name = str(name).strip().lower()
    if not parser_name:
        raise ValueError("Dataset parser name must be non-empty.")

    def decorator(fn: Callable) -> Callable:
        if parser_name in DATASET_MIDI_ROLL_RESOLVERS:
            raise ValueError(f"MIDI-roll resolver for dataset parser '{parser_name}' is already registered.")
        DATASET_MIDI_ROLL_RESOLVERS[parser_name] = fn
        return fn

    return decorator


def resolve_dataset_midi_roll_path(parser_name, metadata, midi_cache_dir):
    """Resolve parser-specific MIDI for the stored discrete pitch-roll target."""
    resolver = DATASET_MIDI_ROLL_RESOLVERS.get(str(parser_name).strip().lower())
    if resolver is None:
        return None, None
    return resolver(metadata=metadata, midi_cache_dir=midi_cache_dir)


def list_dataset_parsers():
    return sorted(DATASET_PARSERS.keys())


def parse_dataset(input_dir, parser_name="slakh", **parser_kwargs):
    parser_name = str(parser_name).strip().lower()
    if parser_name not in DATASET_PARSERS:
        raise ValueError(
            f"Unknown dataset parser '{parser_name}'. "
            f"Available parsers: {', '.join(list_dataset_parsers())}"
        )
    return DATASET_PARSERS[parser_name](input_dir=input_dir, **parser_kwargs)


@register_dataset_parser("slakh")
def parse_slakh(input_dir):
    """Enumerate Slakh stems, preferring FLAC and falling back to WAV."""
    tracks = [
        os.path.join(input_dir, subfolder)
        for subfolder in sorted(os.listdir(input_dir))
    ]
    ban_list = {
        "Chromatic Percussion",
        "Drums",
        "Percussive",
        "Sound Effects",
        "Sound effects",
    }

    records = []
    stem_count = 0
    for track in tqdm(tracks):
        metadata_path = os.path.join(track, "metadata.yaml")
        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)

        for stem_name, stem in meta["stems"].items():
            if stem["inst_class"] in ban_list:
                continue

            stem_dir = os.path.join(track, "stems")
            stem_path = next(
                (
                    os.path.join(stem_dir, stem_name + extension)
                    for extension in SLAKH_AUDIO_EXTENSIONS
                    if os.path.exists(os.path.join(stem_dir, stem_name + extension))
                ),
                None,
            )
            if stem_path is None:
                continue

            midi_path = os.path.join(track, "MIDI", stem_name + ".mid")
            stem_meta = dict(stem)
            stem_meta["dataset_name"] = "slakh"
            stem_meta["track"] = os.path.basename(track)
            stem_meta["stem"] = stem_name
            stem_meta["midi_path"] = midi_path if os.path.exists(midi_path) else None

            records.append(
                {
                    "audio_path": stem_path,
                    "metadata": stem_meta,
                }
            )
            stem_count += 1

    print(f"Found {stem_count} stems in {len(tracks)} tracks.")
    return records

@register_dataset_parser("urmp")
def parse_urmp(
    input_dir,
    split_fractions=DEFAULT_SPLIT_FRACTIONS,
    split_seed=DEFAULT_SPLIT_SEED,
):
    """Enumerate URMP stems and their audio-aligned per-stem Notes files.

    ``Sco_*.mid`` is retained only as metadata: it is score-time MIDI and is
    not time-aligned to the recorded performances. Preparation converts the
    paired ``Notes_*.txt`` file to the MIDI used for labels and synthesis.
    """
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"URMP root directory does not exist: {root}")
    split_fractions = normalize_split_fractions(split_fractions)

    piece_dirs = [
        path for path in sorted(root.iterdir())
        if path.is_dir() and path.name != "Supplementary_Files"
    ]
    if not piece_dirs:
        raise ValueError(f"No URMP piece directories found in {root}.")

    piece_splits = assign_group_splits(
        (path.name for path in piece_dirs), fractions=split_fractions, seed=split_seed
    )

    records = []
    stem_pattern = re.compile(r"^AuSep_(?P<index>\d+)_(?P<instrument>[^_]+)_.+\.wav$", re.IGNORECASE)
    for piece_dir in piece_dirs:
        score_midi_paths = sorted(piece_dir.glob("Sco_*.mid"))
        score_midi_path = score_midi_paths[0] if len(score_midi_paths) == 1 else None
        split = piece_splits[piece_dir.name]
        for audio_path in sorted(piece_dir.glob("AuSep_*.wav")):
            match = stem_pattern.match(audio_path.name)
            if match is None:
                continue
            stem_index = int(match.group("index"))
            instrument_code = match.group("instrument").lower()
            prefix = f"AuSep_{stem_index}_{instrument_code}_"
            suffix = audio_path.name[len(prefix) : -len(audio_path.suffix)]
            notes_path = piece_dir / f"Notes_{stem_index}_{instrument_code}_{suffix}.txt"
            if not notes_path.is_file():
                raise FileNotFoundError(
                    f"URMP stem {audio_path} has no matching Notes annotation: {notes_path}"
                )
            records.append(
                {
                    "audio_path": str(audio_path),
                    "metadata": {
                        "dataset_name": "urmp",
                        "track": piece_dir.name,
                        "stem": f"{stem_index}_{instrument_code}",
                        "stem_index": stem_index,
                        "instrument_code": instrument_code,
                        "program_num": urmp_instrument_program(instrument_code),
                        "urmp_notes_path": str(notes_path),
                        "score_midi_path": None if score_midi_path is None else str(score_midi_path),
                        "split": split,
                        "split_group": piece_dir.name,
                    },
                }
            )

    split_counts = {
        split: sum(record["metadata"]["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    print(
        f"Found {len(records)} URMP stems in {len(piece_dirs)} pieces "
        f"(train={split_counts['train']}, validation={split_counts['validation']}, "
        f"test={split_counts['test']}; fractions={split_fractions}, seed={split_seed})."
    )
    return records

@register_dataset_midi_resolver("urmp")
def resolve_urmp_dataset_midi(metadata, midi_cache_dir):
    return resolve_urmp_reference_midi(metadata, midi_cache_dir)


@register_dataset_parser("guitarset")
def parse_guitarset(
    input_dir,
    split_fractions=DEFAULT_SPLIT_FRACTIONS,
    split_seed=DEFAULT_SPLIT_SEED,
):
    """Enumerate GuitarSet mono-mic recordings with JAMS-derived MIDI.

    All six performer takes and both solo/accompaniment arrangements of one
    named GuitarSet excerpt share a ``split_group`` and therefore cannot leak
    across train, validation, and test.
    """
    root = Path(input_dir).expanduser().resolve()
    audio_dir = root / "audio_mono-mic"
    annotation_dir = root / "annotation"
    if not audio_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            "GuitarSet requires audio_mono-mic/ and annotation/ below the input root; "
            f"got {root}."
        )
    split_fractions = normalize_split_fractions(split_fractions)
    filename_pattern = re.compile(
        r"^(?P<performer>\d{2})_(?P<excerpt>.+)_(?P<part>comp|solo)_mic\.wav$",
        re.IGNORECASE,
    )
    parsed_audio = []
    for audio_path in sorted(audio_dir.glob("*.wav")):
        match = filename_pattern.match(audio_path.name)
        if match is None:
            continue
        performer = match.group("performer")
        excerpt = match.group("excerpt")
        part = match.group("part").lower()
        jams_path = annotation_dir / f"{performer}_{excerpt}_{part}.jams"
        if not jams_path.is_file():
            raise FileNotFoundError(f"GuitarSet recording {audio_path} has no matching JAMS file: {jams_path}")
        parsed_audio.append((audio_path, performer, excerpt, part, jams_path))
    if not parsed_audio:
        raise ValueError(f"No GuitarSet mono-mic WAV files found in {audio_dir}.")

    excerpt_splits = assign_group_splits(
        (excerpt for _, _, excerpt, _, _ in parsed_audio),
        fractions=split_fractions,
        seed=split_seed,
    )
    records = []
    for audio_path, performer, excerpt, part, jams_path in parsed_audio:
        recording_id = audio_path.stem.removesuffix("_mic")
        records.append(
            {
                "audio_path": str(audio_path),
                "metadata": {
                    "dataset_name": "guitarset",
                    "track": recording_id,
                    "stem": "guitar",
                    "program_num": GUITARSET_PROGRAM,
                    "guitarset_performer": performer,
                    "guitarset_excerpt": excerpt,
                    "guitarset_part": part,
                    "guitarset_jams_path": str(jams_path),
                    "split_group": excerpt,
                    "split": excerpt_splits[excerpt],
                },
            }
        )

    split_counts = {
        split: sum(record["metadata"]["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    group_counts = {
        split: sum(assigned_split == split for assigned_split in excerpt_splits.values())
        for split in ("train", "validation", "test")
    }
    print(
        f"Found {len(records)} GuitarSet recordings from {len(excerpt_splits)} excerpts "
        f"(train={split_counts['train']}, validation={split_counts['validation']}, test={split_counts['test']}; "
        f"excerpt groups={group_counts['train']}/{group_counts['validation']}/{group_counts['test']}, "
        f"fractions={split_fractions}, seed={split_seed})."
    )
    return records


@register_dataset_midi_resolver("guitarset")
def resolve_guitarset_dataset_midi(metadata, midi_cache_dir):
    return resolve_guitarset_reference_midi(metadata, midi_cache_dir)


@register_dataset_midi_roll_resolver("guitarset")
def resolve_guitarset_dataset_midi_roll(metadata, midi_cache_dir):
    return resolve_guitarset_discrete_midi(metadata, midi_cache_dir)


@register_dataset_parser("maestro")
def parse_maestro(
    input_dir,
    split_fractions=DEFAULT_SPLIT_FRACTIONS,
    split_seed=DEFAULT_SPLIT_SEED,
):
    """Enumerate MAESTRO v3 performances from its native JSON manifest.

    MAESTRO provides its own official recording-level train/validation/test
    partition, so ``split_fractions`` and ``split_seed`` are intentionally
    ignored. The parser validates every manifest-referenced WAV/MIDI pair.
    """
    del split_fractions, split_seed
    records = load_maestro_records(input_dir)
    split_counts = {
        split: sum(record["metadata"]["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    print(
        f"Found {len(records)} MAESTRO performances "
        f"(train={split_counts['train']}, validation={split_counts['validation']}, "
        f"test={split_counts['test']}; native MAESTRO split)."
    )
    return records
