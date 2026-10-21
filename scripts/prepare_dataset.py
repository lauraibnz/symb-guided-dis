import argparse
import os
import tempfile
from pathlib import Path

import lmdb
from tqdm import tqdm

from symb_guided_dis.autoencoder import music2latent
from symb_guided_dis.dataset.audio import load_audio_chunks, normalize_active_rms_loudness
from symb_guided_dis.dataset.midi import (
    chunk_has_midi_activity,
    load_midi_file_cached,
    require_pretty_midi,
    resolve_example_midi_path,
    transcribe_audio_to_midi,
)
from symb_guided_dis.dataset.parsers import (
    list_dataset_parsers,
    parse_dataset,
    resolve_dataset_midi_path,
    resolve_dataset_midi_roll_path,
)
from symb_guided_dis.dataset.splits import (
    DEFAULT_SPLIT_FRACTIONS,
    DEFAULT_SPLIT_SEED,
    normalize_split_fractions,
)
from symb_guided_dis.dataset.synth import parse_program_exclusion_spec
from symb_guided_dis.dataset.preparation import (
    flush_pending_examples,
    make_stem_rng,
    prepare_structure_aug_chunk_bank,
    prepare_timbre_aug_chunk_bank,
)
from symb_guided_dis.features.clap import DEFAULT_CLAP_MUSIC_CHECKPOINT, CLAPMusicTimbreExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNTH_BINARY = REPO_ROOT / "midi2audio" / "target" / "release" / "midi2audio"
DEFAULT_SOUNDFONT = REPO_ROOT / "midi2audio" / "data" / "TimGM6mb.sf2"
DEFAULT_STRUCT_AUG_EXCLUDE_PROGRAMS = "96-127"
DEFAULT_SPLIT_FRACTIONS_SPEC = ",".join(
    str(DEFAULT_SPLIT_FRACTIONS[name]) for name in ("train", "validation", "test")
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare an LMDB dataset for symb-guided-dis training")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input dataset root. Most parsers expect train/ and validation/; URMP expects its dataset root.",
    )
    parser.add_argument("--output_dir", required=True, help="Output LMDB directory")
    parser.add_argument(
        "--dataset_parser",
        type=str,
        default="slakh",
        choices=list_dataset_parsers(),
        help="Dataset parser used to enumerate audio/MIDI examples.",
    )
    parser.add_argument(
        "--split_fractions",
        default=DEFAULT_SPLIT_FRACTIONS_SPEC,
        help=(
            "Optional train,validation,test fractions for parsers that assign their own splits, "
            f"for example '0.7,0.15,0.15'. Defaults to {DEFAULT_SPLIT_FRACTIONS_SPEC}; "
            "ignored when the input already has train/validation/test directories."
        ),
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed for reproducible parser-assigned train/validation/test splits.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of chunks to encode and write per batch",
    )
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--db_size", type=int, default=200, help="LMDB maximum size in GB")
    parser.add_argument("--sample_rate", type=int, default=44_100, help="Target sample rate in Hz")
    parser.add_argument("--chunk_size", type=float, default=10.0, help="Chunk size in seconds")
    parser.add_argument(
        "--loudness_target_dbfs",
        type=float,
        default=-30.0,
        help="Target active-RMS loudness in dBFS, applied once per source recording.",
    )
    parser.add_argument(
        "--disable_loudness_normalization",
        action="store_true",
        help="Keep source recording levels unchanged.",
    )
    parser.add_argument("--silence_threshold", type=float, default=0.05, help="Peak threshold for amplitude chunk filtering")
    parser.add_argument(
        "--chunk_filter_mode",
        type=str,
        default="auto",
        choices=["auto", "amplitude", "midi"],
        help=(
            "How to skip empty chunks. "
            "'auto' prefers MIDI overlap when symbolic annotations are available, otherwise falls back to amplitude threshold."
        ),
    )
    parser.add_argument(
        "--transcription_backend",
        type=str,
        default="basic_pitch",
        choices=["none", "basic_pitch"],
        help=(
            "Backend used to create MIDI when the dataset parser does not provide it "
            "(default: basic_pitch; use none to require supplied MIDI)."
        ),
    )
    parser.add_argument(
        "--transcription_cache_dir",
        type=str,
        default=None,
        help="Optional directory for caching auto-transcribed MIDI files.",
    )
    parser.add_argument(
        "--save_auto_transcription",
        action="store_true",
        help=(
            "Additionally save automatic-transcription structure targets. Requires "
            "an enabled transcription backend, --save_midi, and --save_struct_aug_latent; writes "
            "midi_roll_<backend> and z_struct_aug_<backend>."
        ),
    )
    parser.add_argument(
        "--emb_model",
        default="music2latent",
        choices=["music2latent", "none"],
        help="Latent codec to use; choose none for audio-only preparation.",
    )
    parser.add_argument("--latents_only", action="store_true", help="Save latents and metadata without raw audio")
    parser.add_argument("--save_midi", action="store_true", help="Save aligned MIDI-roll chunks")
    parser.add_argument(
        "--midi_fps",
        type=float,
        default=100.0,
        help="MIDI-roll frame rate when no latent codec is selected",
    )
    parser.add_argument(
        "--save_struct_aug_latent",
        action="store_true",
        help="Save structure-augmentation latent banks as z_struct_aug",
    )
    parser.add_argument(
        "--struct_aug_synth_binary",
        type=str,
        default=str(DEFAULT_SYNTH_BINARY),
        help="Path to the midi2audio synth binary",
    )
    parser.add_argument(
        "--struct_aug_soundfont",
        type=str,
        default=None,
        help="Optional soundfont path",
    )
    parser.add_argument(
        "--struct_aug_exclude_programs",
        type=str,
        default=DEFAULT_STRUCT_AUG_EXCLUDE_PROGRAMS,
        help="Comma-separated MIDI programs/ranges to exclude from structure augmentation",
    )
    parser.add_argument(
        "--struct_aug_k",
        type=int,
        default=4,
        help="Number of rendered program variants per stem",
    )
    parser.add_argument(
        "--struct_aug_seed",
        type=int,
        default=42,
        help="Seed for deterministic structure-augmentation sampling",
    )
    parser.add_argument(
        "--save_timbre_aug_latent",
        action="store_true",
        help="Save timbre-preserving augmentation latent banks as z_timbre_aug",
    )
    parser.add_argument(
        "--timbre_aug_k",
        type=int,
        default=4,
        help="Number of timbre-preserving variants per stem",
    )
    parser.add_argument("--timbre_aug_ts_min", type=float, default=0.9, help="Minimum tempo factor")
    parser.add_argument("--timbre_aug_ts_max", type=float, default=1.1, help="Maximum tempo factor")
    parser.add_argument("--timbre_aug_pitch_min", type=int, default=-3, help="Minimum pitch shift in semitones")
    parser.add_argument("--timbre_aug_pitch_max", type=int, default=3, help="Maximum pitch shift in semitones")
    parser.add_argument(
        "--timbre_aug_seed",
        type=int,
        default=42,
        help="Seed for deterministic timbre-augmentation sampling",
    )
    parser.add_argument(
        "--save_clap_timbre",
        action="store_true",
        help="Save frozen music-CLAP timbre vectors as clap and clap_timbre_aug.",
    )
    parser.add_argument(
        "--clap_checkpoint",
        default=DEFAULT_CLAP_MUSIC_CHECKPOINT,
        help="Local LAION music-CLAP checkpoint path used with --save_clap_timbre.",
    )
    parser.add_argument(
        "--clap_batch_size",
        type=int,
        default=None,
        help="Waveform chunks per CLAP forward pass (default: --batch_size).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    chunk_filter_mode = str(args.chunk_filter_mode).lower()
    transcription_backend = str(args.transcription_backend).lower()
    require_symbolic_for_output = bool(args.save_midi or args.save_struct_aug_latent)

    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch_size must be > 0, got {args.batch_size}")
    if int(args.sample_rate) <= 0:
        raise ValueError(f"--sample_rate must be > 0, got {args.sample_rate}")
    if float(args.chunk_size) <= 0.0:
        raise ValueError(f"--chunk_size must be > 0, got {args.chunk_size}")
    if float(args.silence_threshold) < 0.0:
        raise ValueError(f"--silence_threshold must be non-negative, got {args.silence_threshold}")
    if args.latents_only and args.emb_model == "none":
        raise ValueError("--latents_only requires a latent codec; use --emb_model music2latent.")
    if args.clap_batch_size is not None and int(args.clap_batch_size) <= 0:
        raise ValueError(f"--clap_batch_size must be > 0, got {args.clap_batch_size}")
    if args.disable_loudness_normalization:
        args.loudness_target_dbfs = None
    if args.save_auto_transcription:
        if transcription_backend == "none":
            raise ValueError("--save_auto_transcription requires --transcription_backend (currently basic_pitch).")
        if not args.save_midi or not args.save_struct_aug_latent:
            raise ValueError(
                "--save_auto_transcription requires --save_midi and --save_struct_aug_latent "
                "to save a matched symbolic pair."
            )

    env = lmdb.open(
        args.output_dir,
        map_size=int(args.db_size) * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    model = music2latent.build(device=args.device) if args.emb_model == "music2latent" else None
    if (args.save_struct_aug_latent or args.save_timbre_aug_latent) and model is None:
        raise ValueError(
            "--save_struct_aug_latent and --save_timbre_aug_latent require --emb_model music2latent."
        )
    clap_extractor = None
    if args.save_clap_timbre:
        clap_extractor = CLAPMusicTimbreExtractor(
            checkpoint_path=args.clap_checkpoint,
            device=args.device,
        )
    clap_batch_size = int(args.clap_batch_size or args.batch_size)

    if args.save_struct_aug_latent:
        synth_binary = Path(args.struct_aug_synth_binary)
        if not synth_binary.exists():
            raise FileNotFoundError(
                f"Synth binary not found at '{synth_binary}'. "
                "Compile midi2audio first or set --struct_aug_synth_binary."
            )
        if args.struct_aug_soundfont is None and DEFAULT_SOUNDFONT.exists():
            args.struct_aug_soundfont = str(DEFAULT_SOUNDFONT)

        excluded_programs = parse_program_exclusion_spec(args.struct_aug_exclude_programs)
        invalid_programs = sorted(program for program in excluded_programs if not 0 <= int(program) <= 127)
        if invalid_programs:
            raise ValueError(f"MIDI programs must be in [0, 127], got {invalid_programs}.")
        allowed_count = 128 - len(excluded_programs)
        if allowed_count <= 0:
            raise ValueError(
                "No allowed structure-augmentation programs remain after exclusions. "
                f"excluded={sorted(excluded_programs)}"
            )
        print(
            "Struct-aug program policy: "
            f"excluded_count={len(excluded_programs)}, allowed_count={allowed_count}"
        )
    else:
        excluded_programs = set()

    pretty_midi_module = None
    midi_cache = {}
    missing_midi_paths = set()
    symbolic_midi_tempdir = tempfile.TemporaryDirectory(prefix="symb_guided_dis_symbolic_midi_")
    auto_transcription_tempdir = None
    if args.save_auto_transcription and args.transcription_cache_dir is None:
        auto_transcription_tempdir = tempfile.TemporaryDirectory(prefix="symb_guided_dis_auto_transcription_")
    auto_transcription_cache_dir = (
        args.transcription_cache_dir
        if args.transcription_cache_dir is not None
        else (None if auto_transcription_tempdir is None else auto_transcription_tempdir.name)
    )
    split_dirs = {
        split: os.path.join(args.input_dir, split)
        for split in ("train", "validation", "test")
    }
    if any(os.path.isdir(split_dir) for split_dir in split_dirs.values()):
        if args.split_fractions != DEFAULT_SPLIT_FRACTIONS_SPEC:
            print("Ignoring --split_fractions because the input has explicit split directories.")
        records_by_split = {}
        for split, split_dir in split_dirs.items():
            if not os.path.isdir(split_dir):
                print(f"Warning: {split_dir} does not exist, skipping.")
                continue
            records_by_split[split] = parse_dataset(split_dir, parser_name=args.dataset_parser)
    else:
        parser_kwargs = {
            "split_fractions": normalize_split_fractions(args.split_fractions),
            "split_seed": int(args.split_seed),
        }
        records_by_split = {"train": [], "validation": [], "test": []}
        for record in parse_dataset(
            args.input_dir,
            parser_name=args.dataset_parser,
            **parser_kwargs,
        ):
            record_split = record.get("metadata", {}).get("split")
            if record_split not in records_by_split:
                raise ValueError(
                    "A parser used without train/validation directories must assign every record "
                    f"a 'train', 'validation', or 'test' metadata split; got {record_split!r}."
                )
            records_by_split[record_split].append(record)

    chunk_id = 0
    for split in ("train", "validation", "test"):
        records = records_by_split.get(split, [])
        if not records:
            print(f"Warning: no {split} records found, skipping.")
            continue

        pending_examples = []
        for record in tqdm(records, desc=f"{split}"):
            audio_path = record["audio_path"]
            meta = dict(record["metadata"])
            track_name = meta.get("track", Path(audio_path).parent.name)
            stem_name = meta.get("stem", Path(audio_path).stem)
            original_midi_path = meta.get("midi_path")
            midi_path = original_midi_path
            midi_roll_path = original_midi_path
            midi_roll_obj = None

            original_program = meta.get("program_num")

            needs_midi_for_example = bool(
                require_symbolic_for_output
                or chunk_filter_mode in {"auto", "midi"}
            )
            midi_source = "dataset" if midi_path else "missing"
            midi_roll_source = midi_source
            if needs_midi_for_example:
                parser_midi_path, parser_midi_source = resolve_dataset_midi_path(
                    args.dataset_parser,
                    metadata=meta,
                    midi_cache_dir=symbolic_midi_tempdir.name,
                )
                if parser_midi_path is not None:
                    midi_path, midi_source = parser_midi_path, parser_midi_source
                else:
                    midi_path, midi_source = resolve_example_midi_path(
                        audio_path=audio_path,
                        midi_path=original_midi_path,
                        transcription_backend=transcription_backend,
                        transcription_cache_dir=args.transcription_cache_dir,
                        transcription_force=False,
                    )

                # A parser may provide separate MIDI for a binary pitch target.
                # GuitarSet supplies a rounded no-bend copy here, while its
                # bend-preserving MIDI remains in ``midi_path`` for structural
                # augmentation synthesis.
                midi_roll_path, midi_roll_source = resolve_dataset_midi_roll_path(
                    args.dataset_parser,
                    metadata=meta,
                    midi_cache_dir=symbolic_midi_tempdir.name,
                )
                if midi_roll_path is None:
                    midi_roll_path, midi_roll_source = midi_path, midi_source
                # Parser-generated MIDI is temporary. Persist its provenance,
                # not its transient path, because the LMDB stores midi_roll.
                if parser_midi_path is None:
                    meta["midi_path"] = midi_path
                else:
                    meta.pop("midi_path", None)
                meta["midi_source"] = midi_source
                meta["midi_roll_source"] = midi_roll_source

            if args.save_auto_transcription and midi_source == "transcription":
                raise ValueError(
                    "--save_auto_transcription requires an independent reference MIDI source. "
                    f"This example only has a {transcription_backend} transcription: {audio_path}"
                )

            waveform, chunks, chunk_samples, truncated_samples = load_audio_chunks(
                audio_path,
                sample_rate=args.sample_rate,
                chunk_size_seconds=args.chunk_size,
            )
            if chunks is None:
                continue

            loudness_metadata = None
            if args.loudness_target_dbfs is not None:
                waveform, loudness_metadata = normalize_active_rms_loudness(
                    waveform,
                    sample_rate=args.sample_rate,
                    target_dbfs=args.loudness_target_dbfs,
                )
                chunks = waveform[:truncated_samples].reshape(-1, chunk_samples).contiguous()

            use_midi_chunk_filter = False
            if chunk_filter_mode == "midi":
                if midi_roll_path is None:
                    raise ValueError(
                        f"--chunk_filter_mode=midi requires MIDI for every example, but none is available for "
                        f"track={track_name}, stem={stem_name}, audio={audio_path}."
                    )
                pretty_midi_module = pretty_midi_module or require_pretty_midi()
                midi_roll_obj = load_midi_file_cached(
                    midi_roll_path,
                    midi_cache=midi_cache,
                    missing_midi_paths=missing_midi_paths,
                    pretty_midi_module=pretty_midi_module,
                )
                use_midi_chunk_filter = True
            elif chunk_filter_mode == "auto" and midi_roll_path is not None:
                pretty_midi_module = pretty_midi_module or require_pretty_midi()
                midi_roll_obj = load_midi_file_cached(
                    midi_roll_path,
                    midi_cache=midi_cache,
                    missing_midi_paths=missing_midi_paths,
                    pretty_midi_module=pretty_midi_module,
                )
                use_midi_chunk_filter = midi_roll_obj is not None

            if args.save_midi and midi_roll_obj is None:
                pretty_midi_module = pretty_midi_module or require_pretty_midi()
                midi_roll_obj = load_midi_file_cached(
                    midi_roll_path,
                    midi_cache=midi_cache,
                    missing_midi_paths=missing_midi_paths,
                    pretty_midi_module=pretty_midi_module,
                )

            auto_midi_path = None
            auto_midi_obj = None
            if args.save_auto_transcription:
                auto_midi_path = transcribe_audio_to_midi(
                    audio_path=audio_path,
                    output_dir=auto_transcription_cache_dir,
                    backend=transcription_backend,
                    force=False,
                )
                pretty_midi_module = pretty_midi_module or require_pretty_midi()
                auto_midi_obj = load_midi_file_cached(
                    auto_midi_path,
                    midi_cache=midi_cache,
                    missing_midi_paths=missing_midi_paths,
                    pretty_midi_module=pretty_midi_module,
                )
                if auto_midi_obj is None:
                    raise RuntimeError(
                        f"Automatic transcription produced no readable MIDI for audio={audio_path}."
                    )

            struct_aug_chunks_bank = None
            struct_aug_programs = None
            if args.save_struct_aug_latent:
                if midi_path is None:
                    raise ValueError(
                        "--save_struct_aug_latent requires MIDI per example. "
                        f"No MIDI was found or transcribed for track={track_name}, stem={stem_name}, audio={audio_path}."
                    )
                pretty_midi_module = pretty_midi_module or require_pretty_midi()
                struct_rng = make_stem_rng(
                    args.struct_aug_seed,
                    split=split,
                    track=track_name,
                    stem=stem_name,
                )
                struct_aug_chunks_bank, struct_aug_programs = prepare_structure_aug_chunk_bank(
                    waveform=waveform,
                    chunks=chunks,
                    truncated_samples=truncated_samples,
                    chunk_samples=chunk_samples,
                    midi_path=midi_path,
                    original_program=original_program,
                    track_name=track_name,
                    stem_name=stem_name,
                    sample_rate=args.sample_rate,
                    struct_aug_k=args.struct_aug_k,
                    synth_binary=args.struct_aug_synth_binary,
                    soundfont=args.struct_aug_soundfont,
                    excluded_programs=excluded_programs,
                    rng=struct_rng,
                    pretty_midi_module=pretty_midi_module,
                )

            auto_struct_aug_chunks_bank = None
            auto_struct_aug_programs = None
            if args.save_auto_transcription:
                auto_struct_rng = make_stem_rng(
                    args.struct_aug_seed,
                    split=split,
                    track=track_name,
                    stem=stem_name,
                )
                auto_struct_aug_chunks_bank, auto_struct_aug_programs = prepare_structure_aug_chunk_bank(
                    waveform=waveform,
                    chunks=chunks,
                    truncated_samples=truncated_samples,
                    chunk_samples=chunk_samples,
                    midi_path=auto_midi_path,
                    original_program=original_program,
                    track_name=track_name,
                    stem_name=stem_name,
                    sample_rate=args.sample_rate,
                    struct_aug_k=args.struct_aug_k,
                    synth_binary=args.struct_aug_synth_binary,
                    soundfont=args.struct_aug_soundfont,
                    excluded_programs=excluded_programs,
                    rng=auto_struct_rng,
                    pretty_midi_module=pretty_midi_module,
                )

            timbre_aug_chunks_bank = None
            timbre_aug_params = None
            if args.save_timbre_aug_latent or args.save_clap_timbre:
                timbre_rng = make_stem_rng(
                    args.timbre_aug_seed,
                    split=split,
                    track=track_name,
                    stem=stem_name,
                )
                timbre_aug_chunks_bank, timbre_aug_params = prepare_timbre_aug_chunk_bank(
                    waveform=waveform,
                    chunks=chunks,
                    sample_rate=args.sample_rate,
                    chunk_samples=chunk_samples,
                    timbre_aug_k=args.timbre_aug_k,
                    ts_min=args.timbre_aug_ts_min,
                    ts_max=args.timbre_aug_ts_max,
                    pitch_min=args.timbre_aug_pitch_min,
                    pitch_max=args.timbre_aug_pitch_max,
                    rng=timbre_rng,
                    track_name=track_name,
                    stem_name=stem_name,
                )

            for chunk_index, chunk in enumerate(chunks):
                start_sec = float(chunk_index) * float(args.chunk_size)
                end_sec = start_sec + float(args.chunk_size)
                if use_midi_chunk_filter:
                    if not chunk_has_midi_activity(midi_roll_obj, start_sec=start_sec, end_sec=end_sec):
                        continue
                elif float(chunk.abs().max()) < float(args.silence_threshold):
                    continue

                chunk_meta = meta.copy()
                chunk_meta["split"] = split
                chunk_meta["chunk_index"] = int(chunk_index)
                chunk_meta["emb_model"] = str(args.emb_model)
                if loudness_metadata is not None:
                    chunk_meta["source_loudness_normalization"] = loudness_metadata
                if args.save_struct_aug_latent:
                    chunk_meta["struct_aug_programs"] = struct_aug_programs
                if args.save_auto_transcription:
                    chunk_meta["automatic_transcription_backend"] = transcription_backend
                    chunk_meta[f"struct_aug_programs_{transcription_backend}"] = auto_struct_aug_programs
                if args.save_timbre_aug_latent or args.save_clap_timbre:
                    chunk_meta["timbre_aug_params"] = [
                        timbre_aug_params[k_idx][chunk_index]
                        for k_idx in range(len(timbre_aug_params))
                    ]

                pending_examples.append(
                    {
                        "chunk": chunk,
                        "metadata": chunk_meta,
                        "struct_aug_chunks": (
                            None if struct_aug_chunks_bank is None else struct_aug_chunks_bank[:, chunk_index, :]
                        ),
                        "auto_struct_aug_chunks": (
                            None
                            if auto_struct_aug_chunks_bank is None
                            else auto_struct_aug_chunks_bank[:, chunk_index, :]
                        ),
                        "timbre_aug_chunks": (
                            None if timbre_aug_chunks_bank is None else timbre_aug_chunks_bank[:, chunk_index, :]
                        ),
                        "midi_roll_obj": midi_roll_obj if args.save_midi else None,
                        "auto_midi_obj": auto_midi_obj if args.save_auto_transcription else None,
                    }
                )

                if len(pending_examples) >= int(args.batch_size):
                    chunk_id = flush_pending_examples(
                        env=env,
                        pending_examples=pending_examples,
                        chunk_id_start=chunk_id,
                        model=model,
                        device=args.device,
                        save_timbre_aug_latent=args.save_timbre_aug_latent,
                        clap_extractor=clap_extractor,
                        clap_batch_size=clap_batch_size,
                        clap_source_sample_rate=args.sample_rate,
                        latents_only=args.latents_only,
                        save_midi=args.save_midi,
                        save_auto_transcription=args.save_auto_transcription,
                        auto_transcription_key=transcription_backend,
                        midi_fps=args.midi_fps,
                        chunk_size=args.chunk_size,
                    )
                    pending_examples = []

        chunk_id = flush_pending_examples(
            env=env,
            pending_examples=pending_examples,
            chunk_id_start=chunk_id,
            model=model,
            device=args.device,
            save_timbre_aug_latent=args.save_timbre_aug_latent,
            clap_extractor=clap_extractor,
            clap_batch_size=clap_batch_size,
            clap_source_sample_rate=args.sample_rate,
            latents_only=args.latents_only,
            save_midi=args.save_midi,
            save_auto_transcription=args.save_auto_transcription,
            auto_transcription_key=transcription_backend,
            midi_fps=args.midi_fps,
            chunk_size=args.chunk_size,
        )

    env.close()
    symbolic_midi_tempdir.cleanup()
    if auto_transcription_tempdir is not None:
        auto_transcription_tempdir.cleanup()
    print(f"Dataset preparation complete. Total chunks saved: {chunk_id}")

if __name__ == "__main__":
    main()
