"""Reusable batching and augmentation helpers for LMDB dataset preparation."""

import hashlib
import os
import random
import tempfile

import torch
import torchaudio

from symb_guided_dis.autoencoder import music2latent
from symb_guided_dis.dataset.audio import DEFAULT_CHUNK_SIZE_SECONDS, DEFAULT_SAMPLE_RATE
from symb_guided_dis.dataset.audio_example import AudioExample, save_many_to_lmdb
from symb_guided_dis.dataset.midi import build_midi_roll_chunk
from symb_guided_dis.dataset.synth import (
    align_waveform_to_target,
    pick_random_programs,
    run_synth_render,
    write_program_shifted_midi,
)
from symb_guided_dis.dataset.transforms import build_timbre_augmented_chunk_bank
from symb_guided_dis.features.clap import extract_clap_timbre_embeddings


DEFAULT_AUGMENTATION_SEED = 42
DEFAULT_STRUCTURE_AUGMENTATION_K = 4
DEFAULT_STRUCTURE_PROGRAM_MIN = 0
DEFAULT_STRUCTURE_PROGRAM_MAX = 127
DEFAULT_TIMBRE_AUGMENTATION_K = 4


def make_stem_rng(base_seed: int | None = DEFAULT_AUGMENTATION_SEED, *, split, track, stem):
    if base_seed is None:
        return random.Random()
    seed_material = f"{split}/{track}/{stem}".encode("utf-8")
    seed_offset = int(hashlib.sha1(seed_material).hexdigest()[:8], 16)
    return random.Random(int(base_seed) + seed_offset)


def prepare_structure_aug_chunk_bank(
    *,
    waveform,
    chunks,
    truncated_samples,
    chunk_samples,
    midi_path,
    original_program,
    track_name,
    stem_name,
    sample_rate,
    synth_binary,
    soundfont,
    excluded_programs,
    rng,
    pretty_midi_module,
    struct_aug_k: int = DEFAULT_STRUCTURE_AUGMENTATION_K,
    program_min: int = DEFAULT_STRUCTURE_PROGRAM_MIN,
    program_max: int = DEFAULT_STRUCTURE_PROGRAM_MAX,
    strict: bool = False,
):
    programs = pick_random_programs(
        rng,
        k=int(struct_aug_k),
        min_program=int(program_min),
        max_program=int(program_max),
        avoid_program=original_program,
        excluded_programs=excluded_programs,
    )

    try:
        if midi_path is None or not os.path.exists(midi_path):
            raise FileNotFoundError(f"MIDI path not found: {midi_path}")

        bank_chunks = []
        with tempfile.TemporaryDirectory(prefix="symb_guided_dis_struct_aug_") as tmpdir:
            for idx, program in enumerate(programs):
                shifted_midi = os.path.join(tmpdir, f"shifted_{idx}.mid")
                rendered_wav = os.path.join(tmpdir, f"rendered_{idx}.wav")

                write_program_shifted_midi(
                    input_midi_path=midi_path,
                    output_midi_path=shifted_midi,
                    target_program=program,
                    pretty_midi_module=pretty_midi_module,
                )
                run_synth_render(
                    midi_path=shifted_midi,
                    wav_path=rendered_wav,
                    program=program,
                    synth_binary=synth_binary,
                    soundfont=soundfont,
                )
                rendered_waveform, rendered_sr = torchaudio.load(rendered_wav)
                if rendered_sr != sample_rate:
                    rendered_waveform = torchaudio.transforms.Resample(rendered_sr, sample_rate)(rendered_waveform)

                rendered_waveform = align_waveform_to_target(rendered_waveform, waveform)
                if rendered_waveform.dim() == 2:
                    rendered_waveform = rendered_waveform.mean(dim=0)
                rendered_waveform = rendered_waveform[:truncated_samples]
                bank_chunks.append(rendered_waveform.reshape(-1, chunk_samples))

        return torch.stack(bank_chunks, dim=0), programs
    except Exception as exc:
        msg = (
            f"Structure augmentation failed for track={track_name}, stem={stem_name}, "
            f"midi={midi_path}, programs={programs}: {exc}"
        )
        if strict:
            raise RuntimeError(msg) from exc
        print(f"Warning: {msg}. Falling back to original chunks.")
        fallback_program = None if original_program is None else int(original_program)
        return chunks.unsqueeze(0).repeat(int(struct_aug_k), 1, 1), [fallback_program] * int(struct_aug_k)


def prepare_timbre_aug_chunk_bank(
    *,
    waveform,
    chunks,
    sample_rate,
    chunk_samples,
    rng,
    track_name,
    stem_name,
    timbre_aug_k: int = DEFAULT_TIMBRE_AUGMENTATION_K,
    ts_min: float = 0.9,
    ts_max: float = 1.1,
    pitch_min: int = -3,
    pitch_max: int = 3,
    strict: bool = False,
):
    try:
        return build_timbre_augmented_chunk_bank(
            waveform,
            sample_rate=sample_rate,
            chunk_samples=chunk_samples,
            num_augments=int(timbre_aug_k),
            ts_min=float(ts_min),
            ts_max=float(ts_max),
            pitch_min=int(pitch_min),
            pitch_max=int(pitch_max),
            rng=rng,
        )
    except Exception as exc:
        msg = f"Timbre augmentation failed for track={track_name}, stem={stem_name}: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        print(f"Warning: {msg}. Falling back to original chunks.")
        bank = chunks.unsqueeze(0).repeat(int(timbre_aug_k), 1, 1)
        params = [
            [{"fallback": "original"} for _ in range(int(chunks.shape[0]))]
            for _ in range(int(timbre_aug_k))
        ]
        return bank, params


def flush_pending_examples(
    *,
    env,
    pending_examples,
    chunk_id_start,
    model,
    device,
    save_timbre_aug_latent: bool = False,
    clap_extractor=None,
    clap_batch_size: int | None = None,
    save_auto_transcription: bool = False,
    auto_transcription_key: str | None = None,
    clap_source_sample_rate: int = DEFAULT_SAMPLE_RATE,
    latents_only: bool = True,
    save_midi: bool = True,
    midi_fps: float = 100.0,
    chunk_size: float = DEFAULT_CHUNK_SIZE_SECONDS,
):
    if not pending_examples:
        return chunk_id_start

    chunks_tensor = torch.stack([item["chunk"] for item in pending_examples]).to(device)
    z_batch = [None] * len(chunks_tensor) if model is None else music2latent.encode_batch(model, chunks_tensor)

    z_struct_aug_batch = [None] * len(pending_examples)
    struct_aug_chunks = [item["struct_aug_chunks"] for item in pending_examples]
    if any(chunks is not None for chunks in struct_aug_chunks):
        struct_aug_bank = torch.stack(struct_aug_chunks).to(device)
        batch_size = int(struct_aug_bank.shape[0])
        bank_size = int(struct_aug_bank.shape[1])
        z_struct_flat = music2latent.encode_batch(
            model,
            struct_aug_bank.reshape(batch_size * bank_size, *struct_aug_bank.shape[2:]),
        )
        z_struct_aug_batch = z_struct_flat.reshape(batch_size, bank_size, *z_struct_flat.shape[1:])

    z_auto_struct_aug_batch = [None] * len(pending_examples)
    auto_struct_aug_chunks = [item["auto_struct_aug_chunks"] for item in pending_examples]
    if save_auto_transcription and any(chunks is not None for chunks in auto_struct_aug_chunks):
        auto_struct_aug_bank = torch.stack(auto_struct_aug_chunks).to(device)
        batch_size = int(auto_struct_aug_bank.shape[0])
        bank_size = int(auto_struct_aug_bank.shape[1])
        z_auto_struct_flat = music2latent.encode_batch(
            model,
            auto_struct_aug_bank.reshape(batch_size * bank_size, *auto_struct_aug_bank.shape[2:]),
        )
        z_auto_struct_aug_batch = z_auto_struct_flat.reshape(
            batch_size, bank_size, *z_auto_struct_flat.shape[1:]
        )

    z_timbre_aug_batch = [None] * len(pending_examples)
    timbre_aug_chunks = [item["timbre_aug_chunks"] for item in pending_examples]
    if save_timbre_aug_latent and any(chunks is not None for chunks in timbre_aug_chunks):
        timbre_aug_bank = torch.stack(timbre_aug_chunks).to(device)
        batch_size = int(timbre_aug_bank.shape[0])
        bank_size = int(timbre_aug_bank.shape[1])
        z_timbre_flat = music2latent.encode_batch(
            model,
            timbre_aug_bank.reshape(batch_size * bank_size, *timbre_aug_bank.shape[2:]),
        )
        z_timbre_aug_batch = z_timbre_flat.reshape(batch_size, bank_size, *z_timbre_flat.shape[1:])

    clap_batch = [None] * len(pending_examples)
    clap_timbre_aug_batch = [None] * len(pending_examples)
    if clap_extractor is not None:
        clap_batch = extract_clap_timbre_embeddings(
            clap_extractor,
            torch.stack([item["chunk"] for item in pending_examples]),
            source_sample_rate=clap_source_sample_rate,
            batch_size=clap_batch_size,
        )
        if any(chunks is not None for chunks in timbre_aug_chunks):
            timbre_aug_bank = torch.stack(timbre_aug_chunks)
            batch_size = int(timbre_aug_bank.shape[0])
            bank_size = int(timbre_aug_bank.shape[1])
            clap_timbre_flat = extract_clap_timbre_embeddings(
                clap_extractor,
                timbre_aug_bank.reshape(batch_size * bank_size, *timbre_aug_bank.shape[2:]),
                source_sample_rate=clap_source_sample_rate,
                batch_size=clap_batch_size,
            )
            clap_timbre_aug_batch = clap_timbre_flat.reshape(batch_size, bank_size, -1)

    entries = []
    chunk_id = int(chunk_id_start)
    for item, chunk_z, chunk_struct_z, chunk_auto_struct_z, chunk_timbre_z, chunk_clap, chunk_clap_timbre_aug in zip(
        pending_examples,
        z_batch,
        z_struct_aug_batch,
        z_auto_struct_aug_batch,
        z_timbre_aug_batch,
        clap_batch,
        clap_timbre_aug_batch,
    ):
        ae = AudioExample(output_type="torch")
        if not latents_only:
            ae.put_array("audio", item["chunk"].cpu())
        ae.put_metadata(item["metadata"])
        if chunk_z is not None:
            ae.put_array("z", chunk_z.cpu())
        if chunk_struct_z is not None:
            ae.put_array("z_struct_aug", chunk_struct_z.cpu())
        if chunk_auto_struct_z is not None:
            ae.put_array(f"z_struct_aug_{auto_transcription_key}", chunk_auto_struct_z.cpu())
        if chunk_timbre_z is not None:
            ae.put_array("z_timbre_aug", chunk_timbre_z.cpu())
        if chunk_clap is not None:
            ae.put_array("clap", chunk_clap.to(torch.float16))
        if chunk_clap_timbre_aug is not None:
            ae.put_array("clap_timbre_aug", chunk_clap_timbre_aug.to(torch.float16))

        if save_midi:
            if chunk_z is not None:
                midi_frames = int(chunk_z.shape[-1])
            else:
                midi_frames = max(1, int(round(float(chunk_size) * float(midi_fps))))
            start_sec = float(item["metadata"]["chunk_index"]) * float(chunk_size)
            end_sec = start_sec + float(chunk_size)
            midi_roll = build_midi_roll_chunk(
                item.get("midi_roll_obj"),
                start_sec=start_sec,
                end_sec=end_sec,
                num_frames=midi_frames,
            )
            ae.put_array("midi_roll", midi_roll)
            if save_auto_transcription:
                auto_midi_roll = build_midi_roll_chunk(
                    item["auto_midi_obj"],
                    start_sec=start_sec,
                    end_sec=end_sec,
                    num_frames=midi_frames,
                )
                ae.put_array(f"midi_roll_{auto_transcription_key}", auto_midi_roll)

        entries.append((f"{chunk_id:08d}", ae))
        chunk_id += 1

    save_many_to_lmdb(env, entries)
    return chunk_id
