import random
from bisect import bisect_right
from collections.abc import Mapping
from pathlib import Path

import lmdb
import torch
from torch.utils.data import Dataset
import gin

from .audio_example import AudioExample, load_from_lmdb
from .synth import parse_program_exclusion_spec


def _normalize_program_exclusion_spec(spec):
    if spec is None:
        return frozenset()
    if isinstance(spec, str):
        return frozenset(parse_program_exclusion_spec(spec))
    if isinstance(spec, (list, tuple, set, frozenset)):
        return frozenset(int(x) for x in spec)
    return frozenset([int(spec)])


@gin.configurable
class BaseDataset(Dataset):
    """Dataset for loading preprocessed audio chunks from LMDB (lazy env init per worker)."""
    
    def __init__(self,
                 lmdb_path,
                 transform=None,
                 return_latents=True,
                 return_midi=True,
                 return_struct_aug_latent=True,
                 return_timbre_aug_latent=True,
                 midi_roll_key="midi_roll",
                 struct_aug_latent_key="z_struct_aug",
                 extra_array_keys=(),
                 struct_aug_exclude_programs=None,
                 struct_aug_filter_fallback="original"):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.return_latents = return_latents
        self.return_midi = return_midi
        self.return_struct_aug_latent = return_struct_aug_latent
        self.return_timbre_aug_latent = return_timbre_aug_latent
        self.midi_roll_key = str(midi_roll_key)
        self.struct_aug_latent_key = str(struct_aug_latent_key)
        if self.midi_roll_key != "midi_roll" and not self.midi_roll_key.startswith("midi_roll_"):
            raise ValueError(
                "midi_roll_key must be 'midi_roll' or use the 'midi_roll_<source>' convention; "
                f"got '{self.midi_roll_key}'."
            )
        if self.struct_aug_latent_key != "z_struct_aug" and not self.struct_aug_latent_key.startswith("z_struct_aug_"):
            raise ValueError(
                "struct_aug_latent_key must be 'z_struct_aug' or use the "
                "'z_struct_aug_<source>' convention; "
                f"got '{self.struct_aug_latent_key}'."
            )
        self.extra_array_keys = tuple(str(key) for key in extra_array_keys)
        self.struct_aug_exclude_programs = _normalize_program_exclusion_spec(struct_aug_exclude_programs)
        self.struct_aug_filter_fallback = str(struct_aug_filter_fallback).lower()
        if self.struct_aug_filter_fallback not in {"original", "all", "error"}:
            raise ValueError(
                "struct_aug_filter_fallback must be one of: original, all, error. "
                f"Got '{struct_aug_filter_fallback}'."
            )
        
        # Don't open LMDB here! Each worker will open its own environment
        self.env = None
        self._nonempty_midi_indices_cache = {}

        # Get length (open temporarily in main process)
        with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.length = txn.stat()['entries']

    def _init_env(self):
        """Lazily open LMDB environment inside each worker process."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,  # safer for multiple workers
                meminit=False
            )

    def __len__(self):
        return self.length

    def close_env(self):
        """Close any open LMDB environment held by this dataset instance."""
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def __getitem__(self, idx):
        # Ensure environment is initialized (works for multiprocessing)
        self._init_env()

        # Load AudioExample from LMDB
        key = f"{idx:08d}"
        example = load_from_lmdb(self.env, key)
        
        if example is None:
            raise IndexError(f"No data found for index {idx}")
        
        result = {'metadata': dict(example.get_metadata() or {})}
        
        # Get audio if available
        audio = example.get('audio', None)
        if audio is not None:
            if self.transform:
                audio = self.transform(audio)
            result['audio'] = audio
        
        # Add latents if available and requested
        if self.return_latents:
            latent = example.get('z', None)
            if latent is not None:
                result['latent'] = latent
            if self.return_struct_aug_latent:
                struct_aug_bank = example.get(self.struct_aug_latent_key, None)
                if struct_aug_bank is not None:
                    if self.struct_aug_latent_key == "z_struct_aug":
                        program_metadata_key = "struct_aug_programs"
                    else:
                        suffix = self.struct_aug_latent_key.removeprefix("z_struct_aug_")
                        program_metadata_key = f"struct_aug_programs_{suffix}"
                    struct_aug_programs = list(result['metadata'].get(program_metadata_key) or [])

                    if hasattr(struct_aug_bank, "dim") and struct_aug_bank.dim() >= 1:
                        k = int(struct_aug_bank.shape[0])
                        if self.struct_aug_exclude_programs:
                            # Keep the bank at its original fixed K.  Filtering the tensor
                            # here would give different samples different K values and break
                            # batch collation.  The model samples only from this mask instead.
                            valid_mask = torch.tensor(
                                [
                                    choice_idx >= len(struct_aug_programs)
                                    or struct_aug_programs[choice_idx] is None
                                    or int(struct_aug_programs[choice_idx])
                                    not in self.struct_aug_exclude_programs
                                    for choice_idx in range(k)
                                ],
                                dtype=torch.bool,
                            )
                            if bool(valid_mask.any()):
                                result['struct_aug_valid_mask'] = valid_mask
                            elif self.struct_aug_filter_fallback == "original":
                                if latent is None:
                                    raise ValueError(
                                        "struct_aug_filter_fallback='original' requires base latent 'z' to exist."
                                    )
                                # A false mask tells the model to use the base latent for this
                                # sample while preserving the fixed bank shape for collation.
                                result['struct_aug_valid_mask'] = valid_mask
                                result['metadata']['struct_aug_filter_fallback'] = "original"
                            elif self.struct_aug_filter_fallback == "all":
                                result['struct_aug_valid_mask'] = torch.ones(k, dtype=torch.bool)
                                result['metadata']['struct_aug_filter_fallback'] = "all"
                            else:
                                raise ValueError(
                                    "All structure augmentations were filtered out for sample "
                                    f"{idx} with excluded programs={sorted(self.struct_aug_exclude_programs)}."
                                )

                    result['z_struct_aug'] = struct_aug_bank
            if self.return_timbre_aug_latent:
                timbre_aug_latent = example.get('z_timbre_aug', None)
                if timbre_aug_latent is not None:
                    result['z_timbre_aug'] = timbre_aug_latent

        # Add MIDI roll if available and requested
        if self.return_midi:
            midi_roll = example.get(self.midi_roll_key, None)
            if midi_roll is not None:
                result['midi_roll'] = midi_roll

        for key in self.extra_array_keys:
            if key in result:
                continue
            value = example.get(key, None)
            if value is not None:
                result[key] = value
        
        return result

    def get_all_metadata(self):
        """
        Load all metadata for the entire dataset.
        Used for building grouped training batches.
        
        Returns:
            list: List of metadata dicts, one per dataset index
        """
        meta_list = []
        # Reuse an existing handle when present; opening the same LMDB twice
        # in one process can fail on some platforms.
        temporary_env = self.env is None
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False) if temporary_env else self.env
        try:
            with env.begin() as txn:
                for idx in range(self.length):
                    key = f"{idx:08d}"
                    example = load_from_lmdb(env, key)
                    if example is not None:
                        meta_list.append(example.get_metadata())
                    else:
                        meta_list.append({})
        finally:
            if temporary_env:
                env.close()
        
        return meta_list

    def get_nonempty_midi_indices(self, split=None):
        """
        Return dataset indices whose selected MIDI-roll key exists and is not all-zero.

        Args:
            split: Optional metadata split filter (for example ``"train"`` or
                ``"validation"``). When provided, only indices from that split
                are returned.

        Returns:
            list[int]: Dataset indices with non-empty MIDI rolls.
        """
        cache_key = split if split is not None else "__all__"
        cached = self._nonempty_midi_indices_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        # Reuse a live environment when the dataset has already been indexed.
        # Opening the same LMDB path twice in one process can fail on some
        # platforms, while worker processes still lazily own their own handles.
        indices = []
        temporary_env = self.env is None
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False) if temporary_env else self.env
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, data in cursor:
                    example = AudioExample.from_bytes(bytes(data))
                    metadata = example.get_metadata() or {}
                    if split is not None and metadata.get("split") != split:
                        continue

                    midi_roll = example.get(self.midi_roll_key, None)
                    if midi_roll is None or midi_roll.numel() == 0:
                        continue
                    if torch.any(midi_roll != 0):
                        indices.append(int(key.decode()))
        finally:
            if temporary_env:
                env.close()

        self._nonempty_midi_indices_cache[cache_key] = tuple(indices)
        return list(indices)

    def __del__(self):
        """Ensure LMDB environment is closed."""
        self.close_env()

    def __getstate__(self):
        """
        Drop the live LMDB handle when pickling for multiprocessing workers.

        Workers reopen their own environment lazily via ``_init_env``.
        """
        self.close_env()
        state = self.__dict__.copy()
        state["env"] = None
        return state


@gin.configurable
class CombinedDataset(Dataset):
    """Read several prepared LMDB datasets as one dataset without copying them.

    Each source retains its own metadata-defined split. Tracks are namespaced
    by source ID so grouped batches cannot pair stems from different datasets
    merely because their source track names happen to match.
    """

    def __init__(self, datasets, **base_dataset_kwargs):
        if not isinstance(datasets, Mapping) or not datasets:
            raise ValueError("datasets must be a non-empty mapping of dataset ID to LMDB path or BaseDataset.")

        self.datasets = {}
        for dataset_id, source in datasets.items():
            dataset_id = str(dataset_id).strip()
            if not dataset_id:
                raise ValueError("CombinedDataset dataset IDs must be non-empty.")
            if dataset_id in self.datasets:
                raise ValueError(f"Duplicate CombinedDataset ID: '{dataset_id}'.")
            if isinstance(source, BaseDataset):
                if base_dataset_kwargs:
                    raise ValueError(
                        "base_dataset_kwargs can only be used when CombinedDataset receives LMDB paths, "
                        "not pre-built BaseDataset instances."
                    )
                dataset = source
            else:
                source_path = Path(source).expanduser().resolve()
                if not source_path.is_dir():
                    raise FileNotFoundError(f"LMDB directory for dataset '{dataset_id}' does not exist: {source_path}")
                dataset = BaseDataset(str(source_path), **base_dataset_kwargs)
            self.datasets[dataset_id] = dataset

        self.dataset_ids = tuple(self.datasets)
        self._offsets = []
        total = 0
        for dataset_id in self.dataset_ids:
            total += len(self.datasets[dataset_id])
            self._offsets.append(total)
        self.length = total
        self._nonempty_midi_indices_cache = {}

    def __len__(self):
        return self.length

    @property
    def dataset_sizes(self):
        return {dataset_id: len(dataset) for dataset_id, dataset in self.datasets.items()}

    def _resolve_index(self, index):
        index = int(index)
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(f"CombinedDataset index {index} is outside [0, {self.length}).")
        source_position = bisect_right(self._offsets, index)
        previous_offset = 0 if source_position == 0 else self._offsets[source_position - 1]
        dataset_id = self.dataset_ids[source_position]
        return dataset_id, index - previous_offset

    @staticmethod
    def _namespace_metadata(metadata, dataset_id, local_index):
        metadata = dict(metadata or {})
        source_track = str(metadata.get("track", "unknown_track"))
        metadata["dataset_id"] = dataset_id
        metadata["source_track"] = source_track
        metadata["track"] = f"{dataset_id}::{source_track}"
        metadata["dataset_local_index"] = int(local_index)
        return metadata

    def __getitem__(self, index):
        dataset_id, local_index = self._resolve_index(index)
        result = dict(self.datasets[dataset_id][local_index])
        result["metadata"] = self._namespace_metadata(result.get("metadata"), dataset_id, local_index)
        return result

    def get_all_metadata(self):
        metadata = []
        for dataset_id, dataset in self.datasets.items():
            metadata.extend(
                self._namespace_metadata(item, dataset_id, local_index)
                for local_index, item in enumerate(dataset.get_all_metadata())
            )
        return metadata

    def get_nonempty_midi_indices(self, split=None):
        cache_key = split if split is not None else "__all__"
        cached = self._nonempty_midi_indices_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        indices = []
        offset = 0
        for dataset in self.datasets.values():
            indices.extend(offset + local_index for local_index in dataset.get_nonempty_midi_indices(split=split))
            offset += len(dataset)
        self._nonempty_midi_indices_cache[cache_key] = tuple(indices)
        return list(indices)

    def close_env(self):
        for dataset in self.datasets.values():
            dataset.close_env()

    def __del__(self):
        self.close_env()

    def __getstate__(self):
        self.close_env()
        return self.__dict__.copy()
