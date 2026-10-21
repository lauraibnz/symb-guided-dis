"""Grouped batch samplers for stem-level cross-timbre training."""

import random
from collections import defaultdict

import torch
from torch.utils.data import Sampler


def resolve_metadata_group(meta, group_key):
    """Resolve a sample group identifier from metadata."""
    if not isinstance(meta, dict):
        raise ValueError(f"Expected metadata dict, got {type(meta).__name__}")

    if isinstance(group_key, (list, tuple)):
        missing = [key for key in group_key if key not in meta]
        if missing:
            raise ValueError(
                f"Missing group keys {missing} in metadata. "
                f"Available keys: {list(meta.keys())}"
            )
        return tuple(meta[key] for key in group_key)

    if group_key not in meta:
        raise ValueError(
            f"Missing group key '{group_key}' in metadata. "
            f"Available keys: {list(meta.keys())}"
        )
    return meta[group_key]


class GroupedStemBatchSampler(Sampler):
    """
    Create batches with G stem-groups and K chunks per stem-group.

    Each group is typically identified by ``("track", "stem")``, so grouped
    batches contain multiple chunks from the same stem. This matches the
    cross-timbre setup where positives should come from different chunks of the
    same stem rather than from coarse program labels.
    """

    def __init__(
        self,
        metadata_list,
        allowed_indices=None,
        groups_per_batch=16,
        samples_per_group=2,
        group_key=("track", "stem"),
        dataset_sampling_weights=None,
        dataset_sampling_exponent=None,
        split=None,
        drop_last=True,
        shuffle=True,
        seed=None,
    ):
        self.groups_per_batch = int(groups_per_batch)
        self.samples_per_group = int(samples_per_group)
        self.group_key = tuple(group_key) if isinstance(group_key, (list, tuple)) else str(group_key)
        self.dataset_sampling_weights = self._normalize_dataset_sampling_weights(
            dataset_sampling_weights
        )
        self.dataset_sampling_exponent = self._normalize_dataset_sampling_exponent(
            dataset_sampling_exponent
        )
        if (
            self.dataset_sampling_weights is not None
            and self.dataset_sampling_exponent is not None
        ):
            raise ValueError(
                "Specify either dataset_sampling_weights or dataset_sampling_exponent, not both."
            )
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.split = split
        self.allowed_indices = set(allowed_indices) if allowed_indices is not None else None

        self.group_indices = defaultdict(list)
        self.group_dataset_ids = {}
        for idx, meta in enumerate(metadata_list):
            if not meta:
                continue
            if self.allowed_indices is not None and idx not in self.allowed_indices:
                continue
            if split is not None and meta.get("split") != split:
                continue
            try:
                group_id = resolve_metadata_group(meta, self.group_key)
            except ValueError:
                continue
            self.group_indices[group_id].append(idx)
            dataset_id = meta.get("dataset_id")
            previous_dataset_id = self.group_dataset_ids.setdefault(group_id, dataset_id)
            if previous_dataset_id != dataset_id:
                raise ValueError(
                    f"Group {group_id!r} spans dataset IDs {previous_dataset_id!r} and {dataset_id!r}. "
                    "Use a dataset-qualified group key."
                )

        self.valid_groups = [
            group_id
            for group_id, indices in self.group_indices.items()
            if len(indices) >= self.samples_per_group
        ]
        self.valid_groups_by_dataset = defaultdict(list)
        for group_id in self.valid_groups:
            self.valid_groups_by_dataset[self.group_dataset_ids[group_id]].append(group_id)
        self.eligible_chunk_counts_by_dataset = {
            dataset_id: sum(
                len(self.group_indices[group_id])
                for group_id in group_ids
            )
            for dataset_id, group_ids in self.valid_groups_by_dataset.items()
        }
        if self.dataset_sampling_exponent is not None:
            # Use only chunks that can actually participate in this grouped
            # sampler: selected split, optional non-empty-MIDI filter, and
            # groups with >=K chunks.
            self.dataset_sampling_weights = {
                dataset_id: float(chunk_count) ** self.dataset_sampling_exponent
                for dataset_id, chunk_count in self.eligible_chunk_counts_by_dataset.items()
            }
        self._validate_dataset_sampling_weights()

        if len(self.valid_groups) < self.groups_per_batch:
            print(
                f"Warning: only {len(self.valid_groups)} groups have >= {self.samples_per_group} chunks. "
                f"Reducing groups_per_batch from {self.groups_per_batch}."
            )
            self.groups_per_batch = len(self.valid_groups)

        self.rng = random.Random(seed)
        self.epoch = 0

        print("\n=== Grouped Stem Batch Sampler ===")
        print(f"Split: {self.split if self.split is not None else 'all'}")
        print(f"group_key={self.group_key}")
        print(f"Valid groups (>={self.samples_per_group} chunks): {len(self.valid_groups)}")
        if self.dataset_sampling_weights is not None:
            print(
                "Valid groups by dataset: "
                f"{ {dataset_id: len(groups) for dataset_id, groups in self.valid_groups_by_dataset.items()} }"
            )
            print(f"Eligible chunks by dataset: {self.eligible_chunk_counts_by_dataset}")
            if self.dataset_sampling_exponent is not None:
                print(
                    "Dataset sampling policy: "
                    f"P(dataset) ∝ eligible_train_chunks ** {self.dataset_sampling_exponent:g}"
                )
            print(f"Dataset group sampling weights: {self.dataset_sampling_weights}")
        print(f"Batch config: G={self.groups_per_batch}, K={self.samples_per_group}")
        print(f"Effective batch size: {self.groups_per_batch * self.samples_per_group}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.seed is not None:
            self.rng = random.Random(self.seed + self.epoch)

    @staticmethod
    def _normalize_dataset_sampling_weights(weights):
        if weights is None:
            return None
        if not isinstance(weights, dict) or not weights:
            raise ValueError("dataset_sampling_weights must be a non-empty mapping or None.")
        normalized = {str(dataset_id): float(weight) for dataset_id, weight in weights.items()}
        if any(weight <= 0.0 for weight in normalized.values()):
            raise ValueError("dataset_sampling_weights must contain only positive weights.")
        return normalized

    @staticmethod
    def _normalize_dataset_sampling_exponent(exponent):
        if exponent is None:
            return None
        exponent = float(exponent)
        if not 0.0 <= exponent <= 1.0:
            raise ValueError(
                "dataset_sampling_exponent must be in [0, 1]: "
                "0 is uniform over datasets and 1 is proportional to eligible chunks. "
                f"Got {exponent}."
            )
        return exponent

    def _validate_dataset_sampling_weights(self):
        if self.dataset_sampling_weights is None:
            return
        available = set(self.valid_groups_by_dataset)
        requested = set(self.dataset_sampling_weights)
        if available != requested:
            raise ValueError(
                "dataset_sampling_weights must name exactly the datasets with valid groups for this split; "
                f"available={sorted(str(name) for name in available)}, "
                f"requested={sorted(self.dataset_sampling_weights)}."
            )

    def _balanced_epoch_groups(self):
        """Return one epoch's groups with the requested dataset proportions.

        The epoch retains the usual number of group draws. Datasets with too
        few valid groups are repeated as needed; larger datasets are
        downsampled for that epoch. Explicit weights therefore make corpus mass
        independent of segmentation, while exponent-derived weights deliberately
        use the eligible chunk count as their dataset-size measure.
        """
        if self.dataset_sampling_weights is None:
            groups = list(self.valid_groups)
            if self.shuffle:
                self.rng.shuffle(groups)
            return groups

        total_draws = len(self.valid_groups)
        total_weight = sum(self.dataset_sampling_weights.values())
        exact_counts = {
            dataset_id: total_draws * weight / total_weight
            for dataset_id, weight in self.dataset_sampling_weights.items()
        }
        draw_counts = {dataset_id: int(count) for dataset_id, count in exact_counts.items()}
        remainder = total_draws - sum(draw_counts.values())
        for dataset_id in sorted(
            exact_counts,
            key=lambda key: (exact_counts[key] - draw_counts[key], key),
            reverse=True,
        )[:remainder]:
            draw_counts[dataset_id] += 1

        groups = []
        for dataset_id, draw_count in draw_counts.items():
            candidates = self.valid_groups_by_dataset[dataset_id]
            if draw_count <= len(candidates):
                groups.extend(self.rng.sample(candidates, k=draw_count))
            else:
                groups.extend(candidates)
                groups.extend(self.rng.choices(candidates, k=draw_count - len(candidates)))
        if self.shuffle:
            self.rng.shuffle(groups)
        return groups

    def _sample_from_group(self, group_id):
        indices = self.group_indices[group_id]
        if len(indices) < self.samples_per_group:
            raise ValueError(
                f"Group {group_id} has only {len(indices)} chunks but requires "
                f"{self.samples_per_group}. This should have been filtered earlier."
            )
        return self.rng.sample(indices, k=self.samples_per_group)

    def __iter__(self):
        if self.groups_per_batch <= 0:
            self.epoch += 1
            return

        if self.seed is not None:
            self.rng = random.Random(self.seed + self.epoch)

        groups = self._balanced_epoch_groups()

        for start in range(0, len(groups), self.groups_per_batch):
            batch_groups = groups[start:start + self.groups_per_batch]
            if len(batch_groups) < self.groups_per_batch and self.drop_last:
                continue

            batch_indices = []
            for group_id in batch_groups:
                batch_indices.extend(self._sample_from_group(group_id))

            if self.shuffle:
                self.rng.shuffle(batch_indices)

            yield batch_indices

        self.epoch += 1

    def __len__(self):
        if self.groups_per_batch <= 0:
            return 0
        n_full = len(self.valid_groups) // self.groups_per_batch
        if (len(self.valid_groups) % self.groups_per_batch) and not self.drop_last:
            return n_full + 1
        return n_full


def collate_stem_grouped(batch, group_key=("track", "stem")):
    """Collate grouped stem batches and expose integer group ids."""
    group_key = tuple(group_key) if isinstance(group_key, (list, tuple)) else str(group_key)

    metadatas = []
    raw_group_ids = []
    for item in batch:
        meta = item.get("metadata", {})
        metadatas.append(meta)
        raw_group_ids.append(resolve_metadata_group(meta, group_key))

    group_to_idx = {}
    group_ids = []
    for group_id in raw_group_ids:
        if group_id not in group_to_idx:
            group_to_idx[group_id] = len(group_to_idx)
        group_ids.append(group_to_idx[group_id])

    result = {
        "metadata": metadatas,
        "group_ids": torch.tensor(group_ids, dtype=torch.long),
    }

    tensor_keys = sorted(
        {
            key
            for item in batch
            for key, value in item.items()
            if key != "metadata" and isinstance(value, torch.Tensor)
        }
    )

    for key in tensor_keys:
        has_key = [key in item for item in batch]
        if any(has_key) and not all(has_key):
            raise ValueError(
                f"Inconsistent grouped batch: some samples contain '{key}' and others do not."
            )
        values = [item[key] for item in batch]
        result[key] = torch.stack(values)

    return result
