"""Main dataset and data item base class definitions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import accumulate
from typing import Any, Callable, NamedTuple, cast

import torch
from shapely.geometry import MultiPolygon, Polygon
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset, Subset


class DataItem:
    """Data structure that contains task data and maybe additional meta data.

    Args:
        data: Tensor containing essential data for model training.
        meta_data: Data structure containing additional data that
            is associated with the data.

    """

    def __init__(
        self, data: torch.Tensor, meta_data: dict[str, torch.Tensor] | None = None
    ) -> None:
        self.data = data
        if meta_data is None:
            meta_data = {}
        self.meta_data = meta_data

    def to(self, *args: Any, **kwargs: Any) -> DataItem:
        """Apply `torch.Tensor.to` to data and metadata tensors.

        For more information consult https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
        """
        data = self.data.to(*args, **kwargs)
        meta_data = {}
        for key, value in self.meta_data.items():
            meta_data[key] = value.to(*args, **kwargs)
        return DataItem(data, meta_data)


class LabelledData(NamedTuple):
    """Class that holds data, metadata and labels."""

    data_item: DataItem
    label: torch.Tensor

    @property
    def data(self) -> torch.Tensor:
        """Get data from data_item."""
        return self.data_item.data

    @property
    def meta_data(self) -> dict[str, torch.Tensor]:
        """Get meta_data from data_items."""
        return self.data_item.meta_data

    @classmethod
    def from_tensor_dict(cls, tensors: dict[str, torch.Tensor]) -> LabelledData:
        """Create Labelled data instance from dict of tensors.

        Args:
            tensors: Dictionary of tensors to create LabelledData from.

        Returns:
            The LabelledData containing the tensors from the dictionary.

        Raises:
            KeyError: If tensors do not contain "data" or "label"
        """
        data = tensors.pop("data")
        label = tensors.pop("label")
        return LabelledData(DataItem(data, tensors), label)

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        """Output LabelledData as dict."""
        tensors = {"data": self.data, "label": self.label}
        tensors.update(self.meta_data)
        return tensors

    def to_tuple(self) -> tuple[DataItem, torch.Tensor]:
        """Output LabelledData as tuple of DataItem and tensor containing labels."""
        return self.data_item, self.label


def custom_collate_fn(batch: Sequence[LabelledData]) -> LabelledData:
    """Collate function for batch creation within data loader.

    Used to create batches from a dataset's DataItem.

    Args:
        batch: List of DataItem from dataset

    Returns:
        New DataItem with batched data.
    """
    batch_tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    tensor_stackability: dict[str, bool] = defaultdict(lambda: True)
    for item in batch:
        for tensor_name, tensor in item.to_tensor_dict().items():
            if tensor_stackability[tensor_name] and bool(
                prev_tensors := batch_tensors[tensor_name]
            ):
                tensor_stackability[tensor_name] = prev_tensors[-1].shape == tensor.shape
            batch_tensors[tensor_name].append(tensor)

    batched_tensors = {
        tensor_name: (
            torch.stack(tensors)
            if tensor_stackability[tensor_name]
            else pad_sequence(tensors, batch_first=True, padding_value=-1)
        )
        for tensor_name, tensors in batch_tensors.items()
    }

    if (
        not tensor_stackability["label"]
        and batched_tensors["label"].shape != batched_tensors["data"].shape
    ):
        batched_tensors["label"] = torch.concat(batch_tensors["label"], 0)

    if (pad_mask := batched_tensors["data"].eq(-1)).any():
        if (aug_mask := batched_tensors.get("mask")) is not None:
            # if pad_mask.dim (B, T, C) != aug_mask.dim (B, T, C) | (B, T)
            # => aug_mask.dim is (B, T), thus pad_mask need to be converted
            if aug_mask.dim() != pad_mask.dim():
                assert pad_mask.all(dim=-1).equal(
                    pad_mask.any(dim=-1)
                ), "There are unknown values outside of padding, please investigate"
                pad_mask = pad_mask.all(dim=-1)
            assert aug_mask.shape == pad_mask.shape, (
                f"Shape of Augumentation mask ({aug_mask.shape}) and "
                f"Pad mask ({pad_mask.shape}) missmatch!"
            )
            batched_tensors["mask"] = torch.logical_or(aug_mask.bool(), pad_mask)
            batched_tensors["pad_mask"] = pad_mask
        else:
            mask = pad_mask.all(dim=-1)
            assert mask.equal(
                pad_mask.any(dim=-1)
            ), "There are unknown values outside of padding, please investigate"
            batched_tensors["mask"] = mask
    return LabelledData.from_tensor_dict(batched_tensors)


class TransformDataset(Dataset[LabelledData]):
    """Wrapper around torch dataset, applying data and target transformations.

    Args:
        dataset: Dataset to wrap.
        data_transforms: List of transforms to apply to data
        target_transforms: List of transforms to apply to targets
        collate_fn: Function to collate list of batches.
        polygons: (Optional) mapping associating a polygon to each entry.
    Raises:
        ValueError: If the wrapped dataset does not have a well-defined length.
    """

    def __init__(
        self,
        dataset: Dataset[LabelledData],
        data_transforms: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        target_transforms: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        collate_fn: Callable[[Sequence[LabelledData]], LabelledData] = custom_collate_fn,
        polygons: dict[int, Polygon | MultiPolygon] | None = None,
    ) -> None:
        self.dataset = dataset
        self.data_transforms = data_transforms if data_transforms else []
        self.target_transforms = target_transforms if target_transforms else []
        self.collate_fn: Callable[[Sequence[LabelledData]], LabelledData] = collate_fn
        self.polygons = polygons

        # We need datasets to have a well-defined length
        try:
            self._dataset_length = len(dataset)  # type: ignore[arg-type]
        except TypeError:
            raise ValueError("Wrapped dataset must have a well-defined size.")

    def _apply_transformations(self, labelled_data: LabelledData) -> LabelledData:
        data, targets = labelled_data.to_tuple()
        for data_transform in self.data_transforms:
            data.data = data_transform(data.data)
        for target_transform in self.target_transforms:
            targets = target_transform(targets)
        return LabelledData(data, targets)

    def __getitem__(self, ix: int) -> LabelledData:
        return self._apply_transformations(self.dataset[ix])

    def __len__(self) -> int:
        return self._dataset_length

    def subset(self, indices: Sequence[int]) -> TransformDataset:
        """Return a subset of the dataset corresponding to the given indices."""
        if self.polygons is None:
            polygons = None
        else:
            polygons = {n: self.polygons[ix] for n, ix in enumerate(indices)}
        return TransformDataset(
            Subset(self.dataset, indices),
            data_transforms=self.data_transforms,
            target_transforms=self.target_transforms,
            collate_fn=self.collate_fn,
            polygons=polygons,
        )

    @classmethod
    def concat(cls, datasets: Sequence[TransformDataset]) -> TransformDataset:
        """Concatenate given datasets to one large dataset."""
        if datasets[0].polygons is None:
            polygons = None
        else:
            assert all(ds.polygons for ds in datasets)
            offsets = [0] + list(accumulate(len(ds) for ds in datasets))
            polygons = {
                n + offset: ds.polygons[n]  # type: ignore[index]
                for offset, ds in zip(offsets, datasets)
                for n in range(len(ds))
            }

        return TransformDataset(
            ConcatDataset([ds.dataset for ds in datasets]),
            data_transforms=datasets[0].data_transforms,
            target_transforms=datasets[0].target_transforms,
            collate_fn=datasets[0].collate_fn,
            polygons=polygons,
        )

    def overlaps(self, ix1: int, ix2: int) -> bool:
        """Check if polygons corresponding to `ix1` and `ix2` overlap."""
        if self.polygons is None:
            raise ValueError("Polygons must be provided in order to check for intersections.")
        return cast(bool, self.polygons[ix1].intersects(self.polygons[ix2]))


class TensorDataset(Dataset[LabelledData]):
    """Simple dataset for serving torch tensor data.

    Args:
        data: List of tuples, consisting of input data and labels.
    """

    def __init__(self, data: Sequence[LabelledData]) -> None:
        self.data: Sequence[LabelledData] = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ix: int) -> LabelledData:
        return self.data[ix]
