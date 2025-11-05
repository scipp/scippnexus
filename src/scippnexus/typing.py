# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

import types
from collections.abc import Callable, KeysView, Mapping
from typing import Any, Protocol, TypeAlias

import numpy.typing as npt


class H5Base(Protocol):
    @property
    def attrs(self) -> dict[str, object]:  # TODO better type hint
        """Attributes of dataset or group"""

    @property
    def name(self) -> str:
        """Name of dataset or group"""

    @property
    def file(self) -> Any:
        """File of dataset or group"""

    @property
    def parent(self) -> H5Group:
        """Parent of dataset or group"""


# TODO Define more required methods
class H5Dataset(H5Base, Protocol):
    """h5py.Dataset-like"""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of a dataset"""

    @property
    def dtype(self) -> list[int]:
        """dtype of a dataset"""

    def read_direct(self, array: npt.NDArray[Any]) -> None:
        """Read dataset into given buffer"""


class H5Group(H5Base, Protocol):
    """h5py.Group-like"""

    def __getitem__(self, index: str | Any) -> H5Dataset | H5Group:
        """Keys in the group"""

    def keys(self) -> KeysView[str]:
        """Keys in the group"""

    def create_dataset(self) -> H5Dataset:
        """Create a dataset"""

    def create_group(self) -> H5Group:
        """Create a group"""

    def visititems(
        self,
        func: Callable[[str], None],
    ) -> None:
        """Apply callable to all items, recursively"""


# Note that scipp does not support dicts yet, but this HDF5 code does, to
# allow for loading blocks of 2d (or higher) data efficiently.
ScippIndex: TypeAlias = (
    types.EllipsisType
    | int
    | tuple[()]
    | slice
    | tuple[str, int | slice]
    | dict[str, int | slice]
)

Definitions = Mapping[str, type]
