# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol


class H5Base(Protocol):
    @property
    def attrs(self) -> dict:
        """Attributes of dataset or group"""

    @property
    def name(self) -> str:
        """Name of dataset or group"""

    @property
    def file(self) -> list[int]:
        """File of dataset or group"""

    @property
    def parent(self) -> H5Group:
        """Parent of dataset or group"""


# TODO Define more required methods
class H5Dataset(H5Base, Protocol):
    """h5py.Dataset-like"""

    @property
    def shape(self) -> list[int]:
        """Shape of a dataset"""

    @property
    def dtype(self) -> list[int]:
        """dtype of a dataset"""

    def read_direct(self, array) -> None:
        """Read dataset into given buffer"""


class H5Group(H5Base, Protocol):
    """h5py.Group-like"""

    def __getitem__(self, index: str | Any) -> H5Dataset | H5Group:
        """Keys in the group"""

    def keys(self) -> list[str]:
        """Keys in the group"""

    def create_dataset(self) -> H5Dataset:
        """Create a dataset"""

    def create_group(self) -> H5Group:
        """Create a group"""

    def visititems(self, func: Callable) -> None:
        """Apply callable to all items, recursively"""


if TYPE_CHECKING:
    from enum import Enum

    class ellipsis(Enum):
        Ellipsis = "..."

else:
    ellipsis = type(Ellipsis)

# Note that scipp does not support dicts yet, but this HDF5 code does, to
# allow for loading blocks of 2d (or higher) data efficiently.
ScippIndex = (
    ellipsis | int | tuple | slice | tuple[str, int | slice] | dict[str, int | slice]
)

Definitions = Mapping[str, type]
