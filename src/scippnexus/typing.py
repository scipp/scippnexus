# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
from typing import Any, Protocol, Union, Tuple, Dict, List, Callable
from typing import TYPE_CHECKING


class H5Base(Protocol):

    @property
    def attrs(self) -> List[int]:
        """Attributes of dataset or group"""

    @property
    def name(self) -> str:
        """Name of dataset or group"""

    @property
    def file(self) -> List[int]:
        """File of dataset or group"""

    @property
    def parent(self) -> H5Group:
        """Parent of dataset or group"""


# TODO Define more required methods
class H5Dataset(H5Base, Protocol):
    """h5py.Dataset-like"""

    @property
    def shape(self) -> List[int]:
        """Shape of a dataset"""

    @property
    def dtype(self) -> List[int]:
        """dtype of a dataset"""

    def read_direct(self, array) -> None:
        """Read dataset into given buffer"""


class H5Group(H5Base, Protocol):
    """h5py.Group-like"""

    def __getitem__(self, index: Union[str, Any]) -> Union[H5Dataset, H5Group]:
        """Keys in the group"""

    def keys(self) -> List[str]:
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
ScippIndex = Union[ellipsis, int, tuple, slice, Tuple[str, Union[int, slice]],
                   Dict[str, Union[int, slice]]]
