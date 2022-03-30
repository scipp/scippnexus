# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Protocol, Union, Tuple, Dict, List, Callable


# TODO Define more required methods
class Dataset(Protocol):
    """h5py.Dataset-like"""
    def shape(self) -> List[int]:
        """Shape of a dataset"""


class Group(Protocol):
    """h5py.Group-like"""
    def visititems(self, func: Callable) -> None:
        """"""


# Note that scipp does not support dicts yet, but this HDF5 code does, to
# allow for loading blocks of 2d (or higher) data efficiently.
ScippIndex = Union[type(Ellipsis), int, slice, Tuple[str, Union[int, slice]],
                   Dict[str, Union[int, slice]]]
