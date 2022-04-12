# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
import warnings
import functools
from typing import List, Union, NoReturn, Any, Dict, Tuple, Protocol
import numpy as np
import scipp as sc
import h5py

from ._hdf5_nexus import _cset_to_encoding, _ensure_str
from ._hdf5_nexus import _ensure_supported_int_type, _warn_latin1_decode
from .typing import H5Group, H5Dataset, ScippIndex
from ._common import to_plain_index
from ._common import convert_time_to_datetime64


class DataArrayLoaderFactory:
    def __call__(self, group: H5Group) -> DataArrayLoader:
        return DataArrayLoader(group)



class DataArrayLoader:
    def __init__(self, group: H5Group):
        self._group = group

    def __getitem__(self, index: ScippIndex):
        pass
