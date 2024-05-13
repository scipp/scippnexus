# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import warnings
from typing import Any

import h5py
import numpy as np


def _cset_to_encoding(cset: int) -> str:
    """
    Converts a HDF5 cset into a python encoding. Allowed values for cset are
    h5py.h5t.CSET_ASCII and h5py.h5t.CSET_UTF8.

    Args:
        cset: The HDF character set to convert

    Returns:
        A string describing the encoding suitable for calls to str(encoding=...)
        Either "ascii" or "utf-8".
    """
    if cset == h5py.h5t.CSET_ASCII:
        return "ascii"
    elif cset == h5py.h5t.CSET_UTF8:
        return "utf-8"
    else:
        raise ValueError(
            f"Unknown character set in HDF5 data file. Expected data "
            f"types are {h5py.h5t.CSET_ASCII=} or "
            f"{h5py.h5t.CSET_UTF8=} but got '{cset}'. "
        )


def _warn_latin1_decode(obj, decoded, error):
    warnings.warn(
        f"Encoding for bytes '{obj}' declared as ascii, "
        f"but contains characters in extended ascii range. Assuming "
        f"extended ASCII (latin-1), but this behavior is not "
        f"specified by the HDF5 or nexus standards and may therefore "
        f"be incorrect. Decoded string using latin-1 is '{decoded}'. "
        f"Error was '{error}'.",
        stacklevel=2,
    )


def _ensure_str(str_or_bytes: str | bytes, encoding: str) -> str:
    """
    See https://docs.h5py.org/en/stable/strings.html for justification about some of
    the operations performed in this method. In particular, variable-length strings
    are returned as `str` from h5py, but need to be encoded using the surrogateescape
    error handler and then decoded using the encoding specified in the nexus file in
    order to get a correctly encoded string in all cases.

    Note that the nexus standard leaves unspecified the behavior of H5T_CSET_ASCII
    for characters >=128. Common extensions are the latin-1 ("extended ascii") character
    set which appear to be used in nexus files from some facilities. Attempt to load
    these strings with the latin-1 extended character set, but warn as this is
    technically unspecified behavior.
    """
    if isinstance(str_or_bytes, str):
        str_or_bytes = str_or_bytes.encode("utf-8", errors="surrogateescape")

    if encoding == "ascii":
        try:
            return str(str_or_bytes, encoding="ascii")
        except UnicodeDecodeError as e:
            decoded = str(str_or_bytes, encoding="latin-1")
            _warn_latin1_decode(str_or_bytes, decoded, str(e))
            return decoded
    else:
        return str(str_or_bytes, encoding)


_map_to_supported_type = {
    'int8': np.int32,
    'int16': np.int32,
    'uint8': np.int32,
    'uint16': np.int32,
    'uint32': np.int32,
    'uint64': np.int64,
}


def _ensure_supported_int_type(dataset_type: Any):
    return _map_to_supported_type.get(dataset_type, dataset_type)
