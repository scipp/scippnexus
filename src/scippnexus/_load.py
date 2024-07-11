# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import contextlib
import io
from os import PathLike

import h5py as h5
import scipp as sc

from .base import (
    DefaultDefinitions,
    DefaultDefinitionsType,
    Group,
    base_definitions,
)
from .file import File
from .typing import Definitions, ScippIndex


def load(
    filename: str | PathLike[str] | io.BytesIO | h5.Group | Group,
    *,
    root: str | None = None,
    select: ScippIndex = (),
    definitions: Definitions | DefaultDefinitionsType = DefaultDefinitions,
) -> sc.DataGroup:
    """TODO"""
    with _open(filename, definitions=definitions) as group:
        if root is not None:
            group = group[root]
        return group[select]


def _open(
    filename: str | PathLike[str] | io.BytesIO | h5.Group | Group,
    definitions: Definitions | DefaultDefinitionsType = DefaultDefinitions,
):
    if isinstance(filename, h5.Group):
        return contextlib.nullcontext(
            Group(
                filename,
                definitions=base_definitions()
                if definitions is DefaultDefinitions
                else definitions,
            )
        )
    if isinstance(filename, Group):
        if definitions is not DefaultDefinitions:
            raise TypeError(
                'Cannot override application definitions. '
                'The `definitions` argument must not be used '
                'When the file is specified as a scippnexus.Group.'
            )
        return contextlib.nullcontext(filename)
    return File(filename, 'r', definitions=definitions)
