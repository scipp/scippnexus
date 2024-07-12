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
) -> sc.DataGroup | sc.DataArray | sc.Dataset:
    """Load a NeXus file.

    This function is a shorthand for opening a file manually.
    That is

    .. code-block:: python

        loaded = snx.load('path/to/nexus_file.nxs')

    is equivalent to

    .. code-block:: python

        with snx.File('path/to/nexus_file.nxs') as f:
            loaded = f[()]

    The additional arguments of ``load`` are used as:

    .. code-block:: python

        loaded = snx.load(
            'path/to/nexus_file.nxs'
            root='entry/instrument',
            select={'x': slice(None, 100)},
            definitions=my_definitions,
        )

    which corresponds to

    .. code-block:: python

        with snx.File('path/to/nexus_file.nxs', definitions=my_definitions) as f:
            loaded = f['entry/instrument']['x', :100]

    Parameters
    ----------
    filename:
        One of:

        - A path to a NeXus file.
        - A file-like object containing a NeXus file.
        - A :class:`h5py.Group`.
        - A :class:`scippnexus.Group`.
    root:
        The root group in the NeXus file to load.
        If not provided

        - Everything is loaded under the given group if ``filename`` is a group.
        - Or the entire file is loaded otherwise.
    select:
        Selects a subset of the data to load.
        Corresponds to the argument passed in brackets when using file objects:
        ``loaded = group[select]``.
        See `Loading groups and datasets
        <../../user-guide/quick-start-guide.html#Loading-groups-and-datasets>`_.
        Defaults to ``()`` which selects the entire data.
    definitions:
        NeXus `application definitions <../../user-guide/application-definitions.rst>`_.
        Defaults to the ScippNexus base definitions.

    Returns
    -------
    :
        The loaded data.
    """
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
