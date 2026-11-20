# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import io
import os
from contextlib import AbstractContextManager
from typing import Any

import h5py

from .base import (
    DefaultDefinitions,
    DefaultDefinitionsType,
    Group,
    base_definitions,
)
from .typing import Definitions


class File(AbstractContextManager[Group], Group):
    def __init__(
        self,
        name: str | os.PathLike[str] | io.BytesIO | h5py.Group,
        *args: Any,
        definitions: Definitions | DefaultDefinitionsType = DefaultDefinitions,
        **kwargs: Any,
    ) -> None:
        r"""Context manager for NeXus files, similar to h5py.File.

        Arguments other than documented are as in :py:class:`h5py.File`.

        Parameters
        ----------
        name:
            Specifies the file to open.
            For Windows paths written in Python source, use a raw string such as
            ``r"C:\data\run.nxs"`` or forward slashes as in ``"C:/data/run.nxs"``.
            Otherwise, Python can interpret backslashes as escape sequences
            before the path reaches ScippNexus.
            If given an open :class:`h5py.File`, the context manager wraps the
            file handle and leaves it open on exit.
        definitions:
            Mapping of NX_class names to application-specific definitions.
            The default is to use the base definitions as defined in the
            NeXus standard.
        """
        defs: Definitions = (
            base_definitions() if definitions is DefaultDefinitions else definitions  # type: ignore[assignment]
        )

        if isinstance(name, h5py.File | h5py.Group):
            if args or kwargs:
                raise TypeError('Cannot provide both h5py.File and other arguments')
            self._file = name
            self._manage_file_context = False
        else:
            self._file = h5py.File(name, *args, **kwargs)
            self._manage_file_context = True
        super().__init__(self._file, definitions=defs)

    def __enter__(self) -> Group:
        if self._manage_file_context:
            self._file.__enter__()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._manage_file_context:
            self._file.close()

    def close(self) -> None:
        self._file.close()
