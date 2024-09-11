# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import io
import os
from contextlib import AbstractContextManager

import h5py

from .base import (
    DefaultDefinitions,
    DefaultDefinitionsType,
    Group,
    base_definitions,
)
from .typing import Definitions


class File(AbstractContextManager, Group):
    def __init__(
        self,
        name: str | os.PathLike[str] | io.BytesIO | h5py.Group,
        *args,
        definitions: Definitions | DefaultDefinitionsType = DefaultDefinitions,
        **kwargs,
    ):
        """Context manager for NeXus files, similar to h5py.File.

        Arguments other than documented are as in :py:class:`h5py.File`.

        Parameters
        ----------
        name:
            Specifies the file to open.
            If this is a :class:`hyp5.File` object, the `:class:`File` will wrap
            this file handle but will not close it when used as a context manager.
        definitions:
            Mapping of NX_class names to application-specific definitions.
            The default is to use the base definitions as defined in the
            NeXus standard.
        """
        if definitions is DefaultDefinitions:
            definitions = base_definitions()

        if isinstance(name, h5py.File | h5py.Group):
            if args or kwargs:
                raise TypeError('Cannot provide both h5py.File and other arguments')
            self._file = name
            self._manage_file_context = False
        else:
            self._file = h5py.File(name, *args, **kwargs)
            self._manage_file_context = True
        super().__init__(self._file, definitions=definitions)

    def __enter__(self):
        if self._manage_file_context:
            self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._manage_file_context:
            self._file.close()

    def close(self):
        self._file.close()
