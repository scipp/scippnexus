# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from contextlib import AbstractContextManager
from typing import Mapping

import h5py

from .base import Group, base_definitions

_default_definitions = object()


class File(AbstractContextManager, Group):
    def __init__(self, *args, definitions: Mapping = _default_definitions, **kwargs):
        """Context manager for NeXus files, similar to h5py.File.

        Arguments other than documented are as in :py:class:`h5py.File`.

        Parameters
        ----------
        definitions:
            Mapping of NX_class names to application-specific definitions.
            The default is to use the base definitions as defined in the
            NeXus standard.
        """
        if definitions is _default_definitions:
            definitions = base_definitions()
        self._file = h5py.File(*args, **kwargs)
        super().__init__(self._file, definitions=definitions)

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    def close(self):
        self._file.close()
