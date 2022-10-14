# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from contextlib import AbstractContextManager
import h5py

from .nexus_classes import NXroot


class File(AbstractContextManager, NXroot):

    def __init__(self, *args, definition=None, **kwargs):
        self._file = h5py.File(*args, **kwargs)
        NXroot.__init__(self, self._file, definition=definition)

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    def close(self):
        self._file.close()
