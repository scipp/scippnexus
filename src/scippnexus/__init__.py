# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

# flake8: noqa
import importlib.metadata
try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .file import File
from .nxobject import NX_class, Field, NXroot
from .nxobject import NexusStructureError
from . import typing
