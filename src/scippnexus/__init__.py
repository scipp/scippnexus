# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

# flake8: noqa
import importlib.metadata
try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import typing
from .file import File
from .nxdata import NXdata
from .nxdetector import NXdetector
from .nxdisk_chopper import NXdisk_chopper
from .nxevent_data import NXevent_data
from .nxlog import NXlog
from .nxmonitor import NXmonitor
from .nxobject import NX_class, NXobject, Field
from .nxobject import NXentry, NXinstrument, NXroot, NXtransformations
from .nxobject import NexusStructureError
from .nxsample import NXsample
from .nxsource import NXsource
