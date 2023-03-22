# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

# flake8: noqa
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .. import typing
#from .nxdata import NXdataStrategy
#from .nxdetector import NXdetectorStrategy
#from .nxlog import NXlogStrategy
from .base import (
    Field,
    Group,
    NexusStructureError,
    NXobject,
    base_definitions,
    create_class,
    create_field,
    group_events_by_detector_number,
)
#from .definition import ApplicationDefinition, make_definition
#from .file import File
from .nexus_classes import *
