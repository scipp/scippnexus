# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import typing
from ._load import load
from .base import (
    Group,
    NexusStructureError,
    NXobject,
    base_definitions,
    create_class,
    create_field,
)
from .field import Attrs, DependsOn, Field
from .file import File
from .nexus_classes import *  # noqa: F403
from .nxtransformations import TransformationChain, compute_positions, zip_pixel_offsets

__all__ = [
    'Attrs',
    'DependsOn',
    'Field',
    'File',
    'Group',
    'NXobject',
    'NexusStructureError',
    'TransformationChain',
    'base_definitions',
    'compute_positions',
    'create_class',
    'create_field',
    'load',
    'typing',
    'zip_pixel_offsets',
]
