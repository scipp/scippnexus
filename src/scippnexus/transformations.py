# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipp as sc
from scipp.scipy import interpolate

from .base import Group, NexusStructureError, NXobject, ScippIndex
from .field import Field, depends_on_to_relative_path
from .file import File
import h5py
from typing import Any


class TransformationError(NexusStructureError):
    pass


# Plan:
# - helper to find an load al nxtransformations
# - do not auto-conv to scipp transform, add special classes for trans and rot
# - consider skipping trans load as part of group load in favor of separate load?
#   or just remove code that follows chains?
# - avoid storing depends_on as weird coord! use dedicated data structure
# - have raw Transform (with vector and offset)
# - translate into scipp transform when building transform

# TODO
# - remove resolved_depends_on mechanism
# - remove Transformation


def _parse_offset(obj: Field | Group) -> sc.Variable | None:
    if (offset := obj.attrs.get('offset')) is None:
        return None
    if (offset_units := obj.attrs.get('offset_units')) is None:
        raise TransformationError(
            f"Found {offset=} but no corresponding 'offset_units' "
            f"attribute at {obj.name}"
        )
    return sc.spatial.translation(value=offset, unit=offset_units)


def _parse_value(
    obj: Field | Group,
    value: sc.Variable | sc.DataArray | sc.DataGroup,
) -> sc.Variable | sc.DataArray:
    if isinstance(value, sc.DataGroup) and (
        isinstance(value.get('value'), sc.DataArray)
    ):
        # Some NXlog groups are split into value, alarm, and connection_status
        # sublogs. We only care about the value.
        value = value['value']
    if not isinstance(value, sc.Variable | sc.DataArray):
        raise TransformationError(f"Failed to load transformation value at {obj.name}")
    return value


class Transform:
    def __init__(
        self, obj: Field | Group, value: sc.Variable | sc.DataArray | sc.DataGroup
    ):
        self.offset = _parse_offset(obj)
        self.vector = sc.vector(value=obj.attrs.get('vector'))
        # TODO This is annoying... what if we keep it, load transform independently,
        # and index before returning?
        # TODO Change NXobject.__getitem__ to never descend into NXtransformations?
        self.depends_on = depends_on_to_relative_path(
            obj.attrs.get('depends_on'), obj.parent.name
        )
        self.transformation_type = obj.attrs.get('transformation_type')
        if self.transformation_type not in ['translation', 'rotation']:
            raise TransformationError(
                f"{self.transformation_type=} attribute at {obj.name},"
                " expected 'translation' or 'rotation'."
            )
        self.value = _parse_value(obj, value)

    # TODO can cache this
    def build(self) -> sc.Variable | sc.DataArray:
        try:
            t = self.value * self.vector
            v = t if isinstance(t, sc.Variable) else t.data
            if self.transformation_type == 'translation':
                v = sc.spatial.translations(dims=v.dims, values=v.values, unit=v.unit)
            elif self.transformation_type == 'rotation':
                v = sc.spatial.rotations_from_rotvecs(v)
            if isinstance(t, sc.Variable):
                t = v
            else:
                t.data = v
            if self.offset is None:
                return t
            if self.transformation_type == 'translation':
                return t * self.offset.to(unit=t.unit, copy=False)
            return t * self.offset
        except (sc.DimensionError, sc.UnitError, TransformationError):
            # TODO We should probably try to return some other data structure and
            # also insert offset and other attributes.
            return self.value


def find_transformation_groups(filename: str) -> list[str]:
    transforms: list[str] = []

    def _collect_transforms(name: str, group: h5py.Group) -> None:
        if group.attrs.get('NX_class') == 'NXtransformations':
            transforms.append(name)

    with h5py.File(filename, 'r') as f:
        # TODO This is slow! No need to visit everything, just recurse groups?
        f.visititems(_collect_transforms)
    return transforms


def _set_recursive(dg: sc.DataGroup, path: str, value: Any) -> None:
    if '/' not in path:
        dg[path] = value
    else:
        first, remainder = path.split('/', maxsplit=1)
        if first not in dg:
            dg[first] = sc.DataGroup()
        _set_recursive(dg[first], remainder, value)


def load_transformations(filename: str) -> sc.DataGroup:
    groups = find_transformation_groups(filename)
    with File(filename, mode='r') as f:
        transforms = sc.DataGroup({group: f[group][()] for group in groups})
    dg = sc.DataGroup()
    for path, value in transforms.items():
        _set_recursive(dg, path, value)
    return dg
