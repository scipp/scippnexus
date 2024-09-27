# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import h5py
import scipp as sc

from .base import Group, NexusStructureError, ScippIndex
from .field import Field, depends_on_to_relative_path
from .file import File
from .typing import H5Base


class TransformationError(NexusStructureError):
    pass


@dataclass
class Transform:
    name: str
    transformation_type: Literal['translation', 'rotation']
    value: sc.DataArray | sc.Variable
    vector: sc.Variable
    depends_on: str
    offset: sc.Variable | None

    def __post_init__(self):
        if self.transformation_type not in ['translation', 'rotation']:
            raise TransformationError(
                f"{self.transformation_type=} attribute at {self.name},"
                " expected 'translation' or 'rotation'."
            )

    @staticmethod
    def from_object(
        obj: Field | Group, value: sc.Variable | sc.DataArray | sc.DataGroup
    ) -> Transform:
        depends_on = depends_on_to_relative_path(
            obj.attrs['depends_on'], obj.parent.name
        )
        return Transform(
            name=obj.name,
            transformation_type=obj.attrs.get('transformation_type'),
            value=_parse_value(obj, value),
            vector=sc.vector(value=obj.attrs.get('vector')),
            depends_on=depends_on,
            offset=_parse_offset(obj),
        )

    def build(self) -> sc.Variable | sc.DataArray:
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


def find_transformations(filename: str) -> list[str]:
    transforms: list[str] = []

    def _collect_transforms(name: str, obj: H5Base) -> None:
        if name.endswith('/depends_on') or 'transformation_type' in obj.attrs:
            transforms.append(f'/{name}')

    with h5py.File(filename, 'r') as f:
        f.visititems(_collect_transforms)
    return transforms


def load_transformations(filename: str) -> sc.DataGroup:
    """
    Load transformations and depends_on fields from a NeXus file.

    Parameters
    ----------
    filename:
        The path to the NeXus file.

    Returns
    -------
    :
        A flat DataGroup with the transformations and depends_on fields.
    """
    groups = find_transformations(filename)
    with File(filename, mode='r', maybe_transformation=_maybe_transformation) as f:
        return sc.DataGroup({group: f[group][()] for group in groups})


def apply_to_transformations(
    dg: sc.DataGroup, func: Callable[[Transform], Transform]
) -> sc.DataGroup:
    def apply_nested(node: Any) -> Any:
        if isinstance(node, sc.DataGroup):
            return node.apply(apply_nested)
        if isinstance(node, Transform):
            return func(node)
        return node

    return dg.apply(apply_nested)


def as_nested(dg: sc.DataGroup) -> sc.DataGroup:
    """
    Convert a flat DataGroup with paths as keys to a nested DataGroup.

    This is useful when loading transformations from a NeXus file, where the
    paths are used as keys to represent the structure of the NeXus file.

    Parameters
    ----------
    dg:
        The flat DataGroup to convert.

    Returns
    -------
    :
        The nested DataGroup.
    """
    out = sc.DataGroup()
    for path, value in dg.items():
        _set_recursive(out, path, value)
    return out


def _maybe_transformation(
    obj: Field | Group,
    value: sc.Variable | sc.DataArray | sc.DataGroup,
    sel: ScippIndex,
) -> sc.Variable | sc.DataArray | sc.DataGroup:
    if obj.attrs.get('transformation_type') is None:
        return value
    try:
        return Transform.from_object(obj, value)
    except KeyError as e:
        warnings.warn(
            UserWarning(f'Invalid transformation, missing attribute {e}'), stacklevel=2
        )
        return value


def _set_recursive(dg: sc.DataGroup, path: str, value: Any) -> None:
    if '/' not in path:
        dg[path] = value
    else:
        path = path.lstrip('/')
        first, remainder = path.split('/', maxsplit=1)
        if first not in dg:
            dg[first] = sc.DataGroup()
        _set_recursive(dg[first], remainder, value)


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
