# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Utilities for loading and working with NeXus transformations.

Transformation chains in NeXus files can be non-local and can thus be challenging to
work with. Additionally, values of transformations can be time-dependent, with each
chain link potentially having a different time-dependent value. In practice the user is
interested in the position and orientation of a component at a specific time or time
range. This may involve evaluating the transformation chain at a specific time, or
applying some heuristic to determine if the changes in the transformation value are
significant or just noise. In combination, the above means that we need to remain
flexible in how we handle transformations, preserving all necessary information from
the source files. This module is therefore structured as follows:

1. :py:class:`Transform` is a dataclass representing a transformation. The raw `value`
   dataset is preserved (instead of directly converting to, e.g., a rotation matrix) to
   facilitate further processing such as computing the mean or variance.
2. :py:func:`load_transformations` loads transformations from a NeXus file into a flat
   :py:class:`scipp.DataGroup`. It can optionally be followed by
   :py:func:`as_nested` to convert the flat structure to a nested one.
3. :py:func:`apply_to_transformations` applies a function to each transformation in a
   :py:class:`scipp.DataGroup`. We imagine that this can be used to
   - Evaluate the transformation at a specific time.
   - Apply filters to remove noise, to avoid having to deal with very small time
     intervals when processing data.

By keeping the loaded transformations in a simple and modifiable format, we can
furthermore manually update the transformations with information from other sources,
such as streamed NXlog values received from a data acquisition system.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import scipp as sc

from .base import Group, NexusStructureError
from .field import DependsOn, Field


class TransformationError(NexusStructureError):
    pass


@dataclass
class Transform:
    name: str
    transformation_type: Literal['translation', 'rotation']
    value: sc.Variable | sc.DataArray | sc.DataGroup
    vector: sc.Variable
    depends_on: DependsOn
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
        depends_on = DependsOn(parent=obj.parent.name, value=obj.attrs['depends_on'])
        return Transform(
            name=obj.name,
            transformation_type=obj.attrs.get('transformation_type'),
            value=_parse_value(value),
            vector=sc.vector(value=obj.attrs['vector']),
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
    value: sc.Variable | sc.DataArray | sc.DataGroup,
) -> sc.Variable | sc.DataArray | sc.DataGroup:
    if isinstance(value, sc.DataGroup) and (
        isinstance(value.get('value'), sc.DataArray)
    ):
        # Some NXlog groups are split into value, alarm, and connection_status
        # sublogs. We only care about the value.
        value = value['value']
    return value
