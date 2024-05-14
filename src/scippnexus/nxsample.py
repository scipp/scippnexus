# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Any

import scipp as sc

from .base import Group, NXobject, ScippIndex, base_definitions_dict
from .field import Field

_matrix_units = {'orientation_matrix': 'one', 'ub_matrix': '1/Angstrom'}


def _fix_unit(name, value):
    if (unit := _matrix_units.get(name)) is not None:
        value.unit = unit
    return value


class NXsample(NXobject):
    """NXsample"""

    def __init__(self, attrs: dict[str, Any], children: dict[str, Field | Group]):
        super().__init__(attrs=attrs, children=children)
        for key in _matrix_units:
            if (field := children.get(key)) is not None:
                field.sizes = {k: field.sizes[k] for k in field.dims[:-2]}
                field.dtype = sc.DType.linear_transform3

    def read_children(self, sel: ScippIndex) -> sc.DataGroup:
        return sc.DataGroup(
            {
                name: _fix_unit(name, self.index_child(child, sel))
                for name, child in self._children.items()
            }
        )


base_definitions_dict['NXsample'] = NXsample
