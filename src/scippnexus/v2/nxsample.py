# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Any, Dict, Union

import scipp as sc

from .base import Field, Group, NXobject, ScippIndex, base_definitions

_matrix_units = {'orientation_matrix': 'one', 'ub_matrix': '1/Angstrom'}


def _fix_unit(name, value):
    if (unit := _matrix_units.get(name)) is not None:
        value.unit = unit
    return value


class NXsample(NXobject):
    """NXsample"""

    def __init__(self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]):
        super().__init__(attrs=attrs, children=children)
        for key in _matrix_units:
            if (field := children.get(key)) is not None:
                field.sizes = {k: field.sizes[k] for k in field.dims[:-2]}
                field.dtype = sc.DType.linear_transform3

    def read_children(self, obj: Group, sel: ScippIndex) -> sc.DataGroup:
        return sc.DataGroup({
            name: _fix_unit(name, self.index_child(child, sel))
            for name, child in obj.items()
        })


base_definitions['NXsample'] = NXsample
