# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Dict, Union

import scipp as sc
from scipp.spatial import linear_transform

from .leaf import Leaf
from .nxobject import ScippIndex

_matrix_units = {
    "orientation_matrix": "one",
    "ub_matrix": "1/Angstrom",
}


class NXsample(Leaf):
    """Sample information, can be read as a dict."""

    def _getitem(
        self, select: ScippIndex
    ) -> Dict[str, Union[sc.Variable, sc.DataArray]]:
        content = super()._getitem(select)
        for key in _matrix_units:
            if (item := content.get(key)) is not None:
                content[key] = linear_transform(
                    value=item.values, unit=_matrix_units[key]
                )
        return content
