# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from .leaf import Leaf
from typing import Dict, Union
import scipp as sc
from scipp.spatial import linear_transform
from .nxobject import ScippIndex

_matrix_units = dict(zip(['orientation_matrix', 'ub_matrix'], ['one', '1/Angstrom']))


class NXsample(Leaf):
    """Sample information, can be read as a dict.
    """

    def _getitem(self,
                 select: ScippIndex) -> Dict[str, Union[sc.Variable, sc.DataArray]]:
        content = super()._getitem(select)
        for key in _matrix_units:
            if (item := content.get(key)) is not None:
                content[key] = linear_transform(value=item.values,
                                                unit=_matrix_units[key])
        return content
