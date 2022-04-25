# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc
from scipp.spatial import linear_transform
from ._common import to_plain_index
from .nxobject import NXobject, ScippIndex


class NXsample(NXobject):
    """Sample information, can be read as a dataset.

    Currently only the 'distance', 'orientation_matrix', and 'ub_matrix' fields are
    loaded.
    """

    @property
    def shape(self):
        return []

    @property
    def dims(self):
        return []

    def _getitem(self, select: ScippIndex) -> sc.Dataset:
        index = to_plain_index([], select)
        if index != tuple():
            raise ValueError("Cannot select slice when loading NXsample")
        ds = sc.Dataset()
        if 'distance' in self:
            ds['distance'] = self['distance'][()]
        for name, unit in zip(['orientation_matrix', 'ub_matrix'],
                              ['one', '1/Angstrom']):
            if (m := self.get(name)) is not None:
                ds[name] = linear_transform(value=m[()].values, unit=unit)
        return ds
