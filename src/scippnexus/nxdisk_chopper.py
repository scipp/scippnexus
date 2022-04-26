# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc
from .nxobject import NXobject, ScippIndex


class NXdisk_chopper(NXobject):
    """Disk chopper information, can be read as a dataset.

    Currently only the 'distance' and 'rotation_speed' fields are loaded.
    """

    @property
    def shape(self):
        return []

    @property
    def dims(self):
        return []

    def _getitem(self, select: ScippIndex) -> sc.Dataset:
        ds = sc.Dataset()
        for name in ['distance', 'rotation_speed']:
            if (field := self.get(name)) is not None:
                ds[name] = field[select]
        return ds
