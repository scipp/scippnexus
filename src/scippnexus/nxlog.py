# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from typing import List, Union
import scipp as sc
from .nxobject import NXobject, ScippIndex
from .nxdata import NXdata, NXdataStrategy


class NXlogStrategy(NXdataStrategy):

    @staticmethod
    def axes(group):
        if (ax := NXdataStrategy.axes(group)) is not None:
            return ax
        # We get the shape from the original dataset, to make sure we do not squeeze
        # dimensions too early
        child_dataset = group._get_child('value')._dataset
        ndim = child_dataset.ndim
        shape = child_dataset.shape
        # The outermost axis in NXlog is pre-defined to 'time' (if present). Note
        # that this may be overriden by an `axes` attribute, if defined for the group.
        if 'time' in group:
            raw_axes = ['time'] + (['.'] * (ndim - 1))
        else:
            raw_axes = ['.'] * ndim
        axes = []
        for i, ax in enumerate(raw_axes):
            # Squeeze dimensions that have size 1 and are not 'time'
            if (ax == 'time') or (shape[i] != 1):
                axes.append(ax)
        return axes

    @staticmethod
    def signal(group):
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXlog uses a "hard-coded" signal name 'value', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        return group.attrs.get('signal', 'value')


class NXlog(NXobject):

    @property
    def shape(self):
        return self._nxbase.shape

    @property
    def dims(self):
        return self._nxbase.dims

    @property
    def unit(self):
        return self._nxbase.unit

    @property
    def _nxbase(self) -> NXdata:
        return NXdata(self._group, strategy=NXlogStrategy)

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        base = self._nxbase
        # Field loads datetime offset attributes automatically, but for NXlog this
        # may apparently be omitted and must then interpreted as relative to epoch.
        base.child_params['time'] = {'is_time': True}
        return base[select]

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        return self._nxbase._get_field_dims(name)
