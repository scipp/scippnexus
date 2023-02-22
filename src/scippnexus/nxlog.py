# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from typing import List, Union

import scipp as sc

from .nxdata import NXdata, NXdataStrategy
from .nxobject import NXobject, ScippIndex


class NXlogStrategy(NXdataStrategy):

    @staticmethod
    def axes2(info):
        if (ax := NXdataStrategy.axes2(info)) is not None:
            return ax
        # We get the shape from the original dataset, to make sure we do not squeeze
        # dimensions too early
        child_dataset = info.datasets['value'].value
        ndim = child_dataset.ndim
        shape = child_dataset.shape
        # The outermost axis in NXlog is pre-defined to 'time' (if present). Note
        # that this may be overridden by an `axes` attribute, if defined for the group.
        if 'time' in info.datasets:
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
    def signal2(info):
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXlog uses a "hard-coded" signal name 'value', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        signal_name = info.attrs.get('signal', 'value')
        if (signal := info.datasets.get(signal_name)) is not None:
            return signal_name, signal
        else:
            return None, None


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
        return NXdata(self._group,
                      strategy=NXlogStrategy,
                      skip=['cue_timestamp_zero', 'cue_index'])

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        base = self._nxbase
        # Field loads datetime offset attributes automatically, but for NXlog this
        # may apparently be omitted and must then interpreted as relative to epoch.
        base.child_params['time'] = {'is_time': True}
        return base[select]

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        return self._nxbase._get_field_dims(name)
