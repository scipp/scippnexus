# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from typing import List, Union
import scipp as sc
from ._common import convert_time_to_datetime64
from .nxobject import NXobject, ScippIndex
from .nxdata import NXdata


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
        axes = ['.'] * self._get_child('value').ndim
        # The outermost axis in NXlog is pre-defined to 'time' (if present). Note
        # that this may be overriden by an `axes` attribute, if defined for the group.
        if 'time' in self:
            axes[0] = 'time'
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXlog uses a "hard-coded" signal name 'value', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        return NXdata(self._group, signal_name_default='value', axes=axes)

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        data = self._nxbase[select]
        # The 'time' field in NXlog contains extra properties 'start' and
        # 'scaling_factor' that are not handled by NXdata. These are used
        # to transform to a datetime-coord.
        if 'time' in self:
            if 'time' not in data.coords:
                raise sc.DimensionError(
                    "NXlog is time-dependent, but failed to load `time` dataset")
            data.coords['time'] = convert_time_to_datetime64(
                raw_times=data.coords.pop('time'),
                start=self['time'].attrs.get('start'),
                scaling_factor=self['time'].attrs.get('scaling_factor'),
                group_path=self['time'].name)
        return data

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        return self._nxbase._get_field_dims(name)
