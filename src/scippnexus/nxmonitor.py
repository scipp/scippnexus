# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import List, Union
import warnings
import scipp as sc
from .nxevent_data import NXevent_data
from .nxobject import NXobject, ScippIndex
from .nxdata import NXdata


class NXmonitor(NXobject):
    @property
    def shape(self) -> List[int]:
        return self._nxbase.shape

    @property
    def dims(self) -> List[str]:
        return self._nxbase.dims

    @property
    def unit(self) -> Union[sc.Unit, None]:
        return self._nxbase.unit

    @property
    def _is_events(self) -> bool:
        return 'event_time_offset' in self

    @property
    def _nxbase(self) -> Union[NXdata, NXevent_data]:
        """Branch between event-mode and histogram-mode monitor."""
        if self._is_events:
            return NXevent_data(self._group)
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXmonitor uses a "hard-coded" signal name 'data', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        return NXdata(self._group, signal_name_default='data')

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        if self._is_events:
            if name in [
                    'event_time_zero', 'event_index', 'event_time_offset', 'event_id'
            ]:
                # Event field is direct child of this class
                return self._nxbase._get_field_dims(name)
            else:
                return self.dims
        return self._nxbase._get_field_dims(name)

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        """
        Load monitor data. Event-mode data takes precedence over histogram-mode data.
        """
        nxbase = self._nxbase
        if isinstance(nxbase, NXevent_data) and 'data' in self:
            warnings.warn(f"Event data present in NXmonitor group {self.name}. "
                          f"Histogram-mode monitor data from this group will be "
                          f"ignored.")
        return nxbase[select]
