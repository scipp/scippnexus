# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any

import scipp as sc

from ._common import (
    convert_time_to_datetime64,
    has_time_unit,
    to_canonical_select,
)
from .base import Group, base_definitions_dict
from .field import Field
from .nxdata import NXdata
from .typing import ScippIndex


class NXlog(NXdata):
    """
    NXlog, a time-series that can be loaded as a DataArray.

    In some cases the NXlog may contain additional time series, such as a connection
    status or alarm. These cannot be handled in a standard way, since the result cannot
    be represented as a single DataArray. Furthermore, they prevent positional
    time-indexing, since the time coord of each time-series is different. We can
    support label-based indexing for this in the future. If additional time-series
    are contained within the NXlog then loading will return a DataGroup of the
    individual time-series (DataArrays).
    """

    def __init__(self, attrs: dict[str, Any], children: dict[str, Field | Group]):
        children = dict(children)
        self._sublogs = []
        self._sublog_children = {}
        for name in children:
            if name.endswith('_time'):
                self._sublogs.append(name[:-5])
        # Extract all fields that belong to sublogs, since they will interfere with the
        # setup logic in the base class (NXdata).
        for name in self._sublogs:
            for k in list(children):
                if k.startswith(name):
                    field = children.pop(k)
                    field.sizes = {
                        'time' if i == 0 else f'dim_{i}': size
                        for i, size in enumerate(field.dataset.shape)
                    }
                    self._sublog_children[k] = field

        super().__init__(
            attrs=attrs,
            children=children,
            fallback_dims=('time',),
            fallback_signal_name='value' if 'value' in children else 'time',
        )

    def read_children(self, sel: ScippIndex) -> sc.DataGroup:
        # Sublogs have distinct time axes (with a different length). Must disable
        # positional indexing.
        if self._sublogs and ('time' in to_canonical_select(list(self.sizes), sel)):
            raise sc.DimensionError(
                "Cannot positionally select time since there are multiple "
                "time fields. Label-based selection is not supported yet."
            )
        dg = super().read_children(sel)
        for name, field in self._sublog_children.items():
            dg[name] = field[sel]
        return dg

    def _time_to_datetime(self, mapping):
        if (time := mapping.get('time')) is not None:
            if time.dtype != sc.DType.datetime64 and has_time_unit(time):
                mapping['time'] = convert_time_to_datetime64(
                    time, start=sc.epoch(unit=time.unit)
                )

    def _assemble_sublog(
        self, dg: sc.DataGroup, name: str, value_name: str | None = None
    ) -> sc.DataArray:
        value_name = name if value_name is None else f'{name}_{value_name}'
        da = sc.DataArray(dg.pop(value_name), coords={'time': dg.pop(f'{name}_time')})
        for k in list(dg):
            if k.startswith(name):
                da.coords[k[len(name) + 1 :]] = dg.pop(k)
        self._time_to_datetime(da.coords)
        return da

    def assemble(self, dg: sc.DataGroup) -> sc.DataGroup | sc.DataArray | sc.Dataset:
        self._time_to_datetime(dg)
        dg = sc.DataGroup(dg)
        sublogs = sc.DataGroup()
        for name in self._sublogs:
            # Somewhat arbitrary definition of which fields is the "value"
            value_name = 'severity' if name == 'alarm' else None
            sublogs[name] = self._assemble_sublog(dg, name, value_name=value_name)
        out = super().assemble(dg)
        return out if not sublogs else sc.DataGroup(value=out, **sublogs)


base_definitions_dict['NXlog'] = NXlog
