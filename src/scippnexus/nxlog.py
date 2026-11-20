# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any

import scipp as sc
from scipp.core import label_based_index_to_positional_index

from ._common import (
    convert_time_to_datetime64,
    has_time_unit,
    to_canonical_select,
)
from .base import Group, base_definitions_dict
from .field import Field
from .nxdata import NXdata
from .typing import ScippIndex


def _chronological_time_slice(
    sizes: dict[str, int], time: sc.Variable, bounds: slice
) -> tuple[str, slice]:
    # Scipp uses the coordinate's direction for label slices. Interpret the bounds
    # in increasing time, then map the positions back to preserve on-disk order.
    if time.sizes['time'] < 2 or time.values[0] <= time.values[-1]:
        return label_based_index_to_positional_index(sizes, time, bounds)
    increasing = sc.array(
        dims=time.dims, values=time.values[::-1], unit=time.unit, dtype=time.dtype
    )
    dim, selected = label_based_index_to_positional_index(sizes, increasing, bounds)
    size = sizes[dim]
    return dim, slice(size - selected.stop, size - selected.start)


class NXlog(NXdata):
    """A NeXus time series.

    An NXlog with auxiliary series such as alarms or connection status loads
    as a DataGroup containing one DataArray per series. A label-based range
    applies to each series' own clock. Bounds are in chronological order, and
    the returned samples keep their on-disk order. Positional indexes and single
    time points are ambiguous across multiple time axes.
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
        canonical_sel = to_canonical_select(list(self.sizes), sel)
        sublog_selections = {}
        if (
            self._sublogs
            and 'time' in canonical_sel
            and (len(canonical_sel) > 1 or canonical_sel['time'] != slice(None))
        ):
            time_index = canonical_sel['time']
            if (
                len(canonical_sel) != 1
                or not isinstance(time_index, slice)
                or not any(
                    isinstance(bound, sc.Variable)
                    for bound in (time_index.start, time_index.stop)
                )
            ):
                raise sc.DimensionError(
                    "Select a time range with label bounds when there are multiple "
                    "time fields."
                )
            time_coord = self._children.get('time')
            if not isinstance(time_coord, Field):
                raise sc.DimensionError("A time field is required for label selection.")
            sel = _chronological_time_slice(self.sizes, time_coord[()], time_index)
            for sublog in self._sublogs:
                time = self._sublog_children[f'{sublog}_time'][()]
                sublog_sel = _chronological_time_slice(time.sizes, time, time_index)
                for name in self._sublog_children:
                    if name.startswith(sublog):
                        sublog_selections[name] = sublog_sel

        dg = super().read_children(sel)
        for name, field in self._sublog_children.items():
            dg[name] = field[sublog_selections.get(name, sel)]
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
