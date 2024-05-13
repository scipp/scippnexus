# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Any

import numpy as np
import scipp as sc

from ._common import to_plain_index
from .base import (
    Group,
    NexusStructureError,
    NXobject,
    ScippIndex,
    base_definitions_dict,
)
from .field import Field

_event_dimension = "event"
_pulse_dimension = "event_time_zero"


def _check_for_missing_fields(fields):
    for field in NXevent_data.mandatory_fields:
        if field not in fields:
            raise NexusStructureError(
                f"Required field {field} not found in NXevent_data"
            )


class NXevent_data(NXobject):
    mandatory_fields = ("event_index", "event_time_offset")
    handled_fields = (
        *mandatory_fields,
        "event_time_zero",
        "event_id",
    )

    def __init__(self, attrs: dict[str, Any], children: dict[str, Field | Group]):
        super().__init__(attrs=attrs, children=children)
        for name, field in children.items():
            if name in ['event_time_zero', 'event_index']:
                field.sizes = {_pulse_dimension: field.dataset.shape[0]}
            elif name in ['event_time_offset', 'event_id']:
                field.sizes = {_event_dimension: field.dataset.shape[0]}

    @property
    def shape(self) -> tuple[int, ...]:
        if (event_index := self._children.get('event_index')) is not None:
            return event_index.shape
        return ()

    @property
    def dims(self) -> tuple[str, ...]:
        return (_pulse_dimension,)[: len(self.shape)]

    @property
    def sizes(self) -> dict[str, int]:
        return dict(zip(self.dims, self.shape, strict=True))

    def field_dims(self, name: str, field: Field) -> tuple[str, ...]:
        if name in ['event_time_zero', 'event_index']:
            return (_pulse_dimension,)
        if name in ['event_time_offset', 'event_id']:
            return (_event_dimension,)
        return None

    def read_children(self, select: ScippIndex) -> sc.DataGroup:
        children = self._children
        if not children:  # TODO Check that select is trivial?
            return sc.DataGroup()

        select = self.convert_label_index_to_positional(select)
        index = to_plain_index([_pulse_dimension], select)

        coords = {}
        if 'event_time_zero' in children:
            coords['event_time_zero'] = children['event_time_zero'][index]

        last_loaded, event_index = self._get_event_index(children, index)
        num_event = children["event_time_offset"].shape[0]
        # Some files contain uint64 "max" indices, which turn into negatives during
        # conversion to int64. This is a hack to get around this.
        event_index[event_index < 0] = num_event

        if len(event_index) > 0:
            event_select = slice(
                event_index[0], event_index[-1] if last_loaded else num_event
            )
        else:
            event_select = slice(0, 0)

        event_time_offset = children['event_time_offset'][event_select]

        event_index = sc.array(
            dims=[_pulse_dimension],
            values=event_index[:-1] if last_loaded else event_index,
            dtype=sc.DType.int64,
            unit=None,
        )

        event_index -= event_index.min()

        dg = sc.DataGroup(
            event_index=event_index,
            event_time_offset=event_time_offset,
            **coords,
        )
        if (event_id := children.get('event_id')) is not None:
            dg['event_id'] = event_id[event_select]
        return dg

    def _get_event_index(self, children: sc.DataGroup, index):
        max_index = self.shape[0]
        if index is Ellipsis or index == ():
            last_loaded = False
        else:
            if isinstance(index, int):
                start, stop, _ = slice(index, None).indices(max_index)
                if start == stop:
                    raise IndexError(f'Index {start} is out of range')
                index = slice(start, start + 1)
            start, stop, stride = index.indices(max_index)
            if stop + stride > max_index:
                last_loaded = False
            elif start == stop:
                last_loaded = True
            else:
                stop += stride
                last_loaded = True
            index = slice(start, stop, stride)

        event_index = children['event_index'][index].values
        return last_loaded, event_index

    def assemble(self, children: sc.DataGroup) -> sc.DataArray:
        _check_for_missing_fields(children)
        event_time_offset = children['event_time_offset']

        coords = {}
        if 'event_time_zero' in children:
            coords['event_time_zero'] = children['event_time_zero']
        event_index = children['event_index']

        # Weights are not stored in NeXus, so use 1s
        weights = sc.ones(
            dims=[_event_dimension],
            shape=event_time_offset.shape,
            unit='counts',
            dtype=np.float32,
        )

        events = sc.DataArray(
            data=weights, coords={'event_time_offset': event_time_offset}
        )
        if (event_id := children.get('event_id')) is not None:
            events.coords['event_id'] = event_id

        # There is some variation in the last recorded event_index in files from
        # different institutions. We try to make sure here that it is what would be the
        # first index of the next pulse. In other words, ensure that event_index
        # includes the bin edge for the last pulse.
        if 'event_time_zero' in coords and coords['event_time_zero'].ndim == 0:
            begins = event_index[_pulse_dimension, 0]
        else:
            begins = event_index

        try:
            binned = sc.bins(data=events, dim=_event_dimension, begin=begins)
        except IndexError as e:
            path = self._children['event_index'].name
            raise NexusStructureError(
                f"Invalid index in NXevent_data at {path}:\n{e}"
            ) from None

        return sc.DataArray(data=binned, coords=coords)


base_definitions_dict['NXevent_data'] = NXevent_data
