# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipp as sc

from .._common import to_plain_index
from .base import (
    Field,
    Group,
    NexusStructureError,
    NXobject,
    ScippIndex,
    asarray,
    base_definitions,
)

_event_dimension = "event"
_pulse_dimension = "event_time_zero"


def _check_for_missing_fields(fields):
    for field in ("event_time_zero", "event_index", "event_time_offset"):
        if field not in fields:
            raise NexusStructureError(
                f"Required field {field} not found in NXevent_data")


class NXevent_data(NXobject):

    def __init__(self, group: Group):
        super().__init__(group)
        for name, field in group._children.items():
            if name in ['event_time_zero', 'event_index']:
                field.sizes = {_pulse_dimension: field.dataset.shape[0]}
            elif name in ['event_time_offset', 'event_id']:
                field.sizes = {_event_dimension: field.dataset.shape[0]}

    @property
    def shape(self) -> Tuple[int]:
        if (event_index := self._group.get('event_index')) is not None:
            return event_index.shape
        return ()

    @property
    def dims(self) -> List[str]:
        return (_pulse_dimension, )[:len(self.shape)]

    @property
    def sizes(self) -> Dict[str, int]:
        return dict(zip(self.dims, self.shape))

    def field_dims(self, name: str, field: Field) -> Tuple[str, ...]:
        if name in ['event_time_zero', 'event_index']:
            return (_pulse_dimension, )
        if name in ['event_time_offset', 'event_id']:
            return (_event_dimension, )
        return None

    def read_children(self, obj: Group, select: ScippIndex) -> sc.DataGroup:
        children = obj
        index = to_plain_index([_pulse_dimension], select)

        if not children:  # TODO Check that select is trivial?
            return sc.DataGroup()

        max_index = self.shape[0]
        event_time_zero = children['event_time_zero'][index]
        if index is Ellipsis or index == tuple():
            last_loaded = False
        else:
            if isinstance(index, int):
                start, stop, _ = slice(index, None).indices(max_index)
                if start == stop:
                    raise IndexError('Index {start} is out of range')
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

        num_event = children["event_time_offset"].shape[0]
        # Some files contain uint64 "max" indices, which turn into negatives during
        # conversion to int64. This is a hack to get around this.
        event_index[event_index < 0] = num_event

        if len(event_index) > 0:
            event_select = slice(event_index[0],
                                 event_index[-1] if last_loaded else num_event)
        else:
            event_select = slice(None)

        if (event_id := children.get('event_id')) is not None:
            event_id = event_id[event_select]
            if event_id.dtype not in [sc.DType.int32, sc.DType.int64]:
                raise NexusStructureError(
                    "NXevent_data contains event_id field with non-integer values")

        event_time_offset = children['event_time_offset'][event_select]

        event_index = sc.array(dims=[_pulse_dimension],
                               values=event_index[:-1] if last_loaded else event_index,
                               dtype=sc.DType.int64,
                               unit=None)

        event_index -= event_index.min()

        dg = sc.DataGroup(event_time_zero=event_time_zero,
                          event_index=event_index,
                          event_time_offset=event_time_offset)
        if event_id is not None:
            dg['event_id'] = event_id
        return dg

    def assemble(self, children: sc.DataGroup) -> sc.DataArray:
        _check_for_missing_fields(children)
        event_time_offset = children['event_time_offset']
        event_time_zero = children['event_time_zero']
        event_index = children['event_index']

        # Weights are not stored in NeXus, so use 1s
        weights = sc.ones(dims=[_event_dimension],
                          shape=event_time_offset.shape,
                          unit='counts',
                          dtype=np.float32)

        events = sc.DataArray(data=weights,
                              coords={'event_time_offset': event_time_offset})
        if (event_id := children.get('event_id')) is not None:
            events.coords['event_id'] = event_id

        # There is some variation in the last recorded event_index in files from
        # different institutions. We try to make sure here that it is what would be the
        # first index of the next pulse. In other words, ensure that event_index
        # includes the bin edge for the last pulse.
        if event_time_zero.ndim == 0:
            begins = event_index[_pulse_dimension, 0]
        else:
            begins = event_index

        try:
            binned = sc.bins(data=events, dim=_event_dimension, begin=begins)
        except IndexError as e:
            raise NexusStructureError(
                f"Invalid index in NXevent_data at {self.name}/event_index:\n{e}.")

        return sc.DataArray(data=binned, coords={'event_time_zero': event_time_zero})

    # TODO now unused
    @staticmethod
    def assemble_as_child(
            event_data: sc.DataArray,
            detector_number: Optional[sc.Variable] = None) -> sc.DataArray:
        grouping = asarray(detector_number)

        if isinstance(event_data, sc.DataGroup):
            raise NexusStructureError("Invalid NXevent_data in NXdetector.")
        if grouping is None:
            event_id = 'event_id'
        else:
            # copy since sc.bin cannot deal with a non-contiguous view
            event_id = grouping.flatten(to='event_id').copy()
        event_data.bins.coords['event_time_zero'] = sc.bins_like(
            event_data, fill_value=event_data.coords['event_time_zero'])
        # After loading raw NXevent_data it is guaranteed that the event table
        # is contiguous and that there is no masking. We can therefore use the
        # more efficient approach of binning from scratch instead of erasing the
        # 'pulse' binning defined by NXevent_data.
        event_data = event_data.bins.constituents['data'].group(event_id)
        # if self._grouping is None:
        #     event_data.coords[self._grouping_key] = event_data.coords.pop('event_id')
        # else:
        #     del event_data.coords['event_id']
        if grouping is None:
            return event_data
        return event_data.fold(dim='event_id', sizes=grouping.sizes)


base_definitions['NXevent_data'] = NXevent_data