# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Dict, List, Tuple, Union

import numpy as np
import scipp as sc

from ._common import to_plain_index
from .nxobject import (
    FieldInfo,
    GroupContentInfo,
    NexusStructureError,
    NXobject,
    NXobjectInfo,
    ScippIndex,
)

_event_dimension = "event"
_pulse_dimension = "event_time_zero"


class NXevent_data(NXobject):
    _field_names = [
        'event_time_zero', 'event_index', 'event_time_offset', 'event_id',
        'cue_timestamp_zero', 'cue_index', 'pulse_height'
    ]

    @staticmethod
    def _make_class_info(info: GroupContentInfo) -> NXobjectInfo:
        """Create info object for this NeXus class."""
        children = {}
        for name in NXevent_data._field_names:
            if (di := info.datasets.pop(name, None)) is not None:
                children[name] = FieldInfo(values=di.value,
                                           dims=NXevent_data._get_field_dims(name))
        return NXobjectInfo(children=children)

    def _read_children(self, children, select: ScippIndex) -> sc.DataGroup:
        self._check_for_missing_fields()
        index = to_plain_index([_pulse_dimension], select)

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

        if not last_loaded:
            event_index = np.append(event_index, num_event)

        event_index = sc.array(dims=[_pulse_dimension],
                               values=event_index,
                               dtype=sc.DType.int64,
                               unit=None)

        event_index -= event_index.min()

        dg = sc.DataGroup(event_time_zero=event_time_zero,
                          event_index=event_index,
                          event_time_offset=event_time_offset)
        if event_id is not None:
            dg['event_id'] = event_id
        return dg

    def _assemble(self, children: sc.DataGroup) -> sc.DataGroup:
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
        # TODO This could probably be simplified if we make use of the sc.bins feature
        # to setup such 'ends' automatically if not provided. Affects also code above.
        if event_time_zero.ndim == 0:
            begins = event_index[_pulse_dimension, 0]
            ends = event_index[_pulse_dimension, 1]
        else:
            begins = event_index[_pulse_dimension, :-1]
            ends = event_index[_pulse_dimension, 1:]

        try:
            binned = sc.bins(data=events, dim=_event_dimension, begin=begins, end=ends)
        except IndexError as e:
            raise NexusStructureError(
                f"Invalid index in NXevent_data at {self.name}/event_index:\n{e}.")

        return sc.DataArray(data=binned, coords={'event_time_zero': event_time_zero})

    @property
    def shape(self) -> Tuple[int]:
        if (event_index := self._info.children.get('event_index')) is not None:
                return event_index.values.shape
        return ()

    @property
    def dims(self) -> List[str]:
        return [_pulse_dimension][:len(self.shape)]

    @property
    def sizes(self) -> Dict[str, int]:
        return dict(zip(self.dims, self.shape))

    @property
    def unit(self) -> None:
        # Binned data, bins do not have a unit
        return None

    @staticmethod
    def _get_field_dims(name: str) -> Union[None, List[str]]:
        if name in ['event_time_zero', 'event_index']:
            return [_pulse_dimension]
        if name in ['event_time_offset', 'event_id']:
            return [_event_dimension]
        return None

    def _check_for_missing_fields(self):
        for field in ("event_time_zero", "event_index", "event_time_offset"):
            if field not in self._info.children:
                raise NexusStructureError(
                    f"Required field {field} not found in NXevent_data")
