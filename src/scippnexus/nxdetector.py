# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import scipp as sc

from .nxdata import NXdata, NXdataInfo, NXdataStrategy
from .nxevent_data import NXevent_data
from .nxobject import (
    Field,
    NexusStructureError,
    NXobject,
    NXobjectInfo,
    ScippIndex,
    asarray,
    is_dataset,
)


class NXdetectorStrategy(NXdataStrategy):

    @staticmethod
    def signal2(info):
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXdetector uses a "hard-coded" signal name 'data', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        # Note the special case of an NXevent_data subgroup named 'data', which we
        # avoid by checking if 'data' is a dataset.
        name, signal = NXdataStrategy.signal2(info)
        if name is not None:
            return name, signal
        if (ds := info.datasets.get('data')) is not None:
            return 'data', ds
        return None, None

    @staticmethod
    def signal(group):
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXdetector uses a "hard-coded" signal name 'data', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        # Note the special case of an NXevent_data subgroup named 'data', which we
        # avoid by checking if 'data' is a dataset.
        if (name := NXdataStrategy.signal(group)) is not None:
            return name
        return 'data' if 'data' in group and is_dataset(group._group['data']) else None


def group_events(*,
                 event_data: sc.DataArray,
                 grouping: Optional[sc.Variable] = None) -> sc.DataArray:
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
    # 'event_time_zero' binning defined by NXevent_data.
    event_data = event_data.bins.constituents['data'].group(event_id)
    #if self._grouping is None:
    #    event_data.coords[self._grouping_key] = event_data.coords.pop('event_id')
    #else:
    #    del event_data.coords['event_id']
    if grouping is None:
        return event_data
    return event_data.fold(dim='event_id', sizes=grouping.sizes)


class NXdetector(NXdata):
    """A detector or detector bank providing an array of values or events.

    If the detector stores event data then the 'detector_number' field (if present)
    is used to map event do detector pixels. Otherwise this returns event data in the
    same format as NXevent_data.
    """
    _detector_number_fields = ['detector_number', 'pixel_id', 'spectrum_index']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, strategy=NXdetectorStrategy)
        self._event_select = tuple()
        self._nxevent_data_fields = [
            'event_time_zero', 'event_index', 'event_time_offset', 'event_id',
            'cue_timestamp_zero', 'cue_index', 'pulse_height'
        ]

    def _make_class_info(self, group_info: GroupContentInfo) -> NXobjectInfo:
        grouping_key = None
        fallback_dims = None
        for key in self._detector_number_fields:
            if (grouping := group_info.datasets.get(key)) is not None:
                grouping_key = key
                if len(grouping.shape) == 0:
                    fallback_dims = ()
                elif len(grouping.shape) == 1:
                    fallback_dims = (key, )
                break
        di = NXdataInfo.from_group_info(info=group_info,
                                        strategy=self._strategy,
                                        fallback_dims=fallback_dims)
        fields = dict(di.field_infos)
        fields.update(group_info.groups)
        info = NXobjectInfo(children=fields)
        if di.signal_name is None:
            info.signal_name = grouping_key
        else:
            info.signal_name = di.signal_name

        event_data = None
        event_entries = []
        for name in list(group_info.groups):
            if group_info.groups[name].nx_class == NXevent_data:
                event_entries.append(name)
                #event_entries.append(group_info.groups[name])
        info.event_entries = event_entries
        # TODO
        #if 'event_time_offset' in self:
        #    # Consumes datasets from self
        #    event_data = NXevent_data._make_class_info(info)
        return info

    @property
    def detector_number(self) -> Optional[Field]:
        for key in self._detector_number_fields:
            if key in self:
                return key
        return None

    @property
    def _event_grouping(self) -> Union[None, Dict[str, Any]]:
        for key in self._detector_number_fields:
            if key in self:
                return {'grouping_key': key, 'grouping': self[key]}
        return {}

    def _nxdata(self, use_event_signal=True) -> NXdata:
        events = self.events
        if use_event_signal and events is not None:
            signal = _EventField(events, self._event_select, **self._event_grouping)
        else:
            signal = None
        skip = None
        if events is not None:
            if events.name == self.name:
                skip = self._nxevent_data_fields
            else:
                skip = [events.name.split('/')[-1]]  # name of the subgroup
        return NXdata(self._group,
                      strategy=NXdetectorStrategy,
                      signal_override=signal,
                      skip=skip)

    @property
    def events(self) -> Union[None, NXevent_data]:
        """Return the underlying NXevent_data group, None if not event data."""
        # The standard is unclear on whether the 'data' field may be NXevent_data or
        # whether the fields of NXevent_data should be stored directly within this
        # NXdetector. Both cases are observed in the wild.
        event_entries = self[NXevent_data]
        if len(event_entries) > 1:
            raise NexusStructureError("No unique NXevent_data entry in NXdetector. "
                                      f"Found {len(event_entries)}.")
        if len(event_entries) == 1:
            # If there is also a signal dataset (not events) it will be ignored
            # (except for possibly using it to deduce shape and dims).
            return next(iter(event_entries.values()))
        if 'event_time_offset' in self:
            return NXevent_data(self._group)
        return None

    @property
    def select_events(self) -> EventSelector:
        """
        Return a proxy object for selecting a slice of the underlying NXevent_data
        group, while keeping wrapping the NXdetector.
        """
        if self._info.events is None:
            raise NexusStructureError(
                "Cannot select events in NXdetector not containing NXevent_data.")
        return EventSelector(self)

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        return self._info.field_dims[name]
        if self.events is not None:
            if name in self._nxevent_data_fields:
                # Event field is direct child of this class
                return self.events._get_field_dims(name)
            if name in self._detector_number_fields:
                # If there is a signal field in addition to the event data it can be
                # used to define dimension labels
                nxdata = self._nxdata(use_event_signal=False)
                if nxdata._signal_name is not None:
                    return nxdata._get_field_dims(name)
                # If grouping is 1-D then we use this name as the dim
                if self._get_child(name).ndim == 1:
                    return [name]
                return None
        return self._nxdata()._get_field_dims(name)

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        return self._nxdata()._getitem(select)

    def _assemble(self, children: sc.DataGroup) -> sc.DataArray:
        children = sc.DataGroup(children)
        return children
        if self._info.event_entries:
            events = children.pop(self._info.event_entries[0])
            grouping_key = self._info.signal_name
            grouping = children[grouping_key]
            event_field = _EventField(events,
                                      event_select=...,
                                      grouping=grouping,
                                      grouping_key=grouping_key)
            print(event_field, grouping)
            signal = event_field[...]
        else:
            signal = children.pop(self._info.signal_name)
        print(signal)
        signal = signal if isinstance(signal, sc.Variable) else signal.data
        coords = children
        coords = {name: asarray(child) for name, child in children.items()}
        print(list(coords.items()))
        da = sc.DataArray(data=signal, coords=coords)
        for name in list(da.coords):
            # TODO building again is inefficient!
            if self._coord_to_attr(da, name, self._info.children[name].build()):
                da.attrs[name] = da.coords.pop(name)
        return da
        # TODO Do not raise here... just change init!
        if len(event_entries) > 1:
            raise NexusStructureError("No unique NXevent_data entry in NXdetector. "
                                      f"Found {len(event_entries)}.")
        if len(event_entries) == 1:
            # If there is also a signal dataset (not events) it will be ignored
            # (except for possibly using it to deduce shape and dims).
            event_data = event_entries[0].build()


def group_events_by_detector_number(dg: sc.DataGroup) -> sc.DataArray:
    for name, value in dg.items():
        if isinstance(
                value, sc.DataArray
        ) and 'event_time_zero' in value.coords and value.bins is not None:
            event_entry = name
            break
    events = dg.pop(event_entry)
    grouping_key = None
    for key in NXdetector._detector_number_fields:
        if (grouping := dg.get(key)) is not None:
            grouping_key = key
            break
    grouping = dg.pop(grouping_key)
    #event_field = _EventField(events,
    #                          event_select=...,
    #                          grouping=grouping,
    #                          grouping_key=grouping_key)
    da = group_events(event_data=events, grouping=asarray(grouping))
    da.coords.update(dg)
    return da
