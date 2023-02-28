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


def group(da: sc.DataArray, groups: sc.Variable) -> sc.DataArray:
    if hasattr(da, 'group'):
        return da.group(groups)
    else:
        return sc.bin(da, groups=[groups])


class EventSelector:
    """A proxy object for creating an NXdetector based on a selection of events.
    """

    def __init__(self, detector):
        self._detector = detector

    def __getitem__(self, select: ScippIndex) -> NXdetector:
        """Return an NXdetector based on a selection (slice) of events."""
        det = copy(self._detector)
        det._event_select = select
        return det


@dataclass
class EventFieldInfo:
    event_data: NXevent_data
    grouping_key: Optional[str] = 'detector_number'
    grouping: Optional[Field] = None

    def build(self) -> EventField:
        return _EventField(nxevent_data=self.event_data,
                           event_select=tuple(),
                           grouping_key=self.grouping_key,
                           grouping=self.grouping)


class _EventField:
    """Field-like wrapper of NXevent_data binned into pixels.

    This has no equivalent in the NeXus format, but represents the conceptual
    event-data "signal" dataset of an NXdetector.
    """

    def __init__(self,
                 nxevent_data: NXevent_data,
                 event_select: ScippIndex,
                 grouping_key: Optional[str] = 'detector_number',
                 grouping: Optional[Field] = None):
        self._nxevent_data = nxevent_data
        self._event_select = event_select
        self._grouping_key = grouping_key
        self._grouping = grouping

    @property
    def name(self) -> str:
        return self._nxevent_data.name

    @property
    def attrs(self):
        return self._nxevent_data.attrs

    @property
    def dims(self):
        if self._grouping is None:
            return [self._grouping_key]
        return self._grouping.dims

    @property
    def shape(self):
        if self._grouping is None:
            raise NexusStructureError(
                "Cannot get shape of NXdetector since no 'detector_number' "
                "field found but detector contains event data.")
        return self._grouping.shape

    @property
    def unit(self) -> None:
        return self._nxevent_data.unit

    def __getitem__(self, select: ScippIndex) -> sc.DataArray:
        event_data = self._nxevent_data[self._event_select]
        if isinstance(event_data, sc.DataGroup):
            raise NexusStructureError("Invalid NXevent_data in NXdetector.")
        if self._grouping is None:
            if select not in (Ellipsis, tuple(), slice(None)):
                raise NexusStructureError(
                    "Cannot load slice of NXdetector since it contains event data "
                    "but no 'detector_number' field, i.e., the shape is unknown. "
                    "Use ellipsis or an empty tuple to load the full detector.")
            # Ideally we would prefer to use np.unique, but a quick experiment shows
            # that this can easily be 100x slower, so it is not an option. In
            # practice most files have contiguous event_id values within a bank
            # (NXevent_data).
            id_min = event_data.bins.coords['event_id'].min()
            id_max = event_data.bins.coords['event_id'].max()
            grouping = sc.arange(dim=self._grouping_key,
                                 unit=None,
                                 start=id_min.value,
                                 stop=id_max.value + 1,
                                 dtype=id_min.dtype)
        else:
            grouping = asarray(self._grouping[select])
            if (self._grouping_key in event_data.coords) and sc.identical(
                    grouping, event_data.coords[self._grouping_key]):
                return event_data
        # copy since sc.bin cannot deal with a non-contiguous view
        event_id = grouping.flatten(to='event_id').copy()
        event_data.bins.coords['event_time_zero'] = sc.bins_like(
            event_data, fill_value=event_data.coords['event_time_zero'])
        # After loading raw NXevent_data it is guaranteed that the event table
        # is contiguous and that there is no masking. We can therefore use the
        # more efficient approach of binning from scratch instead of erasing the
        # 'pulse' binning defined by NXevent_data.
        event_data = group(event_data.bins.constituents['data'], groups=event_id)
        if self._grouping is None:
            event_data.coords[self._grouping_key] = event_data.coords.pop('event_id')
        else:
            del event_data.coords['event_id']
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
        # TODO doesn't popping break fallback?
        event_data = None
        event_entries = []
        for name in list(group_info.groups):
            if group_info.groups[name].nx_class == NXevent_data:
                event_entries.append(group_info.groups.pop(name))
        # TODO Do not raise here... just change init!
        if len(event_entries) > 1:
            raise NexusStructureError("No unique NXevent_data entry in NXdetector. "
                                      f"Found {len(event_entries)}.")
        if len(event_entries) == 1:
            # If there is also a signal dataset (not events) it will be ignored
            # (except for possibly using it to deduce shape and dims).
            event_data = event_entries[0].build()
        # TODO
        #if 'event_time_offset' in self:
        #    # Consumes datasets from self
        #    event_data = NXevent_data._make_class_info(info)
        # parse Nexus:
        # 1. find event data
        # create NXevent_data, consuming dataset infos?
        # 2. find grouping
        #events = EventFieldInfo(event_data=info.groups.pop('events'))
        fallback_dims = None
        for key in self._detector_number_fields:
            if (grouping := group_info.datasets.get(key)) is not None:
                if len(grouping.shape) == 1:
                    fallback_dims = (key, )
                break
        di = NXdataInfo.from_group_info(info=group_info,
                                        strategy=self._strategy,
                                        fallback_dims=fallback_dims)
        print(di)
        fields = dict(di.field_infos)
        fields.update(group_info.groups)
        info = NXobjectInfo(children=fields)
        info.signal_name = di.signal_name

        #info = super()._make_class_info(info=group_info)

        if event_data is None:
            event_field = None
        else:
            event_grouping = {}
            for key in self._detector_number_fields:
                if key in group_info.datasets:
                    #grouping = info.children[key].build()
                    #print(f'{info.children[key]=}')
                    event_grouping = {
                        'grouping_key': key,
                        'grouping': info.children[key].build()
                    }
                    break

            event_field = EventFieldInfo(event_data=event_data, **event_grouping)
        #info = NXdataInfo.from_group_info(info=info, strategy=self._strategy, signal_override=event_field)
        #info.children['events'] = events
        # TODO need to set signal field info (not just name), and same in NXdata
        # NXdata._signal should point to either FieldInfo or EventFieldInfo
        #print(f'{info=}')
        #if event_data is not None:
        #    print(f'{event_data._info.children=}')
        #print(f'{event_field=}')
        if event_field is not None:
            info.children['events'] = event_field
            info.signal_name = 'events'
        #print(f'{info=}')
        return info

    #def _init_info(self, info):
    #    info = NXdataInfo.from_group_info(info=info,
    #                                      strategy=NXdetectorStrategy)
    #    field_dims = info.field_dims
    #    if self.events is not None:
    #        for name in field_dims:
    #            if name in self._nxevent_data_fields:
    #                # Event field is direct child of this class
    #                field_dims[name] = self.events._get_field_dims(name)
    #            if name in self._detector_number_fields:
    #                # If there is a signal field in addition to the event data it can be
    #                # used to define dimension labels
    #                nxdata = self._nxdata(use_event_signal=False)
    #                if nxdata._signal_name is not None:
    #                    field_dims[name] = nxdata._get_field_dims(name)
    #                # If grouping is 1-D then we use this name as the dim
    #                elif self._get_child(name).ndim == 1:
    #                    field_dims[name] = [name]
    #                else:
    #                    field_dims[name] = None
    #    return info

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
        if self.events is None:
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
        signal = children.pop(self._info.signal_name)
        signal = signal if isinstance(signal, sc.Variable) else signal.data
        print(signal)
        return sc.DataArray(data=signal, coords=children)
