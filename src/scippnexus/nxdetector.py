# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

from typing import Optional

import scipp as sc

from .nxdata import NXdata, NXdataInfo, NXdataStrategy
from .nxevent_data import NXevent_data
from .nxobject import (
    Field,
    GroupContentInfo,
    NexusStructureError,
    NXobjectInfo,
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


def _group_events(*,
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
    # if self._grouping is None:
    #     event_data.coords[self._grouping_key] = event_data.coords.pop('event_id')
    # else:
    #     del event_data.coords['event_id']
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

    def _make_class_info(self, group_info: GroupContentInfo) -> NXobjectInfo:
        fallback_dims = None
        for key in self._detector_number_fields:
            if (grouping := group_info.datasets.get(key)) is not None:
                if len(grouping.shape) == 1:
                    fallback_dims = (key, )
                break
        di = NXdataInfo.from_group_info(info=group_info,
                                        strategy=self._strategy,
                                        fallback_dims=fallback_dims)
        fields = dict(di.field_infos)
        fields.update(group_info.groups)
        info = NXobjectInfo(children=fields)
        info.signal_name = di.signal_name
        #if di.signal_name is None:
        #    info.signal_name = grouping_key
        #else:
        #    info.signal_name = di.signal_name

        return info

    @property
    def detector_number(self) -> Optional[Field]:
        for key in self._detector_number_fields:
            if key in self:
                return key
        return None

    def _assemble(self, children: sc.DataGroup):
        # TODO What about dims?
        if self._info.signal_name is None:
            event_entries = _find_event_entries(children)
            if len(event_entries) != 1:
                raise NexusStructureError("Multiple event entries found.")
            event_entry = children[event_entries[0]]
            if 'event_id' in event_entry.bins.coords:
                raise NexusStructureError("xxx")
            children = sc.DataGroup(children)
            da = children.pop(event_entries[0])
            return self._add_coords(da, children)

        return super()._assemble(children)


def _find_event_entries(dg: sc.DataGroup) -> List[str]:
    event_entries = []
    for name, value in dg.items():
        if isinstance(
                value, sc.DataArray
        ) and 'event_time_zero' in value.coords and value.bins is not None:
            event_entries.append(name)
    return event_entries


def group_events_by_detector_number(dg: sc.DataGroup) -> sc.DataArray:
    event_entry = _find_event_entries(dg)[0]
    events = dg.pop(event_entry)
    grouping_key = None
    for key in NXdetector._detector_number_fields:
        if (grouping := dg.get(key)) is not None:
            grouping_key = key
            break
    grouping = None if grouping_key is None else asarray(dg.pop(grouping_key))
    da = _group_events(event_data=events, grouping=grouping)
    # TODO What about _coord_to_attr mapping as NXdata?
    da.coords.update(dg)
    return da
    # TODO
    # The standard is unclear on whether the 'data' field may be NXevent_data or
    # whether the fields of NXevent_data should be stored directly within this
    # NXdetector. Both cases are observed in the wild.
    if len(event_entries) > 1:
        raise NexusStructureError("No unique NXevent_data entry in NXdetector. "
                                  f"Found {len(event_entries)}.")
    if len(event_entries) == 1:
        # If there is also a signal dataset (not events) it will be ignored
        # (except for possibly using it to deduce shape and dims).
        return next(iter(event_entries.values()))
    if 'event_time_offset' in self:
        return NXevent_data(self._group)
