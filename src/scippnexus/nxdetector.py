# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
from copy import copy
from typing import List, Optional, Union
import scipp as sc
from .nxobject import NX_class, NXobject, Field, ScippIndex, NexusStructureError
from .nxdata import NXdata
from .nxevent_data import NXevent_data


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


class _EventField:
    """Field-like wrapper of NXevent_data binned into pixels.

    This has no equivalent in the NeXus format, but represents the conceptual
    event-data "signal" dataset of an NXdetector.
    """
    def __init__(self,
                 nxevent_data: NXevent_data,
                 event_select: ScippIndex,
                 detector_number: Optional[Field] = None):
        self._nxevent_data = nxevent_data
        self._event_select = event_select
        self._detector_number = detector_number

    @property
    def attrs(self):
        return self._nxevent_data.attrs

    @property
    def dims(self):
        if self._detector_number is None:
            return ['detector_number']
        return self._detector_number.dims

    @property
    def shape(self):
        if self._detector_number is None:
            raise NexusStructureError(
                "Cannot get shape of NXdetector since no 'detector_number' "
                "field found but detector contains event data.")
        return self._detector_number.shape

    @property
    def unit(self):
        self._nxevent_data.unit

    def __getitem__(self, select: ScippIndex) -> sc.DataArray:
        event_data = self._nxevent_data[self._event_select]
        if self._detector_number is None:
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
            detector_number = sc.arange(dim='detector_number',
                                        unit=None,
                                        start=id_min.value,
                                        stop=id_max.value + 1,
                                        dtype=id_min.dtype)
        else:
            detector_number = self._detector_number[select]
        event_id = detector_number.flatten(to='event_id')
        event_data.bins.coords['event_time_zero'] = sc.bins_like(
            event_data, fill_value=event_data.coords['event_time_zero'])
        # After loading raw NXevent_data it is guaranteed that the event table
        # is contiguous and that there is no masking. We can therefore use the
        # more efficient approach of binning from scratch instead of erasing the
        # 'pulse' binning defined by NXevent_data.
        event_data = sc.bin(event_data.bins.constituents['data'], groups=[event_id])
        if self._detector_number is None:
            event_data.coords['detector_number'] = event_data.coords.pop('event_id')
        else:
            del event_data.coords['event_id']
        return event_data.fold(dim='event_id', sizes=detector_number.sizes)


class NXdetector(NXobject):
    """A detector or detector bank providing an array of values or events.

    If the detector stores event data then the 'detector_number' field (if present)
    is used to map event do detector pixels. Otherwise this returns event data in the
    same format as NXevent_data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_select = tuple()
        self._nxevent_data_fields = [
            'event_time_zero', 'event_index', 'event_time_offset', 'event_id'
        ]
        self._detector_number_fields = ['detector_number', 'pixel_id']

    @property
    def shape(self) -> List[int]:
        return self._signal.shape

    @property
    def dims(self) -> List[str]:
        return self._signal.dims

    @property
    def unit(self) -> Union[sc.Unit, None]:
        return self._signal.unit

    @property
    def _detector_number(self) -> Union[None, Field]:
        for key in self._detector_number_fields:
            if key in self:
                return self[key]
        return None

    @property
    def _signal(self) -> Union[Field, _EventField]:
        return self._nxdata()._signal

    def _nxdata(self, use_event_signal=True) -> NXdata:
        if use_event_signal and self.events is not None:
            signal = _EventField(self.events, self._event_select, self._detector_number)
        else:
            signal = None
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXdetector uses a "hard-coded" signal name 'data', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        return NXdata(self._group,
                      signal_name_default='data' if 'data' in self else None,
                      signal_override=signal,
                      skip=self._nxevent_data_fields)

    @property
    def events(self) -> Union[None, NXevent_data]:
        """Return the underlying NXevent_data group, None if not event data."""
        # The standard is unclear on whether the 'data' field may be NXevent_data or
        # whether the fields of NXevent_data should be stored directly within this
        # NXdetector. Both cases are observed in the wild.
        event_entries = self.by_nx_class()[NX_class.NXevent_data]
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
                return None
        return self._nxdata()._get_field_dims(name)

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        return self._nxdata()._getitem(select)
