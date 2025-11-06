# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import scipp as sc

from ._common import to_child_select
from .attrs import Attrs
from .base import Group, asvariable
from .field import Field
from .typing import ScippIndex


class EventField:
    def __init__(self, event_data: Group, grouping_name: str, grouping: Field) -> None:
        """Create a field that represents an event data group.

        Parameters
        ----------
        event_data:
            The event data group holding the NXevent_data fields.
        grouping_name:
            The name of the field that contains the grouping information.
        grouping:
            The field that contains the grouping keys. These are IDs corresponding to
            the event_id field of the NXevent_data group, such as the detector_number
            field of an NXdetector.
        """
        self._event_data = event_data
        self._grouping_name = grouping_name
        self._grouping = grouping

    @property
    def attrs(self) -> Attrs:
        return self._event_data.attrs

    @property
    def sizes(self) -> dict[str, int]:
        return {**self._grouping.sizes, **self._event_data.sizes}

    @property
    def dims(self) -> tuple[str, ...]:
        return self._grouping.dims + self._event_data.dims

    @property
    def shape(self) -> tuple[int, ...]:
        return self._grouping.shape + self._event_data.shape

    def __getitem__(self, sel: ScippIndex) -> sc.DataArray:
        event_sel = to_child_select(self.dims, self._event_data.dims, sel)
        events = self._event_data[event_sel]
        detector_sel = to_child_select(self.dims, self._grouping.dims, sel)
        if not isinstance(events, sc.DataArray):
            return events
        da = _group_events(event_data=events, grouping=self._grouping[detector_sel])
        da.coords[self._grouping_name] = da.coords.pop('event_id')
        return da


def _group_events(
    *, event_data: sc.DataArray, grouping: sc.Variable | None = None
) -> sc.DataArray:
    if grouping is None:
        event_id = 'event_id'
    else:
        # copy since sc.bin cannot deal with a non-contiguous view
        grouping = asvariable(grouping)
        event_id = grouping.flatten(to='event_id').copy()
    if 'event_time_zero' in event_data.coords:
        event_data.bins.coords['event_time_zero'] = sc.bins_like(
            event_data, fill_value=event_data.coords['event_time_zero']
        )
    # After loading raw NXevent_data it is guaranteed that the event table
    # is contiguous and that there is no masking. We can therefore use the
    # more efficient approach of binning from scratch instead of erasing the
    # 'event_time_zero' binning defined by NXevent_data.
    event_data = event_data.bins.constituents['data'].group(event_id)
    if grouping is None:
        return event_data
    return event_data.fold(dim='event_id', sizes=grouping.sizes)
