# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import h5py
import numpy as np
import pytest
import scipp as sc

from scippnexus import NXentry, NXevent_data, NXmonitor, NXroot


@pytest.fixture()
def nxroot(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = NXroot(f)
        root.create_class('entry', NXentry)
        yield root


def create_event_data_ids_1234(group):
    group['event_id'] = sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2, 2])
    group['event_time_offset'] = sc.array(dims=[''],
                                          unit='s',
                                          values=[456, 7, 3, 345, 632, 23])
    group['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    group['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, -1000])


def test_negative_event_index_converted_to_num_event(nxroot):
    event_data = nxroot['entry'].create_class('events_0', NXevent_data)
    create_event_data_ids_1234(event_data)
    events = nxroot['entry/events_0'][...]
    assert events.bins.size().values[2] == 3
    assert events.bins.size().values[3] == 0


def test_bad_event_index_causes_load_as_DataGroup(nxroot):
    event_data = nxroot['entry'].create_class('events_0', NXevent_data)
    event_data['event_id'] = sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2])
    event_data['event_time_offset'] = sc.array(dims=[''], unit='s', values=[0, 0, 0, 0])
    event_data['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    event_data['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, 666])
    dg = nxroot['entry/events_0'][...]
    assert isinstance(dg, sc.DataGroup)


def create_event_data_without_event_id(group):
    group['event_time_offset'] = sc.array(dims=[''],
                                          unit='s',
                                          values=[456, 7, 3, 345, 632, 23])
    group['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    group['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, 5])


def test_event_data_without_event_id_can_be_loaded(nxroot):
    event_data = nxroot['entry'].create_class('events_0', NXevent_data)
    create_event_data_without_event_id(event_data)
    da = event_data[...]
    assert len(da.bins.coords) == 1
    assert 'event_time_offset' in da.bins.coords


def test_event_mode_monitor_without_event_id_can_be_loaded(nxroot):
    monitor = nxroot['entry'].create_class('monitor', NXmonitor)
    create_event_data_without_event_id(monitor)
    da = nxroot['entry']['monitor'][...]
    assert len(da.bins.coords) == 1
    assert 'event_time_offset' in da.bins.coords


def test_field_properties(nxroot):
    events = nxroot['entry'].create_class('events_0', NXevent_data)
    events['event_time_offset'] = sc.arange('event', 6, dtype='int64', unit='ns')
    field = nxroot['entry/events_0/event_time_offset']
    assert field.dtype == 'int64'
    assert field.name == '/entry/events_0/event_time_offset'
    assert field.shape == (6, )
    assert field.unit == sc.Unit('ns')


def test_field_dim_labels(nxroot):
    events = nxroot['entry'].create_class('events_0', NXevent_data)
    events['event_time_offset'] = sc.arange('ignored', 2)
    events['event_time_zero'] = sc.arange('ignored', 2)
    events['event_index'] = sc.arange('ignored', 2)
    events['event_id'] = sc.arange('ignored', 2)
    event_data = nxroot['entry/events_0']
    assert event_data['event_time_offset'].dims == ('event', )
    assert event_data['event_time_zero'].dims == ('pulse', )
    assert event_data['event_index'].dims == ('pulse', )
    assert event_data['event_id'].dims == ('event', )
