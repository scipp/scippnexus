# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import h5py
import numpy as np
import pytest
import scipp as sc

import scippnexus as snx


@pytest.fixture()
def h5root():
    """Yield h5py root group (file)"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        yield f


@pytest.fixture()
def nxroot():
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = snx.Group(f, definitions=snx.base_definitions())
        root.create_class('entry', snx.NXentry)
        yield root


def create_event_data_ids_1234(group):
    group['event_id'] = sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2, 2])
    group['event_time_offset'] = sc.array(
        dims=[''], unit='s', values=[456, 7, 3, 345, 632, 23]
    )
    group['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    group['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, -1000])


def test_negative_event_index_converted_to_num_event(nxroot):
    event_data = nxroot['entry'].create_class('events_0', snx.NXevent_data)
    create_event_data_ids_1234(event_data)
    events = nxroot['entry/events_0'][...]
    assert events.bins.size().values[2] == 3
    assert events.bins.size().values[3] == 0


@pytest.mark.filterwarnings("ignore:Failed to load /entry/events_0:UserWarning")
def test_bad_event_index_causes_load_as_DataGroup(nxroot):
    event_data = nxroot['entry'].create_class('events_0', snx.NXevent_data)
    event_data['event_id'] = sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2])
    event_data['event_time_offset'] = sc.array(dims=[''], unit='s', values=[0, 0, 0, 0])
    event_data['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    event_data['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, 666])
    dg = nxroot['entry/events_0'][...]
    assert isinstance(dg, sc.DataGroup)


def create_event_data_without_event_id(group):
    group['event_time_offset'] = sc.array(
        dims=[''], unit='s', values=[456, 7, 3, 345, 632, 23]
    )
    group['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    group['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, 5])


def test_event_data_without_event_id_can_be_loaded(nxroot):
    event_data = nxroot['entry'].create_class('events_0', snx.NXevent_data)
    create_event_data_without_event_id(event_data)
    da = event_data[...]
    assert len(da.bins.coords) == 1
    assert 'event_time_offset' in da.bins.coords


def create_event_data_without_event_time_zero(group):
    group['event_id'] = sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2, 2])
    group['event_time_offset'] = sc.array(
        dims=[''], unit='s', values=[456, 7, 3, 345, 632, 23]
    )
    group['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, 5])


def test_event_data_without_event_time_zero_can_be_loaded(nxroot):
    event_data = nxroot['entry'].create_class('events_0', snx.NXevent_data)
    create_event_data_without_event_time_zero(event_data)
    da = event_data[...]
    assert len(da.bins.coords) == 2
    assert 'event_time_offset' in da.bins.coords


def test_event_mode_monitor_without_event_id_can_be_loaded(nxroot):
    monitor = nxroot['entry'].create_class('monitor', snx.NXmonitor)
    create_event_data_without_event_id(monitor)
    da = monitor[...]['events']
    assert 'event_time_offset' in da.bins.coords


def test_read_empty_nxevent_data(h5root):
    entry = h5root.create_group('entry')
    events = entry.create_group('events')
    events.attrs['NX_class'] = 'NXevent_data'
    root = snx.Group(entry)
    event_data = root['events']
    dg = event_data[()]
    assert sc.identical(dg, sc.DataGroup())


def make_event_data(h5root):
    entry = h5root.create_group('entry')
    events = entry.create_group('events')
    events.attrs['NX_class'] = 'NXevent_data'
    rng = np.random.default_rng(0)
    events['event_id'] = rng.integers(0, 2, size=4)
    events['event_time_offset'] = np.arange(4)
    events['event_time_offset'].attrs['units'] = 'ns'
    events['event_time_zero'] = np.array([100, 200])
    events['event_time_zero'].attrs['units'] = 'ms'
    events['event_index'] = np.array([0, 3])
    return entry


def test_nxevent_data_keys(h5root):
    entry = make_event_data(h5root)
    root = snx.Group(entry)
    event_data = root['events']
    assert set(event_data.keys()) == {
        'event_id',
        'event_time_offset',
        'event_time_zero',
        'event_index',
    }


def test_nxevent_data_children_read_as_variables_with_correct_dims(h5root):
    entry = make_event_data(h5root)
    root = snx.Group(entry, definitions=snx.base_definitions())
    event_data = root['events']
    assert sc.identical(
        event_data['event_id'][()],
        sc.array(dims=['event'], values=[1, 1, 1, 0], unit=None),
    )
    assert sc.identical(
        event_data['event_time_offset'][()],
        sc.array(dims=['event'], values=[0, 1, 2, 3], unit='ns'),
    )
    assert sc.identical(
        event_data['event_time_zero'][()],
        sc.array(dims=['event_time_zero'], values=[100, 200], unit='ms'),
    )
    assert sc.identical(
        event_data['event_index'][()],
        sc.array(dims=['event_time_zero'], values=[0, 3], unit=None),
    )


def test_nxevent_data_dims_and_sizes_ignore_pulse_contents(h5root):
    entry = make_event_data(h5root)
    root = snx.Group(entry, definitions=snx.base_definitions())
    event_data = root['events']
    assert event_data.dims == ('event_time_zero',)
    assert event_data.sizes == {'event_time_zero': 2}


def test_read_nxevent_data(h5root):
    entry = make_event_data(h5root)
    root = snx.Group(entry, definitions=snx.base_definitions())
    event_data = root['events']
    da = event_data[()]
    assert sc.identical(
        da.data.bins.size(),
        sc.array(dims=['event_time_zero'], values=[3, 1], unit=None),
    )
