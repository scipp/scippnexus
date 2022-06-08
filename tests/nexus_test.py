# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import h5py
import numpy as np
import pytest
import scipp as sc
from scippnexus import Field, NXroot, NX_class

# representative sample of UTF-8 test strings from
# https://www.w3.org/2001/06/utf-8-test/UTF-8-demo.html
UTF8_TEST_STRINGS = (
    "∮ E⋅da = Q,  n → ∞, ∑ f(i) = ∏ g(i), ∀x∈ℝ: ⌈x⌉ = −⌊−x⌋, α ∧ ¬β = ¬(¬α ∨ β)",
    "2H₂ + O₂ ⇌ 2H₂O, R = 4.7 kΩ, ⌀ 200 mm",
    "Σὲ γνωρίζω ἀπὸ τὴν κόψη",
)


@pytest.fixture()
def nxroot(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = NXroot(f)
        root.create_class('entry', NX_class.NXentry)
        yield root


def test_nxobject_root(nxroot):
    assert nxroot.nx_class == NX_class.NXroot
    assert set(nxroot.keys()) == {'entry'}


def test_nxobject_create_class_creates_keys(nxroot):
    nxroot.create_class('log', NX_class.NXlog)
    assert set(nxroot.keys()) == {'entry', 'log'}


def test_nxobject_items(nxroot):
    items = nxroot.items()
    assert len(items) == 1
    name, entry = items[0]
    assert name == 'entry'
    entry.create_class('monitor', NX_class.NXmonitor)
    entry.create_class('log', NX_class.NXlog)
    assert {k: v.nx_class
            for k, v in entry.items()} == {
                'log': NX_class.NXlog,
                'monitor': NX_class.NXmonitor
            }


def test_nxobject_entry(nxroot):
    entry = nxroot['entry']
    assert entry.nx_class == NX_class.NXentry
    entry.create_class('events_0', NX_class.NXevent_data)
    entry.create_class('events_1', NX_class.NXevent_data)
    entry.create_class('log', NX_class.NXlog)
    assert set(entry.keys()) == {'events_0', 'events_1', 'log'}


def test_nxobject_log(nxroot):
    da = sc.DataArray(sc.array(dims=['time'], values=[1.1, 2.2, 3.3]),
                      coords={
                          'time':
                          sc.epoch(unit='ns') +
                          sc.array(dims=['time'], unit='s', values=[4.4, 5.5, 6.6]).to(
                              unit='ns', dtype='int64')
                      })
    log = nxroot['entry'].create_class('log', NX_class.NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')
    assert log.nx_class == NX_class.NXlog
    assert sc.identical(log[...], da)


def test_nxlog_length_1(nxroot):
    da = sc.DataArray(
        sc.array(dims=['time'], values=[1.1]),
        coords={
            'time':
            sc.epoch(unit='ns') +
            sc.array(dims=['time'], unit='s', values=[4.4]).to(unit='ns', dtype='int64')
        })
    log = nxroot['entry'].create_class('log', NX_class.NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')
    assert log.nx_class == NX_class.NXlog
    assert sc.identical(log[...], da)


def test_nxlog_length_1_two_dims_no_time_squeezes_all_dims(nxroot):
    da = sc.DataArray(
        sc.array(dims=['time', 'ignored'], values=[[1.1]]),
        coords={
            'time':
            sc.epoch(unit='ns') +
            sc.array(dims=['time'], unit='s', values=[4.4]).to(unit='ns', dtype='int64')
        })
    log = nxroot['entry'].create_class('log', NX_class.NXlog)
    log['value'] = da.data
    assert sc.identical(log[...], sc.DataArray(sc.scalar(1.1)))


def test_nxlog_length_1_two_dims_with_time_squeezes_inner_dim(nxroot):
    da = sc.DataArray(
        sc.array(dims=['time', 'ignored'], values=[[1.1]]),
        coords={
            'time':
            sc.epoch(unit='ns') +
            sc.array(dims=['time'], unit='s', values=[4.4]).to(unit='ns', dtype='int64')
        })
    log = nxroot['entry'].create_class('log', NX_class.NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')
    assert sc.identical(log[...], da['ignored', 0])


def test_nxlog_axes_replaces_time_dim(nxroot):
    da = sc.DataArray(
        sc.array(dims=['time', 'ignored'], values=[[1.1]]),
        coords={
            'time':
            sc.epoch(unit='ns') +
            sc.array(dims=['time'], unit='s', values=[4.4]).to(unit='ns', dtype='int64')
        })
    log = nxroot['entry'].create_class('log', NX_class.NXlog)
    log.attrs['axes'] = ['yy', 'xx']
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')
    expected = sc.DataArray(sc.array(dims=['yy', 'xx'], values=[[1.1]]),
                            coords={'time': da.coords['time'].squeeze()})
    assert sc.identical(log[...], expected)


def test_nxlog_three_dims_with_time_of_length_1(nxroot):
    da = sc.DataArray(
        sc.array(dims=['time', 'a', 'b'], values=np.arange(9.).reshape(1, 3, 3)),
        coords={
            'time':
            sc.epoch(unit='ns') +
            sc.array(dims=['time'], unit='s', values=[4.4]).to(unit='ns', dtype='int64')
        })
    log = nxroot['entry'].create_class('log', NX_class.NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')
    loaded = log[...]
    assert sc.identical(
        loaded.data,
        sc.array(dims=['time', 'dim_1', 'dim_2'], values=np.arange(9.).reshape(1, 3,
                                                                               3)))


def test_nxlog_with_shape_0(nxroot):
    da = sc.DataArray(sc.ones(dims=['time', 'ignored'], shape=(0, 1)),
                      coords={'time': sc.ones(dims=['time'], shape=(0, ), unit='s')})
    log = nxroot['entry'].create_class('log', NX_class.NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time']
    assert sc.identical(log[...], da['ignored', 0])


def test_nxobject_event_data(nxroot):
    event_data = nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    assert event_data.nx_class == NX_class.NXevent_data


def test_nxobject_getting_item_that_does_not_exists_raises_KeyError(nxroot):
    with pytest.raises(KeyError):
        nxroot['abcde']


def test_nxobject_name_property_is_full_path(nxroot):
    nxroot.create_class('monitor', NX_class.NXmonitor)
    nxroot['entry'].create_class('log', NX_class.NXlog)
    nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    assert nxroot.name == '/'
    assert nxroot['monitor'].name == '/monitor'
    assert nxroot['entry'].name == '/entry'
    assert nxroot['entry']['log'].name == '/entry/log'
    assert nxroot['entry']['events_0'].name == '/entry/events_0'


def test_nxobject_grandchild_can_be_accessed_using_path(nxroot):
    nxroot['entry'].create_class('log', NX_class.NXlog)
    assert nxroot['entry/log'].name == '/entry/log'
    assert nxroot['/entry/log'].name == '/entry/log'


def test_nxobject_by_nx_class_of_root_contains_everything(nxroot):
    nxroot.create_class('monitor', NX_class.NXmonitor)
    nxroot['entry'].create_class('log', NX_class.NXlog)
    nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    nxroot['entry'].create_class('events_1', NX_class.NXevent_data)
    classes = nxroot.by_nx_class()
    assert list(classes[NX_class.NXentry]) == ['entry']
    assert list(classes[NX_class.NXmonitor]) == ['monitor']
    assert list(classes[NX_class.NXlog]) == ['log']
    assert set(classes[NX_class.NXevent_data]) == {'events_0', 'events_1'}


def test_nxobject_by_nx_class_contains_only_children(nxroot):
    nxroot.create_class('monitor', NX_class.NXmonitor)
    nxroot['entry'].create_class('log', NX_class.NXlog)
    nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    nxroot['entry'].create_class('events_1', NX_class.NXevent_data)
    classes = nxroot['entry'].by_nx_class()
    assert list(classes[NX_class.NXentry]) == []
    assert list(classes[NX_class.NXmonitor]) == []
    assert list(classes[NX_class.NXlog]) == ['log']
    assert set(classes[NX_class.NXevent_data]) == set(['events_0', 'events_1'])


def test_nxobject_dataset_items_are_returned_as_Field(nxroot):
    events = nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    events['event_time_offset'] = sc.arange('event', 5)
    field = nxroot['entry/events_0/event_time_offset']
    assert isinstance(field, Field)


def test_field_properties(nxroot):
    events = nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    events['event_time_offset'] = sc.arange('event', 6, dtype='int64', unit='ns')
    field = nxroot['entry/events_0/event_time_offset']
    assert field.dtype == 'int64'
    assert field.name == '/entry/events_0/event_time_offset'
    assert field.shape == [6]
    assert field.unit == sc.Unit('ns')


def test_field_dim_labels(nxroot):
    events = nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    events['event_time_offset'] = sc.arange('ignored', 2)
    events['event_time_zero'] = sc.arange('ignored', 2)
    events['event_index'] = sc.arange('ignored', 2)
    events['event_id'] = sc.arange('ignored', 2)
    event_data = nxroot['entry/events_0']
    assert event_data['event_time_offset'].dims == ['event']
    assert event_data['event_time_zero'].dims == ['pulse']
    assert event_data['event_index'].dims == ['pulse']
    assert event_data['event_id'].dims == ['event']
    log = nxroot['entry'].create_class('log', NX_class.NXlog)
    log['value'] = sc.arange('ignored', 2)
    log['time'] = sc.arange('ignored', 2)
    assert log['time'].dims == ['time']
    assert log['value'].dims == ['time']


def test_field_unit_is_none_if_no_units_attribute(nxroot):
    log = nxroot.create_class('log', NX_class.NXlog)
    log['value'] = sc.arange('ignored', 2, unit=None)
    log['time'] = sc.arange('ignored', 2)
    assert log.unit is None
    field = log['value']
    assert field.unit is None


def test_field_getitem_returns_variable_with_correct_size_and_values(nxroot):
    nxroot['field'] = sc.arange('ignored', 6, dtype='int64', unit='ns')
    field = nxroot['field']
    assert sc.identical(
        field[...],
        sc.array(dims=['dim_0'], unit='ns', values=[0, 1, 2, 3, 4, 5], dtype='int64'))
    assert sc.identical(
        field[1:],
        sc.array(dims=['dim_0'], unit='ns', values=[1, 2, 3, 4, 5], dtype='int64'))
    assert sc.identical(
        field[:-1],
        sc.array(dims=['dim_0'], unit='ns', values=[0, 1, 2, 3, 4], dtype='int64'))


@pytest.mark.parametrize("string", UTF8_TEST_STRINGS)
def test_field_of_utf8_encoded_dataset_is_loaded_correctly(nxroot, string):
    nxroot['entry']['title'] = sc.array(dims=['ignored'],
                                        values=[string, string + string])
    title = nxroot['entry/title']
    assert sc.identical(title[...],
                        sc.array(dims=['dim_0'], values=[string, string + string]))


def test_field_of_extended_ascii_in_ascii_encoded_dataset_is_loaded_correctly():
    # When writing, if we use bytes h5py will write as ascii encoding
    # 0xb0 = degrees symbol in latin-1 encoding.
    string = b"run at rot=90" + bytes([0xb0])
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        f['title'] = np.array([string, string + b'x'])
        title = NXroot(f)['title']
        assert sc.identical(
            title[...],
            sc.array(dims=['dim_0'], values=["run at rot=90°", "run at rot=90°x"]))


def test_ms_field_with_second_datetime_attribute_loaded_as_ms_datetime(nxroot):
    nxroot['mytime'] = sc.arange('ignored', 2, unit='ms')
    nxroot['mytime'].attrs['start_time'] = '2022-12-12T12:13:14'
    assert sc.identical(
        nxroot['mytime'][...],
        sc.datetimes(dims=['dim_0'],
                     unit='ms',
                     values=['2022-12-12T12:13:14.000', '2022-12-12T12:13:14.001']))


def test_ns_field_with_second_datetime_attribute_loaded_as_ns_datetime(nxroot):
    nxroot['mytime'] = sc.arange('ignored', 2, unit='ns')
    nxroot['mytime'].attrs['start_time'] = '1970-01-01T00:00:00'
    assert sc.identical(
        nxroot['mytime'][...],
        sc.datetimes(
            dims=['dim_0'],
            unit='ns',
            values=['1970-01-01T00:00:00.000000000', '1970-01-01T00:00:00.000000001']))


def test_second_field_with_ns_datetime_attribute_loaded_as_ns_datetime(nxroot):
    nxroot['mytime'] = sc.arange('ignored', 2, unit='s')
    nxroot['mytime'].attrs['start_time'] = '1984-01-01T00:00:00.000000000'
    assert sc.identical(
        nxroot['mytime'][...],
        sc.datetimes(dims=['dim_0'],
                     unit='ns',
                     values=['1984-01-01T00:00:00', '1984-01-01T00:00:01']))


@pytest.mark.parametrize('timezone,hhmm', [('Z', '12:00'), ('+04', '08:00'),
                                           ('+00', '12:00'), ('-02', '14:00'),
                                           ('+1130', '00:30'), ('-0930', '21:30'),
                                           ('+11:30', '00:30'), ('-09:30', '21:30')])
def test_timezone_information_in_datetime_attribute_is_applied(nxroot, timezone, hhmm):
    nxroot['mytime'] = sc.scalar(value=3, unit='s')
    nxroot['mytime'].attrs['start_time'] = f'1984-01-01T12:00:00{timezone}'
    assert sc.identical(nxroot['mytime'][...],
                        sc.datetime(unit='s', value=f'1984-01-01T{hhmm}:03'))


def test_timezone_information_in_datetime_attribute_preserves_ns_precision(nxroot):
    nxroot['mytime'] = sc.scalar(value=3, unit='s')
    nxroot['mytime'].attrs['start_time'] = '1984-01-01T12:00:00.123456789+0200'
    assert sc.identical(nxroot['mytime'][...],
                        sc.datetime(unit='ns', value='1984-01-01T10:00:03.123456789'))


def test_loads_bare_timestamps_if_multiple_candidate_datetime_offsets_found(nxroot):
    offsets = sc.arange('ignored', 2, unit='ms')
    nxroot['mytime'] = offsets
    nxroot['mytime'].attrs['offset'] = '2022-12-12T12:13:14'
    nxroot['mytime'].attrs['start_time'] = '2022-12-12T12:13:15'
    assert sc.identical(nxroot['mytime'][...], offsets.rename(ignored='dim_0'))


def create_event_data_ids_1234(group):
    group['event_id'] = sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2, 2])
    group['event_time_offset'] = sc.array(dims=[''],
                                          unit='s',
                                          values=[456, 7, 3, 345, 632, 23])
    group['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    group['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, -1000])


def test_negative_event_index_converted_to_num_event(nxroot):
    event_data = nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    create_event_data_ids_1234(event_data)
    events = nxroot['entry/events_0'][...]
    assert events.bins.size().values[2] == 3
    assert events.bins.size().values[3] == 0


def test_bad_event_index_raises_IndexError(nxroot):
    event_data = nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    event_data['event_id'] = sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2])
    event_data['event_time_offset'] = sc.array(dims=[''], unit='s', values=[0, 0, 0, 0])
    event_data['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    event_data['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, 666])
    with pytest.raises(IndexError):
        nxroot['entry/events_0'][...]


def create_event_data_without_event_id(group):
    group['event_time_offset'] = sc.array(dims=[''],
                                          unit='s',
                                          values=[456, 7, 3, 345, 632, 23])
    group['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    group['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, 5])


def test_event_data_without_event_id_can_be_loaded(nxroot):
    event_data = nxroot['entry'].create_class('events_0', NX_class.NXevent_data)
    create_event_data_without_event_id(event_data)
    da = event_data[...]
    assert len(da.bins.coords) == 1
    assert 'event_time_offset' in da.bins.coords


def test_event_mode_monitor_without_event_id_can_be_loaded(nxroot):
    monitor = nxroot['entry'].create_class('monitor', NX_class.NXmonitor)
    create_event_data_without_event_id(monitor)
    da = monitor[...]
    assert len(da.bins.coords) == 1
    assert 'event_time_offset' in da.bins.coords
