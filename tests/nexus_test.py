# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import warnings

import h5py
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

import scippnexus as snx
from scippnexus import (
    NexusStructureError,
    NXdetector,
    NXentry,
    NXevent_data,
    NXlog,
    NXmonitor,
    NXroot,
)

# representative sample of UTF-8 test strings from
# https://www.w3.org/2001/06/utf-8-test/UTF-8-demo.html
UTF8_TEST_STRINGS = (
    "∮ E⋅da = Q,  n → ∞, ∑ f(i) = ∏ g(i), ∀x∈ℝ: ⌈x⌉ = −⌊−x⌋, α ∧ ¬β = ¬(¬α ∨ β)",  # noqa: RUF001
    "2H₂ + O₂ ⇌ 2H₂O, R = 4.7 kΩ, ⌀ 200 mm",
    "Σὲ γνωρίζω ἀπὸ τὴν κόψη",
)


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
        root.create_class('entry', NXentry)
        yield root


def test_nxobject_root(nxroot):
    assert nxroot.nx_class == NXroot
    assert set(nxroot.keys()) == {'entry'}


def test_nxobject_create_class_creates_keys(nxroot):
    nxroot.create_class('log', NXlog)
    assert set(nxroot.keys()) == {'entry', 'log'}


def test_nxobject_create_class_with_string_nx_class(nxroot):
    nxroot.create_class('log', 'NXlog')
    assert set(nxroot.keys()) == {'entry', 'log'}


def test_nxobject_items(nxroot):
    items = nxroot.items()
    assert len(items) == 1
    name, entry = next(iter(items))
    assert name == 'entry'
    entry.create_class('monitor', NXmonitor)
    entry.create_class('log', NXlog)
    assert {k: v.nx_class for k, v in entry.items()} == {
        'log': NXlog,
        'monitor': NXmonitor,
    }


def test_nxobject_iter(nxroot):
    nxroot.create_class('entry2', NXentry)
    # With missing __iter__ this used to raise since Python just calls __getitem__
    # with an int range.
    list(nxroot)
    for _ in nxroot:
        pass


def test_nxobject_entry(nxroot):
    entry = nxroot['entry']
    assert entry.nx_class == NXentry
    entry.create_class('events_0', NXevent_data)
    entry.create_class('events_1', NXevent_data)
    entry.create_class('log', NXlog)
    assert set(entry.keys()) == {'events_0', 'events_1', 'log'}


def test_nx_class_can_be_bytes(h5root):
    log = h5root.create_group('log')
    attr = np.array(b'NXlog', dtype='|S5')
    log.attrs['NX_class'] = attr
    group = snx.Group(log, definitions=snx.base_definitions())
    assert group.nx_class == NXlog


def test_nxobject_event_data(nxroot):
    event_data = nxroot['entry'].create_class('events_0', NXevent_data)
    assert event_data.nx_class == NXevent_data


def test_nxobject_getting_item_that_does_not_exists_raises_KeyError(nxroot):
    with pytest.raises(KeyError):
        nxroot['abcde']


def test_nxobject_name_property_is_full_path(nxroot):
    nxroot.create_class('monitor', NXmonitor)
    nxroot['entry'].create_class('log', NXlog)
    nxroot['entry'].create_class('events_0', NXevent_data)
    assert nxroot.name == '/'
    assert nxroot['monitor'].name == '/monitor'
    assert nxroot['entry'].name == '/entry'
    assert nxroot['entry']['log'].name == '/entry/log'
    assert nxroot['entry']['events_0'].name == '/entry/events_0'


def test_nxobject_grandchild_can_be_accessed_using_path(nxroot):
    nxroot['entry'].create_class('log', NXlog)
    assert nxroot['entry/log'].name == '/entry/log'
    assert nxroot['/entry/log'].name == '/entry/log'


def test_nxobject_getitem_by_class(nxroot):
    nxroot.create_class('monitor', NXmonitor)
    nxroot['entry'].create_class('log', NXlog)
    nxroot['entry'].create_class('events_0', NXevent_data)
    nxroot['entry'].create_class('events_1', NXevent_data)
    assert list(nxroot[NXentry]) == ['entry']
    assert list(nxroot[NXmonitor]) == ['monitor']
    assert list(nxroot['entry'][NXmonitor]) == []  # not nested
    assert list(nxroot[NXlog]) == []  # nested
    assert list(nxroot['entry'][NXlog]) == ['log']
    assert set(nxroot['entry'][NXevent_data]) == {'events_0', 'events_1'}


def test_nxobject_getitem_by_class_get_fields(nxroot):
    nxroot['entry'].create_class('log', NXlog)
    nxroot['entry'].create_class('events_0', NXevent_data)
    nxroot['entry']['field1'] = sc.arange('event', 4.0, unit='ns')
    nxroot['entry']['field2'] = sc.arange('event', 2.0, unit='ns')
    assert list(nxroot[snx.Field]) == []
    assert set(nxroot['entry'][snx.Field]) == {'field1', 'field2'}


def test_nxobject_getitem_by_class_list(nxroot):
    nxroot['entry'].create_class('log', NXlog)
    nxroot['entry'].create_class('events_0', NXevent_data)
    nxroot['entry'].create_class('events_1', NXevent_data)
    nxroot['entry']['field1'] = sc.arange('event', 4.0, unit='ns')
    assert set(nxroot['entry'][[NXlog, NXevent_data]]) == {
        'log',
        'events_0',
        'events_1',
    }
    assert set(nxroot['entry'][[NXlog, snx.Field]]) == {'log', 'field1'}


def test_nxobject_dataset_items_are_returned_as_Field(nxroot):
    events = nxroot['entry'].create_class('events_0', NXevent_data)
    events['event_time_offset'] = sc.arange('event', 5)
    field = nxroot['entry/events_0/event_time_offset']
    assert isinstance(field, snx.Field)


def test_field_properties(nxroot):
    events = nxroot['entry'].create_class('events_0', NXevent_data)
    events['event_time_offset'] = sc.arange('event', 6, dtype='int64', unit='ns')
    field = nxroot['entry/events_0/event_time_offset']
    assert field.dtype == 'int64'
    assert field.name == '/entry/events_0/event_time_offset'
    assert field.shape == (6,)
    assert field.unit == sc.Unit('ns')


def test_field_dim_labels(nxroot):
    events = nxroot['entry'].create_class('events_0', NXevent_data)
    events['event_time_offset'] = sc.arange('ignored', 2)
    events['event_time_zero'] = sc.arange('ignored', 2)
    events['event_index'] = sc.arange('ignored', 2)
    events['event_id'] = sc.arange('ignored', 2)
    event_data = nxroot['entry/events_0']
    assert event_data['event_time_offset'].dims == ('event',)
    assert event_data['event_time_zero'].dims == ('event_time_zero',)
    assert event_data['event_index'].dims == ('event_time_zero',)
    assert event_data['event_id'].dims == ('event',)
    log = nxroot['entry'].create_class('log', NXlog)
    log['value'] = sc.arange('ignored', 2)
    log['time'] = sc.arange('ignored', 2)
    assert log['time'].dims == ('time',)
    assert log['value'].dims == ('time',)


def test_field_unit_is_none_if_no_units_attribute(nxroot):
    log = nxroot.create_class('log', NXlog)
    log['value'] = sc.arange('ignored', 2, unit=None)
    log['time'] = sc.arange('ignored', 2)
    assert log.unit is None
    field = log['value']
    assert field.unit is None


def test_field_errors_with_same_unit_handles_them_with_value(nxroot):
    entry = nxroot.create_class('group', snx.NXentry)
    entry['value'] = sc.array(dims=['ignored'], values=[10.0], unit='m')
    entry['value_errors'] = sc.array(dims=['ignored'], values=[2.0], unit='m')
    value = nxroot['group']['value'][()]
    assert_identical(value, sc.scalar(value=10.0, variance=4.0, unit='m'))


def test_field_errors_with_different_unit_handles_them_individually(nxroot):
    entry = nxroot.create_class('group', snx.NXentry)
    entry['value'] = sc.array(dims=['ignored'], values=[10.0], unit='m')
    entry['value_errors'] = sc.array(dims=['ignored'], values=[200.0], unit='cm')
    value = nxroot['group']['value'][()]
    assert_identical(value, sc.scalar(value=10.0, unit='m'))
    assert 'value_errors' in nxroot['group']
    errors = nxroot['group']['value_errors'][()]
    assert_identical(errors, sc.scalar(value=200.0, unit='cm'))


@pytest.mark.parametrize(
    ('value', 'type_'), [(1.2, np.float32), (123, np.int32), ('abc', str), (True, bool)]
)
def test_field_is_returned_as_python_object_if_shape_empty_and_no_unit(
    nxroot, value, type_
):
    nxroot['field1'] = sc.scalar(value, unit=None, dtype=type_)
    field = nxroot['field1'][()]
    assert isinstance(field, type_)
    assert field == type_(value)


@pytest.mark.parametrize('value', [1.2, 123, True, 'abc'])
def test_field_is_returned_as_variable_if_shape_empty_and_unit(nxroot, value):
    nxroot['field1'] = sc.scalar(value, unit='K')
    field = nxroot['field1'][()]
    assert isinstance(field, sc.Variable)


def test_field_getitem_returns_variable_with_correct_size_and_values(nxroot):
    nxroot['field'] = sc.arange('ignored', 6, dtype='int64', unit='ns')
    field = nxroot['field']
    assert sc.identical(
        field[...],
        sc.array(dims=['dim_0'], unit='ns', values=[0, 1, 2, 3, 4, 5], dtype='int64'),
    )
    assert sc.identical(
        field[1:],
        sc.array(dims=['dim_0'], unit='ns', values=[1, 2, 3, 4, 5], dtype='int64'),
    )
    assert sc.identical(
        field[:-1],
        sc.array(dims=['dim_0'], unit='ns', values=[0, 1, 2, 3, 4], dtype='int64'),
    )


@pytest.mark.parametrize("string", UTF8_TEST_STRINGS)
def test_field_of_utf8_encoded_dataset_is_loaded_correctly(nxroot, string):
    nxroot['entry']['title'] = sc.array(
        dims=['ignored'], values=[string, string + string]
    )
    title = nxroot['entry/title']
    assert sc.identical(
        title[...], sc.array(dims=['dim_0'], values=[string, string + string])
    )


@pytest.mark.filterwarnings("ignore:Encoding for bytes")
def test_field_of_extended_ascii_in_ascii_encoded_dataset_is_loaded_correctly():
    # When writing, if we use bytes h5py will write as ascii encoding
    # 0xb0 = degrees symbol in latin-1 encoding.
    string = b"run at rot=90" + bytes([0xB0])
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        f['title'] = np.array([string, string + b'x'])
        title = snx.Group(f)['title']
        assert sc.identical(
            title[...],
            sc.array(dims=['dim_0'], values=["run at rot=90°", "run at rot=90°x"]),
        )


def test_ms_field_with_second_datetime_attribute_loaded_as_ms_datetime(nxroot):
    nxroot['mytime'] = sc.arange('ignored', 2, unit='ms')
    nxroot['mytime'].dataset.attrs['start_time'] = '2022-12-12T12:13:14'
    assert sc.identical(
        nxroot['mytime'][...],
        sc.datetimes(
            dims=['dim_0'],
            unit='ms',
            values=['2022-12-12T12:13:14.000', '2022-12-12T12:13:14.001'],
        ),
    )


def test_ns_field_with_second_datetime_attribute_loaded_as_ns_datetime(nxroot):
    nxroot['mytime'] = sc.arange('ignored', 2, unit='ns')
    nxroot['mytime'].dataset.attrs['start_time'] = '1970-01-01T00:00:00'
    assert sc.identical(
        nxroot['mytime'][...],
        sc.datetimes(
            dims=['dim_0'],
            unit='ns',
            values=['1970-01-01T00:00:00.000000000', '1970-01-01T00:00:00.000000001'],
        ),
    )


def test_second_field_with_ns_datetime_attribute_loaded_as_ns_datetime(nxroot):
    nxroot['mytime'] = sc.arange('ignored', 2, unit='s')
    nxroot['mytime'].dataset.attrs['start_time'] = '1984-01-01T00:00:00.000000000'
    assert sc.identical(
        nxroot['mytime'][...],
        sc.datetimes(
            dims=['dim_0'],
            unit='ns',
            values=['1984-01-01T00:00:00', '1984-01-01T00:00:01'],
        ),
    )


@pytest.mark.parametrize(
    ('timezone', 'hhmm'),
    [
        ('Z', '12:00'),
        ('+04', '08:00'),
        ('+00', '12:00'),
        ('-02', '14:00'),
        ('+1130', '00:30'),
        ('-0930', '21:30'),
        ('+11:30', '00:30'),
        ('-09:30', '21:30'),
    ],
)
def test_timezone_information_in_datetime_attribute_is_applied(nxroot, timezone, hhmm):
    nxroot['mytime'] = sc.scalar(value=3, unit='s')
    nxroot['mytime'].dataset.attrs['start_time'] = f'1984-01-01T12:00:00{timezone}'
    assert sc.identical(
        nxroot['mytime'][...], sc.datetime(unit='s', value=f'1984-01-01T{hhmm}:03')
    )


def test_timezone_information_in_datetime_attribute_preserves_ns_precision(nxroot):
    nxroot['mytime'] = sc.scalar(value=3, unit='s')
    nxroot['mytime'].dataset.attrs['start_time'] = '1984-01-01T12:00:00.123456789+0200'
    assert sc.identical(
        nxroot['mytime'][...],
        sc.datetime(unit='ns', value='1984-01-01T10:00:03.123456789'),
    )


def test_loads_bare_timestamps_if_multiple_candidate_datetime_offsets_found(nxroot):
    offsets = sc.arange('ignored', 2, unit='ms')
    nxroot['mytime'] = offsets
    nxroot['mytime'].dataset.attrs['offset'] = '2022-12-12T12:13:14'
    nxroot['mytime'].dataset.attrs['start_time'] = '2022-12-12T12:13:15'
    assert sc.identical(nxroot['mytime'][...], offsets.rename(ignored='dim_0'))


def test_length_0_field_with_datetime_attribute_loaded_as_datetime(nxroot):
    nxroot['mytime'] = sc.arange('ignored', 0, unit='ms')
    nxroot['mytime'].dataset.attrs['start_time'] = '2022-12-12T12:13:14'
    assert_identical(
        nxroot['mytime'][...], sc.datetimes(dims=['dim_0'], unit='ms', values=[])
    )


@pytest.mark.skip(reason='Special attributes disabled for now. Do we keep them?')
def test___getattr__for_unique_child_groups(nxroot):
    entry = nxroot['entry']
    with pytest.raises(NexusStructureError):
        entry.log
    entry.create_class('log1', NXlog)
    log = entry.log
    assert log.nx_class == NXlog
    assert log.name == '/entry/log1'
    assert isinstance(log, NXlog)
    entry.create_class('log2', NXlog)
    with pytest.raises(NexusStructureError):
        entry.log


@pytest.mark.skip(reason='Special attributes disabled for now. Do we keep them?')
def test___dir__(nxroot):
    entry = nxroot['entry']
    assert 'log' not in entry.__dir__()
    entry.create_class('log1', NXlog)
    assert 'log' in entry.__dir__()
    entry.create_class('log2', NXlog)
    assert 'log' not in entry.__dir__()


def test___dir__includes_non_dynamic_properties(nxroot):
    entry = nxroot['entry']
    det = entry.create_class('det', NXdetector)
    det.create_class('events', NXevent_data)
    # Ensure we are not replacing __dir__ but adding to it
    assert 'unit' in det.__dir__()


def test_read_recursive(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data['signal'] = np.arange(4)
    data['signal'].attrs['units'] = 'm'
    data['time'] = np.arange(5)
    data['time'].attrs['units'] = 's'
    obj = snx.Group(entry)
    dg = obj[()]
    assert obj.sizes == {'dim_0': None}
    assert 'data' in dg


def test_errors_read_as_variances(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data['signal'] = np.arange(4.0)
    data['signal'].attrs['units'] = 'm'
    data['signal_errors'] = np.arange(4.0)
    data['signal_errors'].attrs['units'] = 'm'
    data['time'] = np.arange(5.0)
    data['time'].attrs['units'] = 's'
    data['time_errors'] = np.arange(5.0)
    data['time_errors'].attrs['units'] = 's'
    obj = snx.Group(data)
    assert set(obj._children.keys()) == {'signal', 'time'}
    dg = obj[()]
    assert dg['signal'].variances is not None
    assert dg['time'].variances is not None
    assert np.array_equal(dg['signal'].variances, np.arange(4.0) ** 2)
    assert np.array_equal(dg['time'].variances, np.arange(5.0) ** 2)


def test_does_not_require_unit_of_errors(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data['signal'] = np.arange(4.0)
    data['signal'].attrs['units'] = 'm'
    data['signal_errors'] = np.arange(4.0)
    # no units on signal_errors
    data['time'] = np.arange(5.0)
    data['time'].attrs['units'] = 's'
    data['time_errors'] = np.arange(5.0)
    # no units on time_errors
    obj = snx.Group(data)
    assert set(obj._children.keys()) == {'signal', 'time'}
    dg = obj[()]
    assert dg['signal'].unit == 'm'
    assert dg['time'].unit == 's'


def test_read_field(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data['signal'] = np.arange(4)
    data['signal'].attrs['units'] = 'm'
    obj = snx.Group(data)
    var = obj['signal'][()]
    assert sc.identical(var, sc.array(dims=['dim_0'], values=np.arange(4), unit='m'))


def test_nxdata_with_signal_axes_indices_reads_as_data_array(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data.attrs['NX_class'] = 'NXdata'
    data.attrs['signal'] = 'signal'
    data.attrs['axes'] = ['time', 'temperature']
    data.attrs['time_indices'] = [0]
    data.attrs['temperature_indices'] = [1]
    ref = sc.DataArray(
        data=sc.ones(dims=['time', 'temperature'], shape=[3, 4], unit='m')
    )
    ref.coords['time'] = sc.array(dims=['time'], values=np.arange(3), unit='s')
    ref.coords['temperature'] = sc.array(
        dims=['temperature'], values=np.arange(4), unit='K'
    )
    data['signal'] = ref.values
    data['signal'].attrs['units'] = str(ref.unit)
    data['time'] = ref.coords['time'].values
    data['time'].attrs['units'] = str(ref.coords['time'].unit)
    data['temperature'] = ref.coords['temperature'].values
    data['temperature'].attrs['units'] = str(ref.coords['temperature'].unit)
    obj = snx.Group(data, definitions=snx.base_definitions())
    da = obj[()]
    assert sc.identical(da, ref)


def test_nxdata_positional_indexing_returns_correct_slice(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data.attrs['NX_class'] = 'NXdata'
    data.attrs['signal'] = 'signal'
    data.attrs['axes'] = ['time', 'temperature']
    data.attrs['time_indices'] = [0]
    data.attrs['temperature_indices'] = [1]
    ref = sc.DataArray(
        data=sc.ones(dims=['time', 'temperature'], shape=[3, 4], unit='m')
    )
    ref.coords['time'] = sc.array(dims=['time'], values=np.arange(3), unit='s')
    ref.coords['temperature'] = sc.array(
        dims=['temperature'], values=np.arange(4), unit='K'
    )
    data['signal'] = ref.values
    data['signal'].attrs['units'] = str(ref.unit)
    data['time'] = ref.coords['time'].values
    data['time'].attrs['units'] = str(ref.coords['time'].unit)
    data['temperature'] = ref.coords['temperature'].values
    data['temperature'].attrs['units'] = str(ref.coords['temperature'].unit)
    obj = snx.Group(data, definitions=snx.base_definitions())
    da = obj['time', 0:2]
    assert sc.identical(da, ref['time', 0:2])


def test_nxdata_label_indexing_returns_correct_slice(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data.attrs['NX_class'] = 'NXdata'
    data.attrs['signal'] = 'signal'
    data.attrs['axes'] = ['time', 'temperature']
    data.attrs['time_indices'] = [0]
    data.attrs['temperature_indices'] = [1]
    ref = sc.DataArray(
        data=sc.ones(dims=['time', 'temperature'], shape=[3, 4], unit='m')
    )
    ref.coords['time'] = sc.array(dims=['time'], values=np.arange(3), unit='s')
    ref.coords['temperature'] = sc.array(
        dims=['temperature'], values=np.arange(4), unit='K'
    )
    data['signal'] = ref.values
    data['signal'].attrs['units'] = str(ref.unit)
    data['time'] = ref.coords['time'].values
    data['time'].attrs['units'] = str(ref.coords['time'].unit)
    data['temperature'] = ref.coords['temperature'].values
    data['temperature'].attrs['units'] = str(ref.coords['temperature'].unit)
    obj = snx.Group(data, definitions=snx.base_definitions())
    da = obj['time', sc.scalar(0, unit='s') : sc.scalar(2, unit='s')]
    assert sc.identical(
        da, ref['time', sc.scalar(0, unit='s') : sc.scalar(2, unit='s')]
    )


def test_nxdata_with_bin_edges_positional_indexing_returns_correct_slice(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data.attrs['NX_class'] = 'NXdata'
    data.attrs['signal'] = 'signal'
    data.attrs['axes'] = ['time', 'temperature']
    data.attrs['time_indices'] = [0]
    data.attrs['temperature_indices'] = [1]
    ref = sc.DataArray(
        data=sc.ones(dims=['time', 'temperature'], shape=[3, 4], unit='m')
    )
    ref.coords['time'] = sc.array(dims=['time'], values=np.arange(3), unit='s')
    ref.coords['temperature'] = sc.array(
        dims=['temperature'], values=np.arange(5), unit='K'
    )
    data['signal'] = ref.values
    data['signal'].attrs['units'] = str(ref.unit)
    data['time'] = ref.coords['time'].values
    data['time'].attrs['units'] = str(ref.coords['time'].unit)
    data['temperature'] = ref.coords['temperature'].values
    data['temperature'].attrs['units'] = str(ref.coords['temperature'].unit)
    obj = snx.Group(data, definitions=snx.base_definitions())
    da = obj['temperature', 0:2]
    assert sc.identical(da, ref['temperature', 0:2])


def test_nxdata_with_bin_edges_label_indexing_returns_correct_slice(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data.attrs['NX_class'] = 'NXdata'
    data.attrs['signal'] = 'signal'
    data.attrs['axes'] = ['time', 'temperature']
    data.attrs['time_indices'] = [0]
    data.attrs['temperature_indices'] = [1]
    ref = sc.DataArray(
        data=sc.ones(dims=['time', 'temperature'], shape=[3, 4], unit='m')
    )
    ref.coords['time'] = sc.array(dims=['time'], values=np.arange(3), unit='s')
    ref.coords['temperature'] = sc.array(
        dims=['temperature'], values=np.arange(5), unit='K'
    )
    data['signal'] = ref.values
    data['signal'].attrs['units'] = str(ref.unit)
    data['time'] = ref.coords['time'].values
    data['time'].attrs['units'] = str(ref.coords['time'].unit)
    data['temperature'] = ref.coords['temperature'].values
    data['temperature'].attrs['units'] = str(ref.coords['temperature'].unit)
    obj = snx.Group(data, definitions=snx.base_definitions())
    da = obj['temperature', sc.scalar(0, unit='K') : sc.scalar(2, unit='K')]
    assert sc.identical(
        da, ref['temperature', sc.scalar(0, unit='K') : sc.scalar(2, unit='K')]
    )


def create_nexus_group_with_data_arrays(h5root, dims, coords):
    entry = h5root.create_group('entry')
    for i, (dim, coord) in enumerate(zip(dims, coords, strict=True)):
        data = entry.create_group(f'data_{i}')
        data.attrs['NX_class'] = 'NXdata'
        data['signal'] = np.arange(5)
        data['signal'].attrs['units'] = 'm'
        data[coord] = np.arange(5)
        data[coord].attrs['units'] = 's'
        data.attrs['signal'] = 'signal'
        data.attrs['axes'] = [dim]
        data.attrs[f'{dim}_indices'] = [0]
    return snx.Group(h5root, definitions=snx.base_definitions())


@pytest.mark.parametrize(
    "slice_",
    [
        ('time', sc.scalar(1, unit='s')),
        ('time', slice(sc.scalar(1, unit='s'), None)),
        ('time', slice(None, sc.scalar(1, unit='s'))),
        {'time': sc.scalar(1, unit='s'), 'x': sc.scalar(1, unit='s')},
    ],
)
@pytest.mark.parametrize(
    ('dims', 'coords'),
    [
        (('time',), ('time2',)),
        (('time', 'time'), ('time2', 'time')),
    ],
)
def test_label_indexing_group_behaves_same_as_indexing_scipp_datagroup(
    h5root, slice_, dims, coords
):
    nx = create_nexus_group_with_data_arrays(h5root, dims, coords)
    dg = nx[()]

    exception = None
    try:
        # Scipp does not support dict slicing,
        # manually slice datagroup in multiple coords
        if isinstance(slice_, dict):
            for s in slice_.items():
                dg = dg[s]
        else:
            dg = dg[slice_]
    except Exception as e:
        exception = type(e)

    if exception:
        with pytest.raises(exception):
            nx[slice_]
    else:
        assert_identical(nx[slice_], dg)


def test_create_field_saves_errors(nxroot):
    entry = nxroot['entry']
    data = sc.array(
        dims=['d0'], values=[1.2, 3.4, 5.6], variances=[0.9, 0.8, 0.7], unit='cm'
    )
    entry.create_field('signal', data)

    loaded = entry['signal'][()]
    # Use allclose instead of identical because the variances are stored as stddevs
    # which loses precision.
    assert sc.allclose(loaded, data.rename_dims(d0='dim_0'))


@pytest.mark.parametrize('nxclass', [NXlog, NXmonitor, NXdetector, NXevent_data])
def test_empty_class_does_not_warn(nxroot, nxclass):
    log = nxroot['entry'].create_class('log', nxclass)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        log[()]


def test_trailing_forward_slash_in_path_does_not_change_file_object(nxroot):
    assert id(nxroot['entry/']) == id(nxroot['entry'])


def test_path_santization(nxroot):
    nxroot['entry'].create_class('log', NXlog)
    assert id(nxroot['/entry/log']) == id(nxroot['/entry//log'])
    assert id(nxroot['entry/log']) == id(nxroot['entry//log'])
