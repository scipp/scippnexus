import h5py
import numpy as np
import pytest
import scipp as sc

import scippnexus.v2 as snx


@pytest.fixture()
def h5root(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        yield f


def test_does_not_see_changes(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data['signal'] = np.arange(4)
    data['time'] = np.arange(4)
    obj = snx.Group(entry)
    dg = obj[()]
    print(list(dg.items()))
    assert obj.sizes == {'dim_0': 4}
    assert 'data' in dg
    entry.create_group('data2')
    assert 'data2' not in dg  # inserted after NXobject creation


def test_read_recursive(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data['signal'] = np.arange(4)
    data['signal'].attrs['units'] = 'm'
    data['time'] = np.arange(5)
    data['time'].attrs['units'] = 's'
    obj = snx.Group(entry)
    dg = obj[()]
    print(list(dg.items()))
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
    assert np.array_equal(dg['signal'].variances, np.arange(4.0)**2)
    assert np.array_equal(dg['time'].variances, np.arange(5.0)**2)


def test_read_field(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data['signal'] = np.arange(4)
    data['signal'].attrs['units'] = 'm'
    obj = snx.Group(data)
    var = obj['signal'][()]
    assert sc.identical(var, sc.array(dims=['dim_0'], values=np.arange(4), unit='m'))


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
        'event_id', 'event_time_offset', 'event_time_zero', 'event_index'
    }


def test_nxevent_data_children_read_as_variables_with_correct_dims(h5root):
    entry = make_event_data(h5root)
    root = snx.Group(entry, definitions=snx.base_definitions)
    event_data = root['events']
    assert sc.identical(event_data['event_id'][()],
                        sc.array(dims=['event'], values=[1, 1, 1, 0], unit=None))
    assert sc.identical(event_data['event_time_offset'][()],
                        sc.array(dims=['event'], values=[0, 1, 2, 3], unit='ns'))
    assert sc.identical(
        event_data['event_time_zero'][()],
        sc.array(dims=['event_time_zero'], values=[100, 200], unit='ms'))
    assert sc.identical(event_data['event_index'][()],
                        sc.array(dims=['event_time_zero'], values=[0, 3], unit=None))


def test_nxevent_data_dims_and_sizes_ignore_pulse_contents(h5root):
    entry = make_event_data(h5root)
    root = snx.Group(entry, definitions=snx.base_definitions)
    event_data = root['events']
    assert event_data.dims == ('event_time_zero', )
    assert event_data.sizes == {'event_time_zero': 2}


def test_read_nxevent_data(h5root):
    entry = make_event_data(h5root)
    root = snx.Group(entry, definitions=snx.base_definitions)
    event_data = root['events']
    da = event_data[()]
    assert sc.identical(da.data.bins.size(),
                        sc.array(dims=['event_time_zero'], values=[3, 1], unit=None))


def test_nxdata_with_signal_axes_indices_reads_as_data_array(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data.attrs['NX_class'] = 'NXdata'
    data.attrs['signal'] = 'signal'
    data.attrs['axes'] = ['time', 'temperature']
    data.attrs['time_indices'] = [0]
    data.attrs['temperature_indices'] = [1]
    ref = sc.DataArray(
        data=sc.ones(dims=['time', 'temperature'], shape=[3, 4], unit='m'))
    ref.coords['time'] = sc.array(dims=['time'], values=np.arange(3), unit='s')
    ref.coords['temperature'] = sc.array(dims=['temperature'],
                                         values=np.arange(4),
                                         unit='K')
    data['signal'] = ref.values
    data['signal'].attrs['units'] = str(ref.unit)
    data['time'] = ref.coords['time'].values
    data['time'].attrs['units'] = str(ref.coords['time'].unit)
    data['temperature'] = ref.coords['temperature'].values
    data['temperature'].attrs['units'] = str(ref.coords['temperature'].unit)
    obj = snx.Group(data, definitions=snx.base_definitions)
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
        data=sc.ones(dims=['time', 'temperature'], shape=[3, 4], unit='m'))
    ref.coords['time'] = sc.array(dims=['time'], values=np.arange(3), unit='s')
    ref.coords['temperature'] = sc.array(dims=['temperature'],
                                         values=np.arange(4),
                                         unit='K')
    data['signal'] = ref.values
    data['signal'].attrs['units'] = str(ref.unit)
    data['time'] = ref.coords['time'].values
    data['time'].attrs['units'] = str(ref.coords['time'].unit)
    data['temperature'] = ref.coords['temperature'].values
    data['temperature'].attrs['units'] = str(ref.coords['temperature'].unit)
    obj = snx.Group(data, definitions=snx.base_definitions)
    da = obj['time', 0:2]
    assert sc.identical(da, ref['time', 0:2])


def test_nxdata_with_bin_edges_positional_indexing_returns_correct_slice(h5root):
    entry = h5root.create_group('entry')
    data = entry.create_group('data')
    data.attrs['NX_class'] = 'NXdata'
    data.attrs['signal'] = 'signal'
    data.attrs['axes'] = ['time', 'temperature']
    data.attrs['time_indices'] = [0]
    data.attrs['temperature_indices'] = [1]
    ref = sc.DataArray(
        data=sc.ones(dims=['time', 'temperature'], shape=[3, 4], unit='m'))
    ref.coords['time'] = sc.array(dims=['time'], values=np.arange(3), unit='s')
    ref.coords['temperature'] = sc.array(dims=['temperature'],
                                         values=np.arange(5),
                                         unit='K')
    data['signal'] = ref.values
    data['signal'].attrs['units'] = str(ref.unit)
    data['time'] = ref.coords['time'].values
    data['time'].attrs['units'] = str(ref.coords['time'].unit)
    data['temperature'] = ref.coords['temperature'].values
    data['temperature'].attrs['units'] = str(ref.coords['temperature'].unit)
    obj = snx.Group(data, definitions=snx.base_definitions)
    da = obj['temperature', 0:2]
    assert sc.identical(da, ref['temperature', 0:2])
