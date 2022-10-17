import h5py
import scipp as sc
from scippnexus import NXroot, NXentry, NXmonitor, NXevent_data
import pytest


@pytest.fixture()
def nxroot(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = NXroot(f)
        root.create_class('entry', NXentry)
        yield root


def test_dense_monitor(nxroot):
    monitor = nxroot['entry'].create_class('monitor', NXmonitor)
    assert monitor.nx_class == NXmonitor
    da = sc.DataArray(
        sc.array(dims=['time_of_flight'], values=[1.0]),
        coords={'time_of_flight': sc.array(dims=['time_of_flight'], values=[1.0])})
    monitor['data'] = da.data
    monitor['data'].attrs['axes'] = 'time_of_flight'
    monitor['time_of_flight'] = da.coords['time_of_flight']
    assert sc.identical(monitor[...], da)


def create_event_data_no_ids(group):
    group.create_field('event_time_offset',
                       sc.array(dims=[''], unit='s', values=[456, 7, 3, 345, 632, 23]))
    group.create_field('event_time_zero',
                       sc.array(dims=[''], unit='s', values=[1, 2, 3, 4]))
    group.create_field('event_index', sc.array(dims=[''],
                                               unit=None,
                                               values=[0, 3, 3, 5]))


def test_loads_event_data_in_current_group(nxroot):
    monitor = nxroot.create_class('monitor1', NXmonitor)
    create_event_data_no_ids(monitor)
    assert monitor.dims == ('pulse', )
    assert monitor.shape == (4, )
    loaded = monitor[...]
    assert sc.identical(
        loaded.bins.size().data,
        sc.array(dims=['pulse'], unit=None, dtype='int64', values=[3, 0, 2, 1]))


def test_loads_event_data_in_child_group(nxroot):
    monitor = nxroot.create_class('monitor1', NXmonitor)
    create_event_data_no_ids(monitor.create_class('events', NXevent_data))
    assert monitor.dims == ('pulse', )
    assert monitor.shape == (4, )
    loaded = monitor[...]
    assert sc.identical(
        loaded.bins.size().data,
        sc.array(dims=['pulse'], unit=None, dtype='int64', values=[3, 0, 2, 1]))
