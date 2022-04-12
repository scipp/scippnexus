import h5py
import scipp as sc
from scippnexus import NXroot, NX_class
from scippnexus.loader import DataArrayLoaderFactory, Selector, ScalarProvider
from scippnexus.typing import H5Group
import pytest


@pytest.fixture()
def nxroot(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = NXroot(f)
        root.create_class('entry', NX_class.NXentry)
        yield root


def data_array_xx_yy() -> sc.DataArray:
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    return da


def data_array_time() -> sc.DataArray:
    return sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2, 3.3]),
        coords={
            'time':
            sc.epoch(unit='ns') +
            sc.array(dims=['time'], unit='s', values=[4.4, 5.5, 6.6]).to(unit='ns',
                                                                         dtype='int64')
        })


def add_data(group: H5Group, name: str, da: sc.DataArray):
    data = group.create_class(name, NX_class.NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx', da.coords['xx'])
    data.create_field('xx2', da.coords['xx2'])
    data.create_field('yy', da.coords['yy'])


def add_log(group: H5Group, name: str, da: sc.DataArray):
    log = group['entry'].create_class(name, NX_class.NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')


def test_get_single_data(nxroot):
    da = data_array_xx_yy()
    add_data(nxroot, 'data1', da)
    add_data(nxroot, 'data2', da + da)
    factory = DataArrayLoaderFactory()
    factory.set_base(lambda x: x['data2'], Selector(nxclass=NX_class.NXdata))
    loader = factory(nxroot)
    assert sc.identical(loader[()], da + da)


def test_combine_data(nxroot):
    da = data_array_xx_yy()
    add_data(nxroot, 'data1', da)
    add_data(nxroot, 'data2', da + da)
    factory = DataArrayLoaderFactory()

    def concat_values_along_z(mapping):
        return sc.concat(list(mapping.values()), dim='z')

    factory.set_base(concat_values_along_z, Selector(nxclass=NX_class.NXdata))
    loader = factory(nxroot)
    assert sc.identical(loader[()], sc.concat([da, da + da], 'z'))


def test_combine_data_and_add_attrs(nxroot):
    da = data_array_xx_yy()
    da.attrs['log1'] = sc.scalar(data_array_time())
    add_data(nxroot, 'data1', da)
    add_data(nxroot, 'data2', da + da)
    add_log(nxroot, 'log1', da.attrs['log1'].value)
    factory = DataArrayLoaderFactory()

    def concat_values_along_z(mapping):
        return sc.concat(list(mapping.values()), dim='z')

    factory.set_base(concat_values_along_z, Selector(nxclass=NX_class.NXdata))
    factory.add_attrs(ScalarProvider, Selector(nxclass=NX_class.NXlog))
    loader = factory(nxroot)
    assert sc.identical(loader[()], sc.concat([da, da + da], 'z'))
