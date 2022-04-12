import h5py
import scipp as sc
from scippnexus import NXroot, NX_class
from scippnexus.loader import DataArrayLoaderFactory, Selector
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


def add_data(group: H5Group, name: str, da: sc.DataArray):
    data = group.create_class(name, NX_class.NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx', da.coords['xx'])
    data.create_field('xx2', da.coords['xx2'])
    data.create_field('yy', da.coords['yy'])


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
