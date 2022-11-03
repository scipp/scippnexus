import h5py
import scipp as sc
from scippnexus import NXroot, NXentry
from scippnexus.definitions.nxcansas import NXcanSAS, SASentry, SASdata
import pytest


@pytest.fixture()
def nxroot(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = NXroot(f)
        root.create_class('entry', NXentry)
        yield root


def test_setitem_SASentry(nxroot):
    nxroot['sasentry'] = SASentry(title='A test', run=12345)
    assert 'sasentry' in nxroot
    entry = nxroot['sasentry']
    assert entry.attrs['definition'] == 'NXcanSAS'
    assert entry['title'][()] == 'A test'
    assert entry['run'][()] == 12345


def test_setitem_SASdata(nxroot):
    data = sc.array(
        dims=['Q'],
        values=[0.1, 0.2, 0.1, 0.4],
        variances=[1.0, 4.0, 9.0, 4.0],  # values chosen for exact sqrt
        unit='1/counts')
    da = sc.DataArray(data=data)
    da.coords['Q'] = sc.linspace('Q', 0, 1, num=5, unit='1/angstrom')
    da.coords['Q'].variances = sc.array(dims=['Q'], values=[1, 1, 4, 4, 1]).values
    nxroot['sasdata'] = SASdata(da, Q_variances='resolutions')
    nxroot._definition = NXcanSAS
    data = nxroot['sasdata']
    assert sc.identical(data[...], da)


def test_setitem_SASdata_raises_if_interpretation_of_variances_not_specified(nxroot):
    data = sc.array(
        dims=['Q'],
        values=[0.1, 0.2, 0.1, 0.4],
        variances=[1.0, 4.0, 9.0, 4.0],  # values chosen for exact sqrt
        unit='1/counts')
    da = sc.DataArray(data=data)
    da.coords['Q'] = sc.linspace('Q', 0, 1, num=5, unit='1/angstrom')
    da.coords['Q'].variances = sc.array(dims=['Q'], values=[1, 1, 4, 4, 1]).values
    with pytest.raises(ValueError):
        nxroot['sasdata'] = SASdata(da)
