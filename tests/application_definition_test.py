import h5py
import pytest
import scipp as sc
from scipp.testing import assert_identical

import scippnexus as snx
from scippnexus.application_definitions import nxcansas


@pytest.fixture()
def nxroot():
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = snx.Group(f, definitions=snx.base_definitions())
        root.create_class('entry', snx.NXentry)
        yield root


def test_setitem_SASentry(nxroot):
    nxroot['sasentry'] = nxcansas.SASentry(title='A test', run=12345)
    assert 'sasentry' in nxroot
    entry = nxroot['sasentry']
    assert entry.attrs['definition'] == 'NXcanSAS'
    assert entry['title'][()] == 'A test'
    assert entry['run'][()] == 12345


@pytest.fixture()
def I_of_Q():
    data = sc.array(
        dims=['Q'],
        values=[0.1, 0.2, 0.1, 0.4],
        variances=[1.0, 4.0, 9.0, 4.0],  # values chosen for exact sqrt
        unit='1/counts',
    )
    da = sc.DataArray(data=data)
    da.coords['Q'] = sc.linspace('Q', 0, 1, num=5, unit='1/angstrom')
    da.coords['Q'].variances = sc.array(dims=['Q'], values=[1, 1, 4, 4, 1]).values
    return da


def test_setitem_SASdata_raises_ValueError_when_given_bin_edges(nxroot, I_of_Q):
    with pytest.raises(ValueError, match='Q is given as bin-edges'):
        nxroot['sasdata'] = nxcansas.SASdata(I_of_Q, Q_variances='resolutions')


def test_setitem_SASdata(nxroot, I_of_Q):
    I_of_Q.coords['Q'] = I_of_Q.coords['Q'][1:]
    nxroot['sasdata'] = nxcansas.SASdata(I_of_Q, Q_variances='resolutions')
    data = nxroot['sasdata']
    assert sc.identical(data[...], I_of_Q)


def test_setitem_SASdata_raises_if_interpretation_of_variances_not_specified(nxroot):
    data = sc.array(
        dims=['Q'],
        values=[0.1, 0.2, 0.1, 0.4],
        variances=[1.0, 4.0, 9.0, 4.0],  # values chosen for exact sqrt
        unit='1/counts',
    )
    da = sc.DataArray(data=data)
    da.coords['Q'] = sc.linspace('Q', 0, 1, num=4, unit='1/angstrom')
    da.coords['Q'].variances = sc.array(dims=['Q'], values=[1, 4, 4, 1]).values
    with pytest.raises(ValueError, match='Q has variances, must specify whether these'):
        nxroot['sasdata'] = nxcansas.SASdata(da)


def test_load_SASdata(nxroot):
    nxroot['sasentry'] = nxcansas.SASentry(title='A test', run=12345)
    entry = nxroot['sasentry']
    da = sc.DataArray(
        sc.array(dims=['Q'], values=[1, 2, 3], unit=''),
        coords={'Q': sc.array(dims=['Q'], values=[1, 2, 3, 4], unit='1/angstrom')},
    )
    group = entry.create_class('sasdata', snx.NXdata)
    group._group.attrs['canSAS_class'] = 'SASdata'
    group._group.attrs['signal'] = 'I'
    group._group.attrs['I_axes'] = 'Q'
    group['I'] = da.data
    group['Q'] = da.coords['Q']
    sasroot = snx.Group(nxroot._group, definitions=nxcansas.definitions)
    loaded = sasroot['sasentry/sasdata'][()]
    assert_identical(loaded, da)


def test_load_SAStransmission_spectrum(nxroot):
    nxroot['sasentry'] = nxcansas.SASentry(title='A test', run=12345)
    entry = nxroot['sasentry']
    spectrum = sc.DataArray(
        sc.array(dims=['lambda'], values=[1, 2, 3], unit='counts'),
        coords={
            'lambda': sc.array(dims=['lambda'], values=[1, 2, 3, 4], unit='angstrom')
        },
    )
    group = entry.create_class('sastransmission_spectrum', snx.NXdata)
    group._group.attrs['canSAS_class'] = 'SAStransmission_spectrum'
    group._group.attrs['signal'] = 'T'
    group._group.attrs['T_axes'] = 'lambda'
    group['T'] = spectrum.data
    group['lambda'] = spectrum.coords['lambda']
    sasroot = snx.Group(nxroot._group, definitions=nxcansas.definitions)
    loaded = sasroot['sasentry/sastransmission_spectrum'][()]
    assert_identical(loaded, spectrum)
