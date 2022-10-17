import h5py
import numpy as np
import scipp as sc
from scippnexus import Field, NXroot, NXentry, NXdata, NXlog
import pytest


@pytest.fixture()
def nxroot(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = NXroot(f)
        root.create_class('entry', NXentry)
        yield root


def test_without_coords(nxroot):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]])
    data = nxroot.create_class('data1', NXdata)
    data.create_field('signal', signal)
    data.attrs['axes'] = signal.dims
    data.attrs['signal'] = 'signal'
    assert sc.identical(data[...], sc.DataArray(signal))


def test_with_coords_matching_axis_names(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx', da.coords['xx'])
    assert sc.identical(data[...], da)


def test_guessed_dim_for_coord_not_matching_axis_name(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = da.data['yy', 1]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx2', da.coords['xx2'])
    assert sc.identical(data[...], da)


def test_multiple_coords(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx', da.coords['xx'])
    data.create_field('xx2', da.coords['xx2'])
    data.create_field('yy', da.coords['yy'])
    assert sc.identical(data[...], da)


def test_slice_of_1d(nxroot):
    da = sc.DataArray(sc.array(dims=['xx'], unit='m', values=[1, 2, 3]))
    da.coords['xx'] = da.data
    da.coords['xx2'] = da.data
    da.coords['scalar'] = sc.scalar(1.2)
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx', da.coords['xx'])
    data.create_field('xx2', da.coords['xx2'])
    data.create_field('scalar', da.coords['scalar'])
    assert sc.identical(data['xx', :2], da['xx', :2])
    assert sc.identical(data[:2], da['xx', :2])


def test_slice_of_multiple_coords(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx', da.coords['xx'])
    data.create_field('xx2', da.coords['xx2'])
    data.create_field('yy', da.coords['yy'])
    assert sc.identical(data['xx', :2], da['xx', :2])


def test_guessed_dim_for_2d_coord_not_matching_axis_name(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = da.data
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx2', da.coords['xx2'])
    assert sc.identical(data[...], da)


def test_skips_axis_if_dim_guessing_finds_ambiguous_shape(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    da.coords['yy2'] = da.data['xx', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('yy2', da.coords['yy2'])
    da = data[...]
    assert 'yy2' not in da.coords


def test_guesses_transposed_dims_for_2d_coord(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = sc.transpose(da.data)
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx2', da.coords['xx2'])
    assert sc.identical(data[...], da)


@pytest.mark.parametrize("indices", [1, [1]], ids=['int', 'list-of-int'])
def test_indices_attribute_for_coord(nxroot, indices):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['yy2'] = da.data['xx', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.attrs['yy2_indices'] = indices
    data.create_field('signal', da.data)
    data.create_field('yy2', da.coords['yy2'])
    assert sc.identical(data[...], da)


@pytest.mark.parametrize("indices", [1, [1]], ids=['int', 'list-of-int'])
def test_indices_attribute_for_coord_with_nontrivial_slice(nxroot, indices):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['yy2'] = da.data['xx', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.attrs['yy2_indices'] = indices
    data.create_field('signal', da.data)
    data.create_field('yy2', da.coords['yy2'])
    assert sc.identical(data['yy', :1], da['yy', :1])


def test_transpose_indices_attribute_for_coord(nxroot):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['xx2'] = sc.transpose(da.data)
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.attrs['xx2_indices'] = [1, 0]
    data.create_field('signal', da.data)
    data.create_field('xx2', da.coords['xx2'])
    assert sc.identical(data[...], da)


def test_auxiliary_signal_is_not_loaded_as_coord(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['xx', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    # We flag 'xx' as auxiliary_signal. It should thus not be loaded as a coord,
    # even though we create the field.
    data.attrs['auxiliary_signals'] = ['xx']
    data.create_field('signal', da.data)
    data.create_field('xx', da.coords['xx'])
    del da.coords['xx']
    assert sc.identical(data[...], da)


def test_field_dims_match_NXdata_dims(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal1'
    data.create_field('signal1', da.data)
    data.create_field('xx', da.coords['xx'])
    data.create_field('xx2', da.coords['xx2'])
    data.create_field('yy', da.coords['yy'])
    assert sc.identical(data['xx', :2].data, data['signal1']['xx', :2])
    assert sc.identical(data['xx', :2].coords['xx'], data['xx']['xx', :2])
    assert sc.identical(data['xx', :2].coords['xx2'], data['xx2']['xx', :2])
    assert sc.identical(data['xx', :2].coords['yy'], data['yy'][:])


def test_field_dims_match_NXdata_dims_when_selected_via_class_name(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal1'
    data.create_field('signal1', da.data)
    data.create_field('xx', da.coords['xx'])
    data.create_field('xx2', da.coords['xx2'])
    data.create_field('yy', da.coords['yy'])
    fields = data[Field]
    assert fields['signal1'].dims == ('xx', 'yy')
    assert fields['xx'].dims == ('xx', )
    assert fields['xx2'].dims == ('xx', )
    assert fields['yy'].dims == ('yy', )


def test_uses_default_field_dims_if_inference_fails(nxroot):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['yy2'] = sc.arange('yy', 4)
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('yy2', da.coords['yy2'])
    assert 'yy2' not in data[()].coords
    assert sc.identical(data['yy2'][()], da.coords['yy2'].rename(yy='dim_0'))


@pytest.mark.parametrize("unit", ['m', 's', None])
def test_create_field_from_variable(nxroot, unit):
    var = sc.array(dims=['xx'], unit=unit, values=[3, 4])
    nxroot.create_field('field', var)
    loaded = nxroot['field'][...]
    # Nexus does not support storing dim labels
    assert sc.identical(loaded, var.rename(xx=loaded.dim))


def test_create_datetime_field_from_variable(nxroot):
    var = sc.datetime(np.datetime64('now'), unit='ns') + sc.arange(
        'time', 1, 4, dtype='int64', unit='ns')
    nxroot.create_field('field', var)
    loaded = nxroot['field'][...]
    # Nexus does not support storing dim labels
    assert sc.identical(loaded, var.rename(time=loaded.dim))


@pytest.mark.parametrize("nx_class", [NXdata, NXlog])
def test_create_class(nxroot, nx_class):
    group = nxroot.create_class('group', nx_class)
    assert group.nx_class == nx_class


@pytest.mark.parametrize("errors_suffix", ['_error', '_errors'])
def test_field_matching_errors_regex_is_loaded_if_no_corresponding_value_field(
        nxroot, errors_suffix):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords[f'xx{errors_suffix}'] = da.data['yy', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field(f'xx{errors_suffix}', da.coords[f'xx{errors_suffix}'])
    assert sc.identical(data[...], da)


@pytest.mark.parametrize("errors_suffix", ['_error', '_errors'])
def test_uncertainties_of_coords_are_loaded(nxroot, errors_suffix):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = sc.array(dims=['xx'],
                               unit='m',
                               values=[1, 2, 3],
                               variances=[1, 4, 9],
                               dtype='float64')
    da.coords['xx2'] = sc.array(dims=['xx'],
                                unit='m',
                                values=[2, 3],
                                variances=[4, 9],
                                dtype='float64')
    da.coords['scalar'] = sc.scalar(value=1.2, variance=4.0, unit='K')
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.attrs['xx2_indices'] = 0
    data.create_field('signal', da.data)
    data.create_field('xx', sc.values(da.coords['xx']))
    data.create_field(f'xx{errors_suffix}', sc.stddevs(da.coords['xx']))
    data.create_field('xx2', sc.values(da.coords['xx2']))
    data.create_field(f'xx2{errors_suffix}', sc.stddevs(da.coords['xx2']))
    data.create_field('scalar', sc.values(da.coords['scalar']))
    data.create_field(f'scalar{errors_suffix}', sc.stddevs(da.coords['scalar']))
    assert sc.identical(data[...], da)


def test_unnamed_extra_dims_of_coords_are_squeezed(nxroot):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]])
    data = nxroot.create_class('data1', NXdata)
    data.create_field('signal', signal)
    data.attrs['axes'] = signal.dims
    data.attrs['signal'] = 'signal'
    # shape=[1]
    data.create_field('scalar', sc.array(dims=['ignored'], values=[1.2]))
    loaded = data[...]
    assert sc.identical(loaded.coords['scalar'], sc.scalar(1.2))
    assert data['scalar'].ndim == 0
    assert data['scalar'].shape == ()
    assert sc.identical(data['scalar'][...], sc.scalar(1.2))


def test_unnamed_extra_dims_of_multidim_coords_are_squeezed(nxroot):
    signal = sc.array(dims=['xx'], unit='m', values=[1.1, 2.2])
    data = nxroot.create_class('data1', NXdata)
    data.create_field('signal', signal)
    data.attrs['axes'] = signal.dims
    data.attrs['signal'] = 'signal'
    # shape=[2,1]
    xx = sc.array(dims=['xx', 'ignored'], values=[[1.1], [2.2]])
    data.create_field('xx', xx)
    loaded = data[...]
    assert sc.identical(loaded.coords['xx'], xx['ignored', 0])
    assert data['xx'].ndim == 1
    assert data['xx'].shape == (2, )
    assert sc.identical(data['xx'][...], xx['ignored', 0])


def test_dims_of_length_1_are_kept_when_axes_specified(nxroot):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1]])
    data = nxroot.create_class('data1', NXdata)
    data.create_field('signal', signal)
    data.attrs['axes'] = ['xx', 'yy']
    data.attrs['signal'] = 'signal'
    loaded = data[...]
    assert sc.identical(loaded.data, signal)
    assert data['signal'].ndim == 2
    assert data['signal'].shape == (1, 1)


def test_dims_of_length_1_are_squeezed_when_no_axes_specified(nxroot):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1]])
    data = nxroot.create_class('data1', NXdata)
    data.create_field('signal', signal)
    data.attrs['signal'] = 'signal'
    loaded = data[...]
    assert sc.identical(loaded.data, sc.scalar(1.1, unit='m'))
    assert data['signal'].ndim == 0
    assert data['signal'].shape == ()


def test_one_dim_of_length_1_is_squeezed_when_no_axes_specified(nxroot):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2]])
    data = nxroot.create_class('data1', NXdata)
    data.create_field('signal', signal)
    data.attrs['signal'] = 'signal'
    loaded = data[...]
    # Note that dimension gets renamed to `dim_0` since no axes are specified
    assert sc.identical(loaded.data,
                        sc.array(dims=['dim_0'], unit='m', values=[1.1, 2.2]))
    assert data['signal'].ndim == 1
    assert data['signal'].shape == (2, )
    assert data['signal'].dims == ('dim_0', )


def test_only_one_axis_specified_for_2d_field(nxroot):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1]])
    data = nxroot.create_class('data1', NXdata)
    data.create_field('signal', signal)
    data.attrs['axes'] = ['zz']
    data.attrs['signal'] = 'signal'
    loaded = data[...]
    assert sc.identical(loaded.data, sc.array(dims=['zz'], unit='m', values=[1.1]))


def test_fields_with_datetime_attribute_are_loaded_as_datetime(nxroot):
    da = sc.DataArray(
        sc.epoch(unit='s') +
        sc.array(dims=['xx', 'yy'], unit='s', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = nxroot.create_class('data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.create_field('signal', da.data)
    data.create_field('xx', da.coords['xx'])
    data.create_field('xx2', da.coords['xx2'])
    data.create_field('yy', da.coords['yy'])
    assert sc.identical(data[...], da)


def test_slicing_with_bin_edge_coord_returns_bin_edges(nxroot):
    da = sc.DataArray(sc.array(dims=['xx'], unit='K', values=[1.1, 2.2, 3.3]))
    da.coords['xx'] = sc.array(dims=['xx'], unit='m', values=[0.1, 0.2, 0.3, 0.4])
    da.coords['xx2'] = sc.array(dims=['xx'], unit='m', values=[0.3, 0.4, 0.5, 0.6])
    data = nxroot.create_class('data', NXdata)
    data.create_field('xx', da.coords['xx'])
    data.create_field('xx2', da.coords['xx2'])
    data.create_field('data', da.data)
    data.attrs['signal'] = 'data'
    data.attrs['axes'] = ['xx']
    data.attrs['xx_indices'] = [0]
    data.attrs['xx2_indices'] = [0]
    assert sc.identical(data[...], da)
    assert sc.identical(data['xx', 0], da['xx', 0])
    assert sc.identical(data['xx', 1], da['xx', 1])
    assert sc.identical(data['xx', 0:1], da['xx', 0:1])
    assert sc.identical(data['xx', 1:3], da['xx', 1:3])
    assert sc.identical(data['xx', 1:1], da['xx', 1:1])  # empty slice
