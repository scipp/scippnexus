import h5py
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

import scippnexus.v2 as snx
from scippnexus.v2 import NXdata, NXlog


@pytest.fixture()
def h5root(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        yield f


@pytest.fixture()
def nxroot(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = snx.Group(f, definitions=snx.base_definitions)
        root.create_class('entry', snx.NXentry)
        yield root


def test_without_coords(h5root):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]])
    data = snx.create_class(h5root, 'data1', snx.NXdata)
    snx.create_field(data, 'signal', signal)
    data.attrs['axes'] = signal.dims
    data.attrs['signal'] = 'signal'
    obj = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(obj[...], sc.DataArray(signal))


def test_with_coords_matching_axis_names(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    data = snx.create_class(h5root, 'data1', snx.NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx', da.coords['xx'])
    group = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(group[...], da)


def test_guessed_dim_for_coord_not_matching_axis_name(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = da.data['yy', 1]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx2', da.coords['xx2'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)


def test_multiple_coords(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx', da.coords['xx'])
    snx.create_field(data, 'xx2', da.coords['xx2'])
    snx.create_field(data, 'yy', da.coords['yy'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)


def test_slice_of_1d(h5root):
    da = sc.DataArray(sc.array(dims=['xx'], unit='m', values=[1, 2, 3]))
    da.coords['xx'] = da.data
    da.coords['xx2'] = da.data
    da.coords['scalar'] = sc.scalar(1.2)
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx', da.coords['xx'])
    snx.create_field(data, 'xx2', da.coords['xx2'])
    snx.create_field(data, 'scalar', da.coords['scalar'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data['xx', :2], da['xx', :2])
    assert sc.identical(data[:2], da['xx', :2])


def test_slice_of_multiple_coords(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx', da.coords['xx'])
    snx.create_field(data, 'xx2', da.coords['xx2'])
    snx.create_field(data, 'yy', da.coords['yy'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data['xx', :2], da['xx', :2])


def test_guessed_dim_for_2d_coord_not_matching_axis_name(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = da.data
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx2', da.coords['xx2'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)


def test_skips_axis_if_dim_guessing_finds_ambiguous_shape(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    da.coords['yy2'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'yy2', da.coords['yy2'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert set(data.dims) == {'dim_0', 'xx', 'yy'}
    dg = data[...]
    assert isinstance(dg, sc.DataGroup)
    assert 'yy2' in dg
    assert set(dg.dims) == {'dim_0', 'xx', 'yy'}


def test_guesses_transposed_dims_for_2d_coord(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = sc.transpose(da.data)
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx2', da.coords['xx2'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)


@pytest.mark.parametrize("indices", [1, [1]], ids=['int', 'list-of-int'])
def test_indices_attribute_for_coord(h5root, indices):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['yy2'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.attrs['yy2_indices'] = indices
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'yy2', da.coords['yy2'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)


@pytest.mark.parametrize("indices", [1, [1]], ids=['int', 'list-of-int'])
def test_indices_attribute_for_coord_with_nontrivial_slice(h5root, indices):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['yy2'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.attrs['yy2_indices'] = indices
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'yy2', da.coords['yy2'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data['yy', :1], da['yy', :1])


def test_transpose_indices_attribute_for_coord(h5root):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['xx2'] = sc.transpose(da.data)
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.attrs['xx2_indices'] = [1, 0]
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx2', da.coords['xx2'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)


def test_auxiliary_signal_causes_load_as_dataset(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    # We flag 'xx' as auxiliary_signal. It should thus not be loaded as a coord,
    # even though we create the field.
    data.attrs['auxiliary_signals'] = ['xx']
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx', da.coords['xx'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert_identical(data[...], sc.Dataset({'signal': da.data, 'xx': da.coords['xx']}))


def test_field_dims_match_NXdata_dims(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal1'
    snx.create_field(data, 'signal1', da.data)
    snx.create_field(data, 'xx', da.coords['xx'])
    snx.create_field(data, 'xx2', da.coords['xx2'])
    snx.create_field(data, 'yy', da.coords['yy'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data['xx', :2].data, data['signal1']['xx', :2])
    assert sc.identical(data['xx', :2].coords['xx'], data['xx']['xx', :2])
    assert sc.identical(data['xx', :2].coords['xx2'], data['xx2']['xx', :2])
    assert sc.identical(data['xx', :2].coords['yy'], data['yy'][:])


def test_field_dims_match_NXdata_dims_when_selected_via_class_name(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal1'
    snx.create_field(data, 'signal1', da.data)
    snx.create_field(data, 'xx', da.coords['xx'])
    snx.create_field(data, 'xx2', da.coords['xx2'])
    snx.create_field(data, 'yy', da.coords['yy'])
    data = snx.Group(data, definitions=snx.base_definitions)
    fields = data[snx.Field]
    assert fields['signal1'].dims == ('xx', 'yy')
    assert fields['xx'].dims == ('xx', )
    assert fields['xx2'].dims == ('xx', )
    assert fields['yy'].dims == ('yy', )


def test_uses_default_field_dims_if_inference_fails(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['yy2'] = sc.arange('yy', 4)
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'yy2', da.coords['yy2'])
    data = snx.Group(data, definitions=snx.base_definitions)
    dg = data[()]
    assert sc.identical(dg['yy2'], da.coords['yy2'].rename(yy='dim_0'))
    assert sc.identical(data['yy2'][()], da.coords['yy2'].rename(yy='dim_0'))


@pytest.mark.parametrize("unit", ['m', 's', None])
def test_create_field_from_variable(h5root, unit):
    var = sc.array(dims=['xx'], unit=unit, values=[3, 4])
    snx.create_field(h5root, 'field', var)
    group = snx.Group(h5root, definitions=snx.base_definitions)
    loaded = group['field'][...]
    # Nexus does not support storing dim labels
    assert sc.identical(loaded, var.rename(xx=loaded.dim))


def test_create_datetime_field_from_variable(h5root):
    var = sc.datetime(np.datetime64('now'), unit='ns') + sc.arange(
        'time', 1, 4, dtype='int64', unit='ns')
    snx.create_field(h5root, 'field', var)
    group = snx.Group(h5root, definitions=snx.base_definitions)
    loaded = group['field'][...]
    # Nexus does not support storing dim labels
    assert sc.identical(loaded, var.rename(time=loaded.dim))


@pytest.mark.parametrize("nx_class", [NXdata, NXlog])
def test_create_class(nxroot, nx_class):
    group = nxroot.create_class('group', nx_class)
    assert group.nx_class == nx_class


@pytest.mark.parametrize("errors_suffix", ['_error', '_errors'])
def test_field_matching_errors_regex_is_loaded_if_no_corresponding_value_field(
        h5root, errors_suffix):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords[f'xx{errors_suffix}'] = da.data['yy', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, f'xx{errors_suffix}', da.coords[f'xx{errors_suffix}'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)


@pytest.mark.parametrize("errors_suffix", ['_error', '_errors'])
def test_uncertainties_of_coords_are_loaded(h5root, errors_suffix):
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
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    data.attrs['xx2_indices'] = 0
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx', sc.values(da.coords['xx']))
    snx.create_field(data, f'xx{errors_suffix}', sc.stddevs(da.coords['xx']))
    snx.create_field(data, 'xx2', sc.values(da.coords['xx2']))
    snx.create_field(data, f'xx2{errors_suffix}', sc.stddevs(da.coords['xx2']))
    snx.create_field(data, 'scalar', sc.values(da.coords['scalar']))
    snx.create_field(data, f'scalar{errors_suffix}', sc.stddevs(da.coords['scalar']))
    data = snx.Group(data, definitions=snx.base_definitions)
    print(data[...], da)
    assert sc.identical(data[...], da)


def test_unnamed_extra_dims_of_coords_are_squeezed(h5root):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]])
    data = snx.create_class(h5root, 'data1', NXdata)
    snx.create_field(data, 'signal', signal)
    data.attrs['axes'] = signal.dims
    data.attrs['signal'] = 'signal'
    # shape=[1]
    snx.create_field(data, 'scalar', sc.array(dims=['ignored'], values=[1.2]))
    data = snx.Group(data, definitions=snx.base_definitions)
    loaded = data[...]
    assert sc.identical(loaded.coords['scalar'], sc.scalar(1.2))
    assert data['scalar'].ndim == 0
    assert data['scalar'].shape == ()
    assert sc.identical(data['scalar'][...], sc.scalar(1.2))


def test_unnamed_extra_dims_of_multidim_coords_are_squeezed(h5root):
    signal = sc.array(dims=['xx'], unit='m', values=[1.1, 2.2])
    data = snx.create_class(h5root, 'data1', NXdata)
    snx.create_field(data, 'signal', signal)
    data.attrs['axes'] = signal.dims
    data.attrs['signal'] = 'signal'
    # shape=[2,1]
    xx = sc.array(dims=['xx', 'ignored'], values=[[1.1], [2.2]])
    snx.create_field(data, 'xx', xx)
    data = snx.Group(data, definitions=snx.base_definitions)
    loaded = data[...]
    assert sc.identical(loaded.coords['xx'], xx['ignored', 0])
    assert data['xx'].ndim == 1
    assert data['xx'].shape == (2, )
    assert sc.identical(data['xx'][...], xx['ignored', 0])


def test_dims_of_length_1_are_kept_when_axes_specified(h5root):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1]])
    data = snx.create_class(h5root, 'data1', NXdata)
    snx.create_field(data, 'signal', signal)
    data.attrs['axes'] = ['xx', 'yy']
    data.attrs['signal'] = 'signal'
    data = snx.Group(data, definitions=snx.base_definitions)
    loaded = data[...]
    assert sc.identical(loaded.data, signal)
    assert data['signal'].ndim == 2
    assert data['signal'].shape == (1, 1)


def test_only_dim_of_length_1_is_squeezed_when_no_axes_specified(h5root):
    signal = sc.array(dims=['xx'], unit='m', values=[1.1])
    data = snx.create_class(h5root, 'data1', NXdata)
    snx.create_field(data, 'signal', signal)
    data.attrs['signal'] = 'signal'
    data = snx.Group(data, definitions=snx.base_definitions)
    loaded = data[...]
    assert sc.identical(loaded.data, sc.scalar(1.1, unit='m'))
    assert data['signal'].ndim == 0
    assert data['signal'].shape == ()


def test_multi_dims_of_length_1_are_kept_when_no_axes_specified(h5root):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1]])
    data = snx.create_class(h5root, 'data1', NXdata)
    snx.create_field(data, 'signal', signal)
    data.attrs['signal'] = 'signal'
    data = snx.Group(data, definitions=snx.base_definitions)
    loaded = data[...]
    assert sc.identical(loaded.data,
                        sc.array(dims=['dim_0', 'dim_1'], unit='m', values=[[1.1]]))
    assert data['signal'].ndim == 2
    assert data['signal'].shape == (1, 1)


def test_one_dim_of_length_1_is_kept_when_no_axes_specified(h5root):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2]])
    data = snx.create_class(h5root, 'data1', NXdata)
    snx.create_field(data, 'signal', signal)
    data.attrs['signal'] = 'signal'
    data = snx.Group(data, definitions=snx.base_definitions)
    loaded = data[...]
    # Note that dimension gets renamed to `dim_0` since no axes are specified
    assert sc.identical(
        loaded.data, sc.array(dims=['dim_0', 'dim_1'], unit='m', values=[[1.1, 2.2]]))
    assert data['signal'].ndim == 2
    assert data['signal'].shape == (1, 2)
    assert data['signal'].dims == ('dim_0', 'dim_1')


def test_only_one_axis_specified_for_2d_field(h5root):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1]])
    data = snx.create_class(h5root, 'data1', NXdata)
    snx.create_field(data, 'signal', signal)
    data.attrs['axes'] = ['zz']
    data.attrs['signal'] = 'signal'
    data = snx.Group(data, definitions=snx.base_definitions)
    loaded = data[...]
    assert sc.identical(loaded.data, sc.array(dims=['zz'], unit='m', values=[1.1]))


def test_fields_with_datetime_attribute_are_loaded_as_datetime(h5root):
    da = sc.DataArray(
        sc.epoch(unit='s') +
        sc.array(dims=['xx', 'yy'], unit='s', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_field(data, 'signal', da.data)
    snx.create_field(data, 'xx', da.coords['xx'])
    snx.create_field(data, 'xx2', da.coords['xx2'])
    snx.create_field(data, 'yy', da.coords['yy'])
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)


def test_slicing_with_bin_edge_coord_returns_bin_edges(h5root):
    da = sc.DataArray(sc.array(dims=['xx'], unit='K', values=[1.1, 2.2, 3.3]))
    da.coords['xx'] = sc.array(dims=['xx'], unit='m', values=[0.1, 0.2, 0.3, 0.4])
    da.coords['xx2'] = sc.array(dims=['xx'], unit='m', values=[0.3, 0.4, 0.5, 0.6])
    data = snx.create_class(h5root, 'data', NXdata)
    snx.create_field(data, 'xx', da.coords['xx'])
    snx.create_field(data, 'xx2', da.coords['xx2'])
    snx.create_field(data, 'data', da.data)
    data.attrs['signal'] = 'data'
    data.attrs['axes'] = ['xx']
    data.attrs['xx_indices'] = [0]
    data.attrs['xx2_indices'] = [0]
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)
    assert sc.identical(data['xx', 0], da['xx', 0])
    assert sc.identical(data['xx', 1], da['xx', 1])
    assert sc.identical(data['xx', 0:1], da['xx', 0:1])
    assert sc.identical(data['xx', 1:3], da['xx', 1:3])
    assert sc.identical(data['xx', 1:1], da['xx', 1:1])  # empty slice


def test_legacy_signal_attr_is_used(h5root):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]])
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = signal.dims
    field = snx.create_field(data, 'mysig', signal)
    field.attrs['signal'] = 1  # legacy way of defining signal
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], sc.DataArray(signal))


def test_invalid_group_signal_attribute_is_ignored(h5root):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]])
    data = snx.create_class(h5root, 'data1', NXdata)
    data.attrs['axes'] = signal.dims
    data.attrs['signal'] = 'signal'
    field = snx.create_field(data, 'mysig', signal)
    field.attrs['signal'] = 1  # legacy way of defining signal
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], sc.DataArray(signal))


def test_legacy_axis_attrs_define_dim_names(h5root):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['yy'] = da.data['xx', 0]
    data = snx.create_class(h5root, 'data1', NXdata)
    signal = snx.create_field(data, 'signal', da.data)
    xx = snx.create_field(data, 'xx', da.coords['xx'])
    yy = snx.create_field(data, 'yy', da.coords['yy'])
    signal.attrs['signal'] = 1
    xx.attrs['axis'] = 1
    yy.attrs['axis'] = 2
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], da)


def test_nested_groups_trigger_fallback_to_load_as_data_group(h5root):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    data = snx.create_class(h5root, 'data1', NXdata)
    snx.create_field(data, 'signal', da.data)
    data.attrs['axes'] = da.dims
    data.attrs['signal'] = 'signal'
    snx.create_class(data, 'nested', NXdata)
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], sc.DataGroup(signal=da.data, nested=sc.DataGroup()))


def test_slicing_raises_given_invalid_index(h5root):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]])
    data = snx.create_class(h5root, 'data1', NXdata)
    snx.create_field(data, 'signal', signal)
    data.attrs['axes'] = signal.dims
    data.attrs['signal'] = 'signal'
    data = snx.Group(data, definitions=snx.base_definitions)
    assert sc.identical(data[...], sc.DataArray(signal))
    with pytest.raises(IndexError):
        data['xx', 2]
    with pytest.raises(sc.DimensionError):
        data['zz', 0]
