import h5py
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

import scippnexus as snx
from scippnexus import NXdetector, NXentry, NXoff_geometry


def make_group(group: h5py.Group) -> snx.Group:
    return snx.Group(group, definitions=snx.base_definitions())


@pytest.fixture()
def h5root():
    """Yield h5py root group (file)"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        yield f


@pytest.fixture()
def nxroot():
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = make_group(f)
        root.create_class('entry', NXentry)
        yield root


def test_returns_as_datagroup_if_no_signal_found(nxroot):
    detector_numbers = sc.array(dims=[''], unit=None, values=np.array([1, 2, 3, 4]))
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_numbers', detector_numbers)
    dg = detector[...]
    assert isinstance(dg, sc.DataGroup)


def test_can_load_fields_if_no_data_found(h5root):
    detector_numbers = sc.array(dims=[''], unit=None, values=np.array([1, 2, 3, 4]))
    detector = snx.create_class(h5root, 'detector0', NXdetector)
    snx.create_field(detector, 'detector_numbers', detector_numbers)
    detector['detector_numbers'][...]


def test_finds_data_from_group_attr(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]])
    )
    da.coords['detector_numbers'] = detector_numbers_xx_yy_1234()
    detector = snx.create_class(h5root, 'detector0', NXdetector)
    snx.create_field(detector, 'detector_numbers', da.coords['detector_numbers'])
    snx.create_field(detector, 'custom', da.data)
    detector.attrs['signal'] = 'custom'
    detector = make_group(detector)
    assert sc.identical(detector[...], da.rename_dims({'xx': 'dim_0', 'yy': 'dim_1'}))


def test_loads_signal_and_events_when_both_found(nxroot):
    detector_number = sc.array(dims=[''], unit=None, values=np.array([1, 2]))
    data = sc.ones(dims=['detector_number'], shape=[2])
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_number)
    detector.create_field('data', data)
    events = detector.create_class('events', snx.NXevent_data)
    events.create_field('event_id', sc.array(dims=[''], unit=None, values=[1]))
    events.create_field('event_time_offset', sc.array(dims=[''], unit='s', values=[1]))
    events.create_field('event_time_zero', sc.array(dims=[''], unit='s', values=[1]))
    events.create_field('event_index', sc.array(dims=[''], unit='None', values=[0]))
    assert detector.sizes == {'detector_number': 2, 'event_time_zero': 1}
    loaded = detector[...]
    assert isinstance(loaded, sc.Dataset)
    assert_identical(loaded['data'].data, data)
    assert loaded['events'].bins is not None


def test_loads_as_data_array_with_embedded_events(nxroot):
    detector_number = sc.array(dims=[''], unit=None, values=np.array([1, 2, 3]))
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_number)
    detector.create_field('event_id', sc.array(dims=[''], unit=None, values=[1]))
    detector.create_field(
        'event_time_offset', sc.array(dims=[''], unit='s', values=[1])
    )
    detector.create_field('event_time_zero', sc.array(dims=[''], unit='s', values=[1]))
    detector.create_field('event_index', sc.array(dims=[''], unit='None', values=[0]))
    assert detector.dims == ('detector_number', 'event_time_zero')
    da = detector[...]
    assert da.bins is not None
    assert_identical(
        da.bins.size(),
        sc.DataArray(
            data=sc.array(dims=['detector_number'], unit=None, values=[1, 0, 0]),
            coords={'detector_number': detector_number.rename({'': 'detector_number'})},
        ),
    )


def detector_numbers_xx_yy_1234():
    return sc.array(dims=['xx', 'yy'], unit=None, values=np.array([[1, 2], [3, 4]]))


def test_loads_data_without_coords(h5root):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]]))
    da.coords['detector_numbers'] = detector_numbers_xx_yy_1234()
    detector = snx.create_class(h5root, 'detector0', NXdetector)
    snx.create_field(detector, 'detector_numbers', da.coords['detector_numbers'])
    snx.create_field(detector, 'data', da.data)
    detector = make_group(detector)
    assert sc.identical(detector[...], da.rename_dims({'xx': 'dim_0', 'yy': 'dim_1'}))


@pytest.mark.parametrize(
    'detector_number_key', ['detector_number', 'pixel_id', 'spectrum_index']
)
def test_detector_number_key_alias(h5root, detector_number_key):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]]))
    da.coords[detector_number_key] = detector_numbers_xx_yy_1234()
    detector = snx.create_class(h5root, 'detector0', NXdetector)
    snx.create_field(detector, detector_number_key, da.coords[detector_number_key])
    snx.create_field(detector, 'data', da.data)
    detector = make_group(detector)
    assert sc.identical(detector[...], da.rename_dims({'xx': 'dim_0', 'yy': 'dim_1'}))


def test_loads_data_with_coords(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]])
    )
    da.coords['detector_numbers'] = detector_numbers_xx_yy_1234()
    da.coords['xx'] = sc.array(dims=['xx'], unit='m', values=[0.1, 0.2])
    detector = snx.create_class(h5root, 'detector0', NXdetector)
    snx.create_field(detector, 'detector_numbers', da.coords['detector_numbers'])
    snx.create_field(detector, 'xx', da.coords['xx'])
    snx.create_field(detector, 'data', da.data)
    detector.attrs['axes'] = ['xx', '.']
    detector = make_group(detector)
    assert sc.identical(detector[...], da.rename_dims({'yy': 'dim_1'}))


def test_slicing_works_as_in_scipp(h5root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2, 3.3], [3.3, 4.4, 5.5]])
    )
    da.coords['detector_numbers'] = sc.array(
        dims=['xx', 'yy'], unit=None, values=np.array([[1, 2, 3], [4, 5, 6]])
    )
    da.coords['xx'] = sc.array(dims=['xx'], unit='m', values=[0.1, 0.2])
    da.coords['xx2'] = sc.array(dims=['xx'], unit='m', values=[0.3, 0.4])
    da.coords['yy'] = sc.array(dims=['yy'], unit='m', values=[0.1, 0.2, 0.3])
    da.coords['2d_edges'] = sc.array(
        dims=['yy', 'xx'], unit='m', values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )
    detector = snx.create_class(h5root, 'detector0', NXdetector)
    snx.create_field(detector, 'detector_numbers', da.coords['detector_numbers'])
    snx.create_field(detector, 'xx', da.coords['xx'])
    snx.create_field(detector, 'xx2', da.coords['xx2'])
    snx.create_field(detector, 'yy', da.coords['yy'])
    snx.create_field(detector, '2d_edges', da.coords['2d_edges'])
    snx.create_field(detector, 'data', da.data)
    detector.attrs['axes'] = ['xx', 'yy']
    detector.attrs['2d_edges_indices'] = [1, 0]
    detector = make_group(detector)
    assert_identical(detector[...], da)
    assert_identical(detector['xx', 0], da['xx', 0])
    assert_identical(detector['xx', 1], da['xx', 1])
    assert_identical(detector['xx', 0:1], da['xx', 0:1])
    assert_identical(detector['yy', 0], da['yy', 0])
    assert_identical(detector['yy', 1], da['yy', 1])
    assert_identical(detector['yy', 0:1], da['yy', 0:1])
    assert_identical(detector['yy', 1:1], da['yy', 1:1])  # empty slice


def create_event_data_ids_1234(group):
    group.create_field(
        'event_id', sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2, 2])
    )
    group.create_field(
        'event_time_offset',
        sc.array(dims=[''], unit='s', values=[456, 7, 3, 345, 632, 23]),
    )
    group.create_field(
        'event_time_zero', sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    )
    group.create_field(
        'event_index', sc.array(dims=[''], unit='None', values=[0, 3, 3, 5])
    )


def test_loads_event_data_mapped_to_detector_numbers_based_on_their_event_id(nxroot):
    detector_numbers = sc.array(
        dims=[''], unit=None, values=np.array([1, 2, 3, 4, 5, 6])
    )
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_numbers)
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    assert detector.sizes == {'detector_number': 6, 'event_time_zero': 4}
    da = detector[...]
    assert sc.identical(
        da.bins.size().data,
        sc.array(
            dims=['detector_number'],
            unit=None,
            dtype='int64',
            values=[2, 3, 0, 1, 0, 0],
        ),
    )
    assert 'event_time_offset' in da.bins.coords
    assert 'event_time_zero' in da.bins.coords


def test_detector_number_fallback_dims_determines_dims_with_event_data(nxroot):
    detector_numbers = sc.array(
        dims=['detector_number'], unit=None, values=np.array([1, 2, 3, 4, 5, 6])
    )
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_numbers)
    detector.create_field('pixel_offset', detector_numbers)
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    da = detector[()]
    assert da.sizes == {'detector_number': 6}
    assert_identical(da.coords['pixel_offset'], detector_numbers)


def test_loads_event_data_with_0d_detector_numbers(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', sc.index(1, dtype='int64'))
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    assert detector.dims == ('event_time_zero',)
    assert detector.shape == (4,)
    da = detector[...]
    assert sc.identical(da.bins.size().data, sc.index(2, dtype='int64'))


def test_loads_event_data_with_2d_detector_numbers(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_numbers_xx_yy_1234())
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    assert detector.sizes == {'dim_0': 2, 'dim_1': 2, 'event_time_zero': 4}
    da = detector[...]
    assert sc.identical(
        da.bins.size().data,
        sc.array(
            dims=['dim_0', 'dim_1'], unit=None, dtype='int64', values=[[2, 3], [0, 1]]
        ),
    )


def test_selecting_pixels_works_with_event_signal(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_numbers_xx_yy_1234())
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    assert detector.sizes == {'dim_0': 2, 'dim_1': 2, 'event_time_zero': 4}
    da = detector['dim_0', 0]
    assert_identical(
        da.bins.size().data,
        sc.array(dims=['dim_1'], unit=None, dtype='int64', values=[2, 3]),
    )


def test_selecting_pixels_works_with_embedded_event_signal(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_numbers_xx_yy_1234())
    create_event_data_ids_1234(detector)
    assert detector.sizes == {'dim_0': 2, 'dim_1': 2, 'event_time_zero': 4}
    da = detector['dim_0', 0]
    assert_identical(
        da.bins.size().data,
        sc.array(dims=['dim_1'], unit=None, dtype='int64', values=[2, 3]),
    )


def test_select_events_slices_underlying_event_data(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_numbers_xx_yy_1234())
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    da = detector['event_time_zero', :2]
    assert sc.identical(
        da.bins.size().data,
        sc.array(
            dims=['dim_0', 'dim_1'], unit=None, dtype='int64', values=[[1, 1], [0, 1]]
        ),
    )
    da = detector['event_time_zero', :3]
    assert sc.identical(
        da.bins.size().data,
        sc.array(
            dims=['dim_0', 'dim_1'], unit=None, dtype='int64', values=[[2, 2], [0, 1]]
        ),
    )
    da = detector['event_time_zero', 3]
    assert sc.identical(
        da.bins.size().data,
        sc.array(
            dims=['dim_0', 'dim_1'], unit=None, dtype='int64', values=[[0, 1], [0, 0]]
        ),
    )
    da = detector[()]
    assert sc.identical(
        da.bins.size().data,
        sc.array(
            dims=['dim_0', 'dim_1'], unit=None, dtype='int64', values=[[2, 3], [0, 1]]
        ),
    )


def test_loading_event_data_without_detector_numbers_does_not_group_events(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    assert detector.dims == ('event_time_zero',)
    da = detector[...]
    assert_identical(
        da.bins.size().data,
        sc.array(
            dims=['event_time_zero'], unit=None, dtype='int64', values=[3, 0, 2, 1]
        ),
    )


def test_loading_event_data_with_det_selection_and_automatic_detector_numbers_raises(
    nxroot,
):
    detector = nxroot.create_class('detector0', NXdetector)
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    assert detector.dims == ('event_time_zero',)
    with pytest.raises(sc.DimensionError):
        detector['detector_number', 0]


def test_loading_event_data_with_full_selection_and_automatic_detector_numbers_works(
    nxroot,
):
    detector = nxroot.create_class('detector0', NXdetector)
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    assert detector.dims == ('event_time_zero',)
    assert tuple(detector[...].shape) == (4,)
    assert tuple(detector[()].shape) == (4,)


def test_event_data_field_dims_labels(nxroot):
    detector_numbers = sc.array(dims=[''], unit=None, values=np.array([1, 2, 3, 4]))
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_numbers)
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))
    assert detector['detector_number'].dims == ('detector_number',)


def test_nxevent_data_without_detector_number_selection_yields_correct_pulses(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))

    class Load:
        def __getitem__(self, select=...):
            da = detector[select]
            assert (
                da.bins.size().sum().value
                == da.bins.constituents['data'].sizes['event']
            )
            return da.bins.size().values

    assert np.array_equal(Load()[...], [3, 0, 2, 1])
    assert np.array_equal(Load()['event_time_zero', 0], 3)
    assert np.array_equal(Load()['event_time_zero', 1], 0)
    assert np.array_equal(Load()['event_time_zero', 3], 1)
    assert np.array_equal(Load()['event_time_zero', -1], 1)
    assert np.array_equal(Load()['event_time_zero', -2], 2)
    assert np.array_equal(Load()['event_time_zero', 0:0], [])
    assert np.array_equal(Load()['event_time_zero', 1:1], [])
    assert np.array_equal(Load()['event_time_zero', 1:-3], [])
    assert np.array_equal(Load()['event_time_zero', 3:3], [])
    assert np.array_equal(Load()['event_time_zero', -1:-1], [])
    assert np.array_equal(Load()['event_time_zero', 0:1], [3])
    assert np.array_equal(Load()['event_time_zero', 0:-3], [3])
    assert np.array_equal(Load()['event_time_zero', -1:], [1])
    assert np.array_equal(Load()['event_time_zero', -2:-1], [2])
    assert np.array_equal(Load()['event_time_zero', -2:], [2, 1])
    assert np.array_equal(Load()['event_time_zero', :-2], [3, 0])


def test_nxevent_data_selection_yields_correct_pulses(nxroot):
    detector_numbers = sc.array(dims=[''], unit=None, values=np.array([1, 2, 3, 4]))
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('detector_number', detector_numbers)
    create_event_data_ids_1234(detector.create_class('events', snx.NXevent_data))

    class Load:
        def __getitem__(self, select=...):
            da = detector[select]
            return da.bins.size().values

    assert np.array_equal(Load()[...], [2, 3, 0, 1])
    assert np.array_equal(Load()['event_time_zero', 0], [1, 1, 0, 1])
    assert np.array_equal(Load()['event_time_zero', 1], [0, 0, 0, 0])
    assert np.array_equal(Load()['event_time_zero', 2], [1, 1, 0, 0])
    assert np.array_equal(Load()['event_time_zero', 3], [0, 1, 0, 0])
    assert np.array_equal(Load()['event_time_zero', -1], [0, 1, 0, 0])
    assert np.array_equal(Load()['event_time_zero', -2], [1, 1, 0, 0])
    assert np.array_equal(Load()['event_time_zero', 0:0], [0, 0, 0, 0])
    assert np.array_equal(Load()['event_time_zero', 1:1], [0, 0, 0, 0])
    assert np.array_equal(Load()['event_time_zero', 1:-3], [0, 0, 0, 0])
    assert np.array_equal(Load()['event_time_zero', 3:3], [0, 0, 0, 0])
    assert np.array_equal(Load()['event_time_zero', -1:-1], [0, 0, 0, 0])
    assert np.array_equal(Load()['event_time_zero', 0:1], [1, 1, 0, 1])
    assert np.array_equal(Load()['event_time_zero', 0:-3], [1, 1, 0, 1])
    assert np.array_equal(Load()['event_time_zero', -1:], [0, 1, 0, 0])
    assert np.array_equal(Load()['event_time_zero', -2:-1], [1, 1, 0, 0])
    assert np.array_equal(Load()['event_time_zero', -2:], [1, 2, 0, 0])
    assert np.array_equal(Load()['event_time_zero', :-2], [1, 1, 0, 1])


def create_off_geometry_detector_numbers_1234(
    group: snx.Group, name: str, detector_faces: bool = True
) -> sc.DataGroup:
    dg = sc.DataGroup()
    off = group.create_class(name, NXoff_geometry)
    # square with point in center
    values = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 0]])
    dg['vertices'] = sc.array(dims=['_', 'comp'], values=values, unit='m')
    # triangles
    dg['winding_order'] = sc.array(
        dims=['winding_order'], values=[0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], unit=None
    )
    dg['faces'] = sc.array(dims=['face'], values=[0, 3, 6, 9], unit=None)
    if detector_faces:
        dg['detector_faces'] = sc.array(
            dims=['face', 'face_index|detector_number'],
            values=[[0, 1], [1, 2], [2, 3], [3, 4]],
            unit=None,
        )
    for name, var in dg.items():
        off[name] = var
    dg['vertices'] = sc.vectors(dims=['vertex'], values=values, unit='m')
    return dg


@pytest.mark.parametrize(
    'detid_name', ['detector_number', 'pixel_id', 'spectrum_index']
)
def test_loads_data_with_coords_and_off_geometry(nxroot, detid_name):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]])
    )
    da.coords['detector_number'] = detector_numbers_xx_yy_1234()
    da.coords['xx'] = sc.array(dims=['xx'], unit='m', values=[0.1, 0.2])
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field(detid_name, da.coords['detector_number'])
    detector.create_field('xx', da.coords['xx'])
    detector.create_field('data', da.data)
    detector._group.attrs['axes'] = ['xx', 'yy']
    expected = create_off_geometry_detector_numbers_1234(detector, name='shape')
    loaded = detector[...]
    assert_identical(loaded.coords['shape'].value, expected)


def test_missing_detector_numbers_given_off_geometry_with_det_faces_loads_as_usual(
    nxroot,
):
    var = sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]])
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('data', var)
    detector._group.attrs['axes'] = ['xx', 'yy']
    expected = create_off_geometry_detector_numbers_1234(detector, name='shape')
    loaded = detector[...]
    assert_identical(loaded.coords['shape'].value, expected)


def test_off_geometry_without_detector_faces_loaded_as_0d_with_multiple_faces(nxroot):
    var = sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]])
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('data', var)
    detector._group.attrs['axes'] = ['xx', 'yy']
    expected = create_off_geometry_detector_numbers_1234(
        detector, name='shape', detector_faces=False
    )
    loaded = detector[...]
    assert_identical(loaded.coords['shape'].value, expected)
    shape = snx.NXoff_geometry.assemble_as_child(loaded.coords['shape'].value)
    assert sc.identical(shape.bins.size(), sc.index(4))


def create_cylindrical_geometry_detector_numbers_1234(
    group: snx.Group, name: str, detector_numbers: bool = True
) -> sc.DataGroup:
    shape = group.create_class(name, snx.NXcylindrical_geometry)
    values = np.array([[0, 0, 0], [0, 1, 0], [3, 0, 0]])
    dg = sc.DataGroup()
    dg['vertices'] = sc.array(dims=['_', 'comp'], values=values, unit='m')
    dg['cylinders'] = sc.array(
        dims=['cylinder', 'vertex_index'], values=[[0, 1, 2], [2, 1, 0]], unit=None
    )
    if detector_numbers:
        dg['detector_number'] = sc.array(
            dims=['detector_number'], values=[0, 1, 1, 0], unit=None
        )
    for name, var in dg.items():
        shape[name] = var
    dg['vertices'] = sc.vectors(dims=['vertex'], values=values, unit='m')
    return dg


def test_cylindrical_geometry_without_detector_numbers_loaded_as_0d(nxroot):
    var = sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]])
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('data', var)
    detector._group.attrs['axes'] = ['xx', 'yy']
    expected = create_cylindrical_geometry_detector_numbers_1234(
        detector, name='shape', detector_numbers=False
    )
    loaded = detector[...]
    assert_identical(loaded.coords['shape'].value, expected)
    shape = snx.NXcylindrical_geometry.assemble_as_child(loaded.coords['shape'].value)
    assert shape.dims == ()
    assert sc.identical(shape.bins.size(), sc.index(2))
    assert sc.identical(
        shape.value,
        sc.Dataset(
            {
                'face1_center': sc.vectors(
                    dims=['cylinder'], values=[[0, 0, 0], [3, 0, 0]], unit='m'
                ),
                'face1_edge': sc.vectors(
                    dims=['cylinder'], values=[[0, 1, 0], [0, 1, 0]], unit='m'
                ),
                'face2_center': sc.vectors(
                    dims=['cylinder'], values=[[3, 0, 0], [0, 0, 0]], unit='m'
                ),
            }
        ),
    )


def test_cylindrical_geometry_with_missing_parent_detector_numbers_loads_as_usual(
    nxroot,
):
    var = sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]])
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('data', var)
    detector._group.attrs['axes'] = ['xx', 'yy']
    expected = create_cylindrical_geometry_detector_numbers_1234(
        detector, name='shape', detector_numbers=True
    )
    loaded = detector[...]
    assert_identical(loaded.coords['shape'].value, expected)


def test_cylindrical_geometry_with_inconsistent_detector_numbers_loads_as_usual(nxroot):
    var = sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1], [3.3]])
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('data', var)
    detector._group.attrs['axes'] = ['xx', 'yy']
    detector.create_field(
        'detector_number', sc.array(dims=var.dims, values=[[1], [2]], unit=None)
    )
    expected = create_cylindrical_geometry_detector_numbers_1234(
        detector, name='shape', detector_numbers=True
    )
    loaded = detector[...]
    assert_identical(loaded.coords['shape'].value, expected)
    detector_number = loaded.coords['detector_number']
    with pytest.raises(snx.NexusStructureError):
        snx.NXcylindrical_geometry.assemble_as_child(
            loaded.coords['shape'].value, detector_number=detector_number
        )


def test_cylindrical_geometry_with_detector_numbers(nxroot):
    var = sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]])
    detector = nxroot.create_class('detector0', NXdetector)
    detector.create_field('data', var)
    detector._group.attrs['axes'] = ['xx', 'yy']
    detector_number = sc.array(dims=var.dims, values=[[1, 2], [3, 4]], unit=None)
    detector.create_field('detector_number', detector_number)
    expected = create_cylindrical_geometry_detector_numbers_1234(
        detector, name='shape', detector_numbers=True
    )
    loaded = detector[...]
    assert_identical(loaded.coords['shape'].value, expected)
    shape = snx.NXcylindrical_geometry.assemble_as_child(
        loaded.coords['shape'].value, detector_number=loaded.coords['detector_number']
    )
    assert shape.dims == detector_number.dims
    for i in [0, 3]:
        assert sc.identical(
            shape.values[i],
            sc.Dataset(
                {
                    'face1_center': sc.vectors(
                        dims=['cylinder'], values=[[0, 0, 0]], unit='m'
                    ),
                    'face1_edge': sc.vectors(
                        dims=['cylinder'], values=[[0, 1, 0]], unit='m'
                    ),
                    'face2_center': sc.vectors(
                        dims=['cylinder'], values=[[3, 0, 0]], unit='m'
                    ),
                }
            ),
        )
    for i in [1, 2]:
        assert sc.identical(
            shape.values[i],
            sc.Dataset(
                {
                    'face1_center': sc.vectors(
                        dims=['cylinder'], values=[[3, 0, 0]], unit='m'
                    ),
                    'face1_edge': sc.vectors(
                        dims=['cylinder'], values=[[0, 1, 0]], unit='m'
                    ),
                    'face2_center': sc.vectors(
                        dims=['cylinder'], values=[[0, 0, 0]], unit='m'
                    ),
                }
            ),
        )


def test_falls_back_to_hdf5_dim_labels(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    xy = sc.array(dims=['x', 'y'], values=[[1, 2], [3, 4]])
    z = sc.array(dims=['z'], values=[1, 2, 3])
    dataset = detector.create_field('xy', xy)
    dataset.dims[0].label = 'x'
    dataset.dims[1].label = 'y'
    dataset = detector.create_field('z', z)
    dataset.dims[0].label = 'z'
    assert detector.sizes == {'x': 2, 'y': 2, 'z': 3}
    dg = detector[()]
    assert_identical(dg['xy'], xy)
    assert_identical(dg['z'], z)


def test_falls_back_to_partial_hdf5_dim_labels(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    xyz = sc.ones(dims=['x', 'dim_1', 'z'], shape=(2, 2, 3))
    dataset = detector.create_field('xyz', xyz)
    dataset.dims[0].label = 'x'
    dataset.dims[2].label = 'z'
    assert detector.sizes == xyz.sizes
    dg = detector[()]
    assert_identical(dg['xyz'], xyz)


def test_squeezes_trailing_when_fall_back_to_partial_hdf5_dim_labels(nxroot):
    detector = nxroot.create_class('detector0', NXdetector)
    x = sc.ones(dims=['x', 'dim_1'], shape=(2, 1))
    dataset = detector.create_field('x', x)
    dataset.dims[0].label = 'x'
    assert detector.sizes == {'x': 2}
    dg = detector[()]
    assert_identical(dg['x'], sc.squeeze(x))


def test_falls_back_to_hdf5_dim_labels_given_unnamed_axes(h5root):
    xy = sc.array(dims=['x', 'y'], values=[[1, 2], [3, 4]])
    z = sc.array(dims=['z'], values=[1, 2, 3])
    detector = snx.create_class(h5root, 'detector0', NXdetector)
    dataset = snx.create_field(detector, 'xy', xy)
    dataset.dims[0].label = 'x'
    dataset.dims[1].label = 'y'
    dataset = snx.create_field(detector, 'z', z)
    dataset.dims[0].label = 'z'
    detector.attrs['axes'] = ['.', '.', '.']
    detector.attrs['xy_indices'] = [0, 1]
    detector.attrs['z_indices'] = [2]
    detector = make_group(detector)
    assert detector.sizes == {'x': 2, 'y': 2, 'z': 3}
    dg = detector[()]
    assert_identical(dg['xy'], xy)
    assert_identical(dg['z'], z)


def test_falls_back_to_hdf5_dim_labels_given_partially_axes(h5root):
    xy = sc.array(dims=['x', 'yy'], values=[[1, 2], [3, 4]])
    z = sc.array(dims=['zz'], values=[1, 2, 3])
    detector = snx.create_class(h5root, 'detector0', NXdetector)
    dataset = snx.create_field(detector, 'xy', xy)
    dataset.dims[0].label = 'x'
    dataset.dims[1].label = 'y'
    dataset = snx.create_field(detector, 'z', z)
    dataset.dims[0].label = 'z'
    detector.attrs['axes'] = ['.', 'yy', 'zz']
    detector.attrs['xy_indices'] = [0, 1]
    detector.attrs['z_indices'] = [2]
    detector = make_group(detector)
    assert detector.sizes == {'x': 2, 'yy': 2, 'zz': 3}
    dg = detector[()]
    assert_identical(dg['xy'], xy)
    assert_identical(dg['z'], z)


@pytest.mark.parametrize('dtype', ('bool', 'int8', 'int16', 'int32', 'int64'))
def test_pixel_masks_interpreted_correctly(dtype):
    bitmask = sc.array(
        dims=['detector_pixel'],
        values=(
            # Set mask 0, 1, 2, and 4 but not 3.
            1 << np.array([0, 1, 2, -1, 4])
            if dtype != 'bool'
            else
            # Set mask 0 only.
            np.array([1, 0, 0, 0, 0])
        )
        .astype(dtype)
        .astype('int64'),
        dtype='int64',
    )
    masks = snx.NXdetector.transform_bitmask_to_dict_of_masks(bitmask)

    assert np.all(masks.get('gap').values == np.array([1, 0, 0, 0, 0]))
    # A 'boolean' bitmask can only define one mask
    if dtype != 'bool':
        assert np.all(masks.get('dead').values == np.array([0, 1, 0, 0, 0]))
        assert np.all(masks.get('under_responding').values == np.array([0, 0, 1, 0, 0]))
        assert np.all(masks.get('noisy').values == np.array([0, 0, 0, 0, 1]))
        assert 'virtual_pixel' not in masks
        assert 'user_defined_pixel' not in masks


def test_pixel_masks_adds_suffix():
    bitmask = sc.array(
        dims=['detector_pixel'],
        values=(1 << np.array([0, 1, 2, -1, 4])),
        dtype='int32',
    )
    masks = snx.NXdetector.transform_bitmask_to_dict_of_masks(bitmask, '_test')
    assert len(masks) == 4
    assert all(k.endswith('_test') for k in masks.keys())


def test_pixel_masks_undefined_are_included():
    bitmask = sc.array(
        dims=['detector_pixel'],
        values=1 << np.array([9, 0]),
        dtype='int32',
    )
    masks = snx.NXdetector.transform_bitmask_to_dict_of_masks(bitmask)
    assert np.all(masks.get('undefined_bit9').values == np.array([1, 0]))
