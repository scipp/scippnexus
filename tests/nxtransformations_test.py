import h5py
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

import scippnexus.v2 as snx
from scippnexus.v2.nxtransformations import NXtransformations


def make_group(group: h5py.Group) -> snx.Group:
    return snx.Group(group, definitions=snx.base_definitions)


@pytest.fixture()
def h5root(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        yield f


def create_detector(group):
    data = sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]])
    detector_numbers = sc.array(dims=['xx', 'yy'],
                                unit=None,
                                values=np.array([[1, 2], [3, 4]]))
    detector = snx.create_class(group, 'detector_0', snx.NXdetector)
    snx.create_field(detector, 'detector_number', detector_numbers)
    snx.create_field(detector, 'data', data)
    return detector


def test_Transformation_with_single_value(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on',
                     sc.scalar('/detector_0/transformations/t1'))
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    value = sc.scalar(6.5, unit='mm')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='mm')
    vector = sc.vector(value=[0, 0, 1])
    t = value * vector
    expected = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    expected = expected * offset
    value = snx.create_field(transformations, 't1', value)
    value.attrs['depends_on'] = '.'
    value.attrs['transformation_type'] = 'translation'
    value.attrs['offset'] = offset.values
    value.attrs['offset_units'] = str(offset.unit)
    value.attrs['vector'] = vector.value

    depends_on = make_group(detector)['depends_on'][()]
    t = make_group(h5root)[depends_on]
    assert t.depends_on is None
    assert sc.identical(t.offset, offset)
    assert sc.identical(t.vector, vector)
    assert_identical(t[()], expected)


def test_chain_with_single_values_and_different_unit(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on',
                     sc.scalar('/detector_0/transformations/t1'))
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    value = sc.scalar(6.5, unit='mm')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='mm')
    vector = sc.vector(value=[0, 0, 1])
    t = value * vector
    value1 = snx.create_field(transformations, 't1', value)
    value1.attrs['depends_on'] = 't2'
    value1.attrs['transformation_type'] = 'translation'
    value1.attrs['offset'] = offset.values
    value1.attrs['offset_units'] = str(offset.unit)
    value1.attrs['vector'] = vector.value
    value2 = snx.create_field(transformations, 't2', value.to(unit='cm'))
    value2.attrs['depends_on'] = '.'
    value2.attrs['transformation_type'] = 'translation'
    value2.attrs['vector'] = vector.value

    t1 = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) * offset
    t2 = sc.spatial.translations(dims=t.dims, values=t.values,
                                 unit=t.unit).to(unit='cm')
    detector = make_group(h5root['detector_0'])
    depends_on = detector[...].coords['depends_on']
    assert_identical(depends_on.value.data, t1)
    assert_identical(depends_on.value.attrs['depends_on'].value, t2)


def test_Transformation_with_multiple_values(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on',
                     sc.scalar('/detector_0/transformations/t1'))
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')})
    log.coords['time'] = sc.epoch(unit='ns') + log.coords['time'].to(unit='ns')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='m')
    vector = sc.vector(value=[0, 0, 1])
    t = log * vector
    t.data = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    expected = t * offset
    value = snx.create_class(transformations, 't1', snx.NXlog)
    snx.create_field(value, 'time', log.coords['time'] - sc.epoch(unit='ns'))
    snx.create_field(value, 'value', log.data)
    value.attrs['depends_on'] = '.'
    value.attrs['transformation_type'] = 'translation'
    value.attrs['offset'] = offset.values
    value.attrs['offset_units'] = str(offset.unit)
    value.attrs['vector'] = vector.value

    depends_on = make_group(detector)['depends_on'][()]
    t = make_group(h5root)[depends_on]
    assert t.depends_on is None
    assert sc.identical(t.offset, offset)
    assert sc.identical(t.vector, vector)
    assert_identical(t[()], expected)


def test_chain_with_multiple_values(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on',
                     sc.scalar('/detector_0/transformations/t1'))
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')})
    log.coords['time'] = sc.epoch(unit='ns') + log.coords['time'].to(unit='ns')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='m')
    vector = sc.vector(value=[0, 0, 1])
    t = log * vector
    t.data = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    value1 = snx.create_class(transformations, 't1', snx.NXlog)
    snx.create_field(value1, 'time', log.coords['time'] - sc.epoch(unit='ns'))
    snx.create_field(value1, 'value', log.data)
    value1.attrs['depends_on'] = 't2'
    value1.attrs['transformation_type'] = 'translation'
    value1.attrs['offset'] = offset.values
    value1.attrs['offset_units'] = str(offset.unit)
    value1.attrs['vector'] = vector.value
    value2 = snx.create_class(transformations, 't2', snx.NXlog)
    snx.create_field(value2, 'time', log.coords['time'] - sc.epoch(unit='ns'))
    snx.create_field(value2, 'value', log.data)
    value2.attrs['depends_on'] = '.'
    value2.attrs['transformation_type'] = 'translation'
    value2.attrs['vector'] = vector.value

    expected = t * offset
    expected.attrs['depends_on'] = sc.scalar(t)
    detector = make_group(detector)
    depends_on = detector[...].coords['depends_on']
    assert sc.identical(depends_on.value, expected)


def test_chain_with_multiple_values_and_different_time_unit(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on',
                     sc.scalar('/detector_0/transformations/t1'))
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    # Making sure to not use nanoseconds since that is used internally and may thus
    # mask bugs.
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')})
    log.coords['time'] = sc.epoch(unit='us') + log.coords['time'].to(unit='us')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='m')
    vector = sc.vector(value=[0, 0, 1])
    t = log * vector
    t.data = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    value1 = snx.create_class(transformations, 't1', snx.NXlog)
    snx.create_field(value1, 'time', log.coords['time'] - sc.epoch(unit='us'))
    snx.create_field(value1, 'value', log.data)
    value1.attrs['depends_on'] = 't2'
    value1.attrs['transformation_type'] = 'translation'
    value1.attrs['offset'] = offset.values
    value1.attrs['offset_units'] = str(offset.unit)
    value1.attrs['vector'] = vector.value
    value2 = snx.create_class(transformations, 't2', snx.NXlog)
    snx.create_field(value2, 'time',
                     log.coords['time'].to(unit='ms') - sc.epoch(unit='ms'))
    snx.create_field(value2, 'value', log.data)
    value2.attrs['depends_on'] = '.'
    value2.attrs['transformation_type'] = 'translation'
    value2.attrs['vector'] = vector.value

    expected = t * offset
    t2 = t.copy()
    t2.coords['time'] = t2.coords['time'].to(unit='ms')
    expected.attrs['depends_on'] = sc.scalar(t2)
    detector = make_group(detector)
    depends_on = detector[...].coords['depends_on']
    assert sc.identical(depends_on.value, expected)


def test_broken_time_dependent_transformation_returns_path_and_transformations(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on',
                     sc.scalar('/detector_0/transformations/t1'))
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')})
    log.coords['time'] = sc.epoch(unit='ns') + log.coords['time'].to(unit='ns')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='m')
    vector = sc.vector(value=[0, 0, 1])
    t = log * vector
    t.data = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    value = snx.create_class(transformations, 't1', snx.NXlog)
    snx.create_field(value, 'time', log.coords['time'] - sc.epoch(unit='ns'))
    # This makes the transform "broken" since "time" has length 2 but data has length 0.
    snx.create_field(value, 'value', log.data[0:0])
    value.attrs['depends_on'] = '.'
    value.attrs['transformation_type'] = 'translation'
    value.attrs['offset'] = offset.values
    value.attrs['offset_units'] = str(offset.unit)
    value.attrs['vector'] = vector.value

    detector = make_group(detector)
    loaded = detector[()]
    assert sc.identical(loaded.coords['depends_on'],
                        sc.scalar('/detector_0/transformations/t1'))
    t = loaded.coords['transformations'].value
    assert isinstance(t, sc.DataGroup)
    # Due to the way NXtransformations works, vital information is stored in the
    # attributes. DataGroup does currently not support attributes, so this information
    # is mostly useless until that is addressed.
    assert 't1' in t


def write_translation(group, name: str, value: sc.Variable, offset: sc.Variable,
                      vector: sc.Variable) -> None:
    dset = snx.create_field(group, name, value)
    dset.attrs['transformation_type'] = 'translation'
    dset.attrs['offset'] = offset.values
    dset.attrs['offset_units'] = str(offset.unit)
    dset.attrs['vector'] = vector.value


def test_nxtransformations_group_single_item(h5root):
    value = sc.scalar(2.4, unit='mm')
    offset = sc.spatial.translation(value=[6, 2, 6], unit='mm')
    vector = sc.vector(value=[0, 1, 1])
    t = value * vector
    expected = (sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) *
                offset)

    transformations = snx.create_class(h5root, 'transformations', NXtransformations)
    write_translation(transformations, 't1', value, offset, vector)

    loaded = make_group(h5root)['transformations'][()]
    assert set(loaded.keys()) == {'t1'}
    assert sc.identical(loaded['t1'], expected)


def test_nxtransformations_group_two_independent_items(h5root):
    transformations = snx.create_class(h5root, 'transformations', NXtransformations)

    value = sc.scalar(2.4, unit='mm')
    offset = sc.spatial.translation(value=[6, 2, 6], unit='mm')
    vector = sc.vector(value=[0, 1, 1])
    t = value * vector
    write_translation(transformations, 't1', value, offset, vector)
    expected1 = (sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) *
                 offset)

    value = value * 0.1
    t = value * vector
    write_translation(transformations, 't2', value, offset, vector)
    expected2 = (sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) *
                 offset)

    loaded = make_group(h5root)['transformations'][()]
    assert set(loaded.keys()) == {'t1', 't2'}
    assert sc.identical(loaded['t1'], expected1)
    assert sc.identical(loaded['t2'], expected2)


def test_nxtransformations_group_single_chain(h5root):
    transformations = snx.create_class(h5root, 'transformations', NXtransformations)

    value = sc.scalar(2.4, unit='mm')
    offset = sc.spatial.translation(value=[6, 2, 6], unit='mm')
    vector = sc.vector(value=[0, 1, 1])
    t = value * vector
    write_translation(transformations, 't1', value, offset, vector)
    expected1 = (sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) *
                 offset)

    value = value * 0.1
    t = value * vector
    write_translation(transformations, 't2', value, offset, vector)
    transformations['t2'].attrs['depends_on'] = 't1'
    expected2 = (sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) *
                 offset)

    loaded = make_group(h5root)['transformations'][()]
    assert set(loaded.keys()) == {'t1', 't2'}
    assert_identical(loaded['t1'], expected1)
    assert_identical(loaded['t2'].data, expected2)
    assert_identical(loaded['t2'].attrs['depends_on'].value, expected1)


def test_slice_transformations(h5root):
    transformations = snx.create_class(h5root, 'transformations', NXtransformations)
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2, 3.3], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22, 33], unit='s')})
    log.coords['time'] = sc.epoch(unit='ns') + log.coords['time'].to(unit='ns')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='m')
    vector = sc.vector(value=[0, 0, 1])
    t = log * vector
    t.data = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    value1 = snx.create_class(transformations, 't1', snx.NXlog)
    snx.create_field(value1, 'time', log.coords['time'] - sc.epoch(unit='ns'))
    snx.create_field(value1, 'value', log.data)
    value1.attrs['transformation_type'] = 'translation'
    value1.attrs['offset'] = offset.values
    value1.attrs['offset_units'] = str(offset.unit)
    value1.attrs['vector'] = vector.value

    expected = t * offset

    assert sc.identical(
        make_group(h5root)['transformations']['time', 1:3]['t1'], expected['time', 1:3])
