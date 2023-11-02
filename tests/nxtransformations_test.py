import h5py
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

import scippnexus as snx
from scippnexus.nxtransformations import NXtransformations, TransformationChainResolver


def make_group(group: h5py.Group) -> snx.Group:
    return snx.Group(group, definitions=snx.base_definitions())


@pytest.fixture()
def h5root():
    """Yield h5py root group (file)"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        yield f


def create_detector(group):
    data = sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]])
    detector_numbers = sc.array(
        dims=['xx', 'yy'], unit=None, values=np.array([[1, 2], [3, 4]])
    )
    detector = snx.create_class(group, 'detector_0', snx.NXdetector)
    snx.create_field(detector, 'detector_number', detector_numbers)
    snx.create_field(detector, 'data', data)
    return detector


def test_Transformation_with_single_value(h5root):
    detector = create_detector(h5root)
    snx.create_field(
        detector, 'depends_on', sc.scalar('/detector_0/transformations/t1')
    )
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

    expected = sc.DataArray(data=expected, coords={'depends_on': sc.scalar('.')})
    detector = make_group(detector)
    depends_on = detector['depends_on'][()]
    assert depends_on == 'transformations/t1'
    t = detector[depends_on][()]
    assert_identical(t, expected)


def test_time_independent_Transformation_with_length_0(h5root):
    detector = create_detector(h5root)
    snx.create_field(
        detector, 'depends_on', sc.scalar('/detector_0/transformations/t1')
    )
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    value = sc.array(dims=['dim_0'], values=[], unit='mm')
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

    expected = sc.DataArray(data=expected, coords={'depends_on': sc.scalar('.')})
    detector = make_group(detector)
    depends_on = detector['depends_on'][()]
    assert depends_on == 'transformations/t1'
    t = detector[depends_on][()]
    assert_identical(t, expected)


def test_depends_on_absolute_path_to_sibling_group_resolved_to_relative_path(h5root):
    det1 = snx.create_class(h5root, 'det1', NXtransformations)
    snx.create_field(det1, 'depends_on', sc.scalar('/det2/transformations/t1'))

    depends_on = make_group(det1)['depends_on'][()]
    assert depends_on == '../det2/transformations/t1'


def test_depends_on_relative_path_unchanged(h5root):
    det1 = snx.create_class(h5root, 'det1', NXtransformations)
    snx.create_field(det1, 'depends_on', sc.scalar('transformations/t1'))

    depends_on = make_group(det1)['depends_on'][()]
    assert depends_on == 'transformations/t1'


def test_depends_on_attr_absolute_path_to_sibling_group_resolved_to_relative_path(
    h5root,
):
    det1 = snx.create_class(h5root, 'det1', NXtransformations)
    transformations = snx.create_class(det1, 'transformations', NXtransformations)
    t1 = snx.create_field(transformations, 't1', sc.scalar(0.1, unit='cm'))
    t1.attrs['depends_on'] = '/det2/transformations/t2'
    t1.attrs['transformation_type'] = 'translation'
    t1.attrs['vector'] = [0, 0, 1]

    loaded = make_group(det1)['transformations/t1'][()]
    assert loaded.coords['depends_on'].value == '../../det2/transformations/t2'


def test_depends_on_attr_relative_path_unchanged(h5root):
    det = snx.create_class(h5root, 'det', NXtransformations)
    transformations = snx.create_class(det, 'transformations', NXtransformations)
    t1 = snx.create_field(transformations, 't1', sc.scalar(0.1, unit='cm'))
    t1.attrs['depends_on'] = '.'
    t1.attrs['transformation_type'] = 'translation'
    t1.attrs['vector'] = [0, 0, 1]

    loaded = make_group(det)['transformations/t1'][()]
    assert loaded.coords['depends_on'].value == '.'
    t1.attrs['depends_on'] = 't2'
    loaded = make_group(det)['transformations/t1'][()]
    assert loaded.coords['depends_on'].value == 't2'


def test_chain_with_single_values_and_different_unit(h5root):
    detector = create_detector(h5root)
    snx.create_field(
        detector, 'depends_on', sc.scalar('/detector_0/transformations/t1')
    )
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
    t2 = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit).to(
        unit='cm'
    )
    detector = make_group(h5root['detector_0'])
    loaded = detector[()]
    depends_on = loaded['depends_on']
    assert depends_on == 'transformations/t1'
    transforms = loaded['transformations']
    assert_identical(transforms['t1'].data, t1)
    assert transforms['t1'].coords['depends_on'].value == 't2'
    assert_identical(transforms['t2'].data, t2)
    assert transforms['t2'].coords['depends_on'].value == '.'


def test_Transformation_with_multiple_values(h5root):
    detector = create_detector(h5root)
    snx.create_field(
        detector, 'depends_on', sc.scalar('/detector_0/transformations/t1')
    )
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')},
    )
    log.coords['time'] = sc.epoch(unit='ns') + log.coords['time'].to(unit='ns')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='m')
    vector = sc.vector(value=[0, 0, 1])
    t = log * vector
    t.data = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    value = snx.create_class(transformations, 't1', snx.NXlog)
    snx.create_field(value, 'time', log.coords['time'] - sc.epoch(unit='ns'))
    snx.create_field(value, 'value', log.data)
    value.attrs['depends_on'] = '.'
    value.attrs['transformation_type'] = 'translation'
    value.attrs['offset'] = offset.values
    value.attrs['offset_units'] = str(offset.unit)
    value.attrs['vector'] = vector.value

    expected = t * offset
    expected.coords['depends_on'] = sc.scalar('.')
    detector = make_group(detector)
    depends_on = detector['depends_on'][()]
    assert depends_on == 'transformations/t1'
    assert_identical(detector[depends_on][()], expected)


def test_chain_with_multiple_values(h5root):
    detector = create_detector(h5root)
    snx.create_field(
        detector, 'depends_on', sc.scalar('/detector_0/transformations/t1')
    )
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')},
    )
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

    expected1 = t * offset
    expected1.coords['depends_on'] = sc.scalar('t2')
    expected2 = t
    expected2.coords['depends_on'] = sc.scalar('.')
    detector = make_group(detector)[()]
    depends_on = detector['depends_on']
    assert depends_on == 'transformations/t1'
    assert_identical(detector['transformations']['t1'], expected1)
    assert_identical(detector['transformations']['t2'], expected2)


def test_chain_with_multiple_values_and_different_time_unit(h5root):
    detector = create_detector(h5root)
    snx.create_field(
        detector, 'depends_on', sc.scalar('/detector_0/transformations/t1')
    )
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    # Making sure to not use nanoseconds since that is used internally and may thus
    # mask bugs.
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')},
    )
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
    snx.create_field(
        value2, 'time', log.coords['time'].to(unit='ms') - sc.epoch(unit='ms')
    )
    snx.create_field(value2, 'value', log.data)
    value2.attrs['depends_on'] = '.'
    value2.attrs['transformation_type'] = 'translation'
    value2.attrs['vector'] = vector.value

    expected1 = t * offset
    expected1.coords['depends_on'] = sc.scalar('t2')

    t2 = t.copy()
    t2.coords['time'] = t2.coords['time'].to(unit='ms')
    expected2 = t2
    expected2.coords['depends_on'] = sc.scalar('.')

    detector = make_group(detector)
    loaded = detector[...]
    depends_on = loaded['depends_on']
    assert depends_on == 'transformations/t1'
    assert_identical(loaded['transformations']['t1'], expected1)
    assert_identical(loaded['transformations']['t2'], expected2)


def test_broken_time_dependent_transformation_returns_datagroup_but_sets_up_depends_on(
    h5root,
):
    detector = create_detector(h5root)
    snx.create_field(
        detector, 'depends_on', sc.scalar('/detector_0/transformations/t1')
    )
    transformations = snx.create_class(detector, 'transformations', NXtransformations)
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')},
    )
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
    t = loaded['transformations']
    assert isinstance(t, sc.DataGroup)
    # Due to the way NXtransformations works, vital information is stored in the
    # attributes. DataGroup does currently not support attributes, so this information
    # is mostly useless until that is addressed.
    t1 = t['t1']
    assert isinstance(t1, sc.DataGroup)
    assert t1.keys() == {'time', 'value'}
    assert loaded['depends_on'] == 'transformations/t1'
    assert_identical(loaded['transformations']['t1'], t1)


def write_translation(
    group, name: str, value: sc.Variable, offset: sc.Variable, vector: sc.Variable
) -> None:
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
    expected = (
        sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) * offset
    )

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
    expected1 = (
        sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) * offset
    )

    value = value * 0.1
    t = value * vector
    write_translation(transformations, 't2', value, offset, vector)
    expected2 = (
        sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) * offset
    )

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
    expected1 = (
        sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) * offset
    )

    value = value * 0.1
    t = value * vector
    write_translation(transformations, 't2', value, offset, vector)
    transformations['t2'].attrs['depends_on'] = 't1'
    expected2 = (
        sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) * offset
    )

    loaded = make_group(h5root)['transformations'][()]
    assert set(loaded.keys()) == {'t1', 't2'}
    assert_identical(loaded['t1'], expected1)
    assert_identical(loaded['t2'].data, expected2)
    assert loaded['t2'].coords['depends_on'].value == 't1'


def test_slice_transformations(h5root):
    transformations = snx.create_class(h5root, 'transformations', NXtransformations)
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2, 3.3], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22, 33], unit='s')},
    )
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
        make_group(h5root)['transformations']['time', 1:3]['t1'], expected['time', 1:3]
    )


def test_TransformationChainResolver_path_handling():
    tree = TransformationChainResolver([{'a': {'b': {'c': 1}}}])
    assert tree['a']['b']['c'].value == 1
    assert tree['a/b/c'].value == 1
    assert tree['/a/b/c'].value == 1
    assert tree['a']['../a/b/c'].value == 1
    assert tree['a/b']['../../a/b/c'].value == 1
    assert tree['a/b']['./c'].value == 1


origin = sc.vector([0, 0, 0], unit='m')
shiftX = sc.spatial.translation(value=[1, 0, 0], unit='m')
rotZ = sc.spatial.rotations_from_rotvecs(sc.vector([0, 0, 90], unit='deg'))


def test_resolve_depends_on_dot():
    tree = TransformationChainResolver([{'depends_on': '.'}])
    assert sc.identical(tree.resolve_depends_on(), origin)


def test_resolve_depends_on_child():
    transform = sc.DataArray(shiftX, coords={'depends_on': sc.scalar('.')})
    tree = TransformationChainResolver([{'depends_on': 'child', 'child': transform}])
    expected = sc.vector([1, 0, 0], unit='m')
    assert sc.identical(tree.resolve_depends_on(), expected)


def test_resolve_depends_on_grandchild():
    transform = sc.DataArray(shiftX, coords={'depends_on': sc.scalar('.')})
    tree = TransformationChainResolver(
        [{'depends_on': 'child/grandchild', 'child': {'grandchild': transform}}]
    )
    expected = sc.vector([1, 0, 0], unit='m')
    assert sc.identical(tree.resolve_depends_on(), expected)


def test_resolve_depends_on_child1_depends_on_child2():
    transform1 = sc.DataArray(shiftX, coords={'depends_on': sc.scalar('child2')})
    transform2 = sc.DataArray(rotZ, coords={'depends_on': sc.scalar('.')})
    tree = TransformationChainResolver(
        [{'depends_on': 'child1', 'child1': transform1, 'child2': transform2}]
    )
    # Note order
    expected = transform2.data * transform1.data * origin
    assert sc.identical(tree.resolve_depends_on(), expected)


def test_resolve_depends_on_grandchild1_depends_on_grandchild2():
    transform1 = sc.DataArray(shiftX, coords={'depends_on': sc.scalar('grandchild2')})
    transform2 = sc.DataArray(rotZ, coords={'depends_on': sc.scalar('.')})
    tree = TransformationChainResolver(
        [
            {
                'depends_on': 'child/grandchild1',
                'child': {'grandchild1': transform1, 'grandchild2': transform2},
            }
        ]
    )
    expected = transform2.data * transform1.data * origin
    assert sc.identical(tree.resolve_depends_on(), expected)


def test_resolve_depends_on_grandchild1_depends_on_child2():
    transform1 = sc.DataArray(shiftX, coords={'depends_on': sc.scalar('../child2')})
    transform2 = sc.DataArray(rotZ, coords={'depends_on': sc.scalar('.')})
    tree = TransformationChainResolver(
        [
            {
                'depends_on': 'child1/grandchild1',
                'child1': {'grandchild1': transform1},
                'child2': transform2,
            }
        ]
    )
    expected = transform2.data * transform1.data * origin
    assert sc.identical(tree.resolve_depends_on(), expected)


def test_compute_positions(h5root):
    instrument = snx.create_class(h5root, 'instrument', snx.NXinstrument)
    detector = create_detector(instrument)
    snx.create_field(detector, 'x_pixel_offset', sc.linspace('xx', -1, 1, 2, unit='m'))
    snx.create_field(detector, 'y_pixel_offset', sc.linspace('yy', -1, 1, 2, unit='m'))
    detector.attrs['axes'] = ['xx', 'yy']
    detector.attrs['x_pixel_offset_indices'] = [0]
    detector.attrs['y_pixel_offset_indices'] = [1]
    snx.create_field(
        detector, 'depends_on', sc.scalar('/instrument/detector_0/transformations/t1')
    )
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
    t2 = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit).to(
        unit='cm'
    )
    root = make_group(h5root)
    loaded = root[()]
    result = snx.compute_positions(loaded)
    origin = sc.vector([0, 0, 0], unit='m')
    assert_identical(
        result['instrument']['detector_0']['position'],
        t2.to(unit='m') * t1.to(unit='m') * origin,
    )
    assert_identical(
        result['instrument']['detector_0']['data'].coords['position'],
        t2.to(unit='m') * t1.to(unit='m') * origin
        + sc.vectors(
            dims=['xx', 'yy'],
            values=[[[-1, -1, 0], [-1, 1, 0]], [[1, -1, 0], [1, 1, 0]]],
            unit='m',
        ),
    )
