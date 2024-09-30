import h5py
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

import scippnexus as snx
from scippnexus.nxtransformations import NXtransformations


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
    snx.create_field(detector, 'depends_on', sc.scalar('transformations/t1'))
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

    detector = make_group(detector)
    depends_on = detector['depends_on'][()].value
    assert depends_on == 'transformations/t1'
    t = detector[depends_on][()].build()
    assert_identical(t, expected)


def test_time_independent_Transformation_with_length_0(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on', sc.scalar('transformations/t1'))
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

    detector = make_group(detector)
    depends_on = detector['depends_on'][()].value
    assert depends_on == 'transformations/t1'
    t = detector[depends_on][()].build()
    assert_identical(t, expected)


def test_depends_on_absolute_path_to_sibling_group_resolved_correctly(h5root):
    det1 = snx.create_class(h5root, 'det1', NXtransformations)
    snx.create_field(det1, 'depends_on', sc.scalar('/det2/transformations/t1'))
    depends_on = make_group(det1)['depends_on'][()]
    assert depends_on.absolute_path() == '/det2/transformations/t1'


def test_depends_on_relative_path_to_sibling_group_resolved_correctly(h5root):
    det1 = snx.create_class(h5root, 'det1', NXtransformations)
    snx.create_field(det1, 'depends_on', sc.scalar('../det2/transformations/t1'))
    depends_on = make_group(det1)['depends_on'][()]
    assert depends_on.absolute_path() == '/det2/transformations/t1'


def test_depends_on_relative_path_unchanged(h5root):
    det1 = snx.create_class(h5root, 'det1', NXtransformations)
    snx.create_field(det1, 'depends_on', sc.scalar('transformations/t1'))

    depends_on = make_group(det1)['depends_on'][()]
    assert depends_on.value == 'transformations/t1'


def test_depends_on_attr_absolute_path_to_sibling_group_preserved(h5root):
    det1 = snx.create_class(h5root, 'det1', NXtransformations)
    transformations = snx.create_class(det1, 'transformations', NXtransformations)
    t1 = snx.create_field(transformations, 't1', sc.scalar(0.1, unit='cm'))
    t1.attrs['depends_on'] = '/det2/transformations/t2'
    t1.attrs['transformation_type'] = 'translation'
    t1.attrs['vector'] = [0, 0, 1]

    loaded = make_group(det1)['transformations/t1'][()]
    assert loaded.depends_on.value == '/det2/transformations/t2'


def test_depends_on_attr_relative_path_unchanged(h5root):
    det = snx.create_class(h5root, 'det', NXtransformations)
    transformations = snx.create_class(det, 'transformations', NXtransformations)
    t1 = snx.create_field(transformations, 't1', sc.scalar(0.1, unit='cm'))
    t1.attrs['depends_on'] = '.'
    t1.attrs['transformation_type'] = 'translation'
    t1.attrs['vector'] = [0, 0, 1]

    loaded = make_group(det)['transformations/t1'][()]
    assert loaded.depends_on.value == '.'
    t1.attrs['depends_on'] = 't2'
    loaded = make_group(det)['transformations/t1'][()]
    assert loaded.depends_on.value == 't2'


def test_chain_with_single_values_and_different_unit(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on', sc.scalar('transformations/t1'))
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
    assert depends_on.value == 'transformations/t1'
    transforms = loaded['transformations']
    assert_identical(transforms['t1'].build(), t1)
    assert transforms['t1'].depends_on.value == 't2'
    assert_identical(transforms['t2'].build(), t2)
    assert transforms['t2'].depends_on.value == '.'


def test_Transformation_with_multiple_values(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on', sc.scalar('transformations/t1'))
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
    detector = make_group(detector)
    depends_on = detector['depends_on'][()]
    assert depends_on.value == 'transformations/t1'
    assert_identical(detector[depends_on.absolute_path()][()].build(), expected)


def test_time_dependent_transform_uses_value_sublog(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on', sc.scalar('transformations/t1'))
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
    # Add alarms with shorter time axis. This will trigger loading as a DataGroup with
    # multiple contained DataArrays.
    alarm = log['time', :2].copy()
    alarm.coords['message'] = sc.array(dims=['time'], values=['alarm 1', 'alarm 2'])
    snx.create_field(value, 'alarm_severity', alarm.data)
    snx.create_field(value, 'alarm_message', alarm.coords['message'])
    snx.create_field(value, 'alarm_time', alarm.coords['time'])
    value.attrs['depends_on'] = '.'
    value.attrs['transformation_type'] = 'translation'
    value.attrs['offset'] = offset.values
    value.attrs['offset_units'] = str(offset.unit)
    value.attrs['vector'] = vector.value

    expected = t * offset
    detector = make_group(detector)
    depends_on = detector['depends_on'][()]
    assert depends_on.value == 'transformations/t1'
    assert_identical(detector[depends_on.absolute_path()][()].build(), expected)


def test_chain_with_multiple_values(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on', sc.scalar('transformations/t1'))
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
    expected2 = t
    detector = make_group(detector)[()]
    depends_on = detector['depends_on']
    assert depends_on.value == 'transformations/t1'
    assert_identical(detector['transformations']['t1'].build(), expected1)
    assert_identical(detector['transformations']['t2'].build(), expected2)


def test_chain_with_multiple_values_and_different_time_unit(h5root):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on', sc.scalar('transformations/t1'))
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

    t2 = t.copy()
    t2.coords['time'] = t2.coords['time'].to(unit='ms')
    expected2 = t2

    detector = make_group(detector)
    loaded = detector[...]
    depends_on = loaded['depends_on']
    assert depends_on.value == 'transformations/t1'
    assert_identical(loaded['transformations']['t1'].build(), expected1)
    assert_identical(loaded['transformations']['t2'].build(), expected2)


@pytest.mark.filterwarnings(
    "ignore:Failed to load /detector_0/transformations/t1:UserWarning"
)
def test_broken_time_dependent_transformation_returns_datagroup_but_sets_up_depends_on(
    h5root,
):
    detector = create_detector(h5root)
    snx.create_field(detector, 'depends_on', sc.scalar('transformations/t1'))
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
    t1 = t['t1'].value
    assert isinstance(t1, sc.DataGroup)
    assert t1.keys() == {'time', 'value'}
    assert loaded['depends_on'].value == 'transformations/t1'
    assert_identical(loaded['transformations']['t1'].value, t1)


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
    transformations['t1'].attrs['depends_on'] = '.'

    loaded = make_group(h5root)['transformations'][()]
    assert set(loaded.keys()) == {'t1'}
    assert sc.identical(loaded['t1'].build(), expected)


def test_nxtransformations_group_two_independent_items(h5root):
    transformations = snx.create_class(h5root, 'transformations', NXtransformations)

    value = sc.scalar(2.4, unit='mm')
    offset = sc.spatial.translation(value=[6, 2, 6], unit='mm')
    vector = sc.vector(value=[0, 1, 1])
    t = value * vector
    write_translation(transformations, 't1', value, offset, vector)
    transformations['t1'].attrs['depends_on'] = '.'
    expected1 = (
        sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) * offset
    )

    value = value * 0.1
    t = value * vector
    write_translation(transformations, 't2', value, offset, vector)
    transformations['t2'].attrs['depends_on'] = '.'
    expected2 = (
        sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit) * offset
    )

    loaded = make_group(h5root)['transformations'][()]
    assert set(loaded.keys()) == {'t1', 't2'}
    assert sc.identical(loaded['t1'].build(), expected1)
    assert sc.identical(loaded['t2'].build(), expected2)


def test_nxtransformations_group_single_chain(h5root):
    transformations = snx.create_class(h5root, 'transformations', NXtransformations)

    value = sc.scalar(2.4, unit='mm')
    offset = sc.spatial.translation(value=[6, 2, 6], unit='mm')
    vector = sc.vector(value=[0, 1, 1])
    t = value * vector
    write_translation(transformations, 't1', value, offset, vector)
    transformations['t1'].attrs['depends_on'] = '.'
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
    assert_identical(loaded['t1'].build(), expected1)
    assert_identical(loaded['t2'].build(), expected2)
    assert loaded['t2'].depends_on.value == 't1'


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
    value1.attrs['depends_on'] = '.'

    expected = t * offset

    assert sc.identical(
        make_group(h5root)['transformations']['time', 1:3]['t1'].build(),
        expected['time', 1:3],
    )


def test_label_slice_transformations(h5root):
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
    value1.attrs['depends_on'] = '.'

    expected = t * offset

    assert sc.identical(
        make_group(h5root)['transformations'][
            'time',
            sc.scalar(22, unit='s').to(unit='ns') : sc.scalar(44, unit='s').to(
                unit='ns'
            ),
        ]['t1'].build(),
        expected[
            'time',
            sc.datetime('1970-01-01T00:00:22', unit='ns') : sc.datetime(
                '1970-01-01T00:00:44', unit='ns'
            ),
        ],
    )


origin = sc.vector([0, 0, 0], unit='m')
shiftX = sc.spatial.translation(value=[1, 0, 0], unit='m')
rotZ = sc.spatial.rotations_from_rotvecs(sc.vector([0, 0, 90], unit='deg'))


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
        t2.to(unit='m')
        * t1.to(unit='m')
        * sc.vectors(
            dims=['xx', 'yy'],
            values=[[[-1, -1, 0], [-1, 1, 0]], [[1, -1, 0], [1, 1, 0]]],
            unit='m',
        ),
    )


def test_compute_positions_with_rotation(h5root):
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
    value = sc.scalar(90.0, unit='deg')
    vector = sc.vector(value=[0, 1, 0])
    rot = snx.create_field(transformations, 't1', value)
    rot.attrs['depends_on'] = '.'
    rot.attrs['transformation_type'] = 'rotation'
    rot.attrs['vector'] = vector.value

    transform = sc.spatial.rotations_from_rotvecs(sc.vector([0, 90, 0], unit='deg'))
    root = make_group(h5root)
    loaded = root[()]
    result = snx.compute_positions(loaded)
    origin = sc.vector([0, 0, 0], unit='m')
    assert_identical(result['instrument']['detector_0']['position'], origin)
    assert_identical(
        result['instrument']['detector_0']['data'].coords['position'],
        transform
        * sc.vectors(
            dims=['xx', 'yy'],
            values=[[[-1, -1, 0], [-1, 1, 0]], [[1, -1, 0], [1, 1, 0]]],
            unit='m',
        ),
    )


@pytest.mark.filterwarnings("ignore:Failed to load /instrument/monitor:UserWarning")
def test_compute_positions_works_for_path_beyond_root(h5root):
    instrument = snx.create_class(h5root, 'instrument', snx.NXinstrument)
    value = sc.scalar(6.5, unit='m')
    vector = sc.vector(value=[0, 0, 1])
    transform1 = snx.create_field(h5root, 't1', value)
    transform1.attrs['depends_on'] = '.'
    transform1.attrs['transformation_type'] = 'translation'
    transform1.attrs['vector'] = vector.value
    transform2 = snx.create_field(instrument, 't2', value)
    transform2.attrs['depends_on'] = '.'
    transform2.attrs['transformation_type'] = 'translation'
    transform2.attrs['vector'] = vector.value
    monitor1 = snx.create_class(instrument, 'monitor1', snx.NXmonitor)
    monitor2 = snx.create_class(instrument, 'monitor2', snx.NXmonitor)
    snx.create_field(monitor1, 'depends_on', '../../t1')
    snx.create_field(monitor2, 'depends_on', '../t2')
    root = make_group(h5root)
    loaded = root[()]
    assert 'position' in snx.compute_positions(loaded)['instrument']['monitor1']
    assert 'position' in snx.compute_positions(loaded)['instrument']['monitor2']
    assert 'position' in snx.compute_positions(loaded['instrument'])['monitor1']
    assert 'position' in snx.compute_positions(loaded['instrument'])['monitor2']


@pytest.mark.filterwarnings("ignore:Failed to load /instrument/monitor:UserWarning")
def test_path_beyond_root_is_fully_resolved_and_can_compute_positions(h5root):
    instrument = snx.create_class(h5root, 'instrument', snx.NXinstrument)
    monitor1 = snx.create_class(instrument, 'monitor1', snx.NXmonitor)
    monitor2 = snx.create_class(instrument, 'monitor2', snx.NXmonitor)
    value = sc.scalar(6.5, unit='m')
    vector = sc.vector(value=[0, 0, 1])
    transform1 = snx.create_field(monitor1, 't1', value)
    transform1.attrs['depends_on'] = '../t2'
    transform1.attrs['transformation_type'] = 'translation'
    transform1.attrs['vector'] = vector.value
    transform2 = snx.create_field(instrument, 't2', value)
    transform2.attrs['depends_on'] = 't3'
    transform2.attrs['transformation_type'] = 'translation'
    transform2.attrs['vector'] = vector.value
    transform3 = snx.create_field(instrument, 't3', value)
    transform3.attrs['depends_on'] = '.'
    transform3.attrs['transformation_type'] = 'translation'
    transform3.attrs['vector'] = vector.value
    # Chain start in current group
    snx.create_field(monitor1, 'depends_on', 't1')
    # Chain start outside current group
    snx.create_field(monitor2, 'depends_on', '../t2')
    root = make_group(h5root)
    mon1 = root['instrument/monitor1'][()]
    assert_identical(snx.compute_positions(mon1)['position'], 3 * value * vector)
    mon2 = root['instrument/monitor2'][()]
    assert_identical(snx.compute_positions(mon2)['position'], 2 * value * vector)


@pytest.mark.filterwarnings("ignore:Failed to load /instrument/monitor:UserWarning")
def test_compute_positions_returns_position_with_unit_meters(h5root):
    instrument = snx.create_class(h5root, 'instrument', snx.NXinstrument)
    value = sc.scalar(6.5, unit='cm')
    vector = sc.vector(value=[0, 0, 1])
    transform = snx.create_field(instrument, 't1', value)
    transform.attrs['depends_on'] = '.'
    transform.attrs['transformation_type'] = 'translation'
    transform.attrs['vector'] = vector.value
    monitor = snx.create_class(instrument, 'monitor', snx.NXmonitor)
    snx.create_field(monitor, 'depends_on', '../t1')
    root = make_group(h5root)
    loaded = root[()]
    mon = snx.compute_positions(loaded)['instrument']['monitor']
    assert mon['position'].unit == 'm'


@pytest.mark.filterwarnings("ignore:Failed to load /instrument/monitor:UserWarning")
def test_compute_positions_handles_chains_with_mixed_units(h5root):
    instrument = snx.create_class(h5root, 'instrument', snx.NXinstrument)
    vector = sc.vector(value=[0, 0, 1])
    t1 = snx.create_field(instrument, 't1', sc.scalar(100, unit='cm'))
    t1.attrs['depends_on'] = 't2'
    t1.attrs['transformation_type'] = 'translation'
    t1.attrs['vector'] = vector.value
    t2 = snx.create_field(instrument, 't2', sc.scalar(1000, unit='mm'))
    t2.attrs['depends_on'] = '.'
    t2.attrs['transformation_type'] = 'translation'
    t2.attrs['vector'] = vector.value
    monitor = snx.create_class(instrument, 'monitor', snx.NXmonitor)
    snx.create_field(monitor, 'depends_on', '../t1')
    root = make_group(h5root)
    loaded = root[()]
    mon = snx.compute_positions(loaded)['instrument']['monitor']
    assert_identical(mon['position'], sc.vector([0, 0, 2], unit='m'))


def test_compute_positions_does_not_apply_time_dependent_transform_to_pixel_offsets(
    h5root,
):
    detector = create_detector(h5root)
    snx.create_field(detector, 'x_pixel_offset', sc.linspace('xx', -1, 1, 2, unit='m'))
    snx.create_field(detector, 'y_pixel_offset', sc.linspace('yy', -1, 1, 2, unit='m'))
    detector.attrs['axes'] = ['xx', 'yy']
    detector.attrs['x_pixel_offset_indices'] = [0]
    detector.attrs['y_pixel_offset_indices'] = [1]
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

    root = make_group(h5root)
    loaded = root[()]
    result = snx.compute_positions(loaded)
    assert 'position' in result['detector_0']
    assert 'position' not in result['detector_0']['data'].coords
    result = snx.compute_positions(loaded, store_transform='transform')
    assert_identical(result['detector_0']['transform'], t * offset)


def test_compute_positions_warns_if_depends_on_is_dead_link(h5root):
    instrument = snx.create_class(h5root, 'instrument', snx.NXinstrument)
    detector = create_detector(instrument)
    snx.create_field(detector, 'depends_on', sc.scalar('transform'))
    root = make_group(h5root)
    with pytest.warns(UserWarning, match='depends_on chain references missing node'):
        loaded = root[()]
    with pytest.warns(UserWarning, match='depends_on chain references missing node'):
        snx.compute_positions(loaded)


_log = sc.DataArray(
    sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
    coords={
        'time': sc.epoch(unit='s') + sc.array(dims=['time'], values=[11, 22], unit='s')
    },
)
_empty_log = _log[0:0].copy()


@pytest.mark.parametrize(
    'logs',
    [(_log, _empty_log), (_empty_log, _log), (_empty_log, _empty_log)],
    ids=['non-empty,empty', 'empty,non-empty', 'empty,empty'],
)
def test_compute_positions_handles_empty_time_dependent_transform_without_error(
    h5root, logs
) -> None:
    log1, log2 = logs
    detector = create_detector(h5root)
    snx.create_field(
        detector, 'depends_on', sc.scalar('/detector_0/transformations/t1')
    )
    transformations = snx.create_class(detector, 'transformations', NXtransformations)

    offset1 = sc.spatial.translation(value=[1, 2, 3], unit='m')
    vector1 = sc.vector(value=[0, 0, 1])
    value1 = snx.create_class(transformations, 't1', snx.NXlog)
    snx.create_field(value1, 'time', log1.coords['time'] - sc.epoch(unit='s'))
    snx.create_field(value1, 'value', log1.data)
    value1.attrs['depends_on'] = 't2'
    value1.attrs['transformation_type'] = 'translation'
    value1.attrs['offset'] = offset1.values
    value1.attrs['offset_units'] = str(offset1.unit)
    value1.attrs['vector'] = vector1.value

    offset2 = sc.spatial.translation(value=[4, 5, 6], unit='m')
    vector2 = sc.vector(value=[0, 1, 1])
    value2 = snx.create_class(transformations, 't2', snx.NXlog)
    snx.create_field(value2, 'time', log2.coords['time'] - sc.epoch(unit='s'))
    snx.create_field(value2, 'value', log2.data)
    value2.attrs['depends_on'] = '.'
    value2.attrs['transformation_type'] = 'translation'
    value2.attrs['offset'] = offset2.values
    value2.attrs['offset_units'] = str(offset2.unit)
    value2.attrs['vector'] = vector2.value

    root = make_group(h5root)
    loaded = root[()]
    with pytest.warns(UserWarning, match='depends_on chain contains empty time-series'):
        result = snx.compute_positions(loaded, store_transform='transform')

    # Even if only some of the logs are empty we cannot return any values.

    transform = result['detector_0']['transform']
    assert transform.sizes == {'time': 0}
    assert transform.dtype == 'float64'
    assert transform.unit == ''
    assert transform.coords['time'].sizes == {'time': 0}
    assert transform.coords['time'].dtype == sc.DType.datetime64

    position = result['detector_0']['position']
    assert position.sizes == {'time': 0}
    assert position.dtype == sc.DType.vector3
    assert position.unit == 'm'
    assert position.coords['time'].sizes == {'time': 0}
    assert position.coords['time'].dtype == sc.DType.datetime64

    assert 'position' not in result['detector_0']['data'].coords


def test_compute_transformation_warns_if_transformation_missing_vector_attr(
    h5root,
) -> None:
    detector = create_detector(h5root)
    snx.create_field(
        detector, 'depends_on', sc.scalar('/detector_0/transformations/t1')
    )
    transformations = snx.create_class(detector, 'transformations', NXtransformations)

    offset1 = sc.spatial.translation(value=[1, 2, 3], unit='m')
    value1 = snx.create_class(transformations, 't1', snx.NXlog)
    snx.create_field(value1, 'time', _log.coords['time'] - sc.epoch(unit='s'))
    snx.create_field(value1, 'value', _log.data)
    value1.attrs['depends_on'] = '.'
    value1.attrs['transformation_type'] = 'rotation'
    value1.attrs['offset'] = offset1.values
    value1.attrs['offset_units'] = str(offset1.unit)

    root = make_group(h5root)
    with pytest.warns(
        UserWarning, match="Invalid transformation, missing attribute 'vector'"
    ):
        root[()]
