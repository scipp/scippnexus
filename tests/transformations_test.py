# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
from ess.reduce import data
from scipp.testing import assert_identical

import scippnexus as snx
from scippnexus import transformations


def test_find_transformation_groups_finds_expected_groups() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    paths = transformations.find_transformation_groups(filename)
    assert paths == [
        'entry/instrument/larmor_detector/depends_on',
        'entry/instrument/larmor_detector/transformations/trans_1',
        'entry/instrument/monitor_1/depends_on',
        'entry/instrument/monitor_1/transformations/trans_3',
        'entry/instrument/monitor_2/depends_on',
        'entry/instrument/monitor_2/transformations/trans_4',
        'entry/instrument/source/depends_on',
        'entry/instrument/source/transformations/trans_2',
    ]


def test_load_transformations_loads_as_flat_datagroup() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    dg = transformations.load_transformations(filename)
    assert list(dg) == ['entry']
    entry = dg['entry']
    assert list(entry) == ['instrument']
    instrument = entry['instrument']
    assert list(instrument) == ['larmor_detector', 'monitor_1', 'monitor_2', 'source']
    for group in instrument.values():
        assert list(group) == ['depends_on', 'transformations']


def test_find_transformations_bifrost() -> None:
    filename = '/home/simon/instruments/bifrost/BIFROST_20240905T122604.h5'
    transformations.find_transformation_groups(filename)


def test_load_transformations_bifrost() -> None:
    filename = '/home/simon/instruments/bifrost/BIFROST_20240905T122604.h5'
    transformations.load_transformations(filename)


def test_scippnexus_can_parse_transformation_chain() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    transforms = transformations.load_transformations(filename)
    dg = snx.load(filename)
    result = snx.compute_positions(
        dg,
        store_position='position',
        store_transform='transform',
        transformations=transforms,
    )
    detector = result['entry']['instrument']['larmor_detector']
    assert 'position' in detector['larmor_detector_events'].coords


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_positions_consistent_with_separate_load() -> None:
    filename = '/home/simon/instruments/bifrost/BIFROST_20240905T122604.h5'
    filename = '/home/simon/instruments/bifrost/268227_00021671.hdf'
    transforms = transformations.load_transformations(filename)
    dg = snx.load(filename)
    expected = snx.compute_positions(
        dg, store_position='position', store_transform='transform'
    )
    result = snx.compute_positions(
        dg,
        store_position='position',
        store_transform='transform',
        transformations=transforms,
    )
    assert_identical(result, expected)
