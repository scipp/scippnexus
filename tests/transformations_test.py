# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

import scippnexus as snx
from scippnexus import transformations

externalfile = pytest.importorskip('externalfile')


@pytest.mark.externalfile()
def test_find_transformation_groups_finds_expected_groups() -> None:
    filename = externalfile.get_path('2023/LOKI_60322-2022-03-02_2205_fixed.nxs')
    paths = transformations.find_transformations(filename)
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


@pytest.mark.externalfile()
def test_load_transformations_loads_as_flat_datagroup() -> None:
    filename = externalfile.get_path('2023/LOKI_60322-2022-03-02_2205_fixed.nxs')
    dg = transformations.load_transformations(filename)
    dg = transformations.as_nested(dg)
    assert list(dg) == ['entry']
    entry = dg['entry']
    assert list(entry) == ['instrument']
    instrument = entry['instrument']
    assert list(instrument) == ['larmor_detector', 'monitor_1', 'monitor_2', 'source']
    for group in instrument.values():
        assert list(group) == ['depends_on', 'transformations']


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.externalfile()
def test_positions_consistent_with_separate_load() -> None:
    # The Bifrost instrument has complex transformation chains so this is a good test.
    filename = externalfile.get_path('2023/BIFROST_873855_00000015.hdf')
    transforms = transformations.load_transformations(filename)
    transforms = transformations.as_nested(transforms)
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
    assert sc.identical(result, expected)
