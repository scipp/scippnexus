# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any

import pytest
import scipp as sc

import scippnexus as snx

externalfile = pytest.importorskip('externalfile')

all_files = [
    '2023/DREAM_baseline_all_dets.nxs',
    '2023/BIFROST_873855_00000015.hdf',
    '2023/DREAM_mccode.h5',
    '2023/LOKI_mcstas_nexus_geometry.nxs',
    '2023/NMX_2e11-rechunk.h5',
    '2023/YMIR_038243_00010244.hdf',
    '2023/amor2020n000346_tweaked.nxs',
    '2023/LOKI_60322-2022-03-02_2205_fixed.nxs',
]


def get_item_at_path(dg: sc.DataGroup, path: str) -> sc.DataArray:
    nodes = path.split('/')
    for node in nodes:
        dg = dg[node]
    return dg


def assert_schema(dg: sc.DataGroup, schema: dict[str, Any]) -> None:
    for name, validate in schema.items():
        validate(get_item_at_path(dg, name))


def validator(
    item_type: type,
    sizes: dict[str, int] | None = None,
    dtype: str | sc.DType | None = None,
) -> None:
    def _validator(item):
        assert isinstance(item, item_type)
        if sizes is not None:
            assert item.sizes == sizes
        if dtype is not None:
            assert item.dtype == dtype

    return _validator


def bins_validator(item_type: type, sizes: dict[str, int] | None = None):
    def _validator(item):
        assert isinstance(item, item_type)
        assert item.bins is not None
        if sizes is not None:
            assert item.sizes == sizes

    return _validator


@pytest.mark.externalfile()
@pytest.mark.parametrize('name', all_files)
def test_files_load_as_data_groups(name):
    with snx.File(externalfile.get_path(name)) as f:
        dg = f[()]
    assert isinstance(dg, sc.DataGroup)


@pytest.mark.externalfile()
@pytest.mark.parametrize('name', all_files)
def test_files_load_as_data_groups_with_no_definitions(name):
    with snx.File(externalfile.get_path(name), definitions={}) as f:
        dg = f[()]
    assert isinstance(dg, sc.DataGroup)


@pytest.mark.externalfile()
def test_amor2020n000346_tweaked():
    with snx.File(externalfile.get_path('2023/amor2020n000346_tweaked.nxs')) as f:
        dg = f[()]
    schema = {}
    schema['entry/instrument/multiblade_detector'] = bins_validator(
        sc.DataArray, {'detector_number': 9216}
    )
    schema['entry/stages/com'] = validator(sc.DataArray, {'time': 1, 'dim_1': 1})
    schema['entry/facility'] = validator(str)
    assert_schema(dg, schema)


@pytest.mark.externalfile()
def test_LOKI_60322_2022_03_02_2205_fixed():
    with snx.File(
        externalfile.get_path('2023/LOKI_60322-2022-03-02_2205_fixed.nxs')
    ) as f:
        dg = f[()]
    schema = {}
    schema['entry/instrument/larmor_detector'] = bins_validator(
        sc.DataArray, {'detector_number': 458752}
    )
    schema['entry/instrument/monitor_1'] = bins_validator(
        sc.DataArray, {'event_time_zero': 1987}
    )
    schema['entry/instrument/monitor_2'] = bins_validator(
        sc.DataArray, {'event_time_zero': 1987}
    )
    schema['entry/instrument/source/transformations/trans_2'] = validator(
        sc.DataArray, {}, sc.DType.translation3
    )

    assert_schema(dg, schema)
