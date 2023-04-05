# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

import scippnexus.v2 as snx

externalfile = pytest.importorskip('externalfile')

all_files = [
    '2023/DREAM_baseline_all_dets.nxs',
    '2023/BIFROST_873855_00000015.hdf',
    '2023/DREAM_mccode.h5',
    '2023/LOKI_mcstas_nexus_geometry.nxs',
    '2023/NMX_2e11-rechunk.h5',
    '2023/YMIR_038243_00010244.hdf',
]


@pytest.mark.externalfile
@pytest.mark.parametrize('name', all_files)
def test_files_load_as_data_groups(name):
    with snx.File(externalfile.get_path(name)) as f:
        dg = f[()]
    assert isinstance(dg, sc.DataGroup)


@pytest.mark.externalfile
@pytest.mark.parametrize('name', all_files)
def test_files_load_as_data_groups_with_no_definitions(name):
    with snx.File(externalfile.get_path(name), definitions={}) as f:
        dg = f[()]
    assert isinstance(dg, sc.DataGroup)
