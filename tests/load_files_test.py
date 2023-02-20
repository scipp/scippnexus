# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
from externalfile import get_path

import scippnexus as snx


@pytest.mark.externalfile
@pytest.mark.parametrize('name', [
    '2023/DREAM_baseline_all_dets.nxs',
    '2023/BIFROST_873855_00000015.hdf',
    '2023/DREAM_mccode.h5',
    '2023/LOKI_mcstas_nexus_geometry.nxs',
    '2023/NMX_2e11-rechunk.h5',
])
def test_files_load_as_data_groups(name):
    with snx.File(get_path(name)) as f:
        dg = f[()]
    assert isinstance(dg, sc.DataGroup)
