import h5py
import numpy as np
import pytest
import scipp as sc
from scipp import spatial
from scipp.testing import assert_identical

import scippnexus as snx


@pytest.fixture()
def nxroot():
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = snx.Group(f, definitions=snx.base_definitions())
        root.create_class('entry', snx.NXentry)
        yield root


def test_ub_matrix_loaded_as_linear_transform_with_inverse_angstrom_unit(nxroot):
    sample = nxroot.create_class('data1', snx.NXsample)
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sample['ub_matrix'] = matrix
    loaded = sample[()]
    assert_identical(
        loaded,
        sc.DataGroup(
            ub_matrix=spatial.linear_transform(value=matrix, unit='1/angstrom')
        ),
    )


def test_ub_matrix_array_can_be_loaded(nxroot):
    sample = nxroot.create_class('data1', snx.NXsample)
    matrices = np.array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 3, 4], [5, 6, 7], [8, 9, 10]]]
    )
    sample['ub_matrix'] = matrices
    loaded = sample[()]
    assert_identical(
        loaded,
        sc.DataGroup(
            ub_matrix=spatial.linear_transforms(
                dims=('dim_0',), values=matrices, unit='1/angstrom'
            )
        ),
    )


def test_orientation_matrix_loaded_as_linear_transform_with_dimensionless_unit(nxroot):
    sample = nxroot.create_class('data1', snx.NXsample)
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sample['orientation_matrix'] = matrix
    loaded = sample[()]
    assert_identical(
        loaded,
        sc.DataGroup(
            orientation_matrix=spatial.linear_transform(value=matrix, unit='')
        ),
    )
