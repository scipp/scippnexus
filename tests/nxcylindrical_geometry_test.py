import h5py
import pytest
import scipp as sc

import scippnexus as snx


@pytest.fixture()
def nxroot():
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = snx.Group(f, snx.base_definitions())
        root.create_class('entry', snx.NXentry)
        yield root


def test_vertices_loaded_as_vector3(nxroot):
    shape = nxroot['entry'].create_class('shape', snx.NXcylindrical_geometry)
    values = [[1, 2, 3], [4, 5, 6]]
    shape['vertices'] = sc.array(dims=['ignored', 'comp'], values=values, unit='mm')
    loaded = shape[()]
    assert sc.identical(
        loaded['vertices'], sc.vectors(dims=['vertex'], values=values, unit='mm')
    )


def test_field_properties(nxroot):
    shape = nxroot['entry'].create_class('shape', snx.NXcylindrical_geometry)
    values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    shape['vertices'] = sc.array(dims=['ignored', 'comp'], values=values, unit='m')
    shape['cylinders'] = sc.array(
        dims=['ignored', 'index'], values=[[0, 1, 2]], unit=None
    )
    shape['detector_number'] = sc.array(dims=['ignored'], values=[], unit=None)
    loaded = shape[()]
    assert loaded['vertices'].dims == ('vertex',)
    assert loaded['cylinders'].dims == ('cylinder', 'vertex_index')
    assert loaded['cylinders'].unit is None
    assert loaded['detector_number'].dims == ('detector_number',)
    assert loaded['detector_number'].unit is None
