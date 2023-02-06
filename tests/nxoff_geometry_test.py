import h5py
import pytest
import scipp as sc

from scippnexus import NXentry, NXoff_geometry, NXroot


@pytest.fixture()
def nxroot(request):
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = NXroot(f)
        root.create_class('entry', NXentry)
        yield root


def test_vertices_loaded_as_vector3(nxroot):
    shape = nxroot['entry'].create_class('shape', NXoff_geometry)
    values = [[1, 2, 3], [4, 5, 6]]
    shape['vertices'] = sc.array(dims=['ignored', 'comp'], values=values, unit='mm')
    loaded = shape[()]
    assert sc.identical(loaded['vertices'],
                        sc.vectors(dims=['vertices'], values=values, unit='mm'))


def test_field_properties(nxroot):
    shape = nxroot['entry'].create_class('shape', NXoff_geometry)
    values = [[1, 2, 3], [4, 5, 6]]
    shape['vertices'] = sc.array(dims=['ignored', 'comp'], values=values, unit='m')
    shape['winding_order'] = sc.array(dims=['ignored'], values=[], unit=None)
    shape['faces'] = sc.array(dims=['ignored'], values=[], unit=None)
    loaded = shape[()]
    assert loaded['vertices'].dims == ('vertices', )
    assert loaded['winding_order'].dims == ('winding_order', )
    assert loaded['winding_order'].unit is None
    assert loaded['faces'].dims == ('faces', )
    assert loaded['faces'].unit is None
