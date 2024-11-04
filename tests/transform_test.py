import pytest
import scipp as sc

from scippnexus import DependsOn
from scippnexus.nxtransformations import Transform, TransformationError


@pytest.fixture()
def depends_on() -> DependsOn:
    return DependsOn(parent='/', value='.')


@pytest.fixture()
def z_vector() -> sc.Variable:
    return sc.vector(value=[0, 0, 1], unit='m')


def test_init_raises_if_transformation_type_is_invalid(depends_on, z_vector) -> None:
    with pytest.raises(TransformationError, match='transformation_type'):
        Transform(
            name='t1',
            transformation_type='trans',
            value=sc.ones(dims=['x', 'y', 'z'], shape=(2, 3, 4)),
            vector=z_vector,
            depends_on=depends_on,
        )


def test_sizes_returns_sizes_of_value(depends_on, z_vector) -> None:
    value = sc.ones(dims=['x', 'y', 'z'], shape=(2, 3, 4))
    transform = Transform(
        name='t1',
        transformation_type='translation',
        value=value,
        vector=z_vector,
        depends_on=depends_on,
    )
    assert transform.sizes == {'x': 2, 'y': 3, 'z': 4}
