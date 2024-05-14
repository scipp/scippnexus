# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipp as sc
from scipp.scipy import interpolate

from .base import Group, NexusStructureError, NXobject, ScippIndex
from .field import Field, depends_on_to_relative_path


class TransformationError(NexusStructureError):
    pass


class NXtransformations(NXobject):
    """Group of transformations."""


class Transformation:
    def __init__(self, obj: Field | Group):  # could be an NXlog
        self._obj = obj

    @property
    def sizes(self) -> dict:
        return self._obj.sizes

    @property
    def dims(self) -> tuple[str, ...]:
        return self._obj.dims

    @property
    def shape(self) -> tuple[int, ...]:
        return self._obj.shape

    @property
    def attrs(self):
        return self._obj.attrs

    @property
    def name(self):
        return self._obj.name

    @property
    def offset(self):
        if (offset := self.attrs.get('offset')) is None:
            return None
        if (offset_units := self.attrs.get('offset_units')) is None:
            raise TransformationError(
                f"Found {offset=} but no corresponding 'offset_units' "
                f"attribute at {self.name}"
            )
        return sc.spatial.translation(value=offset, unit=offset_units)

    @property
    def vector(self) -> sc.Variable:
        return sc.vector(value=self.attrs.get('vector'))

    def __getitem__(self, select: ScippIndex):
        transformation_type = self.attrs.get('transformation_type')
        # According to private communication with Tobias Richter, NeXus allows 0-D or
        # shape=[1] for single values. It is unclear how and if this could be
        # distinguished from a scan of length 1.
        value = self._obj[select]
        return self.make_transformation(value, transformation_type, select)

    def make_transformation(
        self,
        value: sc.Variable | sc.DataArray,
        transformation_type: str,
        select: ScippIndex,
    ):
        try:
            if isinstance(value, sc.DataGroup):
                return value
            t = value * self.vector
            v = t if isinstance(t, sc.Variable) else t.data
            if transformation_type == 'translation':
                v = sc.spatial.translations(dims=v.dims, values=v.values, unit=v.unit)
            elif transformation_type == 'rotation':
                v = sc.spatial.rotations_from_rotvecs(v)
            else:
                raise TransformationError(
                    f"{transformation_type=} attribute at {self.name},"
                    " expected 'translation' or 'rotation'."
                )
            if isinstance(t, sc.Variable):
                t = v
            else:
                t.data = v
            if (offset := self.offset) is None:
                transform = t
            else:
                offset = sc.vector(value=offset.values, unit=offset.unit)
                offset = sc.spatial.translation(value=offset.value, unit=offset.unit)
                if transformation_type == 'translation':
                    offset = offset.to(unit=t.unit, copy=False)
                transform = t * offset
            if (depends_on := self.attrs.get('depends_on')) is not None:
                if not isinstance(transform, sc.DataArray):
                    transform = sc.DataArray(transform)
                transform.coords['depends_on'] = sc.scalar(
                    depends_on_to_relative_path(depends_on, self._obj.parent.name)
                )
            return transform
        except (sc.DimensionError, sc.UnitError, TransformationError):
            # TODO We should probably try to return some other data structure and
            # also insert offset and other attributes.
            return value


def _interpolate_transform(transform, xnew):
    # scipy can't interpolate with a single value
    if transform.sizes["time"] == 1:
        transform = sc.concat([transform, transform], dim="time")
    return interpolate.interp1d(
        transform, "time", kind="previous", fill_value="extrapolate"
    )(xnew=xnew)


def _smaller_unit(a, b):
    if a.unit == b.unit:
        return a.unit
    ratio = sc.scalar(1.0, unit=a.unit).to(unit=b.unit)
    if ratio.value < 1.0:
        return a.unit
    else:
        return b.unit


def combine_transformations(
    chain: list[sc.DataArray | sc.Variable],
) -> sc.DataArray | sc.Variable:
    """
    Take the product of a chain of transformations, handling potentially mismatching
    time-dependence.

    Time-dependent transformations are interpolated to a common time-coordinate.
    """
    total_transform = None
    for transform in chain:
        if total_transform is None:
            total_transform = transform
        elif isinstance(total_transform, sc.DataArray) and isinstance(
            transform, sc.DataArray
        ):
            unit = _smaller_unit(
                transform.coords['time'], total_transform.coords['time']
            )
            total_transform.coords['time'] = total_transform.coords['time'].to(
                unit=unit, copy=False
            )
            transform.coords['time'] = transform.coords['time'].to(
                unit=unit, copy=False
            )
            time = sc.concat(
                [total_transform.coords["time"], transform.coords["time"]], dim="time"
            )
            time = sc.datetimes(values=np.unique(time.values), dims=["time"], unit=unit)
            total_transform = _interpolate_transform(
                transform, time
            ) * _interpolate_transform(total_transform, time)
        else:
            total_transform = transform * total_transform
    if isinstance(total_transform, sc.DataArray):
        time_dependent = [t for t in chain if isinstance(t, sc.DataArray)]
        times = [da.coords['time'][0] for da in time_dependent]
        latest_log_start = sc.reduce(times).max()
        return total_transform['time', latest_log_start:].copy()
    return sc.scalar(1) if total_transform is None else total_transform


def maybe_transformation(
    obj: Field | Group,
    value: sc.Variable | sc.DataArray | sc.DataGroup,
    sel: ScippIndex,
) -> sc.Variable | sc.DataArray | sc.DataGroup:
    """
    Return a loaded field, possibly modified if it is a transformation.

    Transformations are usually stored in NXtransformations groups. However, identifying
    transformation fields in this way requires inspecting the parent group, which
    is cumbersome to implement. Furthermore, according to the NXdetector documentation
    transformations are not necessarily placed inside NXtransformations.
    Instead we use the presence of the attribute 'transformation_type' to identify
    transformation fields.
    """
    if (transformation_type := obj.attrs.get('transformation_type')) is not None:
        return Transformation(obj).make_transformation(value, transformation_type, sel)
    return value


class TransformationChainResolver:
    """
    Resolve a chain of transformations, given depends_on attributes with absolute or
    relative paths.

    A `depends_on` field serves as an entry point into a chain of transformations.
    It points to another entry, based on an absolute or relative path. The target
    entry may have a `depends_on` attribute pointing to the next transform. This
    class follows the paths and resolves the chain of transformations.
    """

    class ChainError(KeyError):
        """Raised when a transformation chain cannot be resolved."""

        pass

    @dataclass
    class Entry:
        name: str
        value: sc.DataGroup

    def __init__(self, stack: list[TransformationChainResolver.Entry]):
        self._stack = stack

    @staticmethod
    def from_root(dg: sc.DataGroup) -> TransformationChainResolver:
        return TransformationChainResolver(
            [TransformationChainResolver.Entry(name='', value=dg)]
        )

    @property
    def name(self) -> str:
        return '/'.join([e.name for e in self._stack])

    @property
    def root(self) -> TransformationChainResolver:
        return TransformationChainResolver(self._stack[0:1])

    @property
    def parent(self) -> TransformationChainResolver:
        if len(self._stack) == 1:
            raise TransformationChainResolver.ChainError(
                "Transformation depends on node beyond root"
            )
        return TransformationChainResolver(self._stack[:-1])

    @property
    def value(self) -> sc.DataGroup:
        return self._stack[-1].value

    def __getitem__(self, path: str) -> TransformationChainResolver:
        base, *remainder = path.split('/', maxsplit=1)
        if base == '':
            node = self.root
        elif base == '.':
            node = self
        elif base == '..':
            node = self.parent
        else:
            try:
                child = self._stack[-1].value[base]
            except KeyError:
                raise TransformationChainResolver.ChainError(
                    f"{base} not found in {self.name}"
                ) from None
            node = TransformationChainResolver(
                [
                    *self._stack,
                    TransformationChainResolver.Entry(name=base, value=child),
                ]
            )
        return node if len(remainder) == 0 else node[remainder[0]]

    def resolve_depends_on(self) -> sc.DataArray | sc.Variable | None:
        """
        Resolve the depends_on attribute of a transformation chain.

        Returns
        -------
        :
            The resolved position in meter, or None if no depends_on was found.
        """
        depends_on = self.value.get('depends_on')
        if depends_on is None:
            return None
        # Note that transformations have to be applied in "reverse" order, i.e.,
        # simply taking math.prod(chain) would be wrong, even if we could
        # ignore potential time-dependence.
        return combine_transformations(self.get_chain(depends_on))

    def get_chain(self, depends_on: str) -> list[sc.DataArray | sc.Variable]:
        if depends_on == '.':
            return []
        node = self[depends_on]
        transform = node.value.copy(deep=False)
        depends_on = '.'
        if isinstance(transform, sc.DataArray):
            if (attr := transform.coords.pop('depends_on', None)) is not None:
                depends_on = attr.value
            # If transform is time-dependent then we keep it is a DataArray, otherwise
            # we convert it to a Variable.
            transform = transform if transform.coords else transform.data
        if transform.dtype in (sc.DType.translation3, sc.DType.affine_transform3):
            transform = transform.to(unit='m', copy=False)
        return [transform, *node.parent.get_chain(depends_on)]


def compute_positions(
    dg: sc.DataGroup,
    *,
    store_position: str = 'position',
    store_transform: str | None = None,
) -> sc.DataGroup:
    """
    Recursively compute positions from depends_on attributes as well as the
    [xyz]_pixel_offset fields of NXdetector groups.

    This function does not operate directly on a NeXus file but on the result of
    loading a NeXus file or sub-group into a scipp.DataGroup. NeXus puts no
    limitations on the structure of the depends_on chains, i.e., they may reference
    parent groups. If this is the case, a call to this function will fail if only the
    subgroup is passed as input.

    Note that this does not consider "legacy" ways of storing positions. In particular,
    ``NXmonitor.distance``, ``NXdetector.distance``, ``NXdetector.polar_angle``, and
    ``NXdetector.azimuthal_angle`` are ignored.

    Note that transformation chains may be time-dependent. In this case it will not
    be applied to the pixel offsets, since the result may consume too much memory and
    the shape is in general incompatible with the shape of the data. Use the
    ``store_transform`` argument to store the resolved transformation chain in this
    case.

    If a transformation chain has an invalid 'depends_on' value, e.g., a path beyond
    the root data group, then the chain is ignored and no position is computed. This
    does not affect other chains.

    Parameters
    ----------
    dg:
        Data group with depends_on entry points into transformation chains.
    store_position:
        Name used to store result of resolving each depends_on chain.
    store_transform:
        If not None, store the resolved transformation chain in this field.

    Returns
    -------
    :
        New data group with added positions.
    """
    # Create resolver at root level, since any depends_on chain may lead to a parent,
    # i.e., we cannot use a resolver at the level of each chain's entry point.
    resolver = TransformationChainResolver.from_root(dg)
    return _with_positions(
        dg,
        store_position=store_position,
        store_transform=store_transform,
        resolver=resolver,
    )


def zip_pixel_offsets(x: dict[str, sc.Variable], /) -> sc.Variable:
    """
    Zip the x_pixel_offset, y_pixel_offset, and z_pixel_offset fields into a vector.

    These fields originate from NXdetector groups. All but x_pixel_offset are optional,
    e.g., for 2D detectors. Zero values for missing fields are assumed.

    Parameters
    ----------
    mapping:
        Mapping (typically a data group, or data array coords) containing
        x_pixel_offset, y_pixel_offset, and z_pixel_offset.

    Returns
    -------
    :
        Vectors with pixel offsets.

    See Also
    --------
    compute_positions
    """
    zero = sc.scalar(0.0, unit=x['x_pixel_offset'].unit)
    return sc.spatial.as_vectors(
        x['x_pixel_offset'],
        x.get('y_pixel_offset', zero),
        x.get('z_pixel_offset', zero),
    )


def _with_positions(
    dg: sc.DataGroup,
    *,
    store_position: str,
    store_transform: str | None = None,
    resolver: TransformationChainResolver,
) -> sc.DataGroup:
    out = sc.DataGroup()
    transform = None
    if 'depends_on' in dg:
        try:
            transform = resolver.resolve_depends_on()
            out[store_position] = transform * sc.vector([0, 0, 0], unit='m')
            if store_transform is not None:
                out[store_transform] = transform
        except TransformationChainResolver.ChainError as e:
            warnings.warn(
                UserWarning(f'depends_on chain references missing node:\n{e}'),
                stacklevel=2,
            )
    for name, value in dg.items():
        if isinstance(value, sc.DataGroup):
            value = _with_positions(
                value,
                store_position=store_position,
                store_transform=store_transform,
                resolver=resolver[name],
            )
        elif (
            isinstance(value, sc.DataArray)
            and 'x_pixel_offset' in value.coords
            # Transform can be time-dependent, do not apply it to offsets since
            # result can be massive and is in general not compatible with the shape
            # of the data.
            and (transform is not None and transform.dims == ())
        ):
            offset = zip_pixel_offsets(value.coords).to(unit='m', copy=False)
            value = value.assign_coords({store_position: transform * offset})
        out[name] = value
    return out
