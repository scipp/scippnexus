# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Utilities for loading and working with NeXus transformations.

Transformation chains in NeXus files can be non-local and can thus be challenging to
work with. Additionally, values of transformations can be time-dependent, with each
chain link potentially having a different time-dependent value. In practice the user is
interested in the position and orientation of a component at a specific time or time
range. This may involve evaluating the transformation chain at a specific time, or
applying some heuristic to determine if the changes in the transformation value are
significant or just noise. In combination, the above means that we need to remain
flexible in how we handle transformations, preserving all necessary information from
the source files. Therefore:

1. :py:class:`Transform` is a dataclass representing a transformation. The raw `value`
   dataset is preserved (instead of directly converting to, e.g., a rotation matrix) to
   facilitate further processing such as computing the mean or variance.
2. Loading a :py:class:`Group` will follow depends_on chains and store them as an
   attribute of the depends_on field. This is done by :py:func:`parse_depends_on_chain`.
3. :py:func:`compute_positions` computes component positions (and transformations). By
   making this an explicit separate step, transformations can be applied to the
   transformations stored by the depends_on field before doing so. We imagine that this
   can be used to

   - Evaluate the transformation at a specific time.
   - Apply filters to remove noise, to avoid having to deal with very small time
     intervals when processing data.

By keeping the loaded transformations in a simple and modifiable format, we can
furthermore manually update the transformations with information from other sources,
such as streamed NXlog values received from a data acquisition system.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import scipp as sc
from scipp.scipy import interpolate

from .base import Group, NexusStructureError, NXobject, base_definitions_dict
from .field import DependsOn, Field


class NXtransformations(NXobject):
    """
    Group of transformations.

    Currently all transformations in the group are loaded. This may lead to redundant
    loads as transformations are also loaded by following depends_on chains.
    """


class TransformationError(NexusStructureError):
    pass


@dataclass
class Transform:
    """In-memory component translation or rotation as described by NXtransformations."""

    name: str
    transformation_type: Literal['translation', 'rotation']
    value: sc.Variable | sc.DataArray | sc.DataGroup
    vector: sc.Variable
    depends_on: DependsOn
    offset: sc.Variable | None = None

    @property
    def sizes(self) -> dict[str, int]:
        """Convenience property to access sizes of the value."""
        return self.value.sizes

    def __post_init__(self):
        if self.transformation_type not in ['translation', 'rotation']:
            raise TransformationError(
                f"{self.transformation_type=} attribute at {self.name},"
                " expected 'translation' or 'rotation'."
            )

    @staticmethod
    def from_object(
        obj: Field | Group, value: sc.Variable | sc.DataArray | sc.DataGroup
    ) -> Transform:
        depends_on = DependsOn(parent=obj.parent.name, value=obj.attrs['depends_on'])
        return Transform(
            name=obj.name,
            transformation_type=obj.attrs.get('transformation_type'),
            value=_parse_value(value),
            vector=sc.vector(value=obj.attrs['vector']),
            depends_on=depends_on,
            offset=_parse_offset(obj),
        )

    def build(self) -> sc.Variable | sc.DataArray:
        """Convert the raw transform into a rotation or translation matrix."""
        t = self.value * self.vector
        v = t if isinstance(t, sc.Variable) else t.data
        if self.transformation_type == 'translation':
            v = sc.spatial.translations(dims=v.dims, values=v.values, unit=v.unit)
        elif self.transformation_type == 'rotation':
            v = sc.spatial.rotations_from_rotvecs(v)
        if isinstance(t, sc.Variable):
            t = v
        else:
            t.data = v
        if self.offset is None:
            return t
        if self.transformation_type == 'translation':
            return t * self.offset.to(unit=t.unit, copy=False)
        return t * self.offset


def _parse_offset(obj: Field | Group) -> sc.Variable | None:
    if (offset := obj.attrs.get('offset')) is None:
        return None
    if (offset_units := obj.attrs.get('offset_units')) is None:
        raise TransformationError(
            f"Found {offset=} but no corresponding 'offset_units' "
            f"attribute at {obj.name}"
        )
    return sc.spatial.translation(value=offset, unit=offset_units)


def _parse_value(
    value: sc.Variable | sc.DataArray | sc.DataGroup,
) -> sc.Variable | sc.DataArray | sc.DataGroup:
    if isinstance(value, sc.DataGroup) and (
        isinstance(value.get('value'), sc.DataArray)
    ):
        # Some NXlog groups are split into value, alarm, and connection_status
        # sublogs. We only care about the value.
        value = value['value']
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
    if any((x.sizes.get('time') == 0) for x in chain):
        warnings.warn(
            UserWarning('depends_on chain contains empty time-series, '), stacklevel=2
        )
        # It is not clear what the dtype should be in this case. As transformations
        # are commonly multiplied onto position vectors, we return an empty array of
        # floats, which can be multiplied by Scipp's vector dtype.
        return sc.DataArray(
            sc.array(dims=['time'], values=[], dtype='float64', unit=''),
            coords={'time': sc.datetimes(dims=['time'], values=[], unit='s')},
        )
    total_transform = None
    for transform in chain:
        if transform.dtype in (sc.DType.translation3, sc.DType.affine_transform3):
            transform = transform.to(unit='m', copy=False)
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
    obj: Field | Group, value: sc.Variable | sc.DataArray | sc.DataGroup
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
    if obj.attrs.get('transformation_type') is None:
        return value
    try:
        return Transform.from_object(obj, value)
    except KeyError as e:
        warnings.warn(
            UserWarning(f'Invalid transformation, missing attribute {e}'), stacklevel=2
        )
        return value


@dataclass
class TransformationChain(DependsOn):
    """
    Represents a chain of transformations references by a depends_on field.

    Loading a group with a depends_on field will try to follow the chain and store the
    transformations as an additional attribute of the in-memory representation of the
    depends_on field.
    """

    transformations: sc.DataGroup = field(default_factory=sc.DataGroup)

    def compute(self) -> sc.Variable | sc.DataArray:
        depends_on = self
        try:
            chain = []
            while (path := depends_on.absolute_path()) is not None:
                chain.append(self.transformations[path])
                depends_on = chain[-1].depends_on
            transform = combine_transformations([t.build() for t in chain])
        except KeyError as e:
            warnings.warn(
                UserWarning(f'depends_on chain references missing node:\n{e}'),
                stacklevel=2,
            )
        else:
            return transform


def parse_depends_on_chain(
    parent: Field | Group, depends_on: DependsOn
) -> TransformationChain | None:
    """Follow a depends_on chain and return the transformations."""
    chain = TransformationChain(depends_on.parent, depends_on.value)
    depends_on = depends_on.value
    try:
        while depends_on != '.':
            transform = parent[depends_on]
            parent = transform.parent
            depends_on = transform.attrs['depends_on']
            chain.transformations[transform.name] = transform[()]
    except KeyError as e:
        warnings.warn(
            UserWarning(f'depends_on chain references missing node {e}'), stacklevel=2
        )
        return None
    return chain


def compute_positions(
    dg: sc.DataGroup,
    *,
    store_position: str = 'position',
    store_transform: str | None = None,
    transformations: sc.DataGroup | None = None,
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
    transformations:
        Optional data group containing transformation chains. If not provided, the
        transformations are looked up in the chains stored within the depends_on field.

    Returns
    -------
    :
        New data group with added positions.
    """
    return _with_positions(
        dg,
        store_position=store_position,
        store_transform=store_transform,
        transformations=transformations,
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
    transformations: sc.DataGroup | None = None,
) -> sc.DataGroup:
    out = sc.DataGroup()
    if (chain := dg.get('depends_on')) is not None:
        if not isinstance(chain, TransformationChain):
            chain = TransformationChain(chain.parent, chain.value)
        if transformations is not None:
            chain = replace(chain, transformations=transformations)
        if (transform := chain.compute()) is not None:
            out[store_position] = transform * sc.vector([0, 0, 0], unit='m')
            if store_transform is not None:
                out[store_transform] = transform
    for name, value in dg.items():
        if isinstance(value, sc.DataGroup):
            value = _with_positions(
                value,
                store_position=store_position,
                store_transform=store_transform,
                transformations=transformations,
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


base_definitions_dict['NXtransformations'] = NXtransformations
