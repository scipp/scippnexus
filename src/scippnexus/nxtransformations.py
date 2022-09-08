# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
from typing import Union
import scipp as sc
import scipp.spatial
import scipp.interpolate
from .nxobject import Field, NXobject, ScippIndex


class TransformationError(Exception):
    pass


def make_transformation(obj, /, path):
    if path.startswith('/'):
        return Transformation(obj.file[path])
    elif path != '.':
        return Transformation(obj.parent[path])
    return None  # end of chain


class Transformation:

    def __init__(self, obj: Union[Field, NXobject]):  # could be an NXlog
        self._obj = obj

    @property
    def attrs(self):
        return self._obj.attrs

    @property
    def name(self):
        return self._obj.name

    @property
    def depends_on(self):
        if (path := self.attrs.get('depends_on')) is not None:
            return make_transformation(self._obj, path)
        return None

    @property
    def offset(self):
        if (offset := self.attrs.get('offset')) is None:
            return None
        if (offset_units := self.attrs.get('offset_units')) is None:
            raise TransformationError(
                f"Found {offset=} but no corresponding 'offset_units' "
                f"attribute at {self.name}")
        return sc.spatial.translation(value=offset, unit=offset_units)

    @property
    def vector(self) -> sc.Variable:
        return sc.vector(value=self.attrs.get('vector'))

    def __getitem__(self, select: ScippIndex):
        transformation_type = self.attrs.get('transformation_type')
        # According to private communication with Tobias Richter, NeXus allows 0-D or
        # shape=[1] for single values. It is unclear how and if this could be
        # distinguished from a scan of length 1.
        t = self._obj[select] * self.vector
        v = t if isinstance(t, sc.Variable) else t.data
        if transformation_type == 'translation':
            v = v.to(unit='m', copy=False)
            v = sc.spatial.translations(dims=v.dims, values=v.values, unit=v.unit)
        elif transformation_type == 'rotation':
            v = sc.spatial.rotations_from_rotvecs(v)
        else:
            raise TransformationError(
                f"{transformation_type=} attribute at {self.name},"
                " expected 'translation' or 'rotation'.")
        if isinstance(t, sc.Variable):
            t = v
        else:
            t.data = v
        if (offset := self.offset) is None:
            return t
        offset = sc.vector(value=offset.values, unit=offset.unit).to(unit='m')
        offset = sc.spatial.translation(value=offset.value, unit=offset.unit)
        return t * offset


def _interpolate_transform(transform, xnew):
    # scipy can't interpolate with a single value
    if transform.sizes["time"] == 1:
        transform = sc.concat([transform, transform], dim="time")
    return sc.interpolate.interp1d(transform,
                                   "time",
                                   kind="previous",
                                   fill_value="extrapolate")(xnew=xnew)


def _smaller_unit(a, b):
    if a.unit == b.unit:
        return a.unit
    ratio = sc.scalar(1.0, unit=a.unit).to(unit=b.unit)
    if ratio.value < 1.0:
        return a.unit
    else:
        return b.unit


def get_full_transformation(depends_on: Field) -> Union[None, sc.DataArray]:
    """
    Get the 4x4 transformation matrix for a component, resulting
    from the full chain of transformations linked by "depends_on"
    attributes
    """
    if (t0 := make_transformation(depends_on, depends_on[()])) is None:
        return None
    transformations = _get_transformations(t0)

    total_transform = sc.spatial.affine_transform(value=np.identity(4), unit=sc.units.m)

    for transform in transformations:
        if isinstance(total_transform, sc.DataArray) and isinstance(
                transform, sc.DataArray):
            unit = _smaller_unit(transform.coords['time'],
                                 total_transform.coords['time'])
            total_transform.coords['time'] = total_transform.coords['time'].to(
                unit=unit, copy=False)
            transform.coords['time'] = transform.coords['time'].to(unit=unit,
                                                                   copy=False)
            time = sc.concat([total_transform.coords["time"], transform.coords["time"]],
                             dim="time")
            time = sc.datetimes(values=np.unique(time.values), dims=["time"], unit=unit)
            total_transform = _interpolate_transform(transform, time) \
                * _interpolate_transform(total_transform, time)
        else:
            total_transform = transform * total_transform
    if isinstance(total_transform, sc.DataArray):
        time_dependent = [t for t in transformations if isinstance(t, sc.DataArray)]
        times = [da.coords['time'][0] for da in time_dependent]
        latest_log_start = sc.reduce(times).max()
        return total_transform['time', latest_log_start:].copy()
    return total_transform


def _get_transformations(transform: Union[Field, NXobject]):
    """Get all transformations in the depends_on chain."""
    transformations = []
    t = transform
    while t is not None:
        transformations.append(t[()])
        t = t.depends_on
    # TODO: this list of transformation should probably be cached in the future
    # to deal with changing beamline components (e.g. pixel positions) during a
    # live data stream (see https://github.com/scipp/scippneutron/issues/76).
    return transformations
