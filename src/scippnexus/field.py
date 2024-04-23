# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import datetime
import posixpath
import re
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import scipp as sc

from scippnexus._common import convert_time_to_datetime64, to_plain_index
from scippnexus._hdf5_nexus import _warn_latin1_decode
from scippnexus.typing import H5Dataset, ScippIndex

from ._cache import cached_property
from .attrs import Attrs

if TYPE_CHECKING:
    from .base import Group


def depends_on_to_relative_path(depends_on: str, parent_path: str) -> str:
    """Replace depends_on paths with relative paths.

    After loading we will generally not have the same root so absolute paths
    cannot be resolved after loading."""
    if depends_on.startswith('/'):
        return posixpath.relpath(depends_on, parent_path)
    return depends_on


def _is_time(obj):
    if (unit := obj.unit) is None:
        return False
    return unit.to_dict().get('powers') == {'s': 1}


def _as_datetime(obj: Any):
    if isinstance(obj, str):
        try:
            # NumPy and scipp cannot handle timezone information. We therefore apply it,
            # i.e., convert to UTC.
            # Would like to use datetime directly, but with Python's datetime we do not
            # get nanosecond precision. Therefore we combine numpy and datetime parsing.
            date_only = 'T' not in obj
            if date_only:
                return sc.datetime(obj)
            date, time = obj.split('T')
            time_and_timezone_offset = re.split(r'Z|\+|-', time)
            time = time_and_timezone_offset[0]
            if len(time_and_timezone_offset) == 1:
                # No timezone, parse directly (scipp based on numpy)
                return sc.datetime(obj)
            else:
                # There is timezone info. Parse with datetime.
                dt = datetime.datetime.fromisoformat(obj)
                dt = dt.replace(microsecond=0)  # handled by numpy
                dt = dt.astimezone(datetime.timezone.utc)
                dt = dt.replace(tzinfo=None).isoformat()
                # We operate with string operations here and thus end up parsing date
                # and time twice. The reason is that the timezone-offset arithmetic
                # cannot be done, e.g., in nanoseconds without causing rounding errors.
                if '.' in time:
                    dt += f".{time.split('.')[1]}"
                return sc.datetime(dt)
        except ValueError:
            pass
    return None


@dataclass
class Field:
    """NeXus field.
    In HDF5 fields are represented as dataset.
    """

    dataset: H5Dataset
    parent: 'Group'
    sizes: Optional[Dict[str, int]] = None
    dtype: Optional[sc.DType] = None
    errors: Optional[H5Dataset] = None

    @cached_property
    def attrs(self) -> Dict[str, Any]:
        """The attributes of the dataset.
        Cannot be used for writing attributes, since they are cached for performance."""
        return MappingProxyType(Attrs(self.dataset.attrs))

    @property
    def dims(self) -> Tuple[str, ...]:
        return tuple(self.sizes.keys())

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.sizes.values())

    @cached_property
    def file(self) -> 'Group':
        return self.parent.file

    def _load_variances(self, var, index):
        stddevs = sc.empty(
            dims=var.dims, shape=var.shape, dtype=var.dtype, unit=var.unit
        )
        try:
            self.errors.read_direct(stddevs.values, source_sel=index)
        except TypeError:
            stddevs.values = self.errors[index].squeeze()
        # According to the standard, errors must have the same shape as the data.
        # This is not the case in all files we observed, is there any harm in
        # attempting a broadcast?
        var.variances = np.broadcast_to(
            sc.pow(stddevs, sc.scalar(2)).values, shape=var.shape
        )

    def __getitem__(self, select: ScippIndex) -> Union[Any, sc.Variable]:
        """
        Load the field as a :py:class:`scipp.Variable` or Python object.

        If the shape is empty and no unit is given this returns a Python object, such
        as a string or integer. Otherwise a :py:class:`scipp.Variable` is returned.

        Parameters
        ----------
        select:
            Scipp-style index: Load the specified slice of the current group.

        Returns
        -------
        :
            Loaded data.
        """
        from .nxtransformations import maybe_transformation

        index = to_plain_index(self.dims, select)
        if isinstance(index, (int, slice)):
            index = (index,)

        base_dims = self.dims
        base_shape = self.shape
        dims = []
        shape = []
        for i, ind in enumerate(index):
            if not isinstance(ind, int):
                dims.append(base_dims[i])
                shape.append(len(range(*ind.indices(base_shape[i]))))

        variable = sc.empty(
            dims=dims,
            shape=shape,
            dtype=self.dtype,
            unit=self.unit,
            with_variances=self.errors is not None,
        )

        # If the variable is empty, return early
        if np.prod(shape) == 0:
            variable = self._maybe_datetime(variable)
            return maybe_transformation(self, value=variable, sel=select)

        if self.dtype == sc.DType.string:
            try:
                strings = self.dataset.asstr()[index]
            except UnicodeDecodeError as e:
                strings = self.dataset.asstr(encoding='latin-1')[index]
                _warn_latin1_decode(self.dataset, strings, str(e))
            variable.values = np.asarray(strings).flatten()
            if self.dataset.name.endswith('depends_on') and variable.ndim == 0:
                variable.value = depends_on_to_relative_path(
                    variable.value, self.dataset.parent.name
                )
        elif variable.values.flags["C_CONTIGUOUS"]:
            # On versions of h5py prior to 3.2, a TypeError occurs in some cases
            # where h5py cannot broadcast data with e.g. shape (20, 1) to a buffer
            # of shape (20,). Note that broadcasting (1, 20) -> (20,) does work
            # (see https://github.com/h5py/h5py/pull/1796).
            # Therefore, we manually squeeze here.
            # A pin of h5py<3.2 is currently required by Mantid and hence scippneutron
            # (see https://github.com/h5py/h5py/issues/1880#issuecomment-823223154)
            # hence this workaround. Once we can use a more recent h5py with Mantid,
            # this try/except can be removed.
            try:
                self.dataset.read_direct(variable.values, source_sel=index)
            except TypeError:
                variable.values = self.dataset[index].squeeze()
            if self.errors is not None:
                self._load_variances(variable, index)
        else:
            variable.values = self.dataset[index]
        if variable.ndim == 0 and variable.unit is None and variable.fields is None:
            # Work around scipp/scipp#2815, and avoid returning NumPy bool
            if isinstance(variable.values, np.ndarray) and variable.dtype != 'bool':
                return variable.values[()]
            else:
                return variable.value
        variable = self._maybe_datetime(variable)
        return maybe_transformation(self, value=variable, sel=select)

    def _maybe_datetime(self, variable: sc.Variable) -> sc.Variable:
        if _is_time(variable):
            starts = []
            for name in self.attrs:
                if (dt := _as_datetime(self.attrs[name])) is not None:
                    starts.append(dt)
            if len(starts) == 1:
                variable = convert_time_to_datetime64(
                    variable,
                    start=starts[0],
                    scaling_factor=self.attrs.get('scaling_factor'),
                )

        return variable

    def __repr__(self) -> str:
        return f'<Nexus field "{self.dataset.name}">'

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def ndim(self) -> int:
        """Total number of dimensions in the dataset.
        See the shape property for potential differences to the value returned by the
        underlying h5py.Dataset.ndim.
        """
        return len(self.shape)

    @cached_property
    def unit(self) -> Union[sc.Unit, None]:
        if (unit := self.attrs.get('units')) is not None:
            try:
                return sc.Unit(unit)
            except sc.UnitError:
                warnings.warn(
                    f"Unrecognized unit '{unit}' for value dataset "
                    f"in '{self.name}'; setting unit as 'dimensionless'",
                    stacklevel=2,
                )
                return sc.units.one
        return None
