# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

import datetime
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import dateutil.parser
import numpy as np
import scipp as sc

from ._common import convert_time_to_datetime64, to_child_select, to_plain_index
from ._hdf5_nexus import _warn_latin1_decode
from .typing import H5Dataset, H5Group, ScippIndex

# What we know:
# 1. We must not do a recursive read, or we will get in trouble for files with many
#    entries. User may just want to access subgroup recursively.
# 2. Some child access needs info from parent:
#    - Field dims
#    - NXevent_data
#    - NXoff_geometry
#    Maybe not... parent can modify dims/customize assembly
# 3. Unless we read shape, attrs, children only once, we will suffer too much overhead.
#    This includes dims/sizes computation.
# 4. Must be able to load coord before loading rest, for label-based indexing

# Desired behaviors:
# - Field should encapsulate "errors" handling
# - NXtransformations should load depends_on as chain (scalar variable with next)
# - NXobject.__setitem__ to set `axes` and `name_indices` attributes?

# Consider:
# - Non-legacy mode would make dim parsing simpler and faster?


def asarray(obj: Union[Any, sc.Variable]) -> sc.Variable:
    return obj if isinstance(obj, sc.Variable) else sc.scalar(obj, unit=None)


class NexusStructureError(Exception):
    """Invalid or unsupported class and field structure in Nexus.
    """
    pass


def is_dataset(obj: Union[H5Group, H5Dataset]) -> bool:
    """Return true if the object is an h5py.Dataset or equivalent.

    Use this instead of isinstance(obj, h5py.Dataset) to ensure that code is compatible
    with other h5py-alike interfaces.
    """
    return hasattr(obj, 'shape')


def _is_time(obj):
    if (unit := obj.unit) is None:
        return False
    return unit.to_dict().get('powers') == {'s': 1}


def _as_datetime(obj: Any):
    if isinstance(obj, str):
        try:
            # NumPy and scipp cannot handle timezone information. We therefore apply it,
            # i.e., convert to UTC.
            # Would like to use dateutil directly, but with Python's datetime we do not
            # get nanosecond precision. Therefore we combine numpy and dateutil parsing.
            date_only = 'T' not in obj
            if date_only:
                return sc.datetime(obj)
            date, time = obj.split('T')
            time_and_timezone_offset = re.split(r'Z|\+|-', time)
            time = time_and_timezone_offset[0]
            if len(time_and_timezone_offset) == 1:
                # No timezone, parse directly (scipp based on numpy)
                return sc.datetime(f'{date}T{time}')
            else:
                # There is timezone info. Parse with dateutil.
                dt = dateutil.parser.isoparse(obj)
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


_scipp_dtype = {
    np.dtype('int8'): sc.DType.int32,
    np.dtype('int16'): sc.DType.int32,
    np.dtype('uint8'): sc.DType.int32,
    np.dtype('uint16'): sc.DType.int32,
    np.dtype('uint32'): sc.DType.int32,
    np.dtype('uint64'): sc.DType.int64,
    np.dtype('int32'): sc.DType.int32,
    np.dtype('int64'): sc.DType.int64,
    np.dtype('float32'): sc.DType.float32,
    np.dtype('float64'): sc.DType.float64,
    np.dtype('bool'): sc.DType.bool,
}


def _dtype_fromdataset(dataset: H5Dataset) -> sc.DType:
    return _scipp_dtype.get(dataset.dtype, sc.DType.string)


@dataclass
class Field:
    dataset: H5Dataset
    sizes: Optional[Dict[str, int]] = None
    dtype: Optional[sc.DType] = None
    errors: Optional[H5Dataset] = None
    _is_time: Optional[bool] = None
    """NeXus field.

    In HDF5 fields are represented as dataset.
    """

    @cached_property
    def attrs(self) -> Dict[str, Any]:
        return dict(self.dataset.attrs) if self.dataset.attrs else dict()

    #def __init__(self,
    #             dataset: H5Dataset,
    #             errors: Optional[H5Dataset] = None,
    #             *,
    #             ancestor,
    #             dims=None,
    #             dtype: Optional[sc.DType] = None,
    #             is_time=None):
    #    self._ancestor = ancestor  # Usually the parent, but may be grandparent, etc.
    #    self.dataset = dataset
    #    self._errors = errors
    #    self._dtype = _dtype_fromdataset(dataset) if dtype is None else dtype
    #    self._shape = self.dataset.shape
    #    if self._dtype == sc.DType.vector3:
    #        self._shape = self._shape[:-1]
    #    self._is_time = is_time
    #    # NeXus treats [] and [1] interchangeably. In general this is ill-defined, but
    #    # the best we can do appears to be squeezing unless the file provides names for
    #    # dimensions. The shape property of this class does thus not necessarily return
    #    # the same as the shape of the underlying dataset.
    #    # TODO Should this logic be in FieldInfo? Or in NXdataInfo?
    #    if dims is not None:
    #        self._dims = tuple(dims)
    #        if len(self._dims) < len(self._shape):
    #            # The convention here is that the given dimensions apply to the shapes
    #            # starting from the left. So we only squeeze dimensions that are after
    #            # len(dims).
    #            self._shape = self._shape[:len(self._dims)] + tuple(
    #                size for size in self._shape[len(self._dims):] if size != 1)
    #    elif (axes := self.attrs.get('axes')) is not None:
    #        self._dims = tuple(axes.split(','))
    #    else:
    #        self._shape = tuple(size for size in self._shape if size != 1)
    #        self._dims = tuple(f'dim_{i}' for i in range(self.ndim))

    @property
    def dims(self) -> Tuple[str]:
        return tuple(self.sizes.keys())

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.sizes.values())

    def _load_variances(self, var, index):
        stddevs = sc.empty(dims=var.dims,
                           shape=var.shape,
                           dtype=var.dtype,
                           unit=var.unit)
        try:
            self.errors.read_direct(stddevs.values, source_sel=index)
        except TypeError:
            stddevs.values = self.errors[index].squeeze()
        # According to the standard, errors must have the same shape as the data.
        # This is not the case in all files we observed, is there any harm in
        # attempting a broadcast?
        var.variances = np.broadcast_to(sc.pow(stddevs, sc.scalar(2)).values,
                                        shape=var.shape)

    def __getitem__(self, select) -> Union[Any, sc.Variable]:
        """Load the field as a :py:class:`scipp.Variable` or Python object.

        If the shape is empty and no unit is given this returns a Python object, such
        as a string or integer. Otherwise a :py:class:`scipp.Variable` is returned.
        """
        index = to_plain_index(self.dims, select)
        if isinstance(index, (int, slice)):
            index = (index, )

        base_dims = self.dims
        base_shape = self.shape
        dims = []
        shape = []
        for i, ind in enumerate(index):
            if not isinstance(ind, int):
                dims.append(base_dims[i])
                shape.append(len(range(*ind.indices(base_shape[i]))))

        variable = sc.empty(dims=dims,
                            shape=shape,
                            dtype=self.dtype,
                            unit=self.unit,
                            with_variances=self.errors is not None)

        # If the variable is empty, return early
        if np.prod(shape) == 0:
            return variable

        if self.dtype == sc.DType.string:
            try:
                strings = self.dataset.asstr()[index]
            except UnicodeDecodeError as e:
                strings = self.dataset.asstr(encoding='latin-1')[index]
                _warn_latin1_decode(self.dataset, strings, str(e))
            variable.values = np.asarray(strings).flatten()
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
        if _is_time(variable):
            starts = []
            for name in self.attrs:
                if (dt := _as_datetime(self.attrs[name])) is not None:
                    starts.append(dt)
            if self._is_time and len(starts) == 0:
                starts.append(sc.epoch(unit=self.unit))
            if len(starts) == 1:
                variable = convert_time_to_datetime64(
                    variable,
                    start=starts[0],
                    scaling_factor=self.attrs.get('scaling_factor'))
        if variable.ndim == 0 and variable.unit is None:
            # Work around scipp/scipp#2815, and avoid returning NumPy bool
            if isinstance(variable.values, np.ndarray) and variable.dtype != 'bool':
                return variable.values[()]
            else:
                return variable.value
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
                warnings.warn(f"Unrecognized unit '{unit}' for value dataset "
                              f"in '{self.name}'; setting unit as 'dimensionless'")
                return sc.units.one
        return None


def _squeezed_field_sizes(dataset: H5Dataset) -> Dict[str, int]:
    shape = tuple(size for size in dataset.shape if size != 1)
    return {f'dim_{i}': size for i, size in enumerate(shape)}


class NXobject:

    def __init__(self, group: Group):
        self._group = group
        for field in group._children.values():
            if isinstance(field, Field):
                field.sizes = _squeezed_field_sizes(field.dataset)
                field.dtype = _dtype_fromdataset(field.dataset)

    @cached_property
    def sizes(self) -> Dict[str, int]:
        # exclude geometry/tansform groups?
        return sc.DataGroup(self._group).sizes

    def index_child(self, child: Union[Field, Group], sel: ScippIndex) -> ScippIndex:
        # Note that this will be similar in NXdata, but there we need to handle
        # bin edges as well.
        child_sel = to_child_select(self.sizes.keys(), child.dims, sel)
        return child[child_sel]

    def read_children(self, obj: Group, sel: ScippIndex) -> sc.DataGroup:
        return sc.DataGroup(
            {name: self.index_child(child, sel)
             for name, child in obj.items()})

    def assemble(self, dg: sc.DataGroup) -> Union[sc.DataGroup, sc.DataArray]:
        return dg


# Group adds children/dims caching, removes __setitem__?
# class Group(WriteableGroup):
class Group(Mapping):

    def __init__(self,
                 group: H5Group,
                 definitions: Optional[Dict[str, NXobject]] = None):
        self._group = group
        self._definitions = {} if definitions is None else definitions

    @cached_property
    def attrs(self) -> Dict[str, Any]:
        # Attrs are not read until needed, to avoid reading all attrs for all subgroups.
        # We may expected a per-subgroup overhead of 1 ms for reading attributes, so if
        # all we want is access one attribute, we may save, e.g., a second for a group
        # with 1000 subgroups.
        return dict(self._group.attrs) if self._group.attrs else dict()

    # TODO
    # should this by Dict[str, Union[H5Group, H5Dataset]]?
    # then we can recreate Group on every access (in principle more repeated init,
    # but maybe better since it "clears" the cache)?
    @cached_property
    def _children(self) -> Dict[str, Union[Field, Group]]:
        # split off special children here?
        # - depends_on
        # - NXoff_geometry and NXcylindrical_geometry
        # - legacy NXgeometry
        # - NXtransformations
        items = {
            name:
            Field(obj) if is_dataset(obj) else Group(obj, definitions=self._definitions)
            for name, obj in self._group.items()
        }
        for suffix in ('_errors', '_error'):
            field_with_errors = [name for name in items if f'{name}{suffix}' in items]
            for name in field_with_errors:
                values = items[name]
                errors = items[f'{name}{suffix}']
                if values.unit == errors.unit and values.dataset.shape == errors.dataset.shape:
                    values.errors = errors.dataset
                    del items[f'{name}{suffix}']
        items = {k: v for k, v in items.items() if not k.startswith('cue_')}
        return items

    @cached_property
    def _nexus(self) -> NXobject:
        return self._definitions.get(self.attrs.get('NX_class'), NXobject)(self)

    def _populate_fields(self) -> None:
        _ = self._nexus

    def _populate_field(self, name: str, field: Field) -> None:
        if field.sizes is not None:
            return
        field.sizes = self._nexus.field_sizes(name, field)
        field.dtype = self._nexus.field_dtype(name, field.dataset)

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[str]:
        return self._children.__iter__()

    def __getitem__(self, sel) -> Union[Field, Group, sc.DataGroup]:
        if isinstance(sel, str):
            child = self._children[sel]
            if isinstance(child, Field):
                self._populate_fields()
                #self._populate_field(sel, child)
            return child
        # Here this is scipp.DataGroup. Child classes like NXdata may return DataArray.
        # (not scipp.DataArray, as that does not support lazy data)
        dg = self._nexus.read_children(self, sel)
        # TODO assemble geometry/transforms/events
        try:
            return self._nexus.assemble(dg)
        except (sc.DimensionError, NexusStructureError) as e:
            return dg

    @cached_property
    def sizes(self) -> Dict[str, int]:
        return self._nexus.sizes

    @property
    def dims(self) -> Tuple[str, ...]:
        return tuple(self.sizes)


def _guess_dims(dims, shape, dataset: H5Dataset):
    """Guess dims of non-signal dataset based on shape.

    Does not check for potential bin-edge coord.
    """
    if shape == dataset.shape:
        return dims
    lut = {}
    for d, s in zip(dims, shape):
        if shape.count(s) == 1:
            lut[s] = d
    try:
        return [lut[s] for s in dataset.shape]
    except KeyError:
        return None


class NXdata(NXobject):

    def __init__(self,
                 group: Group,
                 fallback_dims: Optional[Tuple[str, ...]] = None,
                 fallback_signal_name: Optional[str] = None):
        super().__init__(group)
        self._valid = True
        # Must do full consistency check here, to define self.sizes:
        # - squeeze correctly
        # - check if coord dims are compatible with signal dims
        # - check if there is a signal
        # If not the case, fall back do DataGroup.sizes
        # Can we just set field dims here?
        self._signal_name = None
        self._signal = None
        if (name := group.attrs.get(
                'signal',
                fallback_signal_name)) is not None and name in group._children:
            self._signal_name = name
            self._signal = group._children[name]
        else:
            # Legacy NXdata defines signal not as group attribute, but attr on dataset
            for name, field in group._children.items():
                # What is the meaning of the attribute value? It is undocumented, we simply
                # ignore it.
                if 'signal' in field.attrs:
                    self._signal_name = name
                    self._signal = group._children[name]
                    break

        axes = group.attrs.get('axes')
        signal_axes = None if self._signal is None else self._signal.attrs.get('axes')

        axis_index = {}
        for name, field in group._children.items():
            if (axis := field.attrs.get('axis')) is not None:
                axis_index[name] = axis

        # Apparently it is not possible to define dim labels unless there are
        # corresponding coords. Special case of '.' entries means "no coord".
        def _get_group_dims():
            if axes is not None:
                return [f'dim_{i}' if a == '.' else a for i, a in enumerate(axes)]
            if signal_axes is not None:
                return tuple(signal_axes.split(','))
            if axis_index:
                return [
                    k for k, _ in sorted(axis_index.items(), key=lambda item: item[1])
                ]
            return None

        group_dims = _get_group_dims()

        if group_dims is None:
            group_dims = fallback_dims

        if self._signal is None:
            self._valid = False
        else:
            if group_dims is not None:
                shape = self._signal.dataset.shape
                shape = _squeeze_trailing(group_dims, shape)
                self._signal.sizes = dict(zip(group_dims, shape))

        if axes is not None:
            # Unlike self.dims we *drop* entries that are '.'
            named_axes = [a for a in axes if a != '.']
        elif signal_axes is not None:
            named_axes = signal_axes.split(',')
        # elif fallback_dims is not None:
        #     named_axes = fallback_dims
        else:
            named_axes = []

        # 3. Find field dims
        indices_suffix = '_indices'
        indices_attrs = {
            key[:-len(indices_suffix)]: attr
            for key, attr in group.attrs.items() if key.endswith(indices_suffix)
        }

        dims = np.array(group_dims)
        dims_from_indices = {
            key: tuple(dims[np.array(indices).flatten()])
            for key, indices in indices_attrs.items()
        }

        def get_dims(name, field):
            # Newly written files should always contain indices attributes, but the
            # standard recommends that readers should also make "best effort" guess
            # since legacy files do not set this attribute.
            # TODO signal and errors?
            # TODO aux
            if name in (self._signal_name, ):
                return group_dims
            # if name in [self._signal_name, self._errors_name]:
            #     return self._get_group_dims()  # if None, field determines dims itself
            # if name in list(self.attrs.get('auxiliary_signals', [])):
            #     return self._try_guess_dims(name)
            if (dims := dims_from_indices.get(name)) is not None:
                return dims
            if (axis := axis_index.get(name)) is not None:
                return (group_dims[axis - 1], )
            if name in named_axes:
                # If there are named axes then items of same name are "dimension
                # coordinates", i.e., have a dim matching their name.
                # However, if the item is not 1-D we need more labels. Try to use labels of
                # signal if dimensionality matches.
                if self._signal is not None and len(field.dataset.shape) == len(
                        self._signal.dataset.shape):
                    return group_dims
                return [name]
            if self._signal is not None and group_dims is not None:
                return _guess_dims(group_dims, self._signal.dataset.shape,
                                   field.dataset)

        for name, field in group._children.items():
            if not isinstance(field, Field):
                self._valid = False
            elif (dims := get_dims(name, field)) is not None:
                # The convention here is that the given dimensions apply to the shapes
                # starting from the left. So we only squeeze dimensions that are after
                # len(dims).
                shape = _squeeze_trailing(dims, field.dataset.shape)
                field.sizes = dict(zip(dims, shape))
            elif self._valid:
                s1 = self._signal.sizes
                s2 = field.sizes
                if not set(s2.keys()).issubset(set(s1.keys())):
                    self._valid = False
                elif any(s1[k] != s2[k] for k in s1.keys() & s2.keys()):
                    self._valid = False

        return
        ################

        indices_suffix = '_indices'
        indices_attrs = {
            key[:-len(indices_suffix)]: attr
            for key, attr in group.attrs.items() if key.endswith(indices_suffix)
        }

        dims = np.array(self._dims)
        self._coord_dims = {
            key: tuple(dims[np.array(indices).flatten()])
            for key, indices in indices_attrs.items()
        }
        self._valid = True
        for name, dataset in group._group.items():
            if name not in self._coord_dims:
                # TODO handle squeezing
                if dataset.shape == ():
                    self._coord_dims[name] = ()
                elif name in self._dims:
                    # If there are named axes then items of same name are "dimension
                    # coordinates", i.e., have a dim matching their name.
                    # However, if the item is not 1-D we need more labels. Try to use labels of
                    # signal if dimensionality matches.
                    if dataset.ndim == len(dims):
                        self._coord_dims[name] = self._dims
                    else:
                        self._coord_dims[name] = (name, )
                elif (field_dims := self._guess_dims(name, dataset)) is not None:
                    self._coord_dims[name] = field_dims
                else:
                    self._valid = False

                #elif name in dims:
                #    self._coord_dims[name] = (name, )
                #elif dataset.shape == group._group[self._signal].shape:
                #    self._coord_dims[name] = self._dims
                #elif len(dataset.shape) == 1:
                #    self._coord_dims[name] = (dims[list(self.sizes.values()).index(
                #        dataset.shape[0])], )

    @cached_property
    def sizes(self) -> Dict[str, int]:
        return self._signal.sizes if self._valid else super().sizes

    def _bin_edge_dim(self, coord: Field) -> Union[None, str]:
        if not isinstance(coord, Field):
            return None
        sizes = self.sizes
        for dim, size in zip(coord.dims, coord.shape):
            if (sz := sizes.get(dim)) is not None and sz + 1 == size:
                return dim
        return None

    def index_child(self, child: Union[Field, Group], sel: ScippIndex) -> ScippIndex:
        child_sel = to_child_select(self._group.dims,
                                    child.dims,
                                    sel,
                                    bin_edge_dim=self._bin_edge_dim(child))
        return child[child_sel]

    def assemble(self, dg: sc.DataGroup) -> Union[sc.DataGroup, sc.DataArray]:
        if not self._valid:
            return super().assemble(dg)
        coords = sc.DataGroup(dg)
        signal = coords.pop(self._signal_name)
        da = sc.DataArray(data=signal)
        coords = {name: asarray(coord) for name, coord in coords.items()}
        return self._add_coords(da, coords)

    def _dim_of_coord(self, name: str, coord: sc.Variable) -> Union[None, str]:
        if len(coord.dims) == 1:
            return coord.dims[0]
        if name in coord.dims and name in self.dims:
            return name
        return self._bin_edge_dim(coord)

    def _coord_to_attr(self, da: sc.DataArray, name: str, coord: sc.Variable) -> bool:
        dim_of_coord = self._dim_of_coord(name, coord)
        if dim_of_coord is None:
            return False
        if dim_of_coord not in da.dims:
            return True
        return False

    def _add_coords(self, da: sc.DataArray, coords: sc.DataGroup) -> sc.DataArray:
        da.coords.update(coords)
        for name in coords:
            #if name not in self:
            #    continue
            if self._coord_to_attr(da, name, self._group[name]):
                da.attrs[name] = da.coords.pop(name)
        return da


def _squeeze_trailing(dims: Tuple[str, ...], shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return shape[:len(dims)] + tuple(size for size in shape[len(dims):] if size != 1)


class NXlog(NXdata):

    def __init__(self, group: Group):
        super().__init__(group, fallback_dims=('time', ), fallback_signal_name='value')
        if (time := self._group._children.get('time')) is not None:
            time._is_time = True


class NXdetector(NXdata):

    def __init__(self, group: Group):
        super().__init__(group, fallback_signal_name='data')


base_definitions = {}
base_definitions['NXdata'] = NXdata
base_definitions['NXlog'] = NXlog
base_definitions['NXdetector'] = NXdetector


def create_field(group: H5Group, name: str, data: DimensionedArray,
                 **kwargs) -> H5Dataset:
    if not isinstance(data, sc.Variable):
        return group.create_dataset(name, data=data, **kwargs)
    values = data.values
    if data.dtype == sc.DType.string:
        values = np.array(data.values, dtype=object)
    elif data.dtype == sc.DType.datetime64:
        start = sc.epoch(unit=data.unit)
        values = (data - start).values
    dataset = group.create_dataset(name, data=values, **kwargs)
    if data.unit is not None:
        dataset.attrs['units'] = str(data.unit)
    if data.dtype == sc.DType.datetime64:
        dataset.attrs['start'] = str(start.value)
    return dataset


def create_class(group: H5Group, name: str, nx_class: Union[str, type]) -> H5Group:
    """Create empty HDF5 group with given name and set the NX_class attribute.

    Parameters
    ----------
    name:
        Group name.
    nx_class:
        Nexus class, can be a valid string for the NX_class attribute, or a
        subclass of NXobject, such as NXdata or NXlog.
    """
    group = group.create_group(name)
    attr = nx_class if isinstance(nx_class, str) else nx_class.__name__
    group.attrs['NX_class'] = attr
    return group
