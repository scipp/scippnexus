# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

import datetime
import inspect
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, Union

import dateutil.parser
import numpy as np
import scipp as sc

from .._common import convert_time_to_datetime64, to_child_select, to_plain_index
from .._hdf5_nexus import _warn_latin1_decode
from ..typing import H5Dataset, H5Group, ScippIndex


def asarray(obj: Union[Any, sc.Variable]) -> sc.Variable:
    return obj if isinstance(obj, sc.Variable) else sc.scalar(obj, unit=None)


# TODO move into scipp
class DimensionedArray(Protocol):
    """
    A multi-dimensional array with a unit and dimension labels.

    Could be, e.g., a scipp.Variable or a dimple dataclass wrapping a numpy array.
    """

    @property
    def values(self):
        """Multi-dimensional array of values"""

    @property
    def unit(self):
        """Physical unit of the values"""

    @property
    def dims(self) -> Tuple[str]:
        """Dimension labels for the values"""


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
    parent: Group
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

    @property
    def dims(self) -> Tuple[str]:
        return tuple(self.sizes.keys())

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.sizes.values())

    @cached_property
    def file(self) -> Group:
        return self.parent.file

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
        if variable.ndim == 0 and variable.unit is None and variable.fields is None:
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
    if (shape := dataset.shape) == (1, ):
        return {}
    return {f'dim_{i}': size for i, size in enumerate(shape)}


class NXobject:

    def _init_field(self, field: Field):
        field.sizes = _squeezed_field_sizes(field.dataset)
        field.dtype = _dtype_fromdataset(field.dataset)

    def __init__(self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]):
        from .nxtransformations import Transformation
        self._attrs = attrs
        self._children = children
        self._special_fields = {}
        self._transformations = {}
        for name, field in children.items():
            if name == 'depends_on':
                self._special_fields[name] = field
            if isinstance(field, Field):
                self._init_field(field)
            elif isinstance(field, Transformation):
                if isinstance(field._obj, Field):
                    self._init_field(field._obj)
            elif (nx_class := field.attrs.get('NX_class')) is not None:
                if nx_class in [
                        'NXoff_geometry',
                        'NXcylindrical_geometry',
                        'NXgeometry',
                ]:
                    self._special_fields[name] = field
                elif nx_class == 'NXtransformations':
                    self._special_fields[name] = field
                    self._transformations[name] = field

    @property
    def unit(self) -> Union[None, sc.Unit]:
        raise ValueError(
            f"Group-like {self._attrs.get('NX_class')} has no well-defined unit")

    @cached_property
    def sizes(self) -> Dict[str, int]:
        # exclude geometry/tansform groups?
        return sc.DataGroup(self._children).sizes

    def index_child(
            self, child: Union[Field, Group], sel: ScippIndex
    ) -> Union[sc.Variable, sc.DataArray, sc.Dataset, sc.DataGroup]:
        # Note that this will be similar in NXdata, but there we need to handle
        # bin edges as well.
        child_sel = to_child_select(self.sizes.keys(), child.dims, sel)
        return child[child_sel]

    def read_children(self, obj: Group, sel: ScippIndex) -> sc.DataGroup:
        return sc.DataGroup(
            {name: self.index_child(child, sel)
             for name, child in obj.items()})

    @property
    def detector_number(self) -> Optional[str]:
        return None

    def pre_assemble(self, dg: sc.DataGroup) -> sc.DataGroup:
        for name, field in self._special_fields.items():
            if name == 'depends_on':
                continue
            if name in self._transformations:
                continue
            det_num = self.detector_number
            if det_num is not None:
                det_num = dg[det_num]
            dg[name] = field._nexus.assemble_as_child(dg[name], detector_number=det_num)
        if (depends_on := dg.get('depends_on')) is not None:
            dg['depends_on'] = sc.scalar(depends_on)
        #    transform = self._children[depends_on]
        #    # Avoid loading transform twice if it is a child of the same group
        #    for name, transformations in self._transformations.items():
        #        if transform.name.startswith(transformations.name):
        #            dg['depends_on'] = dg[name][depends_on.split('/')[-1]]
        #            break
        #    else:
        #        dg['depends_on'] = transform[()]
        return dg

    def assemble(self, dg: sc.DataGroup) -> Union[sc.DataGroup, sc.DataArray]:
        return dg


class NXroot(NXobject):
    pass


class Group(Mapping):

    def __init__(self,
                 group: H5Group,
                 definitions: Optional[Dict[str, type]] = None,
                 parent: Optional[Group] = None):
        self._group = group
        self._definitions = DefinitionsDict() if definitions is None else definitions
        if parent is None:
            if group == group.parent:
                self._parent = self
            else:
                self._parent = Group(group.parent, definitions=definitions)
        else:
            self._parent = parent

    @property
    def nx_class(self) -> Optional[type]:
        """The value of the NX_class attribute of the group.

        In case of the subclass NXroot this returns 'NXroot' even if the attribute
        is not actually set. This is to support the majority of all legacy files, which
        do not have this attribute.
        """
        if (nxclass := self.attrs.get('NX_class')) is not None:
            return _nx_class_registry().get(nxclass)
        if self.name == '/':
            return NXroot

    @cached_property
    def attrs(self) -> Dict[str, Any]:
        # Attrs are not read until needed, to avoid reading all attrs for all subgroups.
        # We may expected a per-subgroup overhead of 1 ms for reading attributes, so if
        # all we want is access one attribute, we may save, e.g., a second for a group
        # with 1000 subgroups.
        return dict(self._group.attrs) if self._group.attrs else dict()

    @property
    def name(self) -> str:
        return self._group.name

    @property
    def unit(self) -> Optional[sc.Unit]:
        return self._nexus.unit

    @property
    def parent(self) -> Optional[Group]:
        return self._parent

    @cached_property
    def file(self) -> Optional[Group]:
        return self if self == self.parent else self.parent.file

    @cached_property
    def _children(self) -> Dict[str, Union[Field, Group]]:
        # Transformations should be stored in NXtransformations, which is cumbersome
        # to handle, since we need to check the parent of a transform to tell whether
        # it is a transform. However, we can avoid this by simply treating everything
        # referenced by a 'depends_on' field or attribute as a transform.
        from .nxtransformations import Transformation

        def _make_child(
                name: str, obj: Union[H5Dataset,
                                      H5Group]) -> Union[Transformation, Field, Group]:
            if name == 'depends_on':
                target = obj[()]
                obj = obj.parent[target]
                # TODO Bad, we are recreating the group
                parent = Group(obj.parent, definitions=self._definitions)
            else:
                parent = self
            if is_dataset(obj):
                child = Field(obj, parent=parent)
            else:
                child = Group(obj, parent=parent, definitions=self._definitions)
            return Transformation(child) if name == 'depends_on' else child

        items = {name: _make_child(name, obj) for name, obj in self._group.items()}
        for suffix in ('_errors', '_error'):
            field_with_errors = [name for name in items if f'{name}{suffix}' in items]
            for name in field_with_errors:
                values = items[name]
                errors = items[f'{name}{suffix}']
                if (values.unit == errors.unit
                        and values.dataset.shape == errors.dataset.shape):
                    values.errors = errors.dataset
                    del items[f'{name}{suffix}']
        items = {k: v for k, v in items.items() if not k.startswith('cue_')}
        return items

    @cached_property
    def _nexus(self) -> NXobject:
        return self._definitions.get(self.attrs.get('NX_class'),
                                     group=self)(attrs=self.attrs,
                                                 children=self._children)

    def _populate_fields(self) -> None:
        _ = self._nexus

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[str]:
        return self._children.__iter__()

    @cached_property
    def _is_nxtransformations(self) -> bool:
        return self.attrs.get('NX_class') == 'NXtransformations'

    def _get_children_by_nx_class(
            self, select: Union[type, List[type]]) -> Dict[str, Union[NXobject, Field]]:
        children = {}
        select = tuple(select) if isinstance(select, list) else select
        for key, child in self._children.items():
            nx_class = Field if isinstance(child, Field) else child.nx_class
            if issubclass(nx_class, select):
                children[key] = self[key]
        return children

    def __getitem__(self, sel) -> Union[Field, Group, sc.DataGroup]:
        if isinstance(sel, str):
            # We cannot get the child directly from the HDF5 group, since we need to
            # create the parent group, to ensure that fields get the correct properties
            # such as sizes and dtype.
            if '/' in sel:
                if sel.startswith('/'):
                    return self.file[sel[1:]]
                else:
                    return self[sel.split('/')[0]][sel[sel.index('/') + 1:]]
            child = self._children[sel]
            from .nxtransformations import Transformation
            if isinstance(child, (Field, Transformation)):
                self._populate_fields()
            if self._is_nxtransformations:
                return Transformation(child)
            return child

        def isclass(x):
            return inspect.isclass(x) and issubclass(x, (Field, NXobject))

        if isclass(sel) or (isinstance(sel, list) and len(sel)
                            and all(isclass(x) for x in sel)):
            return self._get_children_by_nx_class(sel)
        # Here this is scipp.DataGroup. Child classes like NXdata may return DataArray.
        # (not scipp.DataArray, as that does not support lazy data)
        dg = self._nexus.read_children(self, sel)
        try:
            dg = self._nexus.pre_assemble(dg)
            return self._nexus.assemble(dg)
        except (sc.DimensionError, NexusStructureError) as e:
            print(e)
            # TODO log warning
            return dg

    # TODO It is not clear if we want to support these convenience methods
    def __setitem__(self, key, value):
        if hasattr(value, '__write_to_nexus_group__'):
            group = create_class(self._group, key, nx_class=value.nx_class)
            value.__write_to_nexus_group__(group)
        else:
            create_field(self._group, key, value)

    def create_field(self, key: str, value: sc.Variable) -> H5Dataset:
        return create_field(self._group, key, value)

    def create_class(self, name, class_name: str) -> Group:
        return Group(create_class(self._group, name, class_name),
                     definitions=self._definitions,
                     parent=self)

    def rebuild(self) -> Group:
        return Group(self._group, definitions=self._definitions, parent=self.parent)

    @cached_property
    def sizes(self) -> Dict[str, int]:
        return self._nexus.sizes

    @property
    def dims(self) -> Tuple[str, ...]:
        return tuple(self.sizes)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.sizes.values())


class NXgeometry(NXobject):

    def __init__(self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]):
        super().__init__(attrs=attrs, children=children)

    @staticmethod
    def assemble_as_child(children: sc.DataGroup,
                          detector_number: Optional[sc.Variable] = None) -> sc.Variable:
        return sc.scalar(children)


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


@lru_cache()
def _nx_class_registry():
    from . import nexus_classes
    return dict(inspect.getmembers(nexus_classes, inspect.isclass))


class DefinitionsDict:

    def __init__(self):
        self._definitions = {}

    def __setitem__(self, nx_class: str, definition: type):
        self._definitions[nx_class] = definition

    def get(self, nx_class: str, group: Group) -> type:
        return self._definitions.get(nx_class, NXobject)


base_definitions = DefinitionsDict()
base_definitions['NXgeometry'] = NXgeometry
