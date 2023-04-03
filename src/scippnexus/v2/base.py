# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

import datetime
import inspect
import posixpath
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, Union, overload

import dateutil.parser
import numpy as np
import scipp as sc

from .._common import convert_time_to_datetime64, to_child_select, to_plain_index
from .._hdf5_nexus import _warn_latin1_decode
from ..typing import H5Dataset, H5Group, ScippIndex


def asvariable(obj: Union[Any, sc.Variable]) -> sc.Variable:
    return obj if isinstance(obj, sc.Variable) else sc.scalar(obj, unit=None)


def depends_on_to_relative_path(depends_on: str, parent_path: str) -> str:
    """Replace depends_on paths with relative paths.

    After loading we will generally not have the same root so absolute paths
    cannot be resolved after loading."""
    if depends_on.startswith('/'):
        return posixpath.relpath(depends_on, parent_path)
    return depends_on


# TODO move into scipp
class DimensionedArray(Protocol):
    """
    A multi-dimensional array with a unit and dimension labels.

    Could be, e.g., a scipp.Variable or a simple dataclass wrapping a numpy array.
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
    """NeXus field.

    In HDF5 fields are represented as dataset.
    """
    dataset: H5Dataset
    parent: Group
    sizes: Optional[Dict[str, int]] = None
    dtype: Optional[sc.DType] = None
    errors: Optional[H5Dataset] = None

    @cached_property
    def attrs(self) -> Mapping[str, Any]:
        """The attributes of the dataset.

        Cannot be used for writing attributes, since they are cached for performance."""
        return MappingProxyType(
            dict(self.dataset.attrs) if self.dataset.attrs else dict())

    @property
    def dims(self) -> Tuple[str, ...]:
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

    def __getitem__(self, select: ScippIndex) -> Union[Any, sc.Variable]:
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
            return self._maybe_datetime(variable)

        if self.dtype == sc.DType.string:
            try:
                strings = self.dataset.asstr()[index]
            except UnicodeDecodeError as e:
                strings = self.dataset.asstr(encoding='latin-1')[index]
                _warn_latin1_decode(self.dataset, strings, str(e))
            variable.values = np.asarray(strings).flatten()
            if self.dataset.name.endswith('depends_on') and variable.ndim == 0:
                variable.value = depends_on_to_relative_path(variable.value,
                                                             self.dataset.parent.name)
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
        from .nxtransformations import maybe_transformation
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
                    scaling_factor=self.attrs.get('scaling_factor'))

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
        """Subclasses should call this in their __init__ method, or ensure that they
        initialize the fields in `children` with the correct sizes and dtypes."""
        self._attrs = attrs
        self._children = children
        for field in children.values():
            if isinstance(field, Field):
                self._init_field(field)

    @property
    def unit(self) -> Union[None, sc.Unit]:
        raise AttributeError(
            f"Group-like {self._attrs.get('NX_class')} has no well-defined unit")

    @cached_property
    def sizes(self) -> Dict[str, int]:
        return sc.DataGroup(self._children).sizes

    def index_child(
            self, child: Union[Field, Group], sel: ScippIndex
    ) -> Union[sc.Variable, sc.DataArray, sc.Dataset, sc.DataGroup]:
        """
        When a Group is indexed, this method is called to index each child.

        The main purpose of this is to translate the Group index to the child index.
        Since the group dimensions (usually given by the signal) may be a superset of
        the child dimensions, we need to translate the group index to a child index.

        The default implementation assumes that the child shape is identical to the
        group shape, for all child dims. Subclasses of NXobject, in particular NXdata,
        override this method to handle bin edges.
        """
        # TODO Could avoid determining sizes if sel is trivial. Merge with
        # NXdata.index_child?
        child_sel = to_child_select(tuple(self.sizes), child.dims, sel)
        return child[child_sel]

    def read_children(self, obj: Group, sel: ScippIndex) -> sc.DataGroup:
        """
        When a Group is indexed, this method is called to read all children.

        The default implementation simply calls index_child on each child and returns
        the result as a DataGroup.

        Subclasses of NXobject, in particular NXevent_data, override this method to
        to implement special logic for reading children with interdependencies, i.e.,
        where reading each child in isolation is not possible.
        """
        return sc.DataGroup(
            {name: self.index_child(child, sel)
             for name, child in obj.items()})

    def assemble(self,
                 dg: sc.DataGroup) -> Union[sc.DataGroup, sc.DataArray, sc.Dataset]:
        """
        When a Group is indexed, this method is called to assemble the read children
        into the result object.

        The default implementation simply returns the DataGroup.

        Subclasses of NXobject, in particular NXdata, override this method to return
        an object with more semantics such as a DataArray or Dataset.
        """
        return dg


class NXroot(NXobject):
    pass


class Group(Mapping):
    """
    A group in a NeXus file.

    This class is a wrapper around an h5py.Group object. It provides a dict-like
    interface to the children of the group, and provides access to the attributes
    of the group. The children are either Field or Group objects, depending on
    whether the child is a dataset or a group, respectively.
    """

    # The implementation of this class is unfortunately relatively complex:

    # 1. NeXus requires "nonlocal" information for interpreting a field. For example,
    #    NXdata attributes define which fields are the signal, and the names of axes.
    #    A field cannot be read without this information, in particular since we want to
    #    support reading slices, using the Scipp dimension-label syntax.
    # 2. HDF5 or h5py performance is not great, and we want to avoid reading the same
    #    attrs or datasets multiple times. We can therefore not rely on "on-the-fly"
    #    interpretation of the file, but need to cache information. An earlier version
    #    of ScippNexus used such a mechanism without caching, which was very slow.

    def __init__(self, group: H5Group, definitions: Optional[Dict[str, type]] = None):
        self._group = group
        self._definitions = {} if definitions is None else definitions
        self._lazy_children = None
        self._lazy_nexus = None

    @property
    def nx_class(self) -> Optional[type]:
        """The value of the NX_class attribute of the group.

        In case of the subclass NXroot this returns :py:class:`NXroot` even if the attr
        is not actually set. This is to support the majority of all legacy files, which
        do not have this attribute.
        """
        if (nxclass := self.attrs.get('NX_class')) is not None:
            return _nx_class_registry().get(nxclass)
        if self.name == '/':
            return NXroot

    @cached_property
    def attrs(self) -> Dict[str, Any]:
        """The attributes of the group.

        Cannot be used for writing attributes, since they are cached for performance."""
        # Attrs are not read until needed, to avoid reading all attrs for all subgroups.
        # We may expected a per-subgroup overhead of 1 ms for reading attributes, so if
        # all we want is access one subgroup, we may save, e.g., a second for a group
        # with 1000 subgroups (or subfields).
        return MappingProxyType(
            dict(self._group.attrs) if self._group.attrs else dict())

    @property
    def name(self) -> str:
        return self._group.name

    @property
    def unit(self) -> Optional[sc.Unit]:
        return self._nexus.unit

    @property
    def parent(self) -> Group:
        return Group(self._group.parent, definitions=self._definitions)

    @cached_property
    def file(self) -> Group:
        return Group(self._group.file, definitions=self._definitions)

    @property
    def _children(self) -> Dict[str, Union[Field, Group]]:
        """Lazily initialized children of the group."""
        if self._lazy_children is None:
            self._lazy_children = self._read_children()
        return self._lazy_children

    def _read_children(self) -> Dict[str, Union[Field, Group]]:

        def _make_child(obj: Union[H5Dataset, H5Group]) -> Union[Field, Group]:
            if is_dataset(obj):
                return Field(obj, parent=self)
            else:
                return Group(obj, definitions=self._definitions)

        items = {name: _make_child(obj) for name, obj in self._group.items()}
        items = {k: v for k, v in items.items() if not k.startswith('cue_')}
        for suffix in ('_errors', '_error'):
            field_with_errors = [name for name in items if f'{name}{suffix}' in items]
            for name in field_with_errors:
                values = items[name]
                errors = items[f'{name}{suffix}']
                if (isinstance(values, Field) and isinstance(errors, Field)
                        and values.unit == errors.unit
                        and values.dataset.shape == errors.dataset.shape):
                    values.errors = errors.dataset
                    del items[f'{name}{suffix}']
        return items

    @property
    def _nexus(self) -> NXobject:
        """Instance of the NXobject subclass corresponding to the NX_class attribute.

        This is used to determine dims, unit, and other attributes of the group and its
        children, as well as defining how children will be read and assembled into the
        result object when the group is indexed.

        Lazily initialized since the NXobject subclass init can be costly.
        """
        if self._lazy_nexus is None:
            self._populate_fields()
        return self._lazy_nexus

    def _populate_fields(self) -> None:
        """Populate the fields of the group.

        Fields are not populated until needed, to avoid reading field and group
        properties for all subgroups. However, when any field is read we must in
        general parse all the field and group properties, since for classes such
        as NXdata the properties of any field may indirectly depend on the properties
        of any other field. For example, field attributes may define which fields are
        axes, and dim labels of other fields can be defined by the names of the axes.
        """
        self._lazy_nexus = self._definitions.get(self.attrs.get('NX_class'),
                                                 NXobject)(attrs=self.attrs,
                                                           children=self._children)

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[str]:
        return self._children.__iter__()

    def _get_children_by_nx_class(
            self, select: Union[type, List[type]]) -> Dict[str, Union[NXobject, Field]]:
        children = {}
        select = tuple(select) if isinstance(select, list) else select
        for key, child in self._children.items():
            nx_class = Field if isinstance(child, Field) else child.nx_class
            if issubclass(nx_class, select):
                children[key] = self[key]
        return children

    @overload
    def __getitem__(self, sel: str) -> Union[Group, Field]:
        ...

    @overload
    def __getitem__(self,
                    sel: ScippIndex) -> Union[sc.DataArray, sc.DataGroup, sc.Dataset]:
        ...

    @overload
    def __getitem__(self, sel: Union[type, List[type]]) -> Dict[str, NXobject]:
        ...

    def __getitem__(self, sel):
        """
        Get a child group or child dataset, a selection of child groups, or load and
        return the current group.

        Three cases are supported:

        - String name: The child group or child dataset of that name is returned.
        - Class such as ``NXdata`` or ``NXlog``: A dict containing all direct children
          with a matching ``NX_class`` attribute are returned. Also accepts a tuple of
          classes. ``Field`` selects all child fields, i.e., all datasets but not
          groups.
        - Scipp-style index: Load the specified slice of the current group, returning
          a :class:`scipp.DataArray` or :class:`scipp.DataGroup`.

        Parameters
        ----------
        name:
            Child name, class, or index.

        Returns
        -------
        :
            Field, group, dict of fields, or loaded data.
        """
        if isinstance(sel, str):
            # We cannot get the child directly from the HDF5 group, since we need to
            # create the parent group, to ensure that fields get the correct properties
            # such as sizes and dtype.
            if '/' in sel:
                if sel.startswith('/'):
                    return self.file[sel[1:]]
                else:
                    grp, path = sel.split('/', 1)
                    return self[grp][path]
            child = self._children[sel]
            if isinstance(child, Field):
                self._populate_fields()
            return child

        def isclass(x):
            return inspect.isclass(x) and issubclass(x, (Field, NXobject))

        if isclass(sel) or (isinstance(sel, list) and len(sel)
                            and all(isclass(x) for x in sel)):
            return self._get_children_by_nx_class(sel)

        dg = self._nexus.read_children(self, sel)
        try:
            dg = self._nexus.assemble(dg)
        except (sc.DimensionError, NexusStructureError) as e:
            self._warn_fallback(e)
        # For a time-dependent transformation in NXtransformations, an NXlog may
        # take the place of the `value` field. In this case, we need to read the
        # properties of the NXlog group to make the actual transformation.
        from .nxtransformations import maybe_transformation
        return maybe_transformation(self, value=dg, sel=sel)

    def _warn_fallback(self, e: Exception) -> None:
        msg = (f"Failed to load {self.name} as {type(self._nexus).__name__}: {e} "
               "Falling back to loading HDF5 group children as scipp.DataGroup.")
        warnings.warn(msg)

    def __setitem__(self, key, value):
        """Set a child group or child dataset.

        Note that due to the caching mechanisms in this class, reading the group
        or its children may not reflect the changes made by this method."""
        if hasattr(value, '__write_to_nexus_group__'):
            group = create_class(self._group, key, nx_class=value.nx_class)
            value.__write_to_nexus_group__(group)
        else:
            create_field(self._group, key, value)

    def create_field(self, key: str, value: sc.Variable) -> H5Dataset:
        """Create a child dataset with given name and value.

        Note that due to the caching mechanisms in this class, reading the group
        or its children may not reflect the changes made by this method."""
        return create_field(self._group, key, value)

    def create_class(self, name: str, class_name: str) -> Group:
        """Create empty HDF5 group with given name and set the NX_class attribute.

        Note that due to the caching mechanisms in this class, reading the group
        or its children may not reflect the changes made by this method.

        Parameters
        ----------
        name:
            Group name.
        nx_class:
            Nexus class, can be a valid string for the NX_class attribute, or a
            subclass of NXobject, such as NXdata or NXlog.
        """
        return Group(create_class(self._group, name, class_name),
                     definitions=self._definitions)

    @cached_property
    def sizes(self) -> Dict[str, int]:
        return self._nexus.sizes

    @property
    def dims(self) -> Tuple[str, ...]:
        return tuple(self.sizes)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.sizes.values())


def create_field(group: H5Group, name: str, data: Union[np.ndarray, DimensionedArray],
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


base_definitions = {}
