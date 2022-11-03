# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
import re
import inspect
import warnings
import datetime
import dateutil.parser
import functools
from typing import overload, List, Union, Any, Dict, Tuple, Protocol, Optional, Callable
import numpy as np
import scipp as sc
import h5py

from ._hdf5_nexus import _cset_to_encoding, _ensure_str
from ._hdf5_nexus import _ensure_supported_int_type, _warn_latin1_decode
from .typing import H5Group, H5Dataset, ScippIndex
from ._common import to_plain_index
from ._common import convert_time_to_datetime64

NXobjectIndex = Union[str, ScippIndex]


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
    def dims(self) -> List[str]:
        """Dimension labels for the values"""


class AttributeManager(Protocol):

    def __getitem__(self, name: str):
        """Get attribute"""


class NexusStructureError(Exception):
    """Invalid or unsupported class and field structure in Nexus.
    """
    pass


class Attrs:
    """HDF5 attributes.
    """

    def __init__(self, attrs: AttributeManager):
        self._attrs = attrs

    def __contains__(self, name: str) -> bool:
        return name in self._attrs

    def __getitem__(self, name: str) -> Any:
        attr = self._attrs[name]
        # Is this check for string attributes sufficient? Is there a better way?
        if isinstance(attr, (str, bytes)):
            cset = self._attrs.get_id(name.encode("utf-8")).get_type().get_cset()
            return _ensure_str(attr, _cset_to_encoding(cset))
        return attr

    def __setitem__(self, name: str, val: Any):
        self._attrs[name] = val

    def __iter__(self):
        yield from self._attrs

    def get(self, name: str, default=None) -> Any:
        return self[name] if name in self else default

    def keys(self):
        return self._attrs.keys()


def _is_time(obj):
    dummy = sc.empty(dims=[], shape=[], unit=obj.unit)
    try:
        dummy.to(unit='s')
        return True
    except sc.UnitError:
        return False


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


class Field:
    """NeXus field.

    In HDF5 fields are represented as dataset.
    """

    def __init__(self, dataset: H5Dataset, *, ancestor, dims=None, is_time=None):
        self._ancestor = ancestor  # Ususally the parent, but may be grandparent, etc.
        self._dataset = dataset
        self._shape = self._dataset.shape
        self._is_time = is_time
        # NeXus treats [] and [1] interchangeably. In general this is ill-defined, but
        # the best we can do appears to be squeezing unless the file provides names for
        # dimensions. The shape property of this class does thus not necessarily return
        # the same as the shape of the underlying dataset.
        if dims is not None:
            self._dims = tuple(dims)
            if len(self._dims) < len(self._shape):
                # The convention here is that the given dimensions apply to the shapes
                # starting from the left. So we only squeeze dimensions that are after
                # len(dims).
                self._shape = self._shape[:len(self._dims)] + tuple(
                    size for size in self._shape[len(self._dims):] if size != 1)
        elif (axes := self.attrs.get('axes')) is not None:
            self._dims = tuple(axes.split(','))
        else:
            self._shape = tuple(size for size in self._shape if size != 1)
            self._dims = tuple(f'dim_{i}' for i in range(self.ndim))

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

        variable = sc.empty(dims=dims, shape=shape, dtype=self.dtype, unit=self.unit)

        # If the variable is empty, return early
        if np.prod(shape) == 0:
            return variable

        if self.dtype == sc.DType.string:
            try:
                strings = self._dataset.asstr()[index]
            except UnicodeDecodeError as e:
                strings = self._dataset.asstr(encoding='latin-1')[index]
                _warn_latin1_decode(self._dataset, strings, str(e))
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
                self._dataset.read_direct(variable.values, source_sel=index)
            except TypeError:
                variable.values = self._dataset[index].squeeze()
        else:
            variable.values = self._dataset[index]
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
        return f'<Nexus field "{self._dataset.name}">'

    @property
    def attrs(self) -> Attrs:
        return Attrs(self._dataset.attrs)

    @property
    def dtype(self) -> str:
        dtype = self._dataset.dtype
        if str(dtype).startswith('str') or h5py.check_string_dtype(dtype):
            dtype = sc.DType.string
        else:
            dtype = sc.DType(_ensure_supported_int_type(str(dtype)))
        return dtype

    @property
    def name(self) -> str:
        return self._dataset.name

    @property
    def file(self) -> NXroot:
        return NXroot(self._dataset.file)

    @property
    def parent(self) -> NXobject:
        return self._ancestor._make(self._dataset.parent)

    @property
    def ndim(self) -> int:
        """Total number of dimensions in the dataset.

        See the shape property for potential differences to the value returned by the
        underlying h5py.Dataset.ndim.
        """
        return len(self.shape)

    @property
    def shape(self) -> List[int]:
        """Shape of the field.

        NeXus may use extra dimensions of length one to store data, such as shape=[1]
        instead of shape=[]. This property returns the *squeezed* shape, dropping all
        length-1 dimensions that are not explicitly named. The returned shape may thus
        be different from the shape of the underlying h5py.Dataset.
        """
        return self._shape

    @property
    def dims(self) -> List[str]:
        return self._dims

    @property
    def unit(self) -> Union[sc.Unit, None]:
        if (unit := self.attrs.get('units')) is not None:
            try:
                return sc.Unit(unit)
            except sc.UnitError:
                warnings.warn(f"Unrecognized unit '{unit}' for value dataset "
                              f"in '{self.name}'; setting unit as 'dimensionless'")
                return sc.units.one
        return None


def is_dataset(obj: Union[H5Group, H5Dataset]) -> bool:
    """Return true if the object is an h5py.Dataset or equivalent.

    Use this instead of isinstance(obj, h5py.Dataset) to ensure that code is compatible
    with other h5py-alike interfaces.
    """
    return hasattr(obj, 'shape')


class NXobject:
    """Base class for all NeXus groups.
    """

    def __init__(self,
                 group: H5Group,
                 *,
                 definition: Any = None,
                 strategy: Optional[Callable] = None):
        self._group = group
        # TODO can strategies replace child-params?
        self.child_params = {}
        self._definition = definition
        if strategy is not None:
            self._strategy = strategy
        elif self._definition is not None:
            self._strategy = self._definition.make_strategy(self)
        else:
            self._strategy = self._default_strategy()

    def _default_strategy(self):
        """
        Default strategy to use when none given and when the application definition
        does not provide one. Override in child classes to set a default.
        """
        return None

    def _make(self, group) -> NXobject:
        if (nx_class := Attrs(group.attrs).get('NX_class')) is not None:
            return _nx_class_registry().get(nx_class,
                                            NXobject)(group,
                                                      definition=self._definition)
        return group  # Return underlying (h5py) group

    def _get_child(
            self,
            name: NXobjectIndex,
            use_field_dims: bool = False) -> Union['NXobject', Field, sc.DataArray]:
        """Get item, with flag to control whether fields dims should be inferred"""
        if name is None:
            raise KeyError("None is not a valid index")
        if isinstance(name, str):
            item = self._group[name]
            if is_dataset(item):
                dims = self._get_field_dims(name) if use_field_dims else None
                return Field(item,
                             dims=dims,
                             ancestor=self,
                             **self.child_params.get(name, {}))
            else:
                return self._make(item)
        da = self._getitem(name)
        if (t := self.depends_on) is not None:
            if isinstance(da, dict):
                da['depends_on'] = t
            else:
                da.coords['depends_on'] = t if isinstance(t,
                                                          sc.Variable) else sc.scalar(t)
        return da

    def _get_children_by_nx_class(
            self, select: Union[type,
                                List[type]]) -> Dict[str, Union['NXobject', Field]]:
        children = {}
        select = tuple(select) if isinstance(select, list) else select
        for key in self.keys():
            if issubclass(type(self._get_child(key)), select):
                # Get child again via __getitem__ so correct field dims are used.
                children[key] = self[key]
        return children

    @overload
    def __getitem__(self, name: str) -> Union['NXobject', Field]:
        ...

    @overload
    def __getitem__(self, name: ScippIndex) -> Union[sc.DataArray, sc.Dataset]:
        ...

    @overload
    def __getitem__(self, name: Union[type, Tuple[type]]) -> Dict[str, 'NXobject']:
        ...

    def __getitem__(self, name):
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
          a :class:`scipp.DataArray` or :class:`scipp.Dataset`.

        Parameters
        ----------
        name:
            Child name, class, or index.

        Returns
        -------
        :
            Field, group, dict of fields, or loaded data.
        """

        def isclass(x):
            return inspect.isclass(x) and issubclass(x, (Field, NXobject))

        if isclass(name) or (isinstance(name, list) and len(name)
                             and all(isclass(x) for x in name)):
            return self._get_children_by_nx_class(name)
        return self._get_child(name, use_field_dims=True)

    def _getitem(self, index: ScippIndex) -> Union[sc.DataArray, sc.Dataset]:
        raise NotImplementedError(f'Loading {self.nx_class} is not supported.')

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        """Subclasses should reimplement this to provide dimension labels for fields."""
        return None

    def __contains__(self, name: str) -> bool:
        return name in self._group

    def get(self, name: str, default=None) -> Union['NXobject', Field, sc.DataArray]:
        return self[name] if name in self else default

    @property
    def attrs(self) -> Attrs:
        return Attrs(self._group.attrs)

    @property
    def name(self) -> str:
        return self._group.name

    @property
    def file(self) -> NXroot:
        return NXroot(self._group.file)

    @property
    def parent(self) -> NXobject:
        return self._make(self._group.parent)

    def _ipython_key_completions_(self) -> List[str]:
        return list(self.keys())

    def __iter__(self):
        yield from self._group

    def keys(self) -> List[str]:
        return self._group.keys()

    def values(self) -> List[Union[Field, 'NXobject']]:
        return [self[name] for name in self.keys()]

    def items(self) -> List[Tuple[str, Union[Field, 'NXobject']]]:
        return list(zip(self.keys(), self.values()))

    @property
    def nx_class(self) -> Optional[type]:
        """The value of the NX_class attribute of the group.

        In case of the subclass NXroot this returns 'NXroot' even if the attribute
        is not actually set. This is to support the majority of all legacy files, which
        do not have this attribute.
        """
        if (nxclass := self.attrs.get('NX_class')) is not None:
            return _nx_class_registry().get(nxclass)

    @property
    def depends_on(self) -> Union[sc.Variable, sc.DataArray, None]:
        if (depends_on := self.get('depends_on')) is not None:
            # Imported late to avoid cyclic import
            from .nxtransformations import get_full_transformation
            return get_full_transformation(depends_on)
        return None

    def __repr__(self) -> str:
        return f'<{type(self).__name__} "{self._group.name}">'

    def create_field(self, name: str, data: DimensionedArray, **kwargs) -> Field:
        if not isinstance(data, sc.Variable):
            return self._group.create_dataset(name, data=data, **kwargs)
        values = data.values
        if data.dtype == sc.DType.string:
            values = np.array(data.values, dtype=object)
        elif data.dtype == sc.DType.datetime64:
            start = sc.epoch(unit=data.unit)
            values = (data - start).values
        dataset = self._group.create_dataset(name, data=values, **kwargs)
        if data.unit is not None:
            dataset.attrs['units'] = str(data.unit)
        if data.dtype == sc.DType.datetime64:
            dataset.attrs['start'] = str(start.value)
        return Field(dataset, dims=data.dims, ancestor=self)

    def create_class(self, name: str, nx_class: Union[str, type]) -> NXobject:
        """Create empty HDF5 group with given name and set the NX_class attribute.

        Parameters
        ----------
        name:
            Group name.
        nx_class:
            Nexus class, can be a valid string for the NX_class attribute, or a
            subclass of NXobject, such as NXdata or NXlog.
        """
        group = self._group.create_group(name)
        attr = nx_class if isinstance(nx_class, str) else nx_class.__name__
        group.attrs['NX_class'] = attr
        return self._make(group)

    def __setitem__(self, name: str, value: Union[Field, NXobject, DimensionedArray]):
        """Create a link or a new field."""
        if isinstance(value, Field):
            self._group[name] = value._dataset
        elif isinstance(value, NXobject):
            self._group[name] = value._group
        elif hasattr(value, '__write_to_nexus_group__'):
            group = self.create_class(name, nx_class=value.nx_class)
            value.__write_to_nexus_group__(group)
        else:
            self.create_field(name, value)

    def __getattr__(self, attr: str) -> Union[Any, 'NXobject']:
        nxclass = _nx_class_registry().get(f'NX{attr}')
        if nxclass is None:
            raise AttributeError(f"'NXobject' object has no attribute {attr}")
        matches = self[nxclass]
        if len(matches) == 0:
            raise NexusStructureError(f"No group with requested NX_class='{nxclass}'")
        if len(matches) == 1:
            return next(iter(matches.values()))
        raise NexusStructureError(f"Multiple keys match {nxclass}, use obj[{nxclass}] "
                                  f"to obtain all matches instead of obj.{attr}.")

    def __dir__(self):
        keys = super().__dir__()
        nxclasses = []
        # Avoiding self.values() since it is more costly, but mainly since there may be
        # edge cases where creation of Field/NXobject may raise on unrelated children.
        for name, val in self._group.items():
            if not is_dataset(val):
                nxclasses.append(self._make(val).nx_class)
        for key in set(nxclasses):
            if key is None:
                continue
            if key in keys:
                continue
            if nxclasses.count(key) == 1:
                keys.append(key.__name__[2:])
        return keys


class NXroot(NXobject):
    """Root of a NeXus file."""

    @property
    def nx_class(self) -> type:
        # As an oversight in the NeXus standard and the reference implementation,
        # the NX_class was never set to NXroot. This applies to essentially all
        # files in existence before 2016, and files written by other implementations
        # that were inspired by the reference implementation. We thus hardcode NXroot:
        return NXroot


@functools.lru_cache()
def _nx_class_registry():
    from . import nexus_classes
    return dict(inspect.getmembers(nexus_classes, inspect.isclass))
