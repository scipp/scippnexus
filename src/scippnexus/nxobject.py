# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
import warnings
from enum import Enum, auto
import functools
from typing import List, Union, NoReturn, Any, Dict, Tuple, Protocol
import dateutil
import numpy as np
import scipp as sc
import h5py

from ._hdf5_nexus import _cset_to_encoding, _ensure_str
from ._hdf5_nexus import _ensure_supported_int_type, _warn_latin1_decode
from .typing import H5Group, H5Dataset, ScippIndex
from ._common import to_plain_index
from ._common import convert_time_to_datetime64

NXobjectIndex = Union[str, ScippIndex]


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


class NX_class(Enum):
    NXdata = auto()
    NXdetector = auto()
    NXdisk_chopper = auto()
    NXentry = auto()
    NXevent_data = auto()
    NXinstrument = auto()
    NXlog = auto()
    NXmonitor = auto()
    NXroot = auto()
    NXsample = auto()
    NXsource = auto()
    NXtransformations = auto()


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
            # datetime.fromisoformat cannot parse time zones and recommends dateutil
            dt = dateutil.parser.isoparse(obj)
            # NumPy and scipp cannot handle timezone information. We therefore strip it,
            # i.e., interpret time as local time. If time is given in UTC this will lead
            # to misleading results since we have no information about the actual time
            # zone.
            dt = dt.replace(tzinfo=None)
            return sc.datetime(np.datetime64(dt), unit='ns')
        except ValueError:
            pass
    return None


class Field:
    """NeXus field.

    In HDF5 fields are represented as dataset.
    """
    def __init__(self, dataset: H5Dataset, dims=None, is_time=None):
        self._dataset = dataset
        self._shape = list(self._dataset.shape)
        self._is_time = is_time
        # NeXus treats [] and [1] interchangeably. In general this is ill-defined, but
        # the best we can do appears to be squeezing unless the file provides names for
        # dimensions. The shape property of this class does thus not necessarily return
        # the same as the shape of the underlying dataset.
        if dims is not None:
            self._dims = dims
            if len(self._dims) < len(self._shape):
                self._shape = [size for size in self._shape if size != 1]
        elif (axes := self.attrs.get('axes')) is not None:
            self._dims = axes.split(',')
        else:
            self._shape = [size for size in self._shape if size != 1]
            self._dims = [f'dim_{i}' for i in range(self.ndim)]

    def __getitem__(self, select) -> sc.Variable:
        index = to_plain_index(self.dims, select)
        if isinstance(index, slice):
            index = (index, )

        shape = list(self.shape)
        for i, ind in enumerate(index):
            shape[i] = len(range(*ind.indices(shape[i])))

        variable = sc.empty(dims=self.dims,
                            shape=shape,
                            dtype=self.dtype,
                            unit=self.unit)
        if self.dtype == sc.DType.string:
            try:
                strings = self._dataset.asstr()[index]
            except UnicodeDecodeError as e:
                strings = self._dataset.asstr(encoding='latin-1')[index]
                _warn_latin1_decode(self._dataset, strings, str(e))
            variable.values = np.asarray(strings).flatten()
        elif variable.values.flags["C_CONTIGUOUS"] and variable.values.size > 0:
            self._dataset.read_direct(variable.values, source_sel=index)
        else:
            variable.values = self._dataset[index]
        if self._is_time or _is_time(variable):
            starts = []
            for name in self.attrs:
                if (dt := _as_datetime(self.attrs[name])) is not None:
                    starts.append(dt)
            if self._is_time and len(starts) == 0:
                starts.append(sc.epoch(unit='ns'))
            if len(starts) == 1:
                variable = convert_time_to_datetime64(
                    variable,
                    start=starts[0],
                    scaling_factor=self.attrs.get('scaling_factor'))
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
        return _make(self._dataset.parent)

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


class NXobject:
    """Base class for all NeXus groups.
    """
    def __init__(self, group: H5Group):
        self._group = group
        self.child_params = {}

    def _get_child(
            self,
            name: NXobjectIndex,
            use_field_dims: bool = False) -> Union['__class__', Field, sc.DataArray]:
        """Get item, with flag to control whether fields dims should be inferred"""
        if name is None:
            raise KeyError("None is not a valid index")
        if isinstance(name, str):
            item = self._group[name]
            if hasattr(item, 'shape'):
                dims = self._get_field_dims(name) if use_field_dims else None
                return Field(item, dims=dims, **self.child_params.get(name, {}))
            else:
                return _make(item)
        da = self._getitem(name)
        if (t := self.depends_on) is not None:
            da.coords['depends_on'] = t if isinstance(t, sc.Variable) else sc.scalar(t)
        return da

    def __getitem__(self,
                    name: NXobjectIndex) -> Union['__class__', Field, sc.DataArray]:
        return self._get_child(name, use_field_dims=True)

    def _getitem(self, index: ScippIndex) -> NoReturn:
        raise NotImplementedError(f'Loading {self.nx_class} is not supported.')

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        """Subclasses should reimplement this to provide dimension labels for fields."""
        return None

    def __contains__(self, name: str) -> bool:
        return name in self._group

    def get(self, name: str, default=None) -> Union['__class__', Field, sc.DataArray]:
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
        return _make(self._group.parent)

    def _ipython_key_completions_(self) -> List[str]:
        return list(self.keys())

    def keys(self) -> List[str]:
        return self._group.keys()

    def values(self) -> List[Union[Field, '__class__']]:
        return [self[name] for name in self.keys()]

    def items(self) -> List[Tuple[str, Union[Field, '__class__']]]:
        return list(zip(self.keys(), self.values()))

    @functools.lru_cache()
    def by_nx_class(self) -> Dict[NX_class, Dict[str, '__class__']]:
        classes = {name: [] for name in _nx_class_registry()}

        # TODO implement visititems for NXobject and merge the two blocks
        def _match_nx_class(_, node):
            if not hasattr(node, 'shape'):
                if (nx_class := node.attrs.get('NX_class')) is not None:
                    if not isinstance(nx_class, str):
                        nx_class = nx_class.decode('UTF-8')
                    if nx_class in _nx_class_registry():
                        classes[nx_class].append(node)

        self._group.visititems(_match_nx_class)

        out = {}
        for nx_class, groups in classes.items():
            names = [group.name.split('/')[-1] for group in groups]
            if len(names) != len(set(names)):  # fall back to full path if duplicate
                names = [group.name for group in groups]
            out[NX_class[nx_class]] = {n: _make(g) for n, g in zip(names, groups)}
        return out

    @property
    def nx_class(self) -> NX_class:
        """The value of the NX_class attribute of the group.

        In case of the subclass NXroot this returns 'NXroot' even if the attribute
        is not actually set. This is to support the majority of all legacy files, which
        do not have this attribute.
        """
        return NX_class[self.attrs['NX_class']]

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
        values = data.values
        if data.dtype == sc.DType.string:
            values = np.array(data.values, dtype=object)
        dataset = self._group.create_dataset(name, data=values, **kwargs)
        if data.unit is not None:
            dataset.attrs['units'] = str(data.unit)
        return Field(dataset, data.dims)

    def create_class(self, name: str, nx_class: NX_class) -> NXobject:
        group = self._group.create_group(name)
        group.attrs['NX_class'] = nx_class.name
        return _make(group)

    def __setitem__(self, name: str, value: Union[Field, NXobject, DimensionedArray]):
        """Create a link or a new field."""
        if isinstance(value, Field):
            self._group[name] = value._dataset
        elif isinstance(value, NXobject):
            self._group[name] = value._group
        else:
            self.create_field(name, value)


class NXroot(NXobject):
    @property
    def nx_class(self) -> NX_class:
        # As an oversight in the NeXus standard and the reference implementation,
        # the NX_class was never set to NXroot. This applies to essentially all
        # files in existence before 2016, and files written by other implementations
        # that were inspired by the reference implementation. We thus hardcode NXroot:
        return NX_class['NXroot']


class NXentry(NXobject):
    pass


class NXinstrument(NXobject):
    pass


class NXtransformations(NXobject):
    pass


def _make(group) -> NXobject:
    if (nx_class := Attrs(group.attrs).get('NX_class')) is not None:
        return _nx_class_registry().get(nx_class, NXobject)(group)
    return group  # Return underlying (h5py) group


@functools.lru_cache()
def _nx_class_registry():
    from .nxevent_data import NXevent_data
    from .nxdata import NXdata
    from .nxdetector import NXdetector
    from .nxdisk_chopper import NXdisk_chopper
    from .nxlog import NXlog
    from .nxmonitor import NXmonitor
    from .nxsample import NXsample
    from .nxsource import NXsource
    return {
        cls.__name__: cls
        for cls in [
            NXroot, NXentry, NXevent_data, NXlog, NXmonitor, NXdata, NXdetector,
            NXsample, NXsource, NXdisk_chopper, NXinstrument, NXtransformations
        ]
    }
