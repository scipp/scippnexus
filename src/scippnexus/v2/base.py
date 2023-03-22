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

    @property
    def parent(self) -> H5Group:
        # TODO Get corrected definitions
        return Group(self.dataset.parent, definitions=base_definitions)

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
        self._special_fields = {}
        self._transformations = {}
        for name, field in group._children.items():
            if isinstance(field, Field):
                field.sizes = _squeezed_field_sizes(field.dataset)
                field.dtype = _dtype_fromdataset(field.dataset)
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

    @property
    def detector_number(self) -> Optional[str]:
        return None

    def pre_assemble(self, dg: sc.DataGroup) -> sc.DataGroup:
        for name, field in self._special_fields.items():
            if name in self._transformations:
                continue
            det_num = self.detector_number
            if det_num is not None:
                det_num = dg[det_num]
            dg[name] = field._nexus.assemble_as_child(dg[name], detector_number=det_num)
        if (depends_on := dg.get('depends_on')) is not None:
            transform = self._group[depends_on]
            # Avoid loading transform twice if it is a child of the same group
            for name, transformations in self._transformations.items():
                if transform.name.startswith(transformations.name):
                    dg['depends_on'] = dg[name][depends_on.split('/')[-1]]
                    break
            else:
                dg['depends_on'] = transform[()]
        return dg

    def assemble(self, dg: sc.DataGroup) -> Union[sc.DataGroup, sc.DataArray]:
        return dg


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

    @property
    def name(self) -> str:
        return self._group.name

    @property
    def parent(self) -> Optional[Group]:
        return Group(self._group.parent,
                     definitions=self._definitions) if self._group.parent else None

    @cached_property
    def _children(self) -> Dict[str, Union[Field, Group]]:
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
                if (values.unit == errors.unit
                        and values.dataset.shape == errors.dataset.shape):
                    values.errors = errors.dataset
                    del items[f'{name}{suffix}']
        items = {k: v for k, v in items.items() if not k.startswith('cue_')}
        return items

    @cached_property
    def _nexus(self) -> NXobject:
        return self._definitions.get(self.attrs.get('NX_class'), NXobject)(self)

    def _populate_fields(self) -> None:
        _ = self._nexus

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[str]:
        return self._children.__iter__()

    def _is_nxtransformations(self) -> bool:
        return self.attrs.get('NX_class') == 'NXtransformations'

    def __getitem__(self, sel) -> Union[Field, Group, sc.DataGroup]:
        if isinstance(sel, str):
            # We cannot get the child directly from the HDF5 group, since we need to
            # create the parent group, to ensure that fields get the correct properties
            # such as sizes and dtype.
            if '/' in sel:
                if sel.startswith('/'):
                    return Group(self._group.file,
                                 definitions=self._definitions)[sel[1:]]
                else:
                    return self[sel.split('/')[0]][sel[sel.index('/') + 1:]]
            child = self._children[sel]
            if isinstance(child, Field):
                self._populate_fields()
            if self._is_nxtransformations():
                from .nxtransformations import Transformation
                return Transformation(child)
            return child
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
        return create_field(self._group, key, value)

    def create_field(self, key: str, value: sc.Variable) -> H5Dataset:
        return create_field(self._group, key, value)

    def create_class(self, name, class_name: str) -> Group:
        return Group(create_class(self._group, name, class_name),
                     definitions=self._definitions)

    def rebuild(self) -> Group:
        return Group(self._group, definitions=self._definitions)

    @cached_property
    def sizes(self) -> Dict[str, int]:
        return self._nexus.sizes

    @property
    def dims(self) -> Tuple[str, ...]:
        return tuple(self.sizes)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.sizes.values())


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
                # What is the meaning of the attribute value? It is undocumented,
                # we simply ignore it.
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

        # Reject fallback dims if they are not compatible with group dims
        if fallback_dims is not None:
            for field in group._children.values():
                if len(fallback_dims) < len(field.shape):
                    fallback_dims = None
                    break

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
        elif fallback_dims is not None:
            named_axes = fallback_dims
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
                # However, if the item is not 1-D we need more labels. Try to use labels
                # of signal if dimensionality matches.
                if self._signal is not None and len(field.dataset.shape) == len(
                        self._signal.dataset.shape):
                    return group_dims
                return (name, )
            if self._signal is not None and group_dims is not None:
                return _guess_dims(group_dims, self._signal.dataset.shape,
                                   field.dataset)

        for name, field in group._children.items():
            if not isinstance(field, Field):
                if name not in self._special_fields:
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
    _detector_number_fields = ['detector_number', 'pixel_id', 'spectrum_index']

    def __init__(self, group: Group):
        super().__init__(group,
                         fallback_dims=('detector_number', ),
                         fallback_signal_name='data')

    @property
    def detector_number(self) -> Optional[str]:
        for name in self._detector_number_fields:
            if name in self._group._children:
                return name


class NXmonitor(NXdata):

    # TODO should read axes of fallback signal?
    def __init__(self, group: Group):
        super().__init__(group, fallback_signal_name='data')


class NXgeometry(NXobject):

    def __init__(self, group: Group):
        super().__init__(group)

    @staticmethod
    def assemble_as_child(children: sc.DataGroup,
                          detector_number: Optional[sc.Variable] = None) -> sc.Variable:
        return sc.scalar(children)


base_definitions = {}
base_definitions['NXdata'] = NXdata
base_definitions['NXlog'] = NXlog
base_definitions['NXdetector'] = NXdetector
base_definitions['NXgeometry'] = NXgeometry
base_definitions['NXmonitor'] = NXmonitor


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


def _group_events(*,
                  event_data: sc.DataArray,
                  grouping: Optional[sc.Variable] = None) -> sc.DataArray:
    if isinstance(event_data, sc.DataGroup):
        raise NexusStructureError("Invalid NXevent_data in NXdetector.")
    if grouping is None:
        event_id = 'event_id'
    else:
        # copy since sc.bin cannot deal with a non-contiguous view
        event_id = grouping.flatten(to='event_id').copy()
    event_data.bins.coords['event_time_zero'] = sc.bins_like(
        event_data, fill_value=event_data.coords['event_time_zero'])
    # After loading raw NXevent_data it is guaranteed that the event table
    # is contiguous and that there is no masking. We can therefore use the
    # more efficient approach of binning from scratch instead of erasing the
    # 'event_time_zero' binning defined by NXevent_data.
    event_data = event_data.bins.constituents['data'].group(event_id)
    # if self._grouping is None:
    #     event_data.coords[self._grouping_key] = event_data.coords.pop('event_id')
    # else:
    #     del event_data.coords['event_id']
    if grouping is None:
        return event_data
    return event_data.fold(dim='event_id', sizes=grouping.sizes)


def _find_event_entries(dg: sc.DataGroup) -> List[str]:
    event_entries = []
    for name, value in dg.items():
        if isinstance(
                value, sc.DataArray
        ) and 'event_time_zero' in value.coords and value.bins is not None:
            event_entries.append(name)
    return event_entries


def group_events_by_detector_number(dg: sc.DataGroup) -> sc.DataArray:
    event_entry = _find_event_entries(dg)[0]
    events = dg.pop(event_entry)
    grouping_key = None
    for key in NXdetector._detector_number_fields:
        if (grouping := dg.get(key)) is not None:
            grouping_key = key
            break
    grouping = None if grouping_key is None else asarray(dg.pop(grouping_key))
    da = _group_events(event_data=events, grouping=grouping)
    # TODO What about _coord_to_attr mapping as NXdata?
    da.coords.update(dg)
    return da
