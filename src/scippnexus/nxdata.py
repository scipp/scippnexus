# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

import uuid
from functools import cached_property
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipp as sc

from ._common import _to_canonical_select, convert_time_to_datetime64, to_child_select
from .base import (
    Group,
    NexusStructureError,
    NXobject,
    asvariable,
    base_definitions_dict,
)
from .field import Field, _is_time
from .nxevent_data import NXevent_data
from .typing import H5Dataset, ScippIndex


def _guess_dims(dims, shape, dataset: H5Dataset):
    """Guess dims of non-signal dataset based on shape."""
    if shape is None:
        return None
    if shape == dataset.shape:
        return dims
    lut = {}
    for d, s in zip(dims, shape):
        if shape.count(s) == 1:
            lut[s] = d
    try:
        return [lut[s] for s in dataset.shape]
    except KeyError:
        try:  # Inner dimension may be bin-edges
            shape = list(dataset.shape)
            shape[-1] -= 1
            return [lut[s] for s in shape]
        except KeyError:
            pass
    return None


class NXdata(NXobject):
    def __init__(
        self,
        attrs: Dict[str, Any],
        children: Dict[str, Union[Field, Group]],
        fallback_dims: Optional[Tuple[str, ...]] = None,
        fallback_signal_name: Optional[str] = None,
    ):
        super().__init__(attrs=attrs, children=children)
        self._valid = True  # True if the children can be assembled
        self._signal_name = None
        self._signal = None
        self._aux_signals = attrs.get('auxiliary_signals', [])

        self._init_signal(
            name=attrs.get('signal', fallback_signal_name), children=children
        )
        if (errors := children.get('errors')) is not None:
            if (
                isinstance(self._signal, Field)
                and isinstance(errors, Field)
                and self._signal.errors is None
                and (errors.unit is None or self._signal.unit == errors.unit)
                and self._signal.dataset.shape == errors.dataset.shape
            ):
                self._signal.errors = errors.dataset
                del children['errors']
        self._init_axes(attrs=attrs, children=children)
        self._init_group_dims(attrs=attrs, fallback_dims=fallback_dims)

        for name, field in children.items():
            self._init_field_dims(name, field)

    def _init_field_dims(self, name: str, field: Union[Field, Group]) -> None:
        if not isinstance(field, Field):
            # If the NXdata contains subgroups we can generally not define valid
            # sizes... except for some non-signal "special fields" that return
            # a DataGroup that will be wrapped in a scalar Variable.
            if name == self._signal_name or name in self._aux_signals:
                return
            if field.attrs.get('NX_class') not in [
                'NXoff_geometry',
                'NXcylindrical_geometry',
                'NXgeometry',
                'NXtransformations',
            ]:
                self._valid = False
        elif (dims := self._get_dims(name, field)) is not None:
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

    def _init_signal(self, name: Optional[str], children):
        # There are multiple ways NeXus can define the "signal" dataset. The latest
        # version uses `signal` attribute on the group (passed as `name`). However,
        # we must give precedence to the `signal` attribute on the dataset, since
        # older files may use that (and the `signal` group attribute is unrelated).
        # Finally, NXlog and NXevent_data can take the role of the signal. In practice
        # those may not be indicate by a `signal` attribute, but we support that
        # anyway since otherwise we would not be able to find NXevent_data signals
        # in many common files.
        if name is not None and name in children:
            self._signal_name = name
            self._signal = children[name]
        # Legacy NXdata defines signal not as group attribute, but attr on dataset
        for name, field in children.items():
            # We ignore the signal value. Usually it is 1, but apparently one could
            # multiple signals. We do not support this, since it is legacy anyway.
            if 'signal' in field.attrs:
                self._signal_name = name
                self._signal = children[name]
                break
        # NXlog or NXevent_data can take the role of the signal.
        for name, field in children.items():
            if name == self._signal_name:
                # Avoid duplicate handling
                continue
            if isinstance(field, EventField) or (
                isinstance(field, Group) and field.nx_class in [NXlog, NXevent_data]
            ):
                if self._signal is None:
                    self._signal_name = name
                    self._signal = field
                else:
                    self._aux_signals.append(name)

    def _init_axes(
        self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]
    ):
        # Latest way of defining axes
        self._axes = attrs.get('axes')
        # Older way of defining axes
        self._signal_axes = (
            None if self._signal is None else self._signal.attrs.get('axes')
        )
        if self._signal_axes is not None:
            self._signal_axes = tuple(self._signal_axes.split(':'))
            # The standard says that the axes should be colon-separated, but some
            # files use comma-separated.
            if len(self._signal_axes) == 1 and self._signal.dataset.ndim > 1:
                self._signal_axes = tuple(self._signal_axes[0].split(','))
        # Another old way of defining axes. Apparently there are two different ways in
        # which this is used: A value of '1' indicates "this is an axis". As this would
        # not allow for determining an order, we have to assume that the signal field
        # has an "axes" attribute that defines the order. We can then ignore the "axis"
        # attributes, since they hold no further information. If there is not "axes"
        # attribute on the signal field then we have to assume that "axis" gives the
        # 1-based index of the axis.
        self._axis_index = {}
        if self._signal_axes is None:
            for name, field in children.items():
                if (axis := field.attrs.get('axis')) is not None:
                    self._axis_index[name] = axis

    def _get_named_axes(self, fallback_dims) -> Tuple[str, ...]:
        if self._axes is not None:
            # Unlike self.dims we *drop* entries that are '.'
            return tuple(a for a in self._axes if a != '.')
        elif self._signal_axes is not None:
            return self._signal_axes
        elif fallback_dims is not None:
            return fallback_dims
        else:
            return ()

    def _get_group_dims(self) -> Optional[Tuple[str, ...]]:
        """Try three ways of defining group dimensions."""
        # Apparently it is not possible to define dim labels unless there are
        # corresponding coords. Special case of '.' entries means "no coord".
        if self._axes is not None:
            return tuple(
                f'dim_{i}' if a == '.' else a for i, a in enumerate(self._axes)
            )
        if self._signal_axes is not None:
            return self._signal_axes
        if self._axis_index:
            return tuple(
                k for k, _ in sorted(self._axis_index.items(), key=lambda item: item[1])
            )
        return None

    def _init_group_dims(
        self, attrs: Dict[str, Any], fallback_dims: Optional[Tuple[str, ...]] = None
    ):
        group_dims = self._get_group_dims()

        if self._signal is None:
            self._valid = False
        elif isinstance(self._signal, EventField):
            group_dims = self._signal.dims
        else:
            if group_dims is not None:
                shape = self._signal.dataset.shape
                # If we have explicit group dims, we can drop trailing 1s.
                shape = _squeeze_trailing(group_dims, shape)
                self._signal.sizes = dict(zip(group_dims, shape))
            elif isinstance(self._signal, Group):
                group_dims = self._signal.dims
            elif fallback_dims is not None:
                shape = self._signal.dataset.shape
                group_dims = [
                    fallback_dims[i] if i < len(fallback_dims) else f'dim_{i}'
                    for i in range(len(shape))
                ]
                self._signal.sizes = dict(zip(group_dims, shape))

        self._group_dims = group_dims
        self._named_axes = self._get_named_axes(fallback_dims)

        indices_suffix = '_indices'
        indices_attrs = {
            key[: -len(indices_suffix)]: attr
            for key, attr in attrs.items()
            if key.endswith(indices_suffix)
        }

        dims = np.array(self._axes)
        self._dims_from_indices = {
            key: tuple(dims[np.array(indices).flatten()])
            for key, indices in indices_attrs.items()
        }

    def _get_dims(self, name, field):
        # Newly written files should always contain indices attributes, but the
        # standard recommends that readers should also make "best effort" guess
        # since legacy files do not set this attribute.
        if name == self._signal_name:
            return self._group_dims
        # Latest way of defining dims
        if (dims := self._dims_from_indices.get(name)) is not None:
            if '.' in dims:
                hdf5_dims = self._dims_from_hdf5(field)
                return tuple(
                    dim if dim != '.' else hdf5_dim
                    for dim, hdf5_dim in zip(dims, hdf5_dims)
                )
            return dims
        # Older way of defining dims via axis attribute
        if (axis := self._axis_index.get(name)) is not None:
            return (self._group_dims[axis - 1],)
        if name in self._aux_signals:
            return _guess_dims(
                self._group_dims, self._signal.dataset.shape, field.dataset
            )
        if name in self._named_axes:
            # If there are named axes then items of same name are "dimension
            # coordinates", i.e., have a dim matching their name.
            # However, if the item is not 1-D we need more labels. Try to use labels
            # of signal if dimensionality matches.
            if isinstance(self._signal, Field) and len(field.dataset.shape) == len(
                self._signal.dataset.shape
            ):
                return self._group_dims
            return (name,)
        if self._signal is not None and self._group_dims is not None:
            signal_shape = (
                self._signal.dataset.shape
                if isinstance(self._signal, Field)
                else (
                    self._signal.shape if isinstance(self._signal, EventField) else None
                )
            )
            return _guess_dims(self._group_dims, signal_shape, field.dataset)
        # While not mandated or recommended by the standard, we can try to find HDF5
        # dim labels as a fallback option for defining dimension labels. Ideally we
        # would like to do so in NXobject._init_field, but this causes significant
        # overhead for small files with many datasets. Defined here, this will only
        # take effect for NXdata, NXdetector, NXlog, and NXmonitor.
        return self._dims_from_hdf5(field)

    def _dims_from_hdf5(self, field):
        hdf5_dims = [dim.label for dim in getattr(field.dataset, 'dims', [])]
        if any(dim != '' for dim in hdf5_dims):
            while hdf5_dims and hdf5_dims[-1] == '':
                hdf5_dims.pop()
            return [f'dim_{i}' if dim == '' else dim for i, dim in enumerate(hdf5_dims)]

    @cached_property
    def sizes(self) -> Dict[str, int]:
        if not self._valid:
            return super().sizes
        sizes = dict(self._signal.sizes)
        for name in self._aux_signals:
            sizes.update(self._children[name].sizes)
        return sizes

    @property
    def unit(self) -> Union[None, sc.Unit]:
        return self._signal.unit if self._valid else super().unit

    def _bin_edge_dim(self, coord: Union[Any, Field]) -> Union[None, str]:
        if not isinstance(coord, Field):
            return None
        sizes = self.sizes
        for dim, size in zip(coord.dims, coord.shape):
            if (sz := sizes.get(dim)) is not None and sz + 1 == size:
                return dim
        return None

    def index_child(self, child: Union[Field, Group], sel: ScippIndex) -> ScippIndex:
        """Same as NXobject.index_child but also handles bin edges."""
        child_sel = to_child_select(
            tuple(self.sizes), child.dims, sel, bin_edge_dim=self._bin_edge_dim(child)
        )
        return child[child_sel]

    def assemble(
        self, dg: sc.DataGroup
    ) -> Union[sc.DataGroup, sc.DataArray, sc.Dataset]:
        if not self._valid:
            raise NexusStructureError("Could not determine signal field or dimensions.")
        dg = dg.copy(deep=False)
        aux = {name: dg.pop(name) for name in self._aux_signals}
        signal = dg.pop(self._signal_name)
        coords = dg
        if isinstance(signal, sc.DataGroup):
            raise NexusStructureError("Signal is not an array-like.")
        da = (
            signal
            if isinstance(signal, sc.DataArray)
            else sc.DataArray(data=asvariable(signal))
        )
        da = self._add_coords(da, coords)
        if aux:
            signals = {self._signal_name: da}
            signals.update(aux)
            if all(
                isinstance(v, (sc.Variable, sc.DataArray)) for v in signals.values()
            ):
                return sc.Dataset(signals)
            return sc.DataGroup(signals)
        return da

    def _dim_of_coord(self, name: str, coord: sc.Variable) -> Union[None, str]:
        if len(coord.dims) == 1:
            return coord.dims[0]
        if name in coord.dims and name in self.dims:
            return name
        return self._bin_edge_dim(coord)

    def _should_be_aligned(
        self, da: sc.DataArray, name: str, coord: sc.Variable
    ) -> bool:
        if name == 'depends_on':
            return True
        dim_of_coord = self._dim_of_coord(name, coord)
        if dim_of_coord is None:
            return True
        if dim_of_coord not in da.dims:
            return False
        return True

    def _add_coords(self, da: sc.DataArray, coords: sc.DataGroup) -> sc.DataArray:
        """Add coords to a data array.

        Sets alignment in the same way as slicing scipp.DataArray would.
        """
        for name, coord in coords.items():
            if not isinstance(coord, sc.Variable):
                da.coords[name] = sc.scalar(coord)
            else:
                da.coords[name] = coord
                # We need the shape *before* slicing to determine dims, so we get the
                # field from the group for the conditional.
                da.coords.set_aligned(
                    name, self._should_be_aligned(da, name, self._children[name])
                )
        return da


def _squeeze_trailing(dims: Tuple[str, ...], shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return shape[: len(dims)] + tuple(size for size in shape[len(dims) :] if size != 1)


class NXlog(NXdata):
    """
    NXlog, a time-series that can be loaded as a DataArray.

    In some cases the NXlog may contain additional time series, such as a connection
    status or alarm. These cannot be handled in a standard way, since the result cannot
    be represented as a single DataArray. Furthermore, they prevent positional
    time-indexing, since the time coord of each time-series is different. We can
    support label-based indexing for this in the future. If additional time-series
    are contained within the NXlog then loading will return a DataGroup of the
    individual time-series (DataArrays).
    """

    def __init__(self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]):
        children = dict(children)
        self._sublogs = []
        self._sublog_children = {}
        for name in children:
            if name.endswith('_time'):
                self._sublogs.append(name[:-5])
        # Extract all fields that belong to sublogs, since they will interfere with the
        # setup logic in the base class (NXdata).
        for name in self._sublogs:
            for k in list(children):
                if k.startswith(name):
                    field = children.pop(k)
                    self._init_field(field)
                    field.sizes = {
                        'time' if i == 0 else f'dim_{i}': size
                        for i, size in enumerate(field.dataset.shape)
                    }
                    self._sublog_children[k] = field

        super().__init__(
            attrs=attrs,
            children=children,
            fallback_dims=('time',),
            fallback_signal_name='value',
        )

    def read_children(self, sel: ScippIndex) -> sc.DataGroup:
        # Sublogs have distinct time axes (with a different length). Must disable
        # positional indexing.
        if self._sublogs and ('time' in _to_canonical_select(list(self.sizes), sel)):
            raise sc.DimensionError(
                "Cannot positionally select time since there are multiple "
                "time fields. Label-based selection is not supported yet."
            )
        dg = super().read_children(sel)
        for name, field in self._sublog_children.items():
            dg[name] = field[sel]
        return dg

    def _time_to_datetime(self, mapping):
        if (time := mapping.get('time')) is not None:
            if time.dtype != sc.DType.datetime64 and _is_time(time):
                mapping['time'] = convert_time_to_datetime64(
                    time, start=sc.epoch(unit=time.unit)
                )

    def _assemble_sublog(
        self, dg: sc.DataGroup, name: str, value_name: Optional[str] = None
    ) -> sc.DataArray:
        value_name = name if value_name is None else f'{name}_{value_name}'
        da = sc.DataArray(dg.pop(value_name), coords={'time': dg.pop(f'{name}_time')})
        for k in list(dg):
            if k.startswith(name):
                da.coords[k[len(name) + 1 :]] = dg.pop(k)
        self._time_to_datetime(da.coords)
        return da

    def assemble(
        self, dg: sc.DataGroup
    ) -> Union[sc.DataGroup, sc.DataArray, sc.Dataset]:
        self._time_to_datetime(dg)
        dg = sc.DataGroup(dg)
        sublogs = sc.DataGroup()
        for name in self._sublogs:
            # Somewhat arbitrary definition of which fields is the "value"
            value_name = 'severity' if name == 'alarm' else None
            sublogs[name] = self._assemble_sublog(dg, name, value_name=value_name)
        out = super().assemble(dg)
        return out if not sublogs else sc.DataGroup(value=out, **sublogs)


def _find_embedded_nxevent_data(
    children: Dict[str, Union[Field, Group]]
) -> Optional[Group]:
    if all(name in children for name in NXevent_data.mandatory_fields):
        parent = children['event_index'].parent._group
        event_group = Group(
            parent, definitions={'NXmonitor': NXevent_data, 'NXdetector': NXevent_data}
        )
        for name in list(children):
            if name in NXevent_data.handled_fields:
                del children[name]
        return event_group


class EventField:
    def __init__(self, event_data: Group, grouping_name: str, grouping: Field) -> None:
        """Create a field that represents an event data group.

        Parameters
        ----------
        event_data:
            The event data group holding the NXevent_data fields.
        grouping_name:
            The name of the field that contains the grouping information.
        grouping:
            The field that contains the grouping keys. These are IDs corresponding to
            the event_id field of the NXevent_data group, such as the detector_number
            field of an NXdetector.
        """
        self._event_data = event_data
        self._grouping_name = grouping_name
        self._grouping = grouping

    @property
    def attrs(self) -> Dict[str, Any]:
        return self._event_data.attrs

    @property
    def sizes(self) -> Dict[str, int]:
        return {**self._grouping.sizes, **self._event_data.sizes}

    @property
    def dims(self) -> Tuple[str, ...]:
        return self._grouping.dims + self._event_data.dims

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._grouping.shape + self._event_data.shape

    def __getitem__(self, sel: ScippIndex) -> sc.DataArray:
        event_sel = to_child_select(self.dims, self._event_data.dims, sel)
        events = self._event_data[event_sel]
        detector_sel = to_child_select(self.dims, self._grouping.dims, sel)
        if not isinstance(events, sc.DataArray):
            return events
        da = _group_events(event_data=events, grouping=self._grouping[detector_sel])
        da.coords[self._grouping_name] = da.coords.pop('event_id')
        return da


class NXdetector(NXdata):
    _detector_number_fields = ['detector_number', 'pixel_id', 'spectrum_index']

    @staticmethod
    def _detector_number(children: Iterable[str]) -> Optional[str]:
        for name in NXdetector._detector_number_fields:
            if name in children:
                return name

    def __init__(self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]):
        fallback_dims = None
        if (det_num_name := NXdetector._detector_number(children)) is not None:
            if (detector_number := children[det_num_name]).dataset.ndim == 1:
                fallback_dims = (det_num_name,)
                detector_number.sizes = {det_num_name: detector_number.dataset.shape[0]}

        if (event_group := _find_embedded_nxevent_data(children)) is not None:
            embedded_events = uuid.uuid4().hex if 'events' in children else 'events'
            children[embedded_events] = event_group
        else:
            embedded_events = None

        def _maybe_event_field(name: str, child: Union[Field, Group]):
            if (
                name == embedded_events
                or (isinstance(child, Group) and child.nx_class == NXevent_data)
            ) and det_num_name is not None:
                event_field = EventField(
                    event_data=child,
                    grouping_name=det_num_name,
                    grouping=children.get(det_num_name),
                )
                return event_field
            return child

        children = {
            name: _maybe_event_field(name, child) for name, child in children.items()
        }

        super().__init__(
            attrs=attrs,
            children=children,
            fallback_dims=fallback_dims,
            fallback_signal_name='data',
        )

    def assemble(self, dg: sc.DataGroup) -> Union[sc.DataArray, sc.Dataset]:
        bitmasks = {
            key[len('pixel_mask') :]: dg.pop(key)
            # tuple because we are going to change the dict over the iteration
            for key in tuple(dg.keys())
            if key.startswith('pixel_mask')
        }

        array_or_dataset = super().assemble(dg)

        for suffix, bitmask in bitmasks.items():
            masks = self.transform_bitmask_to_dict_of_masks(bitmask, suffix)
            for da in (
                array_or_dataset.values()
                if isinstance(array_or_dataset, sc.Dataset)
                else [array_or_dataset]
            ):
                for name, mask in masks.items():
                    da.masks[name] = mask
        return array_or_dataset

    @staticmethod
    def transform_bitmask_to_dict_of_masks(bitmask: sc.Variable, suffix: str = ''):
        bit_to_mask_name = {
            0: 'gap',
            1: 'dead',
            2: 'under_responding',
            3: 'virtual_pixel',
            4: 'noisy',
            6: 'pixel_is_part_of_a_cluster_of_problematic_pixels',
            8: 'user_defined_mask',
            31: 'virtual_pixel',
        }

        number_of_bits_in_dtype = 8 * bitmask.values.dtype.itemsize

        # Bitwise indicator of what masks are present
        masks_present = np.bitwise_or.reduce(bitmask.values.ravel())

        masks = {}
        for bit in range(number_of_bits_in_dtype):
            # Check if the mask associated with the current `bit` is present
            if (masks_present >> bit) % 2:
                name = bit_to_mask_name.get(bit, f'undefined_bit{bit}') + suffix
                masks[name] = sc.array(
                    dims=bitmask.dims,
                    values=(bitmask.values >> bit) % 2,
                    dtype='bool',
                )
        return masks

    @property
    def detector_number(self) -> Optional[str]:
        return self._detector_number(self._children)


class NXmonitor(NXdata):
    def __init__(self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]):
        if (event_group := _find_embedded_nxevent_data(children)) is not None:
            signal = uuid.uuid4().hex if 'events' in children else 'events'
            children[signal] = event_group
        else:
            signal = 'data'
        super().__init__(attrs=attrs, children=children, fallback_signal_name=signal)


def _group_events(
    *, event_data: sc.DataArray, grouping: Optional[sc.Variable] = None
) -> sc.DataArray:
    if grouping is None:
        event_id = 'event_id'
    else:
        # copy since sc.bin cannot deal with a non-contiguous view
        grouping = asvariable(grouping)
        event_id = grouping.flatten(to='event_id').copy()
    event_data.bins.coords['event_time_zero'] = sc.bins_like(
        event_data, fill_value=event_data.coords['event_time_zero']
    )
    # After loading raw NXevent_data it is guaranteed that the event table
    # is contiguous and that there is no masking. We can therefore use the
    # more efficient approach of binning from scratch instead of erasing the
    # 'event_time_zero' binning defined by NXevent_data.
    event_data = event_data.bins.constituents['data'].group(event_id)
    if grouping is None:
        return event_data
    return event_data.fold(dim='event_id', sizes=grouping.sizes)


def _find_event_entries(dg: sc.DataGroup) -> List[str]:
    event_entries = []
    for name, value in dg.items():
        if (
            isinstance(value, sc.DataArray)
            and 'event_time_zero' in value.coords
            and value.bins is not None
        ):
            event_entries.append(name)
    return event_entries


def group_events_by_detector_number(
    dg: sc.DataGroup,
) -> Union[sc.DataArray, sc.Dataset]:
    dg = dg.copy(deep=False)
    grouping_key = None
    for key in NXdetector._detector_number_fields:
        if (grouping := dg.get(key)) is not None:
            grouping_key = key
            break
    grouping = None if grouping_key is None else asvariable(dg.pop(grouping_key))
    grouped_events = sc.DataGroup()
    for event_entry in _find_event_entries(dg):
        events = dg.pop(event_entry)
        grouped_events[event_entry] = _group_events(
            event_data=events, grouping=grouping
        )
    if len(grouped_events) == 1:
        out = next(iter(grouped_events.values()))
    else:
        out = sc.Dataset(grouped_events)
    out.coords.update(dg)
    return out


base_definitions_dict['NXdata'] = NXdata
base_definitions_dict['NXlog'] = NXlog
base_definitions_dict['NXdetector'] = NXdetector
base_definitions_dict['NXmonitor'] = NXmonitor
