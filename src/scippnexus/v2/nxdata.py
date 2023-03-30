# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipp as sc

from .._common import to_child_select
from ..typing import H5Dataset, ScippIndex
from .base import Field, Group, NexusStructureError, NXobject, asarray, base_definitions


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
                 attrs: Dict[str, Any],
                 children: Dict[str, Union[Field, Group]],
                 fallback_dims: Optional[Tuple[str, ...]] = None,
                 fallback_signal_name: Optional[str] = None):
        super().__init__(attrs=attrs, children=children)
        self._valid = True
        # Must do full consistency check here, to define self.sizes:
        # - squeeze correctly
        # - check if coord dims are compatible with signal dims
        # - check if there is a signal
        # If not the case, fall back do DataGroup.sizes
        # Can we just set field dims here?
        self._signal_name = None
        self._signal = None
        self._aux_signals = attrs.get('auxiliary_signals', [])
        if (name := attrs.get('signal',
                              fallback_signal_name)) is not None and name in children:
            self._signal_name = name
            self._signal = children[name]
        else:
            # Legacy NXdata defines signal not as group attribute, but attr on dataset
            for name, field in children.items():
                # What is the meaning of the attribute value? It is undocumented,
                # we simply ignore it.
                if 'signal' in field.attrs:
                    self._signal_name = name
                    self._signal = children[name]
                    break

        axes = attrs.get('axes')
        signal_axes = None if self._signal is None else self._signal.attrs.get('axes')

        axis_index = {}
        for name, field in children.items():
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

        if self._signal is None:
            self._valid = False
        else:
            if group_dims is not None:
                shape = self._signal.dataset.shape
                shape = _squeeze_trailing(group_dims, shape)
                self._signal.sizes = dict(zip(group_dims, shape))
            elif fallback_dims is not None:
                shape = self._signal.dataset.shape
                group_dims = [
                    fallback_dims[i] if i < len(fallback_dims) else f'dim_{i}'
                    for i in range(len(shape))
                ]
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
            for key, attr in attrs.items() if key.endswith(indices_suffix)
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
            if name in (self._signal_name, ):
                return group_dims
            # if name in [self._signal_name, self._errors_name]:
            #     return self._get_group_dims()  # if None, field determines dims itself
            if name in self._aux_signals:
                return _guess_dims(group_dims, self._signal.dataset.shape,
                                   field.dataset)
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

        for name, field in children.items():
            if not isinstance(field, Field):
                # If the NXdata contains subgroups we can generally not define valid
                # sizes... except for some non-signal "special fields" that return
                # a DataGroup that will be wrapped in a scalar Variable.
                if field.attrs.get('NX_class') not in [
                        'NXoff_geometry',
                        'NXcylindrical_geometry',
                        'NXgeometry',
                        'NXtransformations',
                ]:
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

    @property
    def unit(self) -> Union[None, sc.Unit]:
        return self._signal.unit if self._valid else super().unit

    def _bin_edge_dim(self, coord: Field) -> Union[None, str]:
        if not isinstance(coord, Field):
            return None
        sizes = self.sizes
        for dim, size in zip(coord.dims, coord.shape):
            if (sz := sizes.get(dim)) is not None and sz + 1 == size:
                return dim
        return None

    def index_child(self, child: Union[Field, Group], sel: ScippIndex) -> ScippIndex:
        child_sel = to_child_select(tuple(self.sizes),
                                    child.dims,
                                    sel,
                                    bin_edge_dim=self._bin_edge_dim(child))
        return child[child_sel]

    def assemble(self,
                 dg: sc.DataGroup) -> Union[sc.DataGroup, sc.DataArray, sc.Dataset]:
        if not self._valid:
            raise NexusStructureError("Could not determine signal field or dimensions.")
        aux = {name: dg.pop(name) for name in self._aux_signals}
        coords = sc.DataGroup(dg)
        signal = coords.pop(self._signal_name)
        da = sc.DataArray(data=signal)
        da = self._add_coords(da, coords)
        if aux:
            signals = {self._signal_name: da}
            signals.update(aux)
            return sc.Dataset(signals)
        return da

    def _dim_of_coord(self, name: str, coord: sc.Variable) -> Union[None, str]:
        if len(coord.dims) == 1:
            return coord.dims[0]
        if name in coord.dims and name in self.dims:
            return name
        return self._bin_edge_dim(coord)

    def _coord_to_attr(self, da: sc.DataArray, name: str, coord: sc.Variable) -> bool:
        if name == 'depends_on':
            return False
        dim_of_coord = self._dim_of_coord(name, coord)
        if dim_of_coord is None:
            return False
        if dim_of_coord not in da.dims:
            return True
        return False

    def _add_coords(self, da: sc.DataArray, coords: sc.DataGroup) -> sc.DataArray:
        for name, coord in coords.items():
            if not isinstance(coord, sc.Variable):
                da.coords[name] = sc.scalar(coord)
            # We need the shape *before* slicing to determine dims, so we get the
            # field from the group for the conditional.
            elif self._coord_to_attr(da, name, self._children[name]):
                da.attrs[name] = coord
            else:
                da.coords[name] = coord
        return da


def _squeeze_trailing(dims: Tuple[str, ...], shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return shape[:len(dims)] + tuple(size for size in shape[len(dims):] if size != 1)


class NXlog(NXdata):

    def __init__(self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]):
        super().__init__(attrs=attrs,
                         children=children,
                         fallback_dims=('time', ),
                         fallback_signal_name='value')
        if (time := children.get('time')) is not None:
            time._is_time = True


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
            if children[det_num_name].dataset.ndim == 1:
                fallback_dims = ('detector_number', )
        super().__init__(attrs=attrs,
                         children=children,
                         fallback_dims=fallback_dims,
                         fallback_signal_name='data')

    @property
    def detector_number(self) -> Optional[str]:
        return self._detector_number(self._children)


class NXmonitor(NXdata):

    def __init__(self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]):
        super().__init__(attrs=attrs, children=children, fallback_signal_name='data')


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


base_definitions['NXdata'] = NXdata
base_definitions['NXlog'] = NXlog
base_definitions['NXdetector'] = NXdetector
base_definitions['NXmonitor'] = NXmonitor
