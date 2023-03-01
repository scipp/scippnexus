# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union
from warnings import warn

import numpy as np
import scipp as sc

from ._common import to_child_select
from .nxcylindrical_geometry import NXcylindrical_geometry
from .nxobject import (
    DatasetInfo,
    Field,
    FieldInfo,
    GroupContentInfo,
    NexusStructureError,
    NXobject,
    NXobjectInfo,
    ScippIndex,
    asarray,
)
from .nxoff_geometry import NXoff_geometry
from .nxtransformations import NXtransformations
from .typing import H5Dataset, H5Group


def _guess_dims(dims, shape, field: DatasetInfo):
    """Guess dims of non-signal dataset based on shape.

    Does not check for potential bin-edge coord.
    """
    if shape == field.shape:
        return dims
    lut = {}
    for d, s in zip(dims, shape):
        if shape.count(s) == 1:
            lut[s] = d
    try:
        return [lut[s] for s in field.shape]
    except KeyError:
        return None


# Need:
# signal
# errors
# dims
# field dims
# shape
# field errors
@dataclass
class NXdataInfo:
    signal_name: str
    field_infos: Dict[str, FieldInfo]
    #field_dims: Dict[str, Tuple[str]]
    #dims: Tuple[str]
    #shape: Tuple[int]
    #unit:
    #signal: Optional[H5Dataset] # or FieldInfo? or Field?
    #signal_errors: Optional[H5Dataset]

    @staticmethod
    def from_group_info(
            *,
            info: GroupContentInfo,
            fallback_dims: Optional[Tuple[str]] = None,
            strategy) -> DataInfo:
        # 1. Find signal
        signal_name, signal = strategy.signal2(info)
        axes = strategy.axes2(info)

        # 2. Find group dim labels: newest to oldest:
        # - group.axes
        # - group.signal.axes
        # - group field axis attrs
        signal_axes = None if signal is None else signal.attrs.get('axes')

        axis_index = {}
        for name, dataset in info.datasets.items():
            if (axis := dataset.attrs.get('axis')) is not None:
                axis_index[name] = axis

        # TODO consistent list/tuple
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

        if axes is not None:
            # Unlike self.dims we *drop* entries that are '.'
            named_axes = [a for a in axes if a != '.']
        elif signal_axes is not None:
            named_axes = signal_axes.split(',')
        elif fallback_dims is not None:
            named_axes = fallback_dims
        else:
            named_axes = []

        # 3. Find field dims, create FieldInfo
        # - {name}_indices
        # - axis attr
        # - signal/error
        # - auxiliary_signal
        # - name matching dims
        # - guess

        # - {name}_indices
        indices_suffix = '_indices'
        indices_attrs = [
            key[:-len(indices_suffix)] for key in info.attrs.keys()
            if key.endswith(indices_suffix)
        ]

        dims = np.array(group_dims)
        dims_from_indices = {
            key: list(dims[np.array(info.attrs[key + indices_suffix]).flatten()])
            for key in indices_attrs
        }

        def get_dims(name, dataset):
            # Newly written files should always contain indices attributes, but the
            # standard recommends that readers should also make "best effort" guess
            # since legacy files do not set this attribute.
            # TODO signal and errors?
            # TODO aux
            if name in (signal_name, ):
                return group_dims
            if (dims := dims_from_indices.get(name)) is not None:
                return dims
            if (axis := axis_index.get(name)) is not None:
                return (group_dims[axis - 1], )
            if name in named_axes:
                # If there are named axes then items of same name are "dimension
                # coordinates", i.e., have a dim matching their name.
                # However, if the item is not 1-D we need more labels. Try to use labels of
                # signal if dimensionality matches.
                if signal is not None and len(dataset.shape) == len(signal.shape):
                    return group_dims
                return [name]
            if signal is not None and group_dims is not None:
                return _guess_dims(group_dims, signal.shape, dataset)

        field_dims = {name: get_dims(name, ds) for name, ds in info.datasets.items()}

        field_infos = {
            name: FieldInfo(dims=dims, values=info.datasets[name].value)
            for name, dims in field_dims.items()
        }

        for name in field_dims:
            if (errors := strategy.coord_errors(field_dims, name)) is not None:
                field_infos[name].errors = field_infos.pop(errors).values

        return NXdataInfo(signal_name=signal_name, field_infos=field_infos)


class NXdataStrategy:
    """
    Strategy used by :py:class:`scippnexus.NXdata`.

    May be subclassed to customize behavior.
    """
    _error_suffixes = ['_errors', '_error']  # _error is the deprecated suffix

    @staticmethod
    def signal2(info: NXdataInfo) -> Optional[DatasetInfo]:
        """Signal field info."""
        if (name := info.attrs.get('signal')) is not None:
            if (dataset := info.datasets.get(name)) is not None:
                return name, dataset
        # Legacy NXdata defines signal not as group attribute, but attr on dataset
        for name, dataset in info.datasets.items():
            # What is the meaning of the attribute value? It is undocumented, we simply
            # ignore it.
            if 'signal' in dataset.attrs:
                return name, dataset
        return None, None

    @staticmethod
    def axes2(info: NXdataInfo):
        """Names of the axes (dimension labels)."""
        return info.attrs.get('axes')

    @staticmethod
    def axes(group):
        """Names of the axes (dimension labels)."""
        return group.attrs.get('axes')

    @staticmethod
    def signal(group):
        """Name of the signal field."""
        if (name := group.attrs.get('signal')) is not None:
            if name in group:
                return name
        # Legacy NXdata defines signal not as group attribute, but attr on dataset
        for name in group.keys():
            # What is the meaning of the attribute value? It is undocumented, we simply
            # ignore it.
            if 'signal' in group._get_child(name).attrs:
                return name
        return None

    @staticmethod
    def signal_errors(group) -> Optional[str]:
        """Name of the field to use for standard-deviations of the signal."""
        name = f'{NXdataStrategy.signal(group)}_errors'
        if name in group:
            return name
        # This is a legacy named, deprecated in the NeXus format.
        if 'errors' in group:
            return 'errors'

    @staticmethod
    def coord_errors(group, name):
        """Name of the field to use for standard-deviations of a coordinate."""
        errors = [f'{name}{suffix}' for suffix in NXdataStrategy._error_suffixes]
        errors = [x for x in errors if x in group]
        if len(errors) == 0:
            return None
        if len(errors) == 2:
            warn(f"Found {name}_errors as well as the deprecated "
                 f"{name}_error. The latter will be ignored.")
        return errors[0]


class NXdata(NXobject):

    def __init__(
            self,
            group: H5Group,
            *,
            definition=None,
            strategy=None,
            skip: List[str] = None):
        """
        Parameters
        ----------
        skip:
            Names of fields to skip when loading coords.
        """
        super().__init__(group, definition=definition, strategy=strategy)
        self._skip = skip if skip is not None else []

    def _default_strategy(self):
        return NXdataStrategy

    def _make_class_info(self, info: GroupContentInfo) -> NXobjectInfo:
        """Create info object for this NeXus class."""
        di = NXdataInfo.from_group_info(info=info, strategy=self._strategy)
        fields = dict(di.field_infos)
        fields.update(info.groups)
        oi = NXobjectInfo(children=fields)
        oi.signal_name = di.signal_name
        return oi

    def _get_group_dims(self) -> Union[None, List[str]]:
        # Apparently it is not possible to define dim labels unless there are
        # corresponding coords. Special case of '.' entries means "no coord".
        if (axes := self._strategy.axes(self)) is not None:
            return [f'dim_{i}' if a == '.' else a for i, a in enumerate(axes)]
        axes = []
        # Names of axes that have an "axis" attribute serve as dim labels in legacy case
        for name, field in self._group.items():
            if (axis := field.attrs.get('axis')) is not None:
                axes.append((axis, name))
        if axes:
            return [x[1] for x in sorted(axes)]
        return None

    @property
    def sizes(self) -> Dict[str, Union[None, int]]:
        base_sizes = super().sizes
        if self._signal is None:
            return base_sizes
        # special handling to avoid getting 'None' in shape for bin-edge coords
        sizes = self._signal.sizes
        children = sc.DataGroup(self._build_children())
        for child in children.values():
            if hasattr(child, 'sizes'):
                child_sizes = child.sizes
                for dim, sz in sizes.items():
                    if dim in child_sizes:
                        if (child_sizes[dim] != sz) and (child_sizes[dim] != sz + 1):
                            return base_sizes
        return sizes

    @property
    def dims(self) -> List[str]:
        return tuple(self.sizes.keys())

    #    if (d := self._get_group_dims()) is not None:
    #        return d
    #    # Legacy NXdata defines axes not as group attribute, but attr on dataset.
    #    # This is handled by class Field.
    #    if (signal := self._signal) is not None:
    #        return signal.dims
    #    return ()

    @property
    def shape(self) -> List[int]:
        return tuple(self.sizes.values())

    #    if (signal := self._signal) is not None:
    #        return signal.shape
    #    return ()

    @property
    def unit(self) -> Union[sc.Unit, None]:
        return self._signal.unit

    @property
    def _signal_name(self) -> str:
        #return self._info.signal_name
        return self._strategy.signal(self)

    @property
    def _errors_name(self) -> Optional[str]:
        return self._strategy.signal_errors(self)

    @property
    def _signal(self) -> Union[Field, '_EventField', None]:  # noqa: F821
        if self._info.signal_name is None:
            return None
        return self.get(self._info.signal_name)

    def _get_axes(self):
        """Return labels of named axes. Does not include default 'dim_{i}' names."""
        if (axes := self._strategy.axes(self)) is not None:
            # Unlike self.dims we *drop* entries that are '.'
            return [a for a in axes if a != '.']
        elif (signal := self._signal) is not None:
            if (axes := signal.attrs.get('axes')) is not None:
                return axes.split(',')
        return []

    def _guess_dims(self, name: str):
        """Guess dims of non-signal dataset based on shape.

        Does not check for potential bin-edge coord.
        """
        shape = self._get_child(name).shape
        if self.shape == shape:
            return self.dims
        lut = {}
        if self._signal is not None:
            for d, s in zip(self.dims, self.shape):
                if self.shape.count(s) == 1:
                    lut[s] = d
        try:
            dims = [lut[s] for s in shape]
        except KeyError:
            raise NexusStructureError(
                f"Could not determine axis indices for {self.name}/{name}")
        return dims

    def _try_guess_dims(self, name):
        try:
            return self._guess_dims(name)
        except NexusStructureError:
            return None

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        #return self._info.field_dims[name]
        # Newly written files should always contain indices attributes, but the
        # standard recommends that readers should also make "best effort" guess
        # since legacy files do not set this attribute.
        if (indices := self.attrs.get(f'{name}_indices')) is not None:
            return list(np.array(self.dims)[np.array(indices).flatten()])
        if (axis := self._get_child(name).attrs.get('axis')) is not None:
            return (self._get_group_dims()[axis - 1], )
        if name in [self._signal_name, self._errors_name]:
            return self._get_group_dims()  # if None, field determines dims itself
        if name in list(self.attrs.get('auxiliary_signals', [])):
            return self._try_guess_dims(name)
        if name in self._get_axes():
            # If there are named axes then items of same name are "dimension
            # coordinates", i.e., have a dim matching their name.
            # However, if the item is not 1-D we need more labels. Try to use labels of
            # signal if dimensionality matches.
            if self._signal_name in self and self._get_child(name).ndim == len(
                    self.shape):
                return self[self._signal_name].dims
            return [name]
        return self._try_guess_dims(name)

    def _dim_of_coord(self, name: str, coord: Field) -> Union[None, str]:
        if len(coord.dims) == 1:
            return coord.dims[0]
        if name in coord.dims and name in self.dims:
            return name
        return self._bin_edge_dim(coord)

    def _coord_to_attr(self, da: sc.DataArray, name: str, coord: Field) -> bool:
        dim_of_coord = self._dim_of_coord(name, coord)
        if dim_of_coord is None:
            return False
        if dim_of_coord not in da.dims:
            return True
        return False

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        from .nexus_classes import NXgeometry
        signal = self._signal
        if signal is None:
            raise NexusStructureError("No signal field found, cannot load group.")
        signal = signal[select]
        if self._errors_name is not None:
            stddevs = self[self._errors_name][select]
            # According to the standard, errors must have the same shape as the data.
            # This is not the case in all files we observed, is there any harm in
            # attempting a broadcast?
            signal.variances = np.broadcast_to(sc.pow(stddevs, sc.scalar(2)).values,
                                               shape=signal.shape)

        da = sc.DataArray(data=signal) if isinstance(signal, sc.Variable) else signal

        skip = self._skip
        skip += [self._signal_name, self._errors_name]
        skip += list(self.attrs.get('auxiliary_signals', []))
        for name in self:
            if (errors := self._strategy.coord_errors(self, name)) is not None:
                skip += [errors]
        for name in self:
            if name in skip:
                continue
            # It is not entirely clear whether skipping NXtransformations is the right
            # solution. In principle NXobject will load them via the 'depends_on'
            # mechanism, so for valid files this should be sufficient.
            allowed = (Field, NXtransformations, NXcylindrical_geometry, NXoff_geometry,
                       NXgeometry)
            if not isinstance(self._get_child(name), allowed):
                raise NexusStructureError(
                    "Invalid NXdata: may not contain nested groups")

        for name, field in self[Field].items():
            if name in skip:
                continue
            sel = to_child_select(self.dims,
                                  field.dims,
                                  select,
                                  bin_edge_dim=self._bin_edge_dim(field))
            coord: sc.Variable = asarray(self[name][sel])
            if (error_name := self._strategy.coord_errors(self, name)) is not None:
                stddevs = asarray(self[error_name][sel])
                coord.variances = sc.pow(stddevs, sc.scalar(2)).values
            try:
                if self._coord_to_attr(da, name, field):
                    # Like scipp, slicing turns coord into attr if slicing removes the
                    # dim corresponding to the coord.
                    da.attrs[name] = coord
                else:
                    da.coords[name] = coord
            except sc.DimensionError as e:
                raise NexusStructureError(
                    f"Field {name} in NXdata incompatible with dims "
                    f"or shape of signal: {e}") from e

        return da

    def _assemble(self, children: sc.DataGroup) -> sc.DataArray:
        children = sc.DataGroup(children)
        signal = children.pop(self._info.signal_name)
        coords = children
        #coords = {name:asarray(child) for name, child in children.items()}
        print(list(coords.items()))
        da = sc.DataArray(data=signal, coords=coords)
        for name in list(da.coords):
            # TODO building again is inefficient!
            if self._coord_to_attr(da, name, self._info.children[name].build()):
                da.attrs[name] = da.coords.pop(name)
        return da
