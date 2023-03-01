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


@dataclass
class NXdataInfo:
    signal_name: str
    field_infos: Dict[str, FieldInfo]

    @staticmethod
    def from_group_info(*,
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
        # Names of axes that have an "axis" attribute serve as dim labels in legacy case
        signal_axes = None if signal is None else signal.attrs.get('axes')

        axis_index = {}
        for name, dataset in info.datasets.items():
            if (axis := dataset.attrs.get('axis')) is not None:
                axis_index[name] = axis

        # TODO consistent list/tuple
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

    @property
    def shape(self) -> List[int]:
        return tuple(self.sizes.values())

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
