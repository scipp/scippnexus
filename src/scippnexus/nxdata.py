# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import Any

import h5py as h5
import numpy as np
import scipp as sc

from ._cache import cached_property
from ._common import (
    to_child_select,
)
from .base import (
    Group,
    NexusStructureError,
    NXobject,
    asvariable,
    base_definitions_dict,
)
from .event_field import EventField
from .field import Field
from .nxevent_data import NXevent_data
from .typing import ScippIndex


def _guess_dims(
    dims: Sequence[str], shape: Sequence[int] | None, dataset: h5.Dataset
) -> Sequence[str] | None:
    """Guess dims of non-signal dataset based on shape."""
    if shape is None:
        return None
    if shape == dataset.shape:
        return dims
    if 1 < len(dataset.shape) <= len(shape):
        # We allow exact multi-dim matches at the end, even though there is a risk
        # for ambiguity with the transpose. Since image data is very common and many
        # sensors are square, this is a hopefully reasonable compromise.
        if dataset.shape == shape[-len(dataset.shape) :]:
            return dims[-len(dataset.shape) :]
    lut = {s: d for d, s in zip(dims, shape, strict=True) if shape.count(s) == 1}
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
        attrs: dict[str, Any],
        children: dict[str, Field | Group],
        fallback_dims: tuple[str, ...] | None = None,
        fallback_signal_name: str | None = None,
    ) -> None:
        super().__init__(attrs=attrs, children=children)
        self._valid = True  # True if the children can be assembled
        self._signal_name = None
        self._signal = None
        self._aux_signals = list(attrs.get('auxiliary_signals', []))

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

    def _init_field_dims(self, name: str, field: Field | Group) -> None:
        if not isinstance(field, Field):
            # If the NXdata contains subgroups we can generally not define valid
            # sizes... except for some non-signal "special fields" that return
            # a DataGroup that will be wrapped in a scalar Variable.
            if name == self._signal_name or name in self._aux_signals:
                return
            if field.attrs.get('NX_class') not in [
                'NXcite',
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
            field.sizes = dict(zip(dims, shape, strict=False))
        elif self._valid:
            s1 = self._signal.sizes
            s2 = field.sizes
            if not set(s2.keys()).issubset(set(s1.keys())):
                self._valid = False
            elif any(s1[k] != s2[k] for k in s1.keys() & s2.keys()):
                self._valid = False

    def _init_signal(self, name: str | None, children):
        from .nxlog import NXlog

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

    def _init_axes(self, attrs: dict[str, Any], children: dict[str, Field | Group]):
        # Latest way of defining axes
        self._axes = attrs.get('axes')
        # Older way of defining axes
        self._signal_axes = (
            None if self._signal is None else self._signal.attrs.get('axes')
        )
        if self._signal_axes is not None:
            self._signal_axes = tuple(
                self._signal_axes.split(':')
                if isinstance(self._signal_axes, str)
                else self._signal_axes
            )
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

    def _get_named_axes(self, fallback_dims) -> tuple[str, ...]:
        if self._axes is not None:
            # Unlike self.dims we *drop* entries that are '.'
            return tuple(a for a in self._axes if a != '.')
        elif self._signal_axes is not None:
            return self._signal_axes
        elif fallback_dims is not None:
            return fallback_dims
        else:
            return ()

    def _get_group_dims(self) -> tuple[str, ...] | None:
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
        self, attrs: dict[str, Any], fallback_dims: tuple[str, ...] | None = None
    ):
        group_dims = self._get_group_dims()

        if self._signal is None:
            self._valid = False
        elif isinstance(self._signal, EventField):
            # EventField uses dims of detector_number (the grouping) plus dims of event
            # data. The former may be defined by the group dims.
            if group_dims is not None:
                self._signal._grouping.sizes = dict(
                    zip(group_dims, self._signal._grouping.shape, strict=False)
                )
            group_dims = self._signal.dims
        else:
            if group_dims is not None:
                shape = (
                    self._signal.dataset.shape
                    if hasattr(self._signal, 'dataset')
                    else self._signal.shape  # signal is a subgroup
                )
                # If we have explicit group dims, we can drop trailing 1s.
                shape = _squeeze_trailing(group_dims, shape)
                self._signal.sizes = dict(zip(group_dims, shape, strict=False))
            elif isinstance(self._signal, Group):
                group_dims = self._signal.dims
            elif fallback_dims is not None:
                shape = self._signal.dataset.shape
                group_dims = [
                    fallback_dims[i] if i < len(fallback_dims) else f'dim_{i}'
                    for i in range(len(shape))
                ]
                self._signal.sizes = dict(zip(group_dims, shape, strict=True))

        self._group_dims = group_dims
        self._named_axes = self._get_named_axes(fallback_dims)

        indices_suffix = '_indices'
        indices_attrs = {
            key[: -len(indices_suffix)]: attr
            for key, attr in attrs.items()
            if key.endswith(indices_suffix)
        }
        # Datasets referenced by an "indices" attribute will be added as coords.
        self._explicit_coords = list(indices_attrs)

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
                    for dim, hdf5_dim in zip(dims, hdf5_dims, strict=True)
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
    def sizes(self) -> dict[str, int]:
        if not self._valid:
            return super().sizes
        sizes = dict(self._signal.sizes)
        for name in self._aux_signals:
            sizes.update(self._children[name].sizes)
        return sizes

    @property
    def unit(self) -> None | sc.Unit:
        return self._signal.unit if self._valid else super().unit

    def _bin_edge_dim(self, coord: Any | Field) -> None | str:
        if not isinstance(coord, Field):
            return None
        sizes = self.sizes
        for dim, size in zip(coord.dims, coord.shape, strict=True):
            if (sz := sizes.get(dim)) is not None and sz + 1 == size:
                return dim
        return None

    def index_child(self, child: Field | Group, sel: ScippIndex) -> ScippIndex:
        """Same as NXobject.index_child but also handles bin edges."""
        child_sel = to_child_select(
            tuple(self.sizes), child.dims, sel, bin_edge_dim=self._bin_edge_dim(child)
        )
        return child[child_sel]

    def read_children(self, sel: ScippIndex) -> sc.DataGroup:
        sel = self.convert_label_index_to_positional(sel)
        return super().read_children(sel)

    def assemble(self, dg: sc.DataGroup) -> sc.DataGroup | sc.DataArray | sc.Dataset:
        return self._assemble_as_data(dg)

    def _assemble_as_data(
        self, dg: sc.DataGroup
    ) -> sc.DataGroup | sc.DataArray | sc.Dataset:
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
            if all(isinstance(v, sc.Variable | sc.DataArray) for v in signals.values()):
                return sc.Dataset(signals)
            return sc.DataGroup(signals)
        return da

    def _assemble_as_physical_component(
        self, dg: sc.DataGroup, allow_in_coords: list[str]
    ) -> sc.DataGroup:
        """
        Assemble suitable for physical components such as NXdetector or NXmonitor.

        NeXus groups representing physical components typically contain fields or
        subgroups such as 'depends_on', 'NXtransformations', or 'NXoff_geometry'.
        Adding these wrapped into a scalar variable as coordinates would be
        cumbersome. Therefore, we do not assemble the entire NXdetector or NXmonitor
        into a single DataArray (as typically done by NXdata). On the other hand, we
        do want to return a useful DataArray with coordinates. In NeXus the data,
        coordinates, and other datasets or subgroups are all stored in the same group.
        As this is not very useful and user-friendly, we assemble the data and coords
        into a DataArray, which will take the place of the signal field in the output
        DataGroup, while non-signal non-coord fields are simply kept as-is.
        """
        data_items = sc.DataGroup()
        result = sc.DataGroup()
        forward = list(
            chain(
                [self._signal_name],
                self._group_dims or [],
                self._aux_signals,
                self._explicit_coords,
                allow_in_coords,
            )
        )

        for name, value in dg.items():
            if name in forward:
                data_items[name] = value
            else:
                result[name] = value
        assembled_nxdata = self._assemble_as_data(data_items)
        result.update(
            {self._signal_name: assembled_nxdata}
            if isinstance(assembled_nxdata, sc.DataArray)
            else assembled_nxdata
        )
        return result

    def _dim_of_coord(self, name: str, coord: sc.Variable) -> None | str:
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
                if coord.shape == (1,) and not set(coord.dims).issubset(da.dims):
                    da.coords[name] = coord[0]
                else:
                    da.coords[name] = coord
                # We need the shape *before* slicing to determine dims, so we get the
                # field from the group for the conditional.
                da.coords.set_aligned(
                    name, self._should_be_aligned(da, name, self._children[name])
                )
        return da


def _squeeze_trailing(dims: tuple[str, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape[: len(dims)] + tuple(size for size in shape[len(dims) :] if size != 1)


base_definitions_dict['NXdata'] = NXdata
