# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
from typing import List, Union, Optional
from warnings import warn
import scipp as sc
import numpy as np
from ._common import to_child_select
from .typing import H5Group
from .nxobject import Field, NXobject, ScippIndex, NexusStructureError, asarray


class NXdataStrategy:
    """
    Strategy used by :py:class:`scippnexus.NXdata`.

    May be subclassed to customize behavior.
    """
    _error_suffixes = ['_errors', '_error']  # _error is the deprecated suffix

    @staticmethod
    def axes(group):
        """Names of the axes (dimension labels)."""
        return group.attrs.get('axes')

    @staticmethod
    def signal(group):
        """Name of the signal field."""
        if (name := group.attrs.get('signal')) is not None:
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
            signal_override: Union[Field, '_EventField'] = None,  # noqa: F821
            skip: List[str] = None):
        """
        Parameters
        ----------
        signal_override:
            Field-like to use instead of trying to read signal from the file. This is
            used when there is no signal or to provide a signal computed from
            NXevent_data.
        skip:
            Names of fields to skip when loading coords.
        """
        super().__init__(group, definition=definition, strategy=strategy)
        self._signal_override = signal_override
        self._skip = skip if skip is not None else []

    def _default_strategy(self):
        return NXdataStrategy

    @property
    def shape(self) -> List[int]:
        return self._signal.shape

    def _get_group_dims(self) -> Union[None, List[str]]:
        # Apparently it is not possible to define dim labels unless there are
        # corresponding coords. Special case of '.' entries means "no coord".
        if (axes := self._strategy.axes(self)) is not None:
            return [f'dim_{i}' if a == '.' else a for i, a in enumerate(axes)]
        return None

    @property
    def dims(self) -> List[str]:
        if (d := self._get_group_dims()) is not None:
            return d
        # Legacy NXdata defines axes not as group attribute, but attr on dataset.
        # This is handled by class Field.
        return self._signal.dims

    @property
    def unit(self) -> Union[sc.Unit, None]:
        return self._signal.unit

    @property
    def _signal_name(self) -> str:
        return self._strategy.signal(self)

    @property
    def _errors_name(self) -> Optional[str]:
        return self._strategy.signal_errors(self)

    @property
    def _signal(self) -> Union[Field, '_EventField']:  # noqa: F821
        if self._signal_override is not None:
            return self._signal_override
        return self[self._signal_name]

    def _get_axes(self):
        """Return labels of named axes. Does not include default 'dim_{i}' names."""
        if (axes := self._strategy.axes(self)) is not None:
            # Unlike self.dims we *drop* entries that are '.'
            return [a for a in axes if a != '.']
        elif (axes := self._signal.attrs.get('axes')) is not None:
            return axes.split(',')
        return []

    def _guess_dims(self, name: str):
        """Guess dims of non-signal dataset based on shape.

        Does not check for potential bin-edge coord.
        """
        lut = {}
        for d, s in zip(self.dims, self.shape):
            if self.shape.count(s) == 1:
                lut[s] = d
        shape = self._get_child(name).shape
        if self.shape == shape:
            return self.dims
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
        # Newly written files should always contain indices attributes, but the
        # standard recommends that readers should also make "best effort" guess
        # since legacy files do not set this attribute.
        if (indices := self.attrs.get(f'{name}_indices')) is not None:
            return list(np.array(self.dims)[np.array(indices).flatten()])
        if name in [self._signal_name, self._errors_name]:
            return self._get_group_dims()  # if None, field determines dims itself
        if name in list(self.attrs.get('auxiliary_signals', [])):
            return self._try_guess_dims(name)
        if name in self._get_axes():
            # If there are named axes then items of same name are "dimension
            # coordinates", i.e., have a dim matching their name.
            return [name]
        return self._try_guess_dims(name)

    def _bin_edge_dim(self, coord: Field) -> Union[None, str]:
        sizes = dict(zip(self.dims, self.shape))
        for dim, size in zip(coord.dims, coord.shape):
            if dim in sizes and sizes[dim] + 1 == size:
                return dim
        return None

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
        signal = self._signal[select]
        if self._errors_name is not None:
            stddevs = self[self._errors_name][select]
            signal.variances = sc.pow(stddevs, sc.scalar(2)).values

        da = sc.DataArray(data=signal) if isinstance(signal, sc.Variable) else signal

        skip = self._skip
        skip += [self._signal_name, self._errors_name]
        skip += list(self.attrs.get('auxiliary_signals', []))
        for name in self:
            if (errors := self._strategy.coord_errors(self, name)) is not None:
                skip += [errors]

        for name, field in self[Field].items():
            if name in skip:
                continue
            try:
                sel = to_child_select(self.dims,
                                      field.dims,
                                      select,
                                      bin_edge_dim=self._bin_edge_dim(field))
                coord: sc.Variable = asarray(self[name][sel])
                if (error_name := self._strategy.coord_errors(self, name)) is not None:
                    stddevs = self[error_name][sel]
                    coord.variances = sc.pow(stddevs, sc.scalar(2)).values
                if self._coord_to_attr(da, name, field):
                    # Like scipp, slicing turns coord into attr if slicing removes the
                    # dim corresponding to the coord.
                    da.attrs[name] = coord
                else:
                    da.coords[name] = coord
            except sc.DimensionError as e:
                warn(f"Skipped load of axis {field.name} due to:\n{e}")

        return da
