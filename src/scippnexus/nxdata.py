# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
from typing import List, Union
from warnings import warn
import scipp as sc
import numpy as np
from ._common import to_child_select
from .typing import Group
from .nxobject import Field, NXobject, ScippIndex, NexusStructureError


class NXdata(NXobject):
    def __init__(
            self,
            group: Group,
            signal_name_default: str = None,
            signal_override: Union[Field, _EventField] = None,  # noqa: F821
            axes: List[str] = None,
            skip: List[str] = None):
        """
        :param signal_name_default: Default signal name used, if no `signal`
            attribute found in file.
        :param signal_override Signal field-like to use instead of trying to read
            signal from the file. This is used when there is no signal or to provide
            a signal computed from NXevent_data
        :param axes: Default axes used, if no `axes` attribute found in file.
        :param skip: Names of fields to skip when loading coords.
        """
        super().__init__(group)
        self._signal_name_default = signal_name_default
        self._signal_override = signal_override
        self._axes_default = axes
        self._skip = skip if skip is not None else []

    @property
    def shape(self) -> List[int]:
        return self._signal.shape

    def _get_group_dims(self) -> Union[None, List[str]]:
        # Apparently it is not possible to define dim labels unless there are
        # corresponding coords. Special case of '.' entries means "no coord".
        if (axes := self.attrs.get('axes', self._axes_default)) is not None:
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
        if (name := self.attrs.get('signal', self._signal_name_default)) is not None:
            return name
        # Legacy NXdata defines signal not as group attribute, but attr on dataset
        for name in self.keys():
            # TODO What is the meaning of the attribute value?
            if 'signal' in self._get_child(name).attrs:
                return name
        return None

    @property
    def _errors_name(self) -> str:
        if self._signal_name_default is None:
            return 'errors'
        else:
            return f'{self._signal_name_default}_errors'

    @property
    def _signal(self) -> Union[Field, _EventField]:  # noqa: F821
        if self._signal_override is not None:
            return self._signal_override
        return self[self._signal_name]

    def _get_axes(self):
        """Return labels of named axes. Does not include default 'dim_{i}' names."""
        if (axes := self.attrs.get('axes', self._axes_default)) is not None:
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

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        # Newly written files should always contain indices attributes, but the
        # standard recommends that readers should also make "best effort" guess
        # since legacy files do not set this attribute.
        if (indices := self.attrs.get(f'{name}_indices')) is not None:
            return np.array(self.dims)[np.array(indices).flatten()]
        signals = [self._signal_name, self._errors_name]
        signals += list(self.attrs.get('auxiliary_signals', []))
        if name in signals:
            return self._get_group_dims()  # if None, field determines dims itself
        if name in self._get_axes():
            # If there are named axes then items of same name are "dimension
            # coordinates", i.e., have a dim matching their name.
            return [name]
        try:
            return self._guess_dims(name)
        except NexusStructureError:
            return None

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        signal = self._signal[select]
        if self._errors_name in self:
            stddevs = self[self._errors_name][select]
            signal.variances = sc.pow(stddevs, 2).values

        da = sc.DataArray(data=signal) if isinstance(signal, sc.Variable) else signal

        skip = self._skip
        skip += [self._signal_name, self._errors_name]
        skip += list(self.attrs.get('auxiliary_signals', []))

        for name, field in self.items():
            if (not isinstance(field, Field)) or (name in skip):
                continue
            try:
                sel = to_child_select(self.dims, field.dims, select)
                coord = self[name][sel]
                # NeXus treats [] and [1] interchangeably, in general this is
                # ill-defined, but this is the best we can do.
                if coord.shape == [1] and da.sizes.get(coord.dim) != 1:
                    coord = coord.squeeze()
                da.coords[name] = coord
            except sc.DimensionError as e:
                warn(f"Skipped load of axis {field.name} due to:\n{e}")

        return da
