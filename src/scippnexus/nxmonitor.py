# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any

import scipp as sc

from .base import Group, base_definitions_dict
from .field import Field
from .nxdata import NXdata
from .nxevent_data import collect_embedded_nxevent_data


class NXmonitor(NXdata):
    def __init__(self, attrs: dict[str, Any], children: dict[str, Field | Group]):
        signal, children = collect_embedded_nxevent_data(children)
        signal = signal or 'data'
        super().__init__(attrs=attrs, children=children, fallback_signal_name=signal)

    def coord_allow_list(self) -> list[str]:
        """
        Names of datasets that will be treated as coordinates.

        Note that in addition to these, all datasets matching the data's dimensions
        as well as datasets explicitly referenced by an "indices" attribute in the
        group's list of attributes will be treated as coordinates.

        Override in subclasses to customize assembly of datasets into loaded output.
        """
        return ['distance', 'time_of_flight']

    def assemble(self, dg: sc.DataGroup) -> sc.DataGroup:
        return self._assemble_as_physical_component(
            dg, allow_in_coords=self.coord_allow_list()
        )


base_definitions_dict['NXmonitor'] = NXmonitor
