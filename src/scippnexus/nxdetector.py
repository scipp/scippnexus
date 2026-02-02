# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import scipp as sc

from .base import Group, base_definitions_dict
from .event_field import EventField
from .field import Field
from .nxdata import NXdata
from .nxevent_data import NXevent_data, collect_embedded_nxevent_data


class NXdetector(NXdata):
    _detector_number_fields = ('detector_number', 'pixel_id', 'spectrum_index')

    @staticmethod
    def _detector_number(children: Iterable[str]) -> str | None:
        for name in NXdetector._detector_number_fields:
            if name in children:
                return name

    def __init__(self, attrs: dict[str, Any], children: dict[str, Field | Group]):
        children = dict(children)

        fallback_dims = None
        if (det_num_name := NXdetector._detector_number(children)) is not None:
            if (detector_number := children[det_num_name]).dataset.ndim == 1:
                fallback_dims = (det_num_name,)
                detector_number.sizes = {det_num_name: detector_number.dataset.shape[0]}

        embedded_events, children = collect_embedded_nxevent_data(children)

        def _maybe_event_field(name: str, child: Field | Group):
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

    def coord_allow_list(self) -> list[str]:
        """
        Names of datasets that will be treated as coordinates.

        Note that in addition to these, all datasets matching the data's dimensions
        as well as datasets explicitly referenced by an "indices" attribute in the
        group's list of attributes will be treated as coordinates.

        Override in subclasses to customize assembly of datasets into loaded output.
        """
        return [
            *NXdetector._detector_number_fields,
            'time_of_flight',
            'raw_time_of_flight',
            'x_pixel_offset',
            'y_pixel_offset',
            'z_pixel_offset',
            'distance',
            'polar_angle',
            'azimuthal_angle',
            'crate',
            'slot',
            'input',
            'start_time',
            'stop_time',
            'sequence_number',
        ]

    def assemble(self, dg: sc.DataGroup) -> sc.DataGroup:
        bitmasks = {
            key[len('pixel_mask') :]: dg.pop(key)
            # tuple because we are going to change the dict over the iteration
            for key in tuple(dg)
            if key.startswith('pixel_mask')
        }

        out = self._assemble_as_physical_component(
            dg, allow_in_coords=self.coord_allow_list()
        )

        for suffix, bitmask in bitmasks.items():
            masks = self.transform_bitmask_to_dict_of_masks(bitmask, suffix)
            for signal in (self._signal_name, *self._aux_signals):
                for name, mask in masks.items():
                    out[signal].masks[name] = mask
        return out

    @staticmethod
    def transform_bitmask_to_dict_of_masks(bitmask: sc.Variable, suffix: str = ''):
        bit_to_mask_name = {
            0: 'gap_pixel',
            1: 'dead_pixel',
            2: 'underresponding_pixel',
            3: 'overresponding_pixel',
            4: 'noisy_pixel',
            6: 'part_of_a_cluster_of_problematic_pixels',
            8: 'user_defined_mask_pixel',
            31: 'virtual_pixel',
        }

        number_of_bits_in_dtype = 8 * bitmask.values.dtype.itemsize

        # Bitwise indicator of what masks are present
        masks_present = np.bitwise_or.reduce(bitmask.values.ravel())
        one = np.array(1)

        masks = {}
        for bit in range(number_of_bits_in_dtype):
            # Check if the mask associated with the current `bit` is present
            if masks_present & (one << bit):
                name = bit_to_mask_name.get(bit, f'undefined_bit{bit}_pixel') + suffix
                masks[name] = sc.array(
                    dims=bitmask.dims,
                    values=bitmask.values & (one << bit),
                    dtype='bool',
                )
        return masks

    @property
    def detector_number(self) -> str | None:
        return self._detector_number(self._children)


base_definitions_dict['NXdetector'] = NXdetector
