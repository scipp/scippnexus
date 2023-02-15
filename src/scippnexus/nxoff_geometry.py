# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Optional, Tuple, Union

import scipp as sc

from .nxobject import NXobject, ScippIndex


def off_to_shape(*,
                 vertices: sc.Variable,
                 winding_order: sc.Variable,
                 faces: sc.Variable,
                 detector_faces: Optional[sc.Variable] = None) -> sc.Variable:
    # TODO shape and dims should be:
    # [face] if no detector_faces.. or []? The latter! Wrap in scalar binned
    # [detector_number] otherwise (but must get name from parent?)
    # TODO select
    vw = vertices[winding_order.values]
    fvw = sc.bins(begin=faces, data=vw, dim=vw.dim)
    low = fvw.bins.size().min().value
    high = fvw.bins.size().max().value
    if low == high:
        fvw = vw.fold(dim=vertices.dim, sizes={faces.dim: -1, vertices.dim: low})
    else:
        raise NotImplementedError("Conversion from OFF to shape not implemented for "
                                  "inconsistent number of vertices in faces.")
    if detector_faces is None:
        return sc.bins(begin=sc.index(0), dim=faces.dim, data=fvw)
    else:
        shape_index = detector_faces['dummy', 0].copy()
        detid = detector_faces['dummy', 1].copy()
        da = sc.DataArray(shape_index, coords={
            'detector_number': detid
        }).group('detector_number')
        comps = da.bins.constituents
        comps['data'] = fvw[faces.dim, comps['data'].values]
        return sc.DataArray(sc.bins(**comps), coords=da.coords)


class NXoff_geometry(NXobject):
    _dims = {
        'detector_faces': ('face', 'dummy'),
        'vertices': ('vertex', ),
        'winding_order': ('winding_order', ),
        'faces': ('face', )
    }

    @property
    def dims(self) -> Tuple[str]:
        if 'detector_faces' in self:
            return 'TODO'
        return ()

    @property
    def shape(self) -> Tuple[int]:
        if 'detector_faces' in self:
            return 'TODO'
        return ()

    def _get_field_dims(self, name: str) -> Union[None, Tuple[str]]:
        return self._dims.get(name)

    def _get_field_dtype(self, name: str) -> Union[None, sc.DType]:
        if name == 'vertices':
            return sc.DType.vector3
        return None
