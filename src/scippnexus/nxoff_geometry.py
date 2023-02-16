# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Optional, Tuple, Union

import scipp as sc

from .nxobject import NexusStructureError, NXobject


def off_to_shape(*,
                 vertices: sc.Variable,
                 winding_order: sc.Variable,
                 faces: sc.Variable,
                 detector_faces: Optional[sc.Variable] = None,
                 detector_number: Optional[sc.Variable] = None) -> sc.Variable:
    """
    Convert OFF shape description to simpler shape representation.
    """
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
    # TODO check that both or neither are None?
    # TODO no! may be single shape for all detectors! return scalar binned
    if detector_faces is None:  # if detector_number is not None, all have same shape
        return sc.bins(begin=sc.index(0), dim=faces.dim, data=fvw)
    if detector_number is None:
        raise NexusStructureError("`detector_number` not given but NXoff_geometry "
                                  "contains `detector_faces`.")
    shape_index = detector_faces['column', 0].copy()
    detid = detector_faces['column', 1].copy()
    da = sc.DataArray(shape_index, coords={
        'detector_number': detid
    }).group(detector_number.flatten(to='detector_number'))
    comps = da.bins.constituents
    comps['data'] = fvw[faces.dim, comps['data'].values]
    return sc.bins(**comps).fold(dim='detector_number', sizes=detector_number.sizes)


class NXoff_geometry(NXobject):
    _dims = {
        'detector_faces': ('face', 'column'),
        'vertices': ('vertex', ),
        'winding_order': ('winding_order', ),
        'faces': ('face', )
    }

    def _get_field_dims(self, name: str) -> Union[None, Tuple[str]]:
        return self._dims.get(name)

    def _get_field_dtype(self, name: str) -> Union[None, sc.DType]:
        if name == 'vertices':
            return sc.DType.vector3
        return None
