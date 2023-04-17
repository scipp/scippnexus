# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Any, Dict, Optional, Union

import scipp as sc

from .base import Group, NexusStructureError, NXobject, base_definitions_dict
from .field import Field


def off_to_shape(*,
                 vertices: sc.Variable,
                 winding_order: sc.Variable,
                 faces: sc.Variable,
                 detector_faces: Optional[sc.Variable] = None,
                 detector_number: Optional[sc.Variable] = None) -> sc.Variable:
    """
    Convert OFF shape description to simpler shape representation.
    """
    # Vertices in winding order. This duplicates vertices if they are part of more than
    # one faces.
    # TODO Should use this:
    #     vw = vertices[winding_order.values]
    # but NumPy is currently much faster.
    # See https://github.com/scipp/scipp/issues/3044
    vw = sc.vectors(dims=vertices.dims,
                    values=vertices.values[winding_order.values],
                    unit=vertices.unit)
    # Same as above, grouped by face.
    fvw = sc.bins(begin=faces, data=vw, dim=vw.dim)
    low = fvw.bins.size().min().value
    high = fvw.bins.size().max().value
    if low == high:
        # Vertices in winding order, grouped by face. Unlike `fvw` above we now know
        # that each face has the same number of vertices, so we can fold instead of
        # using binned data.
        shapes = vw.fold(dim=vertices.dim, sizes={faces.dim: -1, vertices.dim: low})
    else:
        raise NotImplementedError("Conversion from OFF to shape not implemented for "
                                  "inconsistent number of vertices in faces.")
    if detector_faces is None:  # if detector_number is not None, all have same shape
        return sc.bins(begin=sc.index(0), dim=faces.dim, data=shapes)
    if detector_number is None:
        raise NexusStructureError("`detector_number` not given but NXoff_geometry "
                                  "contains `detector_faces`.")
    shape_index = detector_faces['face_index|detector_number', 0].copy()
    detid = detector_faces['face_index|detector_number', 1].copy()
    da = sc.DataArray(shape_index, coords={
        'detector_number': detid
    }).group(detector_number.flatten(to='detector_number'))
    comps = da.bins.constituents
    comps['data'] = shapes[faces.dim, comps['data'].values]
    return sc.bins(**comps).fold(dim='detector_number', sizes=detector_number.sizes)


class NXoff_geometry(NXobject):
    _dims = {
        'detector_faces': ('face', 'face_index|detector_number'),
        'vertices': ('vertex', ),
        'winding_order': ('winding_order', ),
        'faces': ('face', )
    }

    def __init__(self, attrs: Dict[str, Any], children: Dict[str, Union[Field, Group]]):
        super().__init__(attrs=attrs, children=children)
        for name, field in children.items():
            if isinstance(field, Field):
                field.sizes = dict(zip(self._dims.get(name), field.dataset.shape))
                if name == 'vertices':
                    field.dtype = sc.DType.vector3

    def load_as_array(self,
                      detector_number: Optional[sc.Variable] = None) -> sc.Variable:
        return off_to_shape(**self[()], detector_number=detector_number)

    @staticmethod
    def assemble_as_child(children: sc.DataGroup,
                          detector_number: Optional[sc.Variable] = None) -> sc.Variable:
        return off_to_shape(**children, detector_number=detector_number)


base_definitions_dict['NXoff_geometry'] = NXoff_geometry
