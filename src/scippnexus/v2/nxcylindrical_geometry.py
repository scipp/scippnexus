# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Any, Dict, Optional, Union

import scipp as sc

from .base import Field, Group, NexusStructureError, NXobject, base_definitions


def _parse(*,
           vertices: sc.Variable,
           cylinders: sc.Variable,
           detector_number: Optional[sc.Variable] = None,
           parent_detector_number: Optional[sc.Variable] = None) -> sc.Variable:
    face1_center = cylinders['vertex_index', 0]
    face1_edge = cylinders['vertex_index', 1]
    face2_center = cylinders['vertex_index', 2]
    ds = sc.Dataset()
    ds['face1_center'] = vertices[face1_center.values]
    ds['face1_edge'] = vertices[face1_edge.values]
    ds['face2_center'] = vertices[face2_center.values]
    ds = ds.rename(**{vertices.dim: 'cylinder'})
    if detector_number is None:
        # All cylinders belong to the same shape
        return sc.bins(begin=sc.index(0), dim='cylinder', data=ds)
    if parent_detector_number is None:
        raise NexusStructureError(
            "`detector_number` not given, but "
            "NXcylindrical_geometry contains mapping to `detector_number`.")
    # detector_number gives indices into cylinders, the naming in the NeXus
    # standard appears to be misleading
    if parent_detector_number.values.size != detector_number.values.size:
        raise NexusStructureError(
            "Number of detector numbers in NXcylindrical_geometry "
            "does not match the one given by the parent.")
    detecting_cylinders = ds['cylinder', detector_number.values]
    # One cylinder per detector
    begin = sc.arange('dummy',
                      parent_detector_number.values.size,
                      unit=None,
                      dtype='int64')
    end = begin + sc.index(1)
    shape = sc.bins(begin=begin, end=end, dim='cylinder', data=detecting_cylinders)
    return shape.fold(dim='dummy', sizes=parent_detector_number.sizes)


class NXcylindrical_geometry(NXobject):
    _dims = {
        'vertices': ('vertex', ),
        'detector_number': ('detector_number', ),
        'cylinders': ('cylinder', 'vertex_index')
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
        return _parse(**self[()], parent_detector_number=detector_number)

    @staticmethod
    def assemble_as_child(children: sc.DataGroup,
                          detector_number: Optional[sc.Variable] = None) -> sc.Variable:
        return _parse(**children, parent_detector_number=detector_number)


base_definitions['NXcylindrical_geometry'] = NXcylindrical_geometry
