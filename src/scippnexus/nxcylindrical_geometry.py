# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Tuple, Union

import scipp as sc

from .leaf import Leaf


class NXcylindrical_geometry(Leaf):

    def _get_field_dims(self, name: str) -> Union[None, Tuple[str]]:
        if name in ('vertices', 'detector_number'):
            return (name, )
        if name == 'cylinders':
            return ('cylinders', 'vertices_index')
        return None

    def _get_field_dtype(self, name: str) -> Union[None, sc.DType]:
        if name == 'vertices':
            return sc.DType.vector3
        return None
