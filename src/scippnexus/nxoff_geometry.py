# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Tuple, Union

import scipp as sc

from .leaf import Leaf


class NXoff_geometry(Leaf):
    _dims = {
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
