# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Dict, Union
import scipp as sc
from .nxobject import NXobject, ScippIndex
from ._common import to_plain_index


class Leaf(NXobject):
    """Base class for "leaf" groups than can be loaded as a dict.
    """

    def _getitem(self,
                 select: ScippIndex) -> Dict[str, Union[sc.Variable, sc.DataArray]]:
        from .nexus_classes import NXtransformations
        index = to_plain_index([], select)
        if index != tuple():
            raise ValueError(f"Cannot select slice when loading {type(self).__name__}")
        content = {}
        for key, obj in self.items():
            if key == 'depends_on' or isinstance(obj, NXtransformations):
                continue
            content[key] = obj[()]
        return content
