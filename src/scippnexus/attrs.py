# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from collections.abc import Iterator, Mapping
from typing import Any

import h5py as h5

from ._string import cset_to_encoding, ensure_str


class Attrs(Mapping[str, Any]):
    def __init__(self, attrs: h5.AttributeManager) -> None:
        self._base_attrs = attrs
        self._attrs = dict(attrs)

    def __getitem__(self, name: str) -> Any:
        attr = self._attrs[name]
        # Is this check for string attributes sufficient? Is there a better way?
        if isinstance(attr, str | bytes):
            cset = self._base_attrs.get_id(name.encode("utf-8")).get_type().get_cset()
            return ensure_str(attr, cset_to_encoding(cset))
        return attr

    def __iter__(self) -> Iterator[str]:
        return iter(self._attrs)

    def __len__(self) -> int:
        return len(self._attrs)
