# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from collections.abc import Iterator, Mapping
from typing import Any

from ._hdf5_nexus import _cset_to_encoding, _ensure_str


class Attrs(Mapping):
    def __init__(self, attrs: Mapping):
        self._base_attrs = attrs
        self._attrs = dict(attrs)

    def __getitem__(self, name: str) -> Any:
        attr = self._attrs[name]
        # Is this check for string attributes sufficient? Is there a better way?
        if isinstance(attr, str | bytes):
            cset = self._base_attrs.get_id(name.encode("utf-8")).get_type().get_cset()
            return _ensure_str(attr, _cset_to_encoding(cset))
        return attr

    def __iter__(self) -> Iterator[str]:
        return iter(self._attrs)

    def __len__(self) -> int:
        return len(self._attrs)
