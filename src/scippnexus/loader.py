# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Union, Callable
import scipp as sc

from .typing import H5Group, ScippIndex
from .nxobject import NX_class, NXobject


@dataclass
class Selector:
    nxclass: NX_class
    include: Union[None, List[str]] = None
    include_regexes: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    exclude_regexes: List[str] = field(default_factory=list)


def _load_selected(group: H5Group, selector: Selector) -> List[NXobject]:
    groups = group.by_nx_class()[selector.nxclass]
    # TODO process includes and excludes
    return {name: g[...] for name, g in groups.items()}


class DataArrayLoaderFactory:
    def __init__(self):
        self._base = None
        self._attrs = []

    def set_base(self, func: Callable, selector: Selector):
        self._base = (func, selector)

    def add_attrs(self, func: Callable, selector: Selector):
        self._attrs.append((func, selector))

    def __call__(self, group: H5Group) -> DataArrayLoader:
        return DataArrayLoader(self, group)


class DataArrayLoader:
    def __init__(self, factory: DataArrayLoaderFactory, group: H5Group):
        self._factory = factory
        self._group = group

    def __getitem__(self, index: ScippIndex) -> sc.DataArray:
        # TODO index ignored
        func, selector = self._factory._base
        da = func(_load_selected(self._group, selector))

        return da
