# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Union, Callable
import scipp as sc

from .typing import H5Group, ScippIndex
from .nxobject import NX_class, NXobject, Field
from ._common import to_child_select


@dataclass
class Selector:
    nxclass: NX_class
    include: Union[None, List[str]] = None
    include_regexes: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    exclude_regexes: List[str] = field(default_factory=list)


def _get_selected(group: H5Group, selector: Selector) -> List[NXobject]:
    # TODO Better interface to avoid repeated calls to by_nx_class?
    groups = group.by_nx_class()[selector.nxclass]
    # TODO process includes and excludes
    return groups


def _load_selected(group: H5Group, selector: Selector) -> List[sc.DataArray]:
    groups = _get_selected(group, selector)
    return {name: g[...] for name, g in groups.items()}


# TODO rename ScalarProvider?
# should the callable be hard-coded to sc.scalar, or accept user provided?
class ScalarLoader:
    def __init__(self, node: Union[NXobject, Field]):
        self._node = node[...]  # TODO consider delay load until first use

    @property
    def dims(self):
        return []

    @property
    def shape(self):
        return ()

    def __getitem__(self, select: ScippIndex) -> sc.Variable:
        # TODO Either ignore irrelevant indices, or require callee to filter indices
        if select:
            raise sc.DimensionError(f"Cannot select slice {select} from scalar")
        return sc.scalar(self._node)


class DataArrayLoaderFactory:
    def __init__(self):
        self._base = None
        self._attrs = []

    def set_base(self, func: Callable, selector: Selector):
        self._base = (func, selector)

    def add_attrs(self, func: Callable, selector: Selector):
        # TODO Should also support name transformations
        self._attrs.append((func, selector))

    def __call__(self, group: H5Group) -> DataArrayLoader:
        return DataArrayLoader(self, group)


class DataArrayLoader:
    def __init__(self, factory: DataArrayLoaderFactory, group: H5Group):
        self._factory = factory
        self._group = group

    def __getitem__(self, select: ScippIndex) -> sc.DataArray:
        # TODO select ignored
        # Unclear dividing line to factory. How should we pass, e.g., base and attrs
        # to loader?
        func, selector = self._factory._base
        da = func(_load_selected(self._group, selector))
        for func, selector in self._factory._attrs:
            groups = _get_selected(self._group, selector)
            for name, v in groups.items():
                loader = func(v)
                da.attrs[name] = loader[to_child_select(da.dims,
                                                        loader.dims,
                                                        select=select)]
        return da
