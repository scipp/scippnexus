# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
We avoid functools.cached_property due to a per-instance lock which causes
deadlocks in multi-threaded use of ScippNexus. This lock is removed in
Python 3.12, so we can remove this file once we drop support for Python 3.11.

This file contains a 1:1 backport of Python 3.12's cached_property.
"""

# flake8: noqa: E501
from __future__ import annotations

from collections.abc import Callable
from types import GenericAlias
from typing import Generic, TypeVar

_NOT_FOUND = object()
R = TypeVar("R")


class cached_property(Generic[R]):
    def __init__(self, func: Callable[..., R]) -> None:
        self.func = func
        self.attrname: str | None = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner: object, name: str) -> None:
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance: object, owner: object = None) -> R:
        if instance is None:
            return self  # type: ignore[return-value]
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except (
            AttributeError
        ):  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val: R = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            try:
                cache[self.attrname] = val
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                    f"does not support item assignment for caching {self.attrname!r} property."
                )
                raise TypeError(msg) from None
        return val

    __class_getitem__ = classmethod(GenericAlias)  # type: ignore[var-annotated]
