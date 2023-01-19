# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from abc import ABC, abstractmethod
from typing import Dict

from .nxobject import NXobject


class ApplicationDefinition(ABC):
    """Abstract base class that can be subclassed for custom application definitions."""

    @abstractmethod
    def make_strategy(self, group: NXobject) -> type:
        """Return a strategy that ScippNexus should use to load the given group."""
        ...


def make_definition(mapping: Dict[NXobject, type]) -> ApplicationDefinition:
    """
    Create an application definition from a mapping of NeXus classes to strategies.
    """

    class Definition(ApplicationDefinition):

        def make_strategy(self, group: NXobject) -> type:
            return mapping.get(group.nx_class)

    return Definition()
