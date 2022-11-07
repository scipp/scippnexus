# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from abc import ABC, abstractmethod
from .nxobject import NXobject


class ApplicationDefinition(ABC):
    """Abstract base class that can be subclassed for custom application definitions."""

    @abstractmethod
    def make_strategy(self, group: NXobject) -> type:
        """Return a strategy that ScippNexus should use to load the given group."""
        ...
