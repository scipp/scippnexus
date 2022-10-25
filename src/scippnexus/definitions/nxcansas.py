# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Optional
from ..typing import H5Group


def make_application_definition_strategy(application_definition, strategy):

    class ApplicationDefinitionStrategy(strategy):

        @staticmethod
        def child_strategy(group: H5Group):
            return application_definition.child_strategy(group)

    return ApplicationDefinitionStrategy


class ApplicationDefinition:

    def __init__(self, class_attribute, default=None):
        self._default_class = default
        self._class_attribute = class_attribute
        self._strategies = {}

    def child_strategy(self, group):
        if (definition_class := group.attrs.get(self._class_attribute,
                                                self._default_class)) is not None:
            return self._strategies.get(definition_class)

    def __call__(self, strategy):
        strat = make_application_definition_strategy(self, strategy)
        self._strategies[strategy.__name__] = strat
        return strat


NXcanSAS = ApplicationDefinition('canSAS_class', 'SASroot')


@NXcanSAS
class SASdata:

    @staticmethod
    def dims(group):
        axes = group.attrs.get('axes')
        if isinstance(axes, str):
            axes = axes.split(" ")
        if axes is None:
            axes = group.attrs.get('I_axes')
        if not isinstance(axes, list):
            axes = [axes]
        if axes.count('Q') != 1:
            index = 0
            for i, ax in enumerate(axes):
                if ax == 'Q':
                    axes[i] = f'Q{index}'
                    index += 1
        return tuple(axes)

    @staticmethod
    def axes(group):
        return group.attrs.get('I_axes')

    @staticmethod
    def signal(group):
        return group.attrs.get('signal', 'I')

    @staticmethod
    def signal_errors(group) -> Optional[str]:
        signal_name = group.attrs.get('signal', 'I')
        signal = group._group[signal_name]
        return signal.attrs.get('uncertainties')

    def coord_errors(group, name) -> Optional[str]:
        if name != 'Q':
            return None
        # TODO This naively stores this as Scipp errors, which are just Gaussian.
        # This is probably not correct in all cases.
        uncertainties = group[name].attrs.get('uncertainties')
        resolutions = group[name].attrs.get('resolutions')
        if uncertainties is None:
            return resolutions
        elif resolutions is None:
            return uncertainties
        raise RuntimeError("Cannot handle both uncertainties and resolutions for Q")


@NXcanSAS
class SAStransmission_spectrum:

    @staticmethod
    def dims(group):
        # TODO A valid file should have T_axes, do we need to fallback?
        if (axes := group.attrs.get('T_axes')) is not None:
            return (axes, )
        return ('lambda', )


@NXcanSAS
class SASentry:
    pass


@NXcanSAS
class SASroot:
    pass
