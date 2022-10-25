# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc
from typing import Optional
from ..typing import H5Group


def make_application_definition_strategy(application_definition, strategy):

    class ApplicationDefinitionStrategy(strategy):

        @staticmethod
        def __class_attribute__():
            return application_definition._class_attribute

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

    def __init__(self, data):
        self.data = data

    @property
    def nx_class(self):
        return 'NXdata'

    def __application_definition__(self, group):
        da = self.data
        group.attrs['NX_class'] = self.nx_class
        group.attrs[self.__class_attribute__()] = 'SASdata'
        group.attrs['signal'] = 'I'
        group.attrs['I_axes'] = da.dims
        group.attrs['Q_indices'] = tuple(da.dims.index(d) for d in da.coords['Q'].dims)
        signal = group.create_field('I', sc.values(da.data))
        signal.attrs['uncertainties'] = 'Idev'
        group.create_field('Idev', sc.stddevs(da.data))
        group.create_field('Q', self.data.coords['Q'])

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

    def __init__(self, *, title, run):
        self.title = title
        self.run = run

    @property
    def nx_class(self):
        return 'NXentry'

    def __application_definition__(self, group):
        # TODO automatic mechanism for definition class
        # TODO Should we require from strategies to define the NX_class they apply to?
        group.attrs[self.__class_attribute__()] = 'SASentry'
        group.attrs['version'] = '1.0'
        group.attrs['definition'] = 'NXcanSAS'
        group.create_field('title', self.title)
        group.create_field('run', self.run)


@NXcanSAS
class SASroot:
    pass
