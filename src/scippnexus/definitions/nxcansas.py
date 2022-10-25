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
        # This approach will likely need to be generalized as many application
        # definitions to not define a "class attribute" in the style of canSAS_class,
        # but seem to rely on basic strcture and the NX_class attribute.
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
    nx_class = 'NXdata'

    def __init__(self, data):
        self.data = data

    def __write_to_nexus_group__(self, group):
        da = self.data
        group.attrs['canSAS_class'] = 'SASdata'
        group.attrs['signal'] = 'I'
        group.attrs['I_axes'] = da.dims
        group.attrs['Q_indices'] = tuple(da.dims.index(d) for d in da.coords['Q'].dims)
        signal = group.create_field('I', sc.values(da.data))
        if da.variances is not None:
            signal.attrs['uncertainties'] = 'Idev'
            group.create_field('Idev', sc.stddevs(da.data))
        coord = group.create_field('Q', da.coords['Q'])
        if da.coords['Q'].variances is not None:
            # Note that there is also an "uncertainties" attribute. It is not clear
            # to me what the difference is.
            coord.attrs['resolutions'] = 'Qdev'
            group.create_field('Qdev', sc.stddevs(da.coords['Q']))

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
    nx_class = 'NXentry'

    def __init__(self, *, title, run):
        self.title = title
        self.run = run

    def __write_to_nexus_group__(self, group):
        group.attrs['canSAS_class'] = 'SASentry'
        group.attrs['version'] = '1.0'
        group.attrs['definition'] = 'NXcanSAS'
        group.create_field('title', self.title)
        group.create_field('run', self.run)


@NXcanSAS
class SASroot:
    pass
