# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc
from typing import Optional, Tuple, Union, Literal
from ..nxobject import NXobject
from ..definition import ApplicationDefinition as BaseDef


class ApplicationDefinition(BaseDef):

    def __init__(self, class_attribute: str, default: str = None):
        self._default_class = default
        self._class_attribute = class_attribute
        self._strategies = {}

    def make_strategy(self, group: NXobject):
        # This approach will likely need to be generalized as many application
        # definitions to not define a "class attribute" in the style of canSAS_class,
        # but seem to rely on basic strcture and the NX_class attribute.
        if (definition_class := group.attrs.get(self._class_attribute,
                                                self._default_class)) is not None:
            return self._strategies.get(definition_class)

    def register(self, sas_class):

        def decorator(strategy):
            self._strategies[sas_class] = strategy
            return strategy

        return decorator


NXcanSAS = ApplicationDefinition('canSAS_class', 'SASroot')


class SASdata:
    nx_class = 'NXdata'

    def __init__(self,
                 data: sc.DataArray,
                 Q_variances: Optional[Literal['uncertainties', 'resolutions']] = None):
        self.data = data
        valid = ('uncertainties', 'resolutions')
        if Q_variances not in (None, ) + valid:
            raise ValueError(f"Q_variances must be in {valid}")
        self._variances = Q_variances

    def __write_to_nexus_group__(self, group: NXobject):
        da = self.data
        group.attrs['canSAS_class'] = 'SASdata'
        group.attrs['signal'] = 'I'
        group.attrs['axes'] = da.dims  # for NeXus compliance, same as I_axes
        group.attrs['I_axes'] = da.dims
        group.attrs['Q_indices'] = tuple(da.dims.index(d) for d in da.coords['Q'].dims)
        signal = group.create_field('I', sc.values(da.data))
        # We use the _errors suffix for NeXus compliance, unlike the examples given in
        # NXcanSAS.
        if da.variances is not None:
            signal.attrs['uncertainties'] = 'I_errors'
            group.create_field('I_errors', sc.stddevs(da.data))
        coord = group.create_field('Q', da.coords['Q'])
        if da.coords['Q'].variances is not None:
            if self._variances is None:
                raise ValueError(
                    "Q has variances, must specify whether these represent "
                    "'uncertainties' or 'resolutions' using the 'Q_variances' option'")

            coord.attrs[self._variances] = 'Q_errors'
            group.create_field('Q_errors', sc.stddevs(da.coords['Q']))


@NXcanSAS.register('SASdata')
class SASdataStrategy:

    @staticmethod
    def axes(group: NXobject) -> Tuple[str]:
        return group.attrs.get('I_axes')

    @staticmethod
    def signal(group: NXobject) -> str:
        return group.attrs.get('signal', 'I')

    @staticmethod
    def signal_errors(group: NXobject) -> Optional[str]:
        signal_name = group.attrs.get('signal', 'I')
        signal = group._group[signal_name]
        return signal.attrs.get('uncertainties')

    def coord_errors(group: NXobject, name: str) -> Optional[str]:
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


@NXcanSAS.register('SAStransmission_spectrum')
class SAStransmission_spectrumStrategy:

    @staticmethod
    def dims(group: NXobject) -> Tuple[str]:
        # TODO A valid file should have T_axes, do we need to fallback?
        if (axes := group.attrs.get('T_axes')) is not None:
            return (axes, )
        return ('lambda', )


class SASentry:
    nx_class = 'NXentry'

    def __init__(self, *, title: str, run: Union[str, int]):
        self.title = title
        self.run = run

    def __write_to_nexus_group__(self, group: NXobject):
        group.attrs['canSAS_class'] = 'SASentry'
        group.attrs['version'] = '1.0'
        group.attrs['definition'] = 'NXcanSAS'
        group.create_field('title', self.title)
        group.create_field('run', self.run)


@NXcanSAS.register('SASentry')
class SASentryStrategy:
    pass


@NXcanSAS.register('SASroot')
class SASrootStrategy:
    pass
