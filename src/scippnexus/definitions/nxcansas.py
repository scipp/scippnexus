# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Optional
from ..typing import H5Group

# required children
# required attrs
# meaning of special attrs (e.g., errors)
# consider structure verification out of scope, just deal with leaf class,
# such as SASData!?


class ApplicationDefinitionStrategy:

    def __init__(self, group: H5Group):
        self._group = group

    def child_strategy(self, group: H5Group):
        if (definition_class := group.attrs.get(self.class_attribute)) is not None:
            return self.strategies.get(definition_class)


class NXcanSAS(ApplicationDefinitionStrategy):
    class_attribute = 'canSAS_class'


class SASdata(NXcanSAS):

    @property
    def dims(self):
        axes = self._group.attrs.get('axes')
        if isinstance(axes, str):
            axes = axes.split(" ")
        if axes is None:
            axes = self._group.attrs.get('I_axes')
        if not isinstance(axes, list):
            axes = [axes]
        if axes.count('Q') != 1:
            index = 0
            for i, ax in enumerate(axes):
                if ax == 'Q':
                    axes[i] = f'Q{index}'
                    index += 1
        return tuple(axes)

    @property
    def axes(self):
        return self._group.attrs.get('I_axes')

    @property
    def signal(self):
        return self._group.attrs.get('signal', 'I')

    @property
    def signal_errors(self) -> Optional[str]:
        signal_name = self._group.attrs.get('signal', 'I')
        signal = self._group._group[signal_name]
        return signal.attrs.get('uncertainties')

    def coord_errors(self, name) -> Optional[str]:
        if name != 'Q':
            return None
        # TODO This naively stores this as Scipp errors, which are just Gaussian.
        # This is probably not correct in all cases.
        uncertainties = self._group[name].attrs.get('uncertainties')
        resolutions = self._group[name].attrs.get('resolutions')
        if uncertainties is None:
            return resolutions
        elif resolutions is None:
            return uncertainties
        raise RuntimeError("Cannot handle both uncertainties and resolutions for Q")


class SAStransmission_spectrum(NXcanSAS):

    @property
    def dims(self):
        # TODO A valid file should have T_axes, do we need to fallback?
        if (axes := self._group.attrs.get('T_axes')) is not None:
            return (axes, )
        return ('lambda', )


class SASentry(NXcanSAS):
    pass


class SASroot(NXcanSAS):
    pass


NXcanSAS.classes = [SASroot, SASentry, SASdata, SAStransmission_spectrum]
NXcanSAS.strategies = {cls.__name__: cls for cls in NXcanSAS.classes}
