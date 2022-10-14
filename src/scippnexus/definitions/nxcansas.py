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


class Definition:

    @classmethod
    def definition_for_group(cls, group: H5Group):
        if (definition_class := group.attrs.get(cls.class_attribute)) is not None:
            return cls.definitions.get(definition_class)


class SASobject:

    def __init__(self, group: H5Group):
        self._group = group


class SASdata(SASobject):

    @property
    def signal_errors(self) -> Optional[str]:
        signal_name = self._group.attrs.get('signal', 'I')
        signal = self._group[signal_name]
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


class SASentry(SASobject):
    pass


class SASroot(SASobject):
    pass


class NXcanSAS(Definition):
    class_attribute = 'canSAS_class'
    classes = [SASroot, SASentry, SASdata]
    definitions = {cls.__name__: cls for cls in classes}
