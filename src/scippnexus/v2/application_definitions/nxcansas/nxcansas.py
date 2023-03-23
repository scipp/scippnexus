# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Literal, Optional, Union

import scipp as sc

from ....typing import H5Group
from ...base import Group, NXdata, NXobject, base_definitions, create_field


class SASentry:
    nx_class = 'NXentry'

    def __init__(self, *, title: str, run: Union[str, int]):
        self.title = title
        self.run = run

    def __write_to_nexus_group__(self, group: H5Group):
        group.attrs['canSAS_class'] = 'SASentry'
        group.attrs['version'] = '1.0'
        group.attrs['definition'] = 'NXcanSAS'
        create_field(group, 'title', self.title)
        create_field(group, 'run', self.run)


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

    def __write_to_nexus_group__(self, group: H5Group):
        da = self.data
        group.attrs['canSAS_class'] = 'SASdata'
        group.attrs['signal'] = 'I'
        group.attrs['axes'] = da.dims  # for NeXus compliance, same as I_axes
        group.attrs['I_axes'] = da.dims
        group.attrs['Q_indices'] = tuple(da.dims.index(d) for d in da.coords['Q'].dims)
        # TODO writing Field should deal with variances
        signal = create_field(group, 'I', sc.values(da.data))
        # We use the _errors suffix for NeXus compliance, unlike the examples given in
        # NXcanSAS.
        if da.variances is not None:
            signal.attrs['uncertainties'] = 'I_errors'
            create_field(group, 'I_errors', sc.stddevs(da.data))
        if da.coords.is_edges('Q'):
            raise ValueError(
                "Q is given as bin-edges, but NXcanSAS requires Q points (such as "
                "bin centers).")
        coord = create_field(group, 'Q', da.coords['Q'])
        if da.coords['Q'].variances is not None:
            if self._variances is None:
                raise ValueError(
                    "Q has variances, must specify whether these represent "
                    "'uncertainties' or 'resolutions' using the 'Q_variances' option'")

            coord.attrs[self._variances] = 'Q_errors'
            create_field(group, 'Q_errors', sc.stddevs(da.coords['Q']))


class _SASdata(NXdata):

    def __init__(self, group: Group):
        fallback_dims = group.attrs.get('I_axes')
        if fallback_dims is not None:
            fallback_dims = (fallback_dims, )
        super().__init__(group, fallback_dims=fallback_dims, fallback_signal_name='I')

    # TODO Mechanism for custom error names
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


class _SAStransmission_spectrum(NXdata):

    def __init__(self, group: Group):
        # TODO A valid file should have T_axes, do we need to fallback?
        super().__init__(group,
                         fallback_dims=(group.attrs.get('T_axes', 'lambda'), ),
                         fallback_signal_name='T')


class NXcanSAS:

    def get(self, key: type, group: Group) -> type:
        if (cls := group.attrs.get('canSAS_class')) is not None:
            if cls == 'SASdata':
                return _SASdata
            if cls == 'SAStransmission_spectrum':
                return _SAStransmission_spectrum
        return base_definitions.get(key, group)


definitions = NXcanSAS()
