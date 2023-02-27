# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from .nxdata import NXdata, NXdataStrategy


class NXlogStrategy(NXdataStrategy):

    @staticmethod
    def axes2(info):
        if (ax := NXdataStrategy.axes2(info)) is not None:
            return ax
        # We get the shape from the original dataset, to make sure we do not squeeze
        # dimensions too early
        value = info.datasets.get('value')
        if value is None:
            return ('time', )
        child_dataset = value.value
        ndim = child_dataset.ndim
        shape = child_dataset.shape
        # The outermost axis in NXlog is pre-defined to 'time' (if present). Note
        # that this may be overridden by an `axes` attribute, if defined for the group.
        if 'time' in info.datasets:
            raw_axes = ['time'] + (['.'] * (ndim - 1))
        else:
            raw_axes = ['.'] * ndim
        axes = []
        for i, ax in enumerate(raw_axes):
            # Squeeze dimensions that have size 1 and are not 'time'
            if (ax == 'time') or (shape[i] != 1):
                axes.append(ax)
        return axes

    @staticmethod
    def signal2(info):
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXlog uses a "hard-coded" signal name 'value', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        signal_name = info.attrs.get('signal', 'value')
        if (signal := info.datasets.get(signal_name)) is not None:
            return signal_name, signal
        else:
            return None, None


class NXlog(NXdata):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.child_params['time'] = {'is_time': True}

    def _default_strategy(self):
        return NXlogStrategy
