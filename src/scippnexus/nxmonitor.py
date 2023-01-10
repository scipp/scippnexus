# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Dict, Union

from .nxdetector import NXdetector
from .nxobject import Field


class NXmonitor(NXdetector):

    @property
    def _event_grouping(self) -> Dict[str, Union[str, Field]]:
        # Unlike NXdetector, NXmonitor does not group by 'detector_number'. We pass
        # grouping information that matches the underlying binning of NXevent_data
        # such that no addition binning will need to be performed. That is, the by-pulse
        # binning present in the file (in NXevent_data) is preserved.
        return {
            'grouping_key': 'event_time_zero',
            'grouping': self.events['event_time_zero']
        }
