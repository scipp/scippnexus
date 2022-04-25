# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Dict, Union
from .nxobject import Field
from .nxdetector import NXdetector


class NXmonitor(NXdetector):
    @property
    def _event_grouping(self) -> Dict[str, Union[str, Field]]:
        return {
            'groups_key': 'event_time_zero',
            'groups': self.events['event_time_zero']
        }
