# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from .base import NXobject, NXroot  # noqa: F401
from .nxcylindrical_geometry import NXcylindrical_geometry  # noqa: F401
from .nxdata import NXdata, NXdetector, NXlog, NXmonitor  # noqa: F401
from .nxevent_data import NXevent_data  # noqa: F401
from .nxoff_geometry import NXoff_geometry  # noqa: F401
from .nxsample import NXsample  # noqa: F401
from .nxtransformations import NXtransformations  # noqa: F401


class NXentry(NXobject):
    """Entry in a NeXus file."""


class NXinstrument(NXobject):
    """Group of instrument-related information."""


class NXaperture(NXobject):
    """NXaperture"""


class NXattenuator(NXobject):
    """NXattenuator"""


class NXbeam(NXobject):
    """NXbeam"""


class NXbeam_stop(NXobject):
    """NXbeam_stop"""


class NXbending_magnet(NXobject):
    """NXbending_magnet"""


class NXcapillary(NXobject):
    """NXcapillary"""


class NXcite(NXobject):
    """NXcite"""


class NXcollection(NXobject):
    """NXcollection"""


class NXcollimator(NXobject):
    """NXcollimator"""


class NXcrystal(NXobject):
    """NXcrystal"""


class NXdetector_group(NXobject):
    """NXdetector_group"""


class NXdetector_module(NXobject):
    """NXdetector_module"""


class NXdisk_chopper(NXobject):
    """NXdisk_chopper"""


class NXenvironment(NXobject):
    """NXenvironment"""


class NXfermi_chopper(NXobject):
    """NXfermi_chopper"""


class NXfilter(NXobject):
    """NXfilter"""


class NXflipper(NXobject):
    """NXflipper"""


class NXfresnel_zone_plate(NXobject):
    """NXfresnel_zone_plate"""


class NXgeometry(NXobject):
    """NXgeometry"""


class NXgrating(NXobject):
    """NXgrating"""


class NXguide(NXobject):
    """NXguide"""


class NXinsertion_device(NXobject):
    """NXinsertion_device"""


class NXmirror(NXobject):
    """NXmirror"""


class NXmoderator(NXobject):
    """NXmoderator"""


class NXmonochromator(NXobject):
    """NXmonochromator"""


class NXnote(NXobject):
    """NXnote"""


class NXorientation(NXobject):
    """NXorientation"""


class NXparameters(NXobject):
    """NXparameters"""


class NXpdb(NXobject):
    """NXpdb"""


class NXpinhole(NXobject):
    """NXpinhole"""


class NXpolarizer(NXobject):
    """NXpolarizer"""


class NXpositioner(NXobject):
    """NXpositioner"""


class NXprocess(NXobject):
    """NXprocess"""


class NXreflections(NXobject):
    """NXreflections"""


class NXsample_component(NXobject):
    """NXsample_component"""


class NXsensor(NXobject):
    """NXsensor"""


class NXshape(NXobject):
    """NXshape"""


class NXsource(NXobject):
    """NXsource"""


class NXslit(NXobject):
    """NXslit"""


class NXsubentry(NXobject):
    """NXsubentry"""


class NXtranslation(NXobject):
    """NXtranslation"""


class NXuser(NXobject):
    """NXuser"""


class NXvelocity_selector(NXobject):
    """NXvelocity_selector"""


class NXxraylens(NXobject):
    """NXxraylens"""


# Not included in list of NeXus classes since this is the "base" of all others
del NXobject
