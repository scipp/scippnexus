# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass

from . import NXsource, create_field
from .typing import H5Group


@dataclass(kw_only=True)
class ESSSourceMetadata:
    """ESS Source Metadata default values.

    Example
    -------
    >>> import scippnexus as snx
    >>> # Write the metadata in the file.
    >>> with snx.File('output.nxs', 'w') as file:
    ...     file['source'] = ESSSourceMetadata()
    ...
    >>> # Read the metadata from the file.
    >>> with snx.File('output.nxs') as file:
    ...     print(file['source/name'][()])
    ...
    European Spallation Source
    >>>
    """

    nx_class = NXsource

    def __write_to_nexus_group__(self, group: H5Group):
        name_field = create_field(group, 'name', 'European Spallation Source')
        name_field.attrs['short_name'] = 'ESS'
        create_field(group, 'type', 'Spallation Neutron Source')
        create_field(group, 'probe', 'neutron')
