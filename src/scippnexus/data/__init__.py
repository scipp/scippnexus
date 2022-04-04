# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
_version = '1'

__all__ = ['get_path']


def _make_pooch():
    import pooch
    return pooch.create(
        path=pooch.os_cache('scippnexus'),
        env='SCIPPNEXUS_DATA_DIR',
        retry_if_failed=3,
        base_url='https://public.esss.dk/groups/scipp/scippnexus/{version}/',
        version=_version,
        registry={
            'PG3_4844_event.nxs': 'md5:d5ae38871d0a09a28ae01f85d969de1e',
        })


_pooch = _make_pooch()


def get_path(name: str) -> str:
    """
    Return the path to a data file bundled with scippnexus.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _pooch.fetch(name)
