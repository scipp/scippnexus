# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

__all__ = ['get_path']


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache('scippnexus-externalfile'),
        env='SCIPPNEXUS_DATA_DIR',
        retry_if_failed=3,
        base_url='login.esss.dk:/mnt/groupdata/scipp/testdata/scippnexus/',
        registry={
            '2023/DREAM_baseline_all_dets.nxs': 'md5:1e1a3141c6785d25777b456e2a653f42',
            '2023/BIFROST_873855_00000015.hdf': 'md5:eb180b09d265c308e81c4a4885662bbd',
            '2023/DREAM_mccode.h5': 'md5:ebe4be53f20e139e865bc20b264bdceb',
            '2023/LOKI_mcstas_nexus_geometry.nxs': 'md5:f431d9775a53caffeebe9b879189b17c',  # noqa: E501
            '2023/NMX_2e11-rechunk.h5': 'md5:1174c208614b2e7a5faddc284b41d2c9',
            '2023/YMIR_038243_00010244.hdf': 'md5:cefb04b6d4d36f16e7f329a6045ad129',
            '2023/amor2020n000346_tweaked.nxs': 'md5:4e07ccc87b5c6549e190bc372c298e83',
            '2023/LOKI_60322-2022-03-02_2205_fixed.nxs': 'md5:e174d87dff10cbdcc2a33d093ae1e8ce',  # noqa: E501
        },
    )


_pooch = _make_pooch()


def sshdownloader(url, output_file, pooch):
    from subprocess import call

    cmd = ['scp', f'{url}', f'{output_file}']
    call(cmd)  # noqa: S603


def get_path(name: str) -> str:
    """
    Get path of file "downloaded" via SSH from login.esss.dk.

    You must have setup SSH agent for passwordless login for this to work.
    """
    return _pooch.fetch(name, downloader=sshdownloader)
