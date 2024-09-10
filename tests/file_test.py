# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import io
from pathlib import Path

import h5py as h5
import pytest

import scippnexus as snx


def is_closed(file: snx.File) -> bool:
    try:
        file.create_class('x', 'NXentry')
    except ValueError as err:
        return 'Unable to synchronously create group' in err.args[0]
    return False


@pytest.mark.parametrize('path_type', [str, Path])
def test_load_entry_from_filename(tmp_path, path_type):
    with h5.File(tmp_path / 'test.nxs', 'w') as f:
        snx.create_class(f, 'entry', snx.NXentry)

    with snx.File(path_type(tmp_path / 'test.nxs'), 'r') as f:
        assert f.keys() == {'entry'}
    assert is_closed(f)


def test_load_entry_from_buffer(tmp_path):
    buffer = io.BytesIO()
    with h5.File(buffer, 'w') as f:
        snx.create_class(f, 'entry', snx.NXentry)
    buffer.seek(0)

    with snx.File(buffer, 'r') as f:
        assert f.keys() == {'entry'}
    assert is_closed(f)


def test_load_entry_from_h5py_group_root(tmp_path):
    with h5.File('test.nxs', 'w', driver='core', backing_store=False) as h5_file:
        snx.create_class(h5_file, 'entry', snx.NXentry)
        with snx.File(h5_file) as snx_file:
            assert snx_file.keys() == {'entry'}
        # Not great, but we don't want to close the h5py file ourselves.
        assert not is_closed(snx_file)


def test_load_entry_from_h5py_group_not_root(tmp_path):
    with h5.File('test.nxs', 'w', driver='core', backing_store=False) as h5_file:
        entry = snx.create_class(h5_file, 'entry', snx.NXentry)
        snx.create_class(entry, 'user', snx.NXuser)
        with snx.File(h5_file['entry']) as snx_file:
            assert snx_file.keys() == {'user'}
        # Not great, but we don't want to close the h5py file ourselves.
        assert not is_closed(snx_file)


def test_file_from_h5py_group_does_not_allow_extra_args(tmp_path):
    with h5.File('test.nxs', 'w', driver='core', backing_store=False) as h5_file:
        snx.create_class(h5_file, 'entry', snx.NXentry)
        with pytest.raises(
            TypeError, match='Cannot provide both h5py.File and other arguments'
        ):
            snx.File(h5_file, 'r')
