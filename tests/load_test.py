# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import io
from pathlib import Path

import h5py as h5
import pytest
import scipp as sc
import scipp.testing

import scippnexus as snx


@pytest.fixture()
def reference_data() -> sc.DataGroup:
    return sc.DataGroup(
        entry=sc.DataGroup(
            log=sc.DataArray(
                data=sc.arange('time', -0.5, 1.5, 0.5, unit='rad'),
                coords={
                    'time': sc.array(
                        dims=['time'], values=[100, 200, 300, 400], unit='s'
                    )
                    + sc.epoch(unit='s')
                },
            ),
            instrument=sc.DataGroup(
                detector=sc.DataGroup(
                    data=sc.DataArray(
                        data=sc.array(
                            dims=['detector_number'], values=[24, 56, 19], unit='counts'
                        ),
                        coords={
                            'detector_number': sc.array(
                                dims=['detector_number'], values=[1, 3, 4], unit=None
                            )
                        },
                    )
                )
            ),
        )
    )


@pytest.fixture()
def nexus_buffer(reference_data) -> io.BytesIO:
    buffer = io.BytesIO()
    with snx.File(buffer, 'w') as root:
        entry = root.create_class('entry', snx.NXentry)

        ref_log = reference_data['entry']['log']
        log = entry.create_class('log', snx.NXlog)
        log.create_field('value', ref_log.data)
        log.create_field('time', ref_log.coords['time'])

        ref_data = reference_data['entry']['instrument']['detector']['data']
        instrument = entry.create_class('instrument', snx.NXinstrument)
        detector = instrument.create_class('detector', snx.NXdetector)
        detector.create_field('detector_number', ref_data.coords['detector_number'])
        detector.create_field('data', ref_data.data)
    return buffer


def test_load_all_from_buffer(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    loaded = snx.load(nexus_buffer)
    sc.testing.assert_identical(loaded, reference_data)


@pytest.mark.parametrize('path_type', [str, Path])
def test_load_all_from_file(
    path_type: type,
    nexus_buffer: io.BytesIO,
    reference_data: sc.DataGroup,
    tmp_path: Path,
) -> None:
    filename = tmp_path / 'test.nxs'
    filename.write_bytes(nexus_buffer.getvalue())
    loaded = snx.load(path_type(filename))
    sc.testing.assert_identical(loaded, reference_data)


def test_load_all_from_snx_group(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    with snx.File(nexus_buffer, 'r') as f:
        loaded = snx.load(f)
    sc.testing.assert_identical(loaded, reference_data)


def test_load_all_from_nested_snx_group(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    with snx.File(nexus_buffer, 'r') as f:
        loaded = snx.load(f['entry'])
    sc.testing.assert_identical(loaded, reference_data['entry'])


def test_load_all_from_h5_group(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    with h5.File(nexus_buffer, 'r') as f:
        loaded = snx.load(f)
    sc.testing.assert_identical(loaded, reference_data)


def test_load_all_from_nested_h5_group(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    with h5.File(nexus_buffer, 'r') as f:
        loaded = snx.load(f['entry'])
    sc.testing.assert_identical(loaded, reference_data['entry'])


def test_load_from_root_single_segment(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    loaded = snx.load(nexus_buffer, root='entry')
    sc.testing.assert_identical(loaded, reference_data['entry'])


def test_load_from_root_multiple_segments(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    loaded = snx.load(nexus_buffer, root='entry/instrument/detector')
    sc.testing.assert_identical(
        loaded, reference_data['entry']['instrument']['detector']
    )


def test_load_nonexistent_root(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    with pytest.raises(KeyError):
        snx.load(nexus_buffer, root='entry/does-not-exist')


def test_load_select_index_range(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    loaded = snx.load(nexus_buffer, select={'time': slice(1, None)})
    sc.testing.assert_identical(loaded, reference_data['time', 1:])


def test_load_select_index(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    loaded = snx.load(nexus_buffer, select={'time': 2})
    sc.testing.assert_identical(loaded, reference_data['time', 2])


def test_load_from_root_select_index_range(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    loaded = snx.load(
        nexus_buffer,
        root='entry/instrument',
        select={'detector_number': slice(None, 2)},
    )
    sc.testing.assert_identical(
        loaded, reference_data['entry']['instrument']['detector_number', :2]
    )


def test_load_select_unknown_dim(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    with pytest.raises(sc.DimensionError):
        snx.load(nexus_buffer, select={'y': 3})


class NXdetectorTimes10(snx.NXdetector):
    def assemble(self, dg: sc.DataGroup) -> sc.DataGroup:
        return 10 * super().assemble(dg)


def test_load_from_buffer_with_explicit_base_definitions(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    loaded = snx.load(nexus_buffer, definitions=snx.base_definitions())
    sc.testing.assert_identical(loaded, reference_data)


def test_load_from_buffer_with_custom_definitions(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    definitions = {**snx.base_definitions(), 'NXdetector': NXdetectorTimes10}
    loaded = snx.load(nexus_buffer, definitions=definitions)
    expected = reference_data.copy()
    expected['entry']['instrument']['detector']['data'] *= 10
    sc.testing.assert_identical(loaded, expected)


def test_load_from_h5_group_with_custom_definitions(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    definitions = {**snx.base_definitions(), 'NXdetector': NXdetectorTimes10}
    with h5.File(nexus_buffer, 'r') as f:
        loaded = snx.load(f, definitions=definitions)
    expected = reference_data.copy()
    expected['entry']['instrument']['detector']['data'] *= 10
    sc.testing.assert_identical(loaded, expected)


def test_load_from_snx_group_rejects_new_definitions(
    nexus_buffer: io.BytesIO, reference_data: sc.DataGroup
) -> None:
    definitions = {**snx.base_definitions(), 'NXdetector': NXdetectorTimes10}
    with pytest.raises(TypeError, match='Cannot override application definitions'):
        with snx.File(nexus_buffer, 'r') as f:
            snx.load(f, definitions=definitions)
