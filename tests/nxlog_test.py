# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import h5py
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

import scippnexus as snx
from scippnexus import NXentry, NXlog

# representative sample of UTF-8 test strings from
# https://www.w3.org/2001/06/utf-8-test/UTF-8-demo.html
UTF8_TEST_STRINGS = (
    "∮ E⋅da = Q,  n → ∞, ∑ f(i) = ∏ g(i), ∀x∈ℝ: ⌈x⌉ = −⌊−x⌋, α ∧ ¬β = ¬(¬α ∨ β)",  # noqa: RUF001
    "2H₂ + O₂ ⇌ 2H₂O, R = 4.7 kΩ, ⌀ 200 mm",
    "Σὲ γνωρίζω ἀπὸ τὴν κόψη",
)


@pytest.fixture()
def h5root():
    """Yield h5py root group (file)"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        yield f


@pytest.fixture()
def nxroot():
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = snx.Group(f, definitions=snx.base_definitions())
        root.create_class('entry', NXentry)
        yield root


def test_nxobject_log(h5root):
    da = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2, 3.3]),
        coords={
            'time': sc.epoch(unit='ns')
            + sc.array(dims=['time'], unit='s', values=[4.4, 5.5, 6.6]).to(
                unit='ns', dtype='int64'
            )
        },
    )
    log = snx.create_class(h5root, 'log', NXlog)
    snx.create_field(log, 'value', da.data)
    snx.create_field(log, 'time', da.coords['time'] - sc.epoch(unit='ns'))
    log = snx.Group(log, definitions=snx.base_definitions())
    assert sc.identical(log[...], da)


def test_nxlog_with_missing_value_uses_time_as_value(nxroot):
    time = sc.epoch(unit='ns') + sc.array(
        dims=['time'], unit='s', values=[4.4, 5.5, 6.6]
    ).to(unit='ns', dtype='int64')
    log = nxroot['entry'].create_class('log', NXlog)
    log['time'] = time - sc.epoch(unit='ns')
    loaded = log[()]
    assert_identical(loaded, sc.DataArray(data=time))


def test_nxlog_length_1(h5root):
    nxroot = snx.Group(h5root, definitions=snx.base_definitions())
    da = sc.DataArray(
        sc.array(dims=['time'], values=[1.1]),
        coords={
            'time': sc.epoch(unit='ns')
            + sc.array(dims=['time'], unit='s', values=[4.4]).to(
                unit='ns', dtype='int64'
            )
        },
    )
    log = nxroot.create_class('log', NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')
    assert sc.identical(log[...], da)


def test_nxlog_length_1_two_dims_no_time_defaults_inner_dim_name(nxroot):
    var = sc.array(dims=['time', 'ignored'], values=[[1.1]])
    log = nxroot['entry'].create_class('log', NXlog)
    log['value'] = var
    assert_identical(log[...], sc.DataArray(var.rename(ignored='dim_1')))


def test_nxlog_length_1_two_dims_with_time_defaults_inner_dim_name(nxroot):
    da = sc.DataArray(
        sc.array(dims=['time', 'ignored'], values=[[1.1]]),
        coords={
            'time': sc.epoch(unit='ns')
            + sc.array(dims=['time'], unit='s', values=[4.4]).to(
                unit='ns', dtype='int64'
            )
        },
    )
    log = nxroot['entry'].create_class('log', NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')
    assert sc.identical(log[...], da.rename(ignored='dim_1'))


def test_nxlog_axes_replaces_time_dim(nxroot):
    da = sc.DataArray(
        sc.array(dims=['time', 'ignored'], values=[[1.1]]),
        coords={
            'time': sc.epoch(unit='ns')
            + sc.array(dims=['time'], unit='s', values=[4.4]).to(
                unit='ns', dtype='int64'
            )
        },
    )
    log = nxroot['entry'].create_class('log', NXlog)
    log._group.attrs['axes'] = ['yy', 'xx']
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')
    expected = sc.DataArray(
        sc.array(dims=['yy', 'xx'], values=[[1.1]]),
        coords={'time': da.coords['time'].squeeze()},
    )
    assert sc.identical(log[...], expected)


def test_nxlog_three_dims_with_time_of_length_1(nxroot):
    da = sc.DataArray(
        sc.array(dims=['time', 'a', 'b'], values=np.arange(9.0).reshape(1, 3, 3)),
        coords={
            'time': sc.epoch(unit='ns')
            + sc.array(dims=['time'], unit='s', values=[4.4]).to(
                unit='ns', dtype='int64'
            )
        },
    )
    log = nxroot['entry'].create_class('log', NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time'] - sc.epoch(unit='ns')
    loaded = log[...]
    assert_identical(
        loaded.data,
        sc.array(
            dims=['time', 'dim_1', 'dim_2'], values=np.arange(9.0).reshape(1, 3, 3)
        ),
    )


def test_nxlog_with_shape_0(nxroot):
    da = sc.DataArray(
        sc.ones(dims=['time', 'ignored'], shape=(0, 1)),
        coords={'time': sc.ones(dims=['time'], shape=(0,), unit='s')},
    )
    log = nxroot['entry'].create_class('log', NXlog)
    log['value'] = da.data
    log['time'] = da.coords['time']
    da.coords['time'] = sc.datetimes(dims=['time'], values=[], unit='ns')
    assert_identical(log[...], da.rename(ignored='dim_1'))


def test_log_with_connection_status_raises_with_positional_and_label_indexing(h5root):
    da = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2, 3.3]),
        coords={
            'time': sc.array(
                dims=['time'], unit='s', values=[44, 55, 66], dtype='int64'
            )
        },
    )
    log = snx.create_class(h5root, 'log', NXlog)
    snx.create_field(log, 'value', da.data)
    snx.create_field(log, 'time', da.coords['time'])
    connection_status = da['time', :2].copy()
    snx.create_field(log, 'connection_status', connection_status.data)
    snx.create_field(log, 'connection_status_time', connection_status.coords['time'])
    log = snx.Group(log, definitions=snx.base_definitions())
    with pytest.raises(sc.DimensionError):
        log['time', :2]
    with pytest.raises(sc.DimensionError):
        log['time', : sc.scalar(60, unit='s')]
    with pytest.raises(sc.DimensionError):
        log[:2]
    with pytest.raises(sc.DimensionError):
        log[:]


@pytest.mark.parametrize('sublog_length', [0, 1, 2])
def test_log_with_connection_status_loaded_as_datagroup_containing_data_arrays(
    h5root, sublog_length
):
    da = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2, 3.3]),
        coords={
            'time': sc.array(
                dims=['time'], unit='s', values=[44, 55, 66], dtype='int64'
            )
        },
    )
    log = snx.create_class(h5root, 'log', NXlog)
    snx.create_field(log, 'value', da.data)
    snx.create_field(log, 'time', da.coords['time'])
    connection_status = da['time', :sublog_length].copy()
    snx.create_field(log, 'connection_status', connection_status.data)
    snx.create_field(log, 'connection_status_time', connection_status.coords['time'])
    log = snx.Group(log, definitions=snx.base_definitions())
    loaded = log[()]
    da.coords['time'] = sc.epoch(unit='s') + da.coords['time']
    connection_status.coords['time'] = (
        sc.epoch(unit='s') + connection_status.coords['time']
    )
    assert_identical(loaded['value'], da)
    assert_identical(loaded['connection_status'], connection_status)


def test_log_with_alarm_loaded_as_datagroup_containing_data_arrays(h5root):
    da = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2, 3.3]),
        coords={
            'time': sc.array(
                dims=['time'], unit='s', values=[44, 55, 66], dtype='int64'
            )
        },
    )
    log = snx.create_class(h5root, 'log', NXlog)
    snx.create_field(log, 'value', da.data)
    snx.create_field(log, 'time', da.coords['time'])
    alarm = da['time', :2].copy()
    alarm.coords['message'] = sc.array(dims=['time'], values=['alarm 1', 'alarm 2'])
    snx.create_field(log, 'alarm_severity', alarm.data)
    snx.create_field(log, 'alarm_message', alarm.coords['message'])
    snx.create_field(log, 'alarm_time', alarm.coords['time'])
    log = snx.Group(log, definitions=snx.base_definitions())
    loaded = log[()]
    da.coords['time'] = sc.epoch(unit='s') + da.coords['time']
    alarm.coords['time'] = sc.epoch(unit='s') + alarm.coords['time']
    assert_identical(loaded['value'], da)
    assert_identical(loaded['alarm'], alarm)
