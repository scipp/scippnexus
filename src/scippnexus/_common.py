# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

import numpy as np
import scipp as sc

from .typing import ScippIndex


def convert_time_to_datetime64(
    raw_times: sc.Variable,
    start: str | None = None,
    scaling_factor: float | np.float64 = None,
) -> sc.Variable:
    """
    The nexus standard allows an arbitrary scaling factor to be inserted
    between the numbers in the `time` series and the unit of time reported
    in the nexus attribute.

    The times are also relative to a given log start time, which might be
    different for each log. If this log start time is not available, the start of the
    unix epoch (1970-01-01T00:00:00) is used instead.

    See https://manual.nexusformat.org/classes/base_classes/NXlog.html

    Args:
        raw_times: The raw time data from a nexus file.
        start: Optional, the start time of the log in an ISO8601
            string. If not provided, defaults to the beginning of the
            unix epoch (1970-01-01T00:00:00).
        scaling_factor: Optional, the scaling factor between the provided
            time series data and the unit of the raw_times Variable. If
            not provided, defaults to 1 (a no-op scaling factor).
    """
    if (
        raw_times.dtype in (sc.DType.float64, sc.DType.float32)
    ) or scaling_factor is not None:
        unit = sc.units.ns
    else:
        # determine more precise unit
        ratio = sc.scalar(1.0, unit=start.unit) / sc.scalar(
            1.0, unit=raw_times.unit
        ).to(unit=start.unit)
        unit = start.unit if ratio.value < 1.0 else raw_times.unit

    if scaling_factor is None:
        times = raw_times
    else:
        times = raw_times * sc.scalar(value=scaling_factor)
    return start.to(unit=unit, copy=False) + times.to(
        dtype=sc.DType.int64, unit=unit, copy=False
    )


def _to_canonical_select(dims: list[str], select: ScippIndex) -> dict[str, int | slice]:
    """Return selection as dict with explicit dim labels"""

    def check_1d():
        if len(dims) != 1:
            raise sc.DimensionError(
                f"Dataset has multiple dimensions {dims}, "
                "specify the dimension to index."
            )

    if select is Ellipsis:
        return {}
    if isinstance(select, tuple) and len(select) == 0:
        return {}
    if isinstance(select, tuple) and isinstance(select[0], str):
        key, sel = select
        return {key: sel}
    if isinstance(select, tuple):
        check_1d()
        if len(select) != 1:
            raise sc.DimensionError(
                f"Dataset has single dimension {dims}, "
                "but multiple indices {select} were specified."
            )
        return {dims[0]: select[0]}
    elif isinstance(select, int | sc.Variable) or isinstance(select, slice):
        check_1d()
        return {dims[0]: select}
    if not isinstance(select, dict):
        raise IndexError(f"Cannot process index {select}.")
    return select.copy()


def to_plain_index(dims: list[str], select: ScippIndex) -> int | slice | tuple:
    """
    Given a valid "scipp" index 'select', return an equivalent plain numpy-style index.
    """
    select = _to_canonical_select(dims, select)
    index = [slice(None)] * len(dims)
    for key, sel in select.items():
        if key not in dims:
            raise sc.DimensionError(
                f"'{key}' used for indexing not found in dataset dims {dims}."
            )
        index[dims.index(key)] = sel
    if len(index) == 1:
        return index[0]
    return tuple(index)


def to_child_select(
    dims: list[str],
    child_dims: list[str],
    select: ScippIndex,
    bin_edge_dim: str | None = None,
) -> ScippIndex:
    """
    Given a valid "scipp" index 'select' for a Nexus class, return a selection for a
    child field of the class, which may have fewer dimensions.

    This removes any selections that apply to the parent but not the child.
    """
    select = _to_canonical_select(dims, select)
    for d in dims:
        if d not in child_dims and d in select:
            del select[d]
    for dim in select:
        if dim == bin_edge_dim:
            index = select[dim]
            if isinstance(index, int):
                select[dim] = slice(index, index + 2)
            elif index.stop > index.start:
                select[dim] = slice(index.start, index.stop + 1)
    return select
