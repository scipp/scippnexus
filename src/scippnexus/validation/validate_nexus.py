# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""This script can be used to check the validity of a Nexus file."""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable, Optional, Union

import h5py
import numpy as np


@dataclass
class Status:
    name: str
    message: str
    level: Optional[int] = 0


def error_reporter(error_id: int, register=None):

    def decorator(f: Callable[[Union[h5py.Group, h5py.Dataset]], Optional[Status]]):

        def func(obj: Union[h5py.Group, h5py.Dataset]):
            if (status := f(obj)):
                report(error_id, status.name, status.message, status.level)

        if register is not None:
            register.append(func)

        return func

    return decorator


_dataset_checks = []
_group_checks = []


def dataset_check(error_id: int):
    return error_reporter(error_id, _dataset_checks)


def group_check(error_id: int):
    return error_reporter(error_id, _group_checks)


def report(error_id: int, name: str, message: str, level: Optional[int] = 0):
    print(f'SNX-{error_id:04d} level={level} {name}: {message}')


def nx_class(group: h5py.Group) -> Optional[str]:
    if (nx_class := group.attrs.get('NX_class')) is not None:
        return nx_class if isinstance(nx_class, str) else nx_class.decode()


def allow_int_without_unit(dataset: h5py.Dataset):
    name = dataset.name.split('/')[-1]
    if 'index' in name:
        return True
    if name in ['event_id', 'event_index', 'detector_number']:
        return True
    parent_class = nx_class(dataset.parent)
    if parent_class == 'NXoff_geometry':
        return name in ['faces', 'winding_order', 'detector_faces']
    if parent_class == 'NXcylindrical_geometry':
        return name in ['cylinders', 'detector_number']
    if parent_class == 'NXdisk_chopper':
        return name in ['slits']
    return False


@dataset_check(2)
def check_units(dataset: h5py.Dataset):
    numeric = np.issubdtype(dataset.dtype, np.number)
    integral = np.issubdtype(dataset.dtype, np.integer)
    if numeric:
        if 'units' not in dataset.attrs:
            if integral:
                if not allow_int_without_unit(dataset):
                    return Status(
                        dataset.name,
                        'Possibly missing units attribute for integral dataset',
                        level=1)
            else:
                return Status(dataset.name,
                              'Missing units attribute for floating-point dataset')
    elif 'units' in dataset.attrs:
        return Status(dataset.name, 'Found units attribute for non-numeric dataset')


@dataset_check(4)
def check_transformation_offset_units(dataset: h5py.Dataset):
    if dataset.attrs.get('transformation_type') is not None:
        if dataset.attrs.get('offset') is not None:
            if dataset.attrs.get('offset_units') is None:
                return Status(
                    dataset.name,
                    'Missing offset_units attribute for transformation with offset')


@dataset_check(5)
def check_shape(dataset: h5py.Dataset):
    parent_class = nx_class(dataset.parent)
    if parent_class == 'NXlog':
        if dataset.ndim > 1 and dataset.parent.attrs.get('axes') is None:
            return Status(
                dataset.name,
                'Missing axes attribute for NXlog with dataset of more than 1 dimension'
            )


@group_check(1)
def check_deprecated_group(group: h5py.Group):
    deprecated = ['NXgeometry', 'NXshape', 'NXorientation', 'NXtranslation']
    if (nxcls := nx_class(group)) is not None:
        if nxcls in deprecated:
            return Status(group.name, f'{nx_class} is deprecated')


@group_check(3)
def check_depends_on(group: h5py.Group):
    for obj in group.values():
        if (nxcls := nx_class(obj)) is not None:
            if nxcls == 'NXtransformations' and 'depends_on' not in group:
                return Status(
                    group.name,
                    'Group contains NXtransformations but no `depends_on` entry point.',
                    level=1)


@group_check(6)
def check_nx_class(group: h5py.Group):
    if 'NX_class' not in group.attrs:
        return Status(group.name, 'Missing NX_class attribute')
    if not isinstance((nxcls := group.attrs['NX_class']), str):
        return Status(group.name,
                      f'NX_class attribute is not a string but {type(nxcls)}',
                      level=1)


def check_dataset(dataset: h5py.Dataset):
    for check in _dataset_checks:
        check(dataset)


def check_group(group: h5py.Group):
    for check in _group_checks:
        check(group)


def check(name: str, obj: Union[h5py.Group, h5py.Dataset]):
    if hasattr(obj, 'shape'):
        check_dataset(obj)
    else:
        check_group(obj)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('file', help='Nexus file to validate')
    args = parser.parse_args()
    filename = args.file
    with h5py.File(filename, 'r') as f:
        f.visititems(check)


if __name__ == '__main__':
    sys.exit(main())
