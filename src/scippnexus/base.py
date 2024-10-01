# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, overload

import numpy as np
import scipp as sc
from scipp.core import label_based_index_to_positional_index

from ._cache import cached_property
from ._common import to_child_select
from .attrs import Attrs
from .field import Field
from .typing import H5Dataset, H5Group, ScippIndex


def asvariable(obj: Any | sc.Variable) -> sc.Variable:
    return obj if isinstance(obj, sc.Variable) else sc.scalar(obj, unit=None)


class NexusStructureError(Exception):
    """Invalid or unsupported class and field structure in Nexus."""

    pass


def is_dataset(obj: H5Group | H5Dataset) -> bool:
    """Return true if the object is an h5py.Dataset or equivalent.

    Use this instead of isinstance(obj, h5py.Dataset) to ensure that code is compatible
    with other h5py-alike interfaces.
    """
    return hasattr(obj, 'shape')


_scipp_dtype = {
    np.dtype('int8'): sc.DType.int32,
    np.dtype('int16'): sc.DType.int32,
    np.dtype('uint8'): sc.DType.int32,
    np.dtype('uint16'): sc.DType.int32,
    np.dtype('uint32'): sc.DType.int32,
    np.dtype('uint64'): sc.DType.int64,
    np.dtype('int32'): sc.DType.int32,
    np.dtype('int64'): sc.DType.int64,
    np.dtype('float32'): sc.DType.float32,
    np.dtype('float64'): sc.DType.float64,
    np.dtype('bool'): sc.DType.bool,
}


def _dtype_fromdataset(dataset: H5Dataset) -> sc.DType:
    return _scipp_dtype.get(dataset.dtype, sc.DType.string)


def _squeezed_field_sizes(dataset: H5Dataset) -> dict[str, int]:
    if (shape := dataset.shape) == (1,):
        return {}
    return {f'dim_{i}': size for i, size in enumerate(shape)}


class NXobject:
    def _init_field(self, field: Field):
        if field.sizes is None:
            field.sizes = _squeezed_field_sizes(field.dataset)
        field.dtype = _dtype_fromdataset(field.dataset)

    def __init__(self, attrs: dict[str, Any], children: dict[str, Field | Group]):
        """Subclasses should call this in their __init__ method, or ensure that they
        initialize the fields in `children` with the correct sizes and dtypes."""
        self._attrs = attrs
        self._children = children
        for field in children.values():
            if isinstance(field, Field):
                self._init_field(field)

    @property
    def unit(self) -> None | sc.Unit:
        raise AttributeError(
            f"Group-like {self._attrs.get('NX_class')} has no well-defined unit"
        )

    @cached_property
    def sizes(self) -> dict[str, int]:
        return sc.DataGroup(self._children).sizes

    def index_child(
        self, child: Field | Group, sel: ScippIndex
    ) -> sc.Variable | sc.DataArray | sc.Dataset | sc.DataGroup:
        """
        When a Group is indexed, this method is called to index each child.

        The main purpose of this is to translate the Group index to the child index.
        Since the group dimensions (usually given by the signal) may be a superset of
        the child dimensions, we need to translate the group index to a child index.

        The default implementation assumes that the child shape is identical to the
        group shape, for all child dims. Subclasses of NXobject, in particular NXdata,
        override this method to handle bin edges.
        """
        # TODO Could avoid determining sizes if sel is trivial. Merge with
        # NXdata.index_child?
        child_sel = to_child_select(tuple(self.sizes), child.dims, sel)
        return child[child_sel]

    def read_children(self, sel: ScippIndex) -> sc.DataGroup:
        """
        When a Group is indexed, this method is called to read all children.

        The default implementation simply calls index_child on each child and returns
        the result as a DataGroup.

        Subclasses of NXobject, in particular NXevent_data, override this method to
        to implement special logic for reading children with interdependencies, i.e.,
        where reading each child in isolation is not possible.
        """
        return sc.DataGroup(
            {
                name: self.index_child(child, sel)
                for name, child in self._children.items()
            }
        )

    def assemble(self, dg: sc.DataGroup) -> sc.DataGroup | sc.DataArray | sc.Dataset:
        """
        When a Group is indexed, this method is called to assemble the read children
        into the result object.

        The default implementation simply returns the DataGroup.

        Subclasses of NXobject, in particular NXdata, override this method to return
        an object with more semantics such as a DataArray or Dataset.
        """
        return dg

    def convert_label_index_to_positional(self, sel):
        if isinstance(sel, dict):
            return dict(self.convert_label_index_to_positional(s) for s in sel.items())
        if (
            isinstance(sel, tuple)
            and len(sel) > 1
            and (
                # Scalar label index
                isinstance((index := sel[1]), sc.Variable)
                or (
                    # Slice label index
                    isinstance(index, slice)
                    and (
                        isinstance(index.start, sc.Variable)
                        or isinstance(index.stop, sc.Variable)
                    )
                )
            )
        ):
            if (dim := sel[0]) in self.sizes:
                if (coord := self._children.get(dim)) is not None:
                    return label_based_index_to_positional_index(
                        self.sizes, coord[()], index
                    )
                if isinstance(self._signal, Field):
                    # If it is *not* a field, the translation can happen in subgroup
                    raise sc.DimensionError(
                        f'Invalid slice dimension: \'{dim}\': '
                        f'no coordinate for that dimension. '
                        f'Coordinates are {tuple(self._children.keys())}'
                    )

        # It is not a label index, or the index will be translated in subgroup.
        # Either way, pass it on.
        return sel


class NXroot(NXobject):
    pass


class Group(Mapping):
    """
    A group in a NeXus file.

    This class is a wrapper around an h5py.Group object. It provides a dict-like
    interface to the children of the group, and provides access to the attributes
    of the group. The children are either Field or Group objects, depending on
    whether the child is a dataset or a group, respectively.
    """

    # The implementation of this class is unfortunately relatively complex:

    # 1. NeXus requires "nonlocal" information for interpreting a field. For example,
    #    NXdata attributes define which fields are the signal, and the names of axes.
    #    A field cannot be read without this information, in particular since we want to
    #    support reading slices, using the Scipp dimension-label syntax.
    # 2. HDF5 or h5py performance is not great, and we want to avoid reading the same
    #    attrs or datasets multiple times. We can therefore not rely on "on-the-fly"
    #    interpretation of the file, but need to cache information. An earlier version
    #    of ScippNexus used such a mechanism without caching, which was very slow.

    def __init__(self, group: H5Group, definitions: dict[str, type] | None = None):
        self._group = group
        self._definitions = {} if definitions is None else definitions
        self._lazy_children = None
        self._lazy_nexus = None

    @property
    def nx_class(self) -> type | None:
        """The value of the NX_class attribute of the group.

        In case of the subclass NXroot this returns :py:class:`NXroot` even if the attr
        is not actually set. This is to support the majority of all legacy files, which
        do not have this attribute.
        """
        if (nxclass := self.attrs.get('NX_class')) is not None:
            return _nx_class_registry().get(nxclass)
        if self.name == '/':
            return NXroot

    @cached_property
    def attrs(self) -> dict[str, Any]:
        """The attributes of the group.

        Cannot be used for writing attributes, since they are cached for performance."""
        # Attrs are not read until needed, to avoid reading all attrs for all subgroups.
        # We may expected a per-subgroup overhead of 1 ms for reading attributes, so if
        # all we want is access one subgroup, we may save, e.g., a second for a group
        # with 1000 subgroups (or subfields).
        return MappingProxyType(Attrs(self._group.attrs))

    @property
    def name(self) -> str:
        return self._group.name

    @property
    def unit(self) -> sc.Unit | None:
        return self._nexus.unit

    @property
    def parent(self) -> Group:
        return Group(self._group.parent, definitions=self._definitions)

    @cached_property
    def file(self) -> Group:
        return Group(self._group.file, definitions=self._definitions)

    @property
    def _children(self) -> dict[str, Field | Group]:
        """Lazily initialized children of the group."""
        if self._lazy_children is None:
            self._lazy_children = self._read_children()
        return self._lazy_children

    def _read_children(self) -> dict[str, Field | Group]:
        def _make_child(obj: H5Dataset | H5Group) -> Field | Group:
            if is_dataset(obj):
                return Field(obj, parent=self)
            else:
                return Group(obj, definitions=self._definitions)

        items = {name: _make_child(obj) for name, obj in self._group.items()}
        # In the case of NXevent_data, the `cue_` fields are unusable, since
        # the definition is broken (the cue_index points into the
        # event_time_offset/event_id fields, instead of the
        # event_time_zero/event_index fields). In the case of NXlog they may
        # be some utility if we deal with extremely long time-series that
        # could be leveraged for label-based indexing in the future.
        items = {k: v for k, v in items.items() if not k.startswith('cue_')}
        for suffix in ('_errors', '_error'):
            field_with_errors = [name for name in items if f'{name}{suffix}' in items]
            for name in field_with_errors:
                values = items[name]
                errors = items[f'{name}{suffix}']
                if (
                    isinstance(values, Field)
                    and isinstance(errors, Field)
                    and (values.unit == errors.unit or errors.unit is None)
                    and values.dataset.shape == errors.dataset.shape
                ):
                    values.errors = errors.dataset
                    del items[f'{name}{suffix}']
        return items

    @property
    def _nexus(self) -> NXobject:
        """Instance of the NXobject subclass corresponding to the NX_class attribute.

        This is used to determine dims, unit, and other attributes of the group and its
        children, as well as defining how children will be read and assembled into the
        result object when the group is indexed.

        Lazily initialized since the NXobject subclass init can be costly.
        """
        self._populate_fields()
        return self._lazy_nexus

    def _populate_fields(self) -> None:
        """Populate the fields of the group.

        Fields are not populated until needed, to avoid reading field and group
        properties for all subgroups. However, when any field is read we must in
        general parse all the field and group properties, since for classes such
        as NXdata the properties of any field may indirectly depend on the properties
        of any other field. For example, field attributes may define which fields are
        axes, and dim labels of other fields can be defined by the names of the axes.
        """
        if self._lazy_nexus is not None:
            return
        self._lazy_nexus = self._definitions.get(self.attrs.get('NX_class'), NXobject)(
            attrs=self.attrs, children=self._children
        )

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[str]:
        return self._children.__iter__()

    def _get_children_by_nx_class(
        self, select: type | list[type]
    ) -> dict[str, NXobject | Field]:
        children = {}
        select = tuple(select) if isinstance(select, list) else select
        for key, child in self._children.items():
            nx_class = Field if isinstance(child, Field) else child.nx_class
            if nx_class is not None and issubclass(nx_class, select):
                children[key] = self[key]
        return children

    @overload
    def __getitem__(self, sel: str) -> Group | Field: ...

    @overload
    def __getitem__(
        self, sel: ScippIndex
    ) -> sc.DataArray | sc.DataGroup | sc.Dataset: ...

    @overload
    def __getitem__(self, sel: type | list[type]) -> dict[str, NXobject]: ...

    def __getitem__(self, sel):
        """
        Get a child group or child dataset, a selection of child groups, or load and
        return the current group.

        Three cases are supported:

        - String name: The child group or child dataset of that name is returned.
        - Class such as ``NXdata`` or ``NXlog``: A dict containing all direct children
          with a matching ``NX_class`` attribute are returned. Also accepts a tuple of
          classes. ``Field`` selects all child fields, i.e., all datasets but not
          groups.
        - Scipp-style index: Load the specified slice of the current group, returning
          a :class:`scipp.DataArray` or :class:`scipp.DataGroup`.

        Parameters
        ----------
        sel:
            Child name, class, or index.

        Returns
        -------
        :
            Field, group, dict of fields, or loaded data.
        """
        if isinstance(sel, str):
            # We cannot get the child directly from the HDF5 group, since we need to
            # create the parent group, to ensure that fields get the correct properties
            # such as sizes and dtype.
            if '/' in sel:
                sel_path = PurePosixPath(sel)
                if sel_path.is_absolute():
                    return self.file[sel_path.relative_to('/').as_posix()]
                # If the path is a single name, we can directly access the child
                elif len(sel_path.parts) == 1:
                    return self[sel_path.as_posix()]
                else:
                    grp = sel_path.parts[0]
                    return self[grp][sel_path.relative_to(grp).as_posix()]
            elif sel == '..':
                return self.parent
            child = self._children[sel]
            if isinstance(child, Field):
                self._populate_fields()
            return child

        def isclass(x):
            return inspect.isclass(x) and issubclass(x, Field | NXobject)

        if isclass(sel) or (
            isinstance(sel, list) and len(sel) and all(isclass(x) for x in sel)
        ):
            return self._get_children_by_nx_class(sel)

        dg = self._nexus.read_children(sel)
        if not dg:
            # Bail out early to avoid fallback warnings. Everything is optional in
            # NeXus so we cannot assume that the group is invalid (in contrast to
            # likely partially incomplete groups that will fail to assemble below).
            return dg
        try:
            dg = self._nexus.assemble(dg)
        except (sc.DimensionError, NexusStructureError) as e:
            self._warn_fallback(e)
        # For a time-dependent transformation in NXtransformations, an NXlog may
        # take the place of the `value` field. In this case, we need to read the
        # properties of the NXlog group to make the actual transformation.
        from .nxtransformations import maybe_transformation, parse_depends_on_chain

        if isinstance(dg, sc.DataGroup) and 'depends_on' in dg:
            if (chain := parse_depends_on_chain(self, dg['depends_on'])) is not None:
                dg['depends_on'] = chain

        return maybe_transformation(self, value=dg)

    def _warn_fallback(self, e: Exception) -> None:
        msg = (
            f"Failed to load {self.name} as {type(self._nexus).__name__}: {e} "
            "Falling back to loading HDF5 group children as scipp.DataGroup."
        )
        warnings.warn(msg, stacklevel=2)

    def __setitem__(self, key, value):
        """Set a child group or child dataset.

        Note that due to the caching mechanisms in this class, reading the group
        or its children may not reflect the changes made by this method."""
        if hasattr(value, '__write_to_nexus_group__'):
            group = create_class(self._group, key, nx_class=value.nx_class)
            value.__write_to_nexus_group__(group)
        else:
            create_field(self._group, key, value)

    def create_field(self, key: str, value: sc.Variable) -> H5Dataset:
        """Create a child dataset with given name and value.

        Note that due to the caching mechanisms in this class, reading the group
        or its children may not reflect the changes made by this method.

        Returns
        -------
        :
            The created dataset of the values.
            If errors are written to the file, their dataset is not returned.
        """
        return create_field(self._group, key, value)

    def create_class(self, name: str, class_name: str) -> Group:
        """Create empty HDF5 group with given name and set the NX_class attribute.

        Note that due to the caching mechanisms in this class, reading the group
        or its children may not reflect the changes made by this method.

        Parameters
        ----------
        name:
            Group name.
        nx_class:
            Nexus class, can be a valid string for the NX_class attribute, or a
            subclass of NXobject, such as NXdata or NXlog.
        """
        return Group(
            create_class(self._group, name, class_name), definitions=self._definitions
        )

    @cached_property
    def sizes(self) -> dict[str, int]:
        return self._nexus.sizes

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self.sizes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.sizes.values())


def _create_field_params_numpy(data: np.ndarray):
    return data, None, {}


def _create_field_params_string(data: sc.Variable):
    return np.array(data.values, dtype=object), None, {}


def _create_field_params_datetime(data: sc.Variable):
    start = sc.epoch(unit=data.unit)
    return (data - start).values, None, {'start': str(start.value)}


def _create_field_params_number(data: sc.Variable):
    errors = sc.stddevs(data).values if data.variances is not None else None
    return data.values, errors, {}


def create_field(
    group: H5Group, name: str, data: np.ndarray | sc.Variable, **kwargs
) -> H5Dataset:
    if not isinstance(data, sc.Variable):
        values, errors, attrs = _create_field_params_numpy(data)
    elif data.dtype == sc.DType.string:
        values, errors, attrs = _create_field_params_string(data)
    elif data.dtype == sc.DType.datetime64:
        values, errors, attrs = _create_field_params_datetime(data)
    else:
        values, errors, attrs = _create_field_params_number(data)

    if isinstance(data, sc.Variable) and data.unit:
        attrs['units'] = str(data.unit)

    values_dataset = group.create_dataset(name, data=values, **kwargs)
    values_dataset.attrs.update(attrs)
    if errors is not None:
        errors_dataset = group.create_dataset(name + '_errors', data=errors, **kwargs)
        errors_dataset.attrs.update(attrs)

    return values_dataset


def create_class(group: H5Group, name: str, nx_class: str | type) -> H5Group:
    """Create empty HDF5 group with given name and set the NX_class attribute.

    Parameters
    ----------
    name:
        Group name.
    nx_class:
        Nexus class, can be a valid string for the NX_class attribute, or a
        subclass of NXobject, such as NXdata or NXlog.
    """
    group = group.create_group(name)
    attr = nx_class if isinstance(nx_class, str | bytes) else nx_class.__name__
    group.attrs['NX_class'] = attr
    return group


@lru_cache
def _nx_class_registry():
    from . import nexus_classes

    return dict(inspect.getmembers(nexus_classes, inspect.isclass))


base_definitions_dict = {}


def base_definitions() -> dict[str, type]:
    """Return a dict of all base definitions.

    This is a copy of the base definitions dict, so that it can be modified without
    affecting the original.
    """
    return dict(base_definitions_dict)


class DefaultDefinitionsType:
    def __repr__(self) -> str:
        return "DefaultDefinitions"


DefaultDefinitions = DefaultDefinitionsType()
