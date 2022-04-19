# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
import scipp as sc

from .nxobject import NXobject

import dask.threaded
from dask.array.core import normalize_chunks
from dask.base import DaskMethodsMixin


def make_graph(nxobject):
    dsk = {}
    name = str(nxobject.name)
    dsk[f'original-{name}'] = nxobject

    def getter(obj, indices):
        return obj[indices]

    dsk[(name, 0)] = (getter, f'original-{name}', {nxobject.dims[0]: slice(0, 1)})
    dsk[(name, 1)] = (getter, f'original-{name}', {nxobject.dims[0]: slice(1, 2)})
    return dsk


class SliceWrapper:
    def __init__(self, da):
        self.da = da

    def __getitem__(self, indices):
        out = self.da
        slices = zip(self.da.dims, indices)
        for k, v in slices:
            out = out[k, v]
        return SliceWrapper(out)


class NXArrayAdapter:
    def __init__(self, nxobject):
        self._nxobject = nxobject

    @property
    def dtype(self):
        import numpy as np
        return np.dtype('float64')

    @property
    def shape(self):
        return self._nxobject.shape

    def __getitem__(self, indices):
        print(f'getitem {indices=} {dict(zip(self._nxobject.dims, indices))}')
        return SliceWrapper(self._nxobject[dict(zip(self._nxobject.dims, indices))])


class NXDaskArray(DaskMethodsMixin):
    def __init__(self, nxobject):
        adapter = NXArrayAdapter(nxobject)
        from dask.array import from_array
        self._dask_array = from_array(adapter, chunks=2, asarray=False)

    def __dask_graph__(self):
        return self._dask_array.__dask_graph__()

    def __dask_keys__(self):
        return self._dask_array.__dask_keys__()

    def __dask_postcompute__(self):
        def finalize(results, *extra_args):
            print(f'{results=} {extra_args=}')
            chunks = [sc.concat([x.da for x in inner], 'yy') for inner in results]
            return sc.concat(chunks, 'xx')

        return finalize, ()

    __dask_scheduler__ = staticmethod(dask.get)

    def __getitem__(self, select):
        print(f'{select=}')
        dim, ind = select
        from copy import copy
        selected = copy(self)
        selected._dask_array = self._dask_array[ind]
        return selected


class NXCollection:
    def __init__(self, nxobject: NXobject, chunks='auto'):
        chunks = normalize_chunks(chunks=chunks, shape=nxobject.shape)
        self._dim = nxobject.dims[0]
        self._dask = make_graph(nxobject)
        print(chunks)
        # self._dask = graph_from_arraylike(nxobject, chunks=chunks,
        # shape=nxobject.shape, name=nxobject.name)

    def __dask_graph__(self):
        return self._dask

    def __dask_keys__(self):
        return [key for key in self._dask if not isinstance(key, str)]

    def __dask_postcompute__(self):
        def finalize(results, *extra_args):
            print(f'{results=} {extra_args=}')
            return sc.concat(results, self._dim)

        return finalize, ()

    def __dask_postpersist__(self):
        print('persist')
        pass

    __dask_scheduler__ = staticmethod(dask.threaded.get)
    __dask_scheduler__ = staticmethod(dask.get)

    def __getitem__(self, select):
        pass


#    dict(  # doctest: +SKIP
#            graph_from_arraylike(arr, chunks=((2, 2), (3, 3)), shape=(4,6),
#                                 name="X", inline_array=False)
#        )
#    {"original-X": arr,
#     ('X', 0, 0): (getter, 'original-X', (slice(0, 2), slice(0, 3))),
#     ('X', 1, 0): (getter, 'original-X', (slice(2, 4), slice(0, 3))),
#     ('X', 1, 1): (getter, 'original-X', (slice(2, 4), slice(3, 6))),
#     ('X', 0, 1): (getter, 'original-X', (slice(0, 2), slice(3, 6)))}
