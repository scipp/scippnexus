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
        # dask sometimes slices the chunks to save memory, so we need to support
        # plain slicing syntax
        return SliceWrapper(self._nxobject[dict(zip(self._nxobject.dims, indices))])


def from_nxobject(nxobject, chunks='auto'):
    adapter = NXArrayAdapter(nxobject)
    from dask.array import from_array
    da = from_array(adapter, chunks=chunks, asarray=False)
    return NXDaskArray(dims=nxobject.dims, dask_array=da)


class NXDaskArray(DaskMethodsMixin):
    def __init__(self, dims, dask_array):
        self._dims = dims
        self._dask_array = dask_array

    def __dask_graph__(self):
        return self._dask_array.__dask_graph__()

    def __dask_keys__(self):
        return self._dask_array.__dask_keys__()

    def __dask_layers__(self):
        return self._dask_array.__dask_layers__()

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        from dask.optimization import cull
        dsk2, _ = cull(dsk, keys)
        return dsk2

    def __dask_postcompute__(self):
        def finalize(results, *extra_args):
            print(f'{results=} {extra_args=}')
            if len(self._dims) == 0:
                return results
            elif len(self._dims) == 1:
                chunks = [x.da for x in results]
            else:  # TODO will fail for more than 2
                chunks = [
                    sc.concat([x.da for x in inner], self._dims[1]) for inner in results
                ]
            return sc.concat(chunks, self._dims[0])

        return finalize, ()

    __dask_scheduler__ = staticmethod(dask.get)

    def __getitem__(self, select):
        print(f'{select=}')
        dim, ind = select
        dims = list(self._dims)
        if isinstance(ind, int):
            del dims[dims.index(dim)]
        return NXDaskArray(dims=dims, dask_array=self._dask_array[ind])


# graph builder?
# Can we just prepare layers and call from_collections once we have the file?
# but we do not know the number of chunks or number of attributes?

#
# name = 'add-' + tokenize(self, other)
# layer = {(name, i): (add, input_key, other)
#          for i, input_key in enumerate(self.__dask_keys__())}
# graph = HighLevelGraph.from_collections(name, layer, dependencies=[self])
# return new_collection(name, graph)


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
