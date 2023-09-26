from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

'''
from typing import Callable, Optional

from .tensor import Tensor
from .tensor_data import Index, Shape, Storage, Strides
'''
# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """
    #fn = njit()(fn)
    #print("\033[0;44m this is _map func\033[0m")
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        #print("\033[0;44m this is _map func\033[0m")
        if np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape):
            for i in prange(len(out)):
                #index = np.empty(len(out_))
                out[i] = fn(in_storage[i])
        else:
            for i in prange(len(out)):
                #out[i] = fn(in_storage[i])
                out_index = np.empty(len(out_shape), dtype=np.int32)
                in_index = np.empty(len(in_shape), dtype=np.int32)
                #out_index = np.zeros(MAX_DIMS, dtype=np.int32)
                #in_index = np.zeros(MAX_DIMS, dtype=np.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                #print("in_index is ", in_shape, "in_strides is ", in_strides)
                data = in_storage[index_to_position(in_index, in_strides)]
                map_data = fn(data)
                out[index_to_position(out_index, out_strides)] = map_data

        #raise NotImplementedError('Need to implement for Task 3.1')
    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """
    #fn = njit()(fn)
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # if np.array_equal(out_shape, a_shape) and np.array_equal(a_shape, b_shape):
        #    if np.array_equal(out_strides, a_strides) and np.array_equal(a_strides, b_strides):
        #        for i in prange(len(out)):
        #            index = np.empty(len(a_shape), dtype=np.int32)
        #            to_index(i, out_shape, index)
        #            idx = index_to_position(index, out_strides)
        #            out[idx] = fn(a_storage[idx], b_storage[idx])
        # else:
        for i in prange(len(out)):
            a_index = np.empty(len(a_shape), dtype=np.int32)
            b_index = np.empty(len(b_shape), dtype=np.int32)
            out_index = np.empty(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_data = a_storage[index_to_position(a_index, a_strides)]
            b_data = b_storage[index_to_position(b_index, b_strides)]
            map_data = fn(a_data, b_data)
            out[index_to_position(out_index, out_strides)] = map_data

        #raise NotImplementedError('Need to implement for Task 3.1')


    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        for oi in prange(len(out)):
            out_index = np.empty(len(out_shape), dtype=np.int32)
            a_index = np.empty(len(a_shape), dtype=np.int32)
            to_index(oi, out_shape, out_index)
            to_index(oi, out_shape, a_index)
            out_i = index_to_position(out_index, out_strides)
            a_i = index_to_position(a_index, a_strides)

            dimred = a_shape[reduce_dim]
            
            for ai in range(dimred):
                out[out_i] = fn(a_storage[a_i + ai*a_strides[reduce_dim]], out[out_i])
            
            #out[out_i] = fn(a_storage[a_i + (dimred-1)*a_strides[reduce_dim]], out[out_i])
        #raise NotImplementedError('Need to implement for Task 3.1')

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    n = len(out)
    l = len(out_shape) - 2
    for i in prange(n):
        out_index = np.empty(len(out_shape), dtype=np.int32)
        to_index(i, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)
        # a_index = np.empty(len(a_shape), dtype=np.int32)
        # b_index = np.empty(len(b_shape), dtype=np.int32)

        for k in prange(a_shape[-1]):            #a_shape[-1] is the reduced axis
            #out_index = np.empty(len(out_shape), dtype=np.int32)
            #to_index(i, out_shape, out_index)
            #out_pos = index_to_position(out_index, out_strides)
            a_index = np.empty(len(a_shape), dtype=np.int32)
            b_index = np.empty(len(b_shape), dtype=np.int32)
            for j in prange(l):
                a_index[j] = b_index[j] = out_index[j]
            a_index[-1] = k
            a_index[-2] = out_index[-2]
            b_index[-2] = k
            b_index[-1] = out_index[-1]
            broadcast_index(a_index, out_shape, a_shape, a_index)
            broadcast_index(b_index, out_shape, b_shape, b_index)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out[out_pos] = out[out_pos] + a_storage[a_pos]*b_storage[b_pos]
    
    #raise NotImplementedError('Need to implement for Task 3.2')


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
