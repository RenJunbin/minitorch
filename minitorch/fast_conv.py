from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides
    #input.make()
    #Tensor.make
    '''
    B: batch
    T: width
    C: input_channels
    K: k_width
    #@njit(inline="always")
    def unroll_chan(input, B, T, C, K):

        out = [[[input[b, c, t+k] if t+k < T else 0
                 for c in range(in_channels)
                 for k in range(kw)]
                 for t in range(output.shape[-1])]
                 for b in range(batch)]
        return Tensor.make(
            storage=out,
            shape=(B, T, C*K),          # shape = (batch_, width, in_channels*kw)
            backend=input.backend
        )
    new_weight_shape = np.array(out_channels_, in_channels_*kw, dtype=np.int32)     # shape = (out_channels, in_channels*kw)
    unroll_input = unroll_chan(input, batch, width, in_channels, kw)
    tensor_weight = Tensor.make(weight, new_weight_shape, weight_strides).permute(1, 0)
    
    out = unroll_input.__matmul__(tensor_weight)._tensor._storage[out_size]
    '''
    '''
    unroll_input = np.empty(shape=(batch, out_shape[-1], in_channels*kw))
    for b in prange(batch):
        for t in prange(out_shape[-1]):
            input_index = np.empty(len(input_shape))
            for k in range(kw):
                for c in range(in_channels):
                    if t + k < width:
                        unroll_input[b, t, c*kw+k] = input[index_to_position(input_index, input_strides)]
    unroll_input = Tensor.make(
        storage=unroll_input.tolist(),
        shape=unroll_input.shape
    )
    '''
    #kw = (kw, ) if reverse else (kw-1, -1, -1)
    for num in prange(out_size):
        out_index = np.empty(len(out_shape), dtype=np.int32)
        # input_index = np.empty(len(input_shape), dtype=np.int32)
        # weight_index = np.empty(len(weight_shape), dtype=np.int32)

        to_index(num, out_shape, out_index)
        # input_index[0] = out_index[0]       # batchNum is ready, inputChannel, inwidth are not ready
        # weight_index[0] = out_index[1]      # outChannel is ready, inputChannel, kw are not ready
        
        #kw = (kw, ) if reverse else (kw-1, -1, -1)
        for ic in prange(in_channels):
            input_index = np.empty(len(input_shape), dtype=np.int32)
            weight_index = np.empty(len(weight_shape), dtype=np.int32)
            for k in range(kw):
                weight_index[0] = out_index[1]
                weight_index[1] = ic
                weight_index[2] = k #if reverse==False else kw-k-1

                input_index[0] = out_index[0]
                input_index[1] = ic
                input_index[2] = k + out_index[2] if not reverse else out_index[2] - k

                if 0<=input_index[2] < width:
                    out[num] = out[num] + \
                        input[index_to_position(input_index, input_strides)] \
                        * weight[index_to_position(weight_index, weight_strides)]

    # TODO: Implement for Task 4.1.
    #raise NotImplementedError('Need to implement for Task 4.1')


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        # batch, out_channels, w
        # batch, in_channels, w
        # out_channels, in_channels2, kw
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),       # in_channels, out_channels, kw
            grad_weight.size,
            *new_input.tuple(),         # in_channels, batch, w
            *new_grad_output.tuple(),   # out_channels, in_channels, kw
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),        # batch, in_channels, w
            grad_input.size,
            *grad_output.tuple(),       
            *new_weight.tuple(),        # in_channels, out_channels, kw
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    for num in prange(out_size):
        out_index = np.empty(len(out_shape), dtype=np.int32)
        to_index(num, out_shape, out_index)     # out_channels is ready
        # out_index[0] is batch,[1] is out_channels, [2] is height, [3] is weigth
        # in_index[0] is batch,[1] is in_channels, [2] is height, [3] is weigth
        # weight_index[0] is out_channels, [1] is in_channels, [2] is kh, [3] is kw
        for ic in prange(in_channels):
            input_index = np.empty(len(input_shape), dtype=np.int32)
            weight_index = np.empty(len(input_shape), dtype=np.int32)
            for h in range(kh):
                for w in range(kw):
                    weight_index[0] = out_index[1]  #out_channels
                    weight_index[1] = ic
                    weight_index[2] = h
                    weight_index[3] = w

                    input_index[0] = out_index[0]
                    input_index[1] = ic
                    input_index[2] = h + out_index[2] if not reverse else out_index[2] - h
                    input_index[3] = w + out_index[3] if not reverse else out_index[3] - w

                    if 0<=input_index[2] < width and 0<=input_index[3] < height:
                        out[num] = out[num] + \
                            input[index_to_position(input_index, input_strides)] \
                            * weight[index_to_position(weight_index, weight_strides)]
    '''
    for num in prange(out_size):
        out_index = np.empty(len(out_shape), dtype=np.int32)
        to_index(num, out_shape, out_index)
  
        for ic in prange(in_channels):
            input_index = np.empty(len(input_shape), dtype=np.int32)
            weight_index = np.empty(len(weight_shape), dtype=np.int32)
            for k in range(kw):
                weight_index[0] = out_index[1]
                weight_index[1] = ic
                weight_index[2] = k #if reverse==False else kw-k-1

                input_index[0] = out_index[0]
                input_index[1] = ic
                input_index[2] = k + out_index[2] if not reverse else out_index[2] - k

                if input_index[2] < width:
                    out[num] = out[num] + \
                        input[index_to_position(input_index, input_strides)] \
                        * weight[index_to_position(weight_index, weight_strides)]
    '''
    
    
    #pass
    # TODO: Implement for Task 4.2.
    #raise NotImplementedError('Need to implement for Task 4.2')


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
