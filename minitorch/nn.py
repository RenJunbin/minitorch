from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    tiled_h = height // kh
    tiled_w = width // kw
    # TODO: Implement for Task 4.3.
    '''
    new_height = height // kh
    new_width = width // kw
    strides = input._tensor.strides
    #new_strides = input.view(batch, channel, new_height, new_width, kh*kw)._tensor.strides
    new_strides = (strides[0], strides[1], kh*strides[2], kw, strides[2], strides[3])
    output = Tensor.make(
                storage=input._tensor._storage,
                shape=(batch, channel, new_height, new_width, kh, kw),
                strides=new_strides,
                backend=input.backend).contiguous().view(batch, channel, new_height, new_width, kh*kw).contiguous()
    return (
        output, 
        new_height, new_width
    )
    '''
    input = input.contiguous()
    input = input.view(batch, channel, tiled_h, kh, tiled_w, kw)
    input = input.permute(0, 1, 2, 4, 3, 5)
    input = input.contiguous()
    input = input.view(batch, channel, tiled_h, tiled_w, kw*kh)

    return input, tiled_h, tiled_w
    #raise NotImplementedError('Need to implement for Task 4.3')


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    kh, kw = kernel
    tiled_input, tiled_height, tiled_width = tile(input, kernel)
    '''
    outbuffer = [0 for p in range(batch*channel*tiled_height*tiled_width)]
    #print("tiled_input is", tiled_input)
    for b in range(batch):
        for c in range(channel):
            for th in range(tiled_height):
                for tw in range(tiled_width):
                    values = 0
                    for o in range(kh*kw):
                        values += tiled_input[b, c, th, tw, o]

                    outbuffer[
                        b*channel*tiled_height*tiled_width +
                        c*tiled_height*tiled_width +
                        th*tiled_width +
                        tw  
                    ] = values / (kh*kw)
    return Tensor.make(
        storage=outbuffer,
        shape=(batch,channel, tiled_height, tiled_width),
        backend=input.backend
    )
    '''
    input, tiled_h, tiled_w = tile(input, kernel)
    input = input.mean(4)
    input = input.view(batch, channel, tiled_h, tiled_w)
    return input
    #raise NotImplementedError('Need to implement for Task 4.3')


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        ctx.save_for_backward(input, dim)
        dim = int(dim[0])
        return max_reduce(input, dim)
        #raise NotImplementedError('Need to implement for Task 4.4')

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        input, dim = ctx.saved_values
        dim = int(dim[0])
        return (grad_output*argmax(input, dim), 1.0)
        #raise NotImplementedError('Need to implement for Task 4.4')


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))

class SoftMax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        # TODO: Implement for Task 4.4.
        #ctx.save_for_backward(input, dim)
        dim = int(dim[0])
        m = max(input, dim)
        e_input = input.f.exp_map(input - m)
        sum_along_axis = e_input.sum(dim)
        output = e_input / sum_along_axis
        ctx.save_for_backward(input, dim, output)
        return output
        #raise NotImplementedError('Need to implement for Task 4.4')

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        # TODO: Implement for Task 4.4.
        input, dim, output = ctx.saved_values
        return (output * (grad_output - (grad_output*output).sum(dim)), 0.0)
        #return (output * (grad_output - grad_output*output), 0.0)
        #return (grad_output*argmax(input, dim), 1.0)
        #raise NotImplementedError('Need to implement for Task 4.4')

class LogSoftMax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        # TODO: Implement for Task 4.4.
        #ctx.save_for_backward(input, dim)
        m = max(input, dim)
        e_input = input.f.exp_map(input - m)
        sum = e_input.sum(dim)
        #output = input - sum.log() - m
        output = input - sum.log() - m
        ctx.save_for_backward(input, dim, output)
        return output
        #return e_input / sum_along_axis
        #raise NotImplementedError('Need to implement for Task 4.4')

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        # TODO: Implement for Task 4.4.
        input, dim, _ = ctx.saved_values
        dim = int(dim[0])
        #output = softmax(input, dim)
        #grad_output.log()
        #dim = int(dim[0])
        #ones = grad_output.ones(input.shape)
        #o = ones - input.exp()/input.exp().sum(dim)
        #output = output.exp()
        m = max(input, dim)
        e_input = input.f.exp_map(input - m)
        sum_along_axis = e_input.sum(dim)
        output = e_input / sum_along_axis
        #ctx.save_for_backward(input, dim, output)
        #grad = grad_output - grad_output*output.sum(dim) - grad_output*argmax(input, dim)
        grad = grad_output - output*grad_output.sum(dim)
        return grad, 0.0
        #raise NotImplementedError('Need to implement for Task 4.4')

def softmax(input: Tensor, dim: int) -> Tensor:
    #raise Exception("this is an error")
    return SoftMax.apply(input, input._ensure_tensor(dim))

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    return LogSoftMax.apply(input, input._ensure_tensor(dim))
# def softmax(input: Tensor, dim: int) -> Tensor:
#     r"""
#     Compute the softmax as a tensor.


#    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

#     $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

#     Args:
#         input : input tensor
#         dim : dimension to apply softmax

#     Returns:
#         softmax tensor
#     """
#     # TODO: Implement for Task 4.4.
#     '''
#     m = max(input, dim)
#     e_input = input.f.exp_map(input - m)
#     #e_input = input.exp()
#     sum_along_axis = e_input.sum(dim)
#     #input.sum
#     output = e_input / sum_along_axis
#     '''
#     #return input.f.exp_map(input - max(input, dim)) / input.f.exp_map(input - max(input, dim)).sum(dim)
#     input = input.exp() - max(input, dim)
#     sum_along_axis = input.sum(dim)
#     return input / sum_along_axis
#     #raise NotImplementedError('Need to implement for Task 4.4')


# def logsoftmax(input: Tensor, dim: int) -> Tensor:
#     r"""
#     Compute the log of the softmax as a tensor.

#     $z_i = x_i - \log \sum_i e^{x_i}$

#     See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

#     Args:
#         input : input tensor
#         dim : dimension to apply log-softmax

#     Returns:
#          log of softmax tensor
#     """
#     # TODO: Implement for Task 4.4.
#     '''
#     m = max(input, dim)
#     e_input = input.f.exp_map(input - m)
#     sum = e_input.sum(dim)
#     output = input - sum.log() - m
#     return output
#     '''
#     return softmax(input, dim).log()
#     #raise NotImplementedError('Need to implement for Task 4.4')


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    tiled_input, tiled_h, tiled_w = tile(input, kernel)
    tiled_input = max(tiled_input, 4)
    tiled_input = tiled_input.view(batch, channel, tiled_h, tiled_w)
    return tiled_input
    #raise NotImplementedError('Need to implement for Task 4.4')


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if ignore:
        return input
    else:
        rand_matrix = rand(
            shape=input.shape,
            backend=input.backend
        )
        for i in range(len(input._tensor._storage)):
            if rand_matrix._tensor._storage[i] < rate:
                input._tensor._storage[i] = 0
        return input
    #raise NotImplementedError('Need to implement for Task 4.4')
