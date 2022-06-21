import torch
import torch.nn as nn
import torch.nn.functional as F

from opacus.grad_sample.utils import register_grad_sampler
from opacus.utils.tensor_utils import unfold2d
import numpy as np

class StdConv2d(nn.Conv2d):

  def forward(self, x):

    pre_normalized_w = self.weight

    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)

    std = torch.sqrt(v + 1e-10)

                                                                                                                                       
    w = (w - m) / std

    self.w = w
    self.std = std
    self.normalized_w = w.data

    return F.conv2d(x, w, bias = self.bias, stride = self.stride, padding = self.padding, dilation = self.dilation, groups = self.groups)

#mainly copied from https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/conv.py
def compute_conv_grad_sample(
    layer: StdConv2d,
    activations: torch.Tensor,
    backprops: torch.Tensor,
):
    """
    Computes per sample gradients for convolutional layers
    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    n = activations.shape[0]
    # get activations and backprops in shape depending on the Conv layer
    activations = unfold2d(
        activations,
        kernel_size=layer.kernel_size,
        padding=layer.padding,
        stride=layer.stride,
        dilation=layer.dilation,
    )

    backprops = backprops.reshape(n, -1, activations.shape[-1])
    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    grad_sample = torch.einsum("noq,npq->nop", backprops, activations)
    # rearrange the above tensor and extract diagonals.
    grad_sample = grad_sample.view(
        n,
        layer.groups,
        -1,
        layer.groups,
        int(layer.in_channels / layer.groups),
        np.prod(layer.kernel_size),
    )
    grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
    shape = [n] + list(layer.weight.shape)

    ret = {layer.weight: grad_sample.view(shape)}

    return ret, backprops.shape


@register_grad_sampler(StdConv2d)
def custom_compute_stdconv2d_grad_sample(
    layer: StdConv2d, A: torch.Tensor, B: torch.Tensor
) -> None:


    ret, backprops_shape = compute_conv_grad_sample(layer, A, B)
    unnormalized_grad = ret[layer.weight] 

    # print(unnormalized_grad.shape, layer.normalized_w.shape)
    # exit()
    grad = (1/layer.std)*(unnormalized_grad - torch.mean(unnormalized_grad*layer.normalized_w, dim=[2, 3, 4], keepdim=True)*layer.normalized_w)
    grad = grad - torch.mean(grad, dim=[2, 3, 4], keepdim=True)

    ret[layer.weight] = grad

    if layer.bias is not None:
        ret[layer.bias] = torch.sum(B.reshape(backprops_shape), dim=2)

    return ret
