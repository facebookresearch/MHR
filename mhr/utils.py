# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch


def batch6DFromXYZ(r, return_9D=False) -> torch.Tensor:
    """
    Generate a matrix representing a rotation defined by a XYZ-Euler
    rotation.

    Args:
        r: ... x 3 rotation vectors

    Returns:
        ... x 6
    """
    rc = torch.cos(r)
    rs = torch.sin(r)
    cx = rc[..., 0]
    cy = rc[..., 1]
    cz = rc[..., 2]
    sx = rs[..., 0]
    sy = rs[..., 1]
    sz = rs[..., 2]

    result = torch.empty(list(r.shape[:-1]) + [3, 3], dtype=r.dtype).to(r.device)

    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cx * sz + sx * sy * cz
    result[..., 0, 2] = sx * sz + cx * sy * cz
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = -sx * cz + cx * sy * sz
    result[..., 2, 0] = -sy
    result[..., 2, 1] = sx * cy
    result[..., 2, 2] = cx * cy

    if not return_9D:
        return torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1)
    else:
        return result


class SparseLinear(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, sparse_mask, bias=True, load_with_cuda=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if not load_with_cuda:
            # Sometimes, this crashes on cpu...
            self.sparse_indices = torch.nn.Parameter(
                sparse_mask.nonzero().T, requires_grad=False
            )  # 2 x K
        else:
            self.sparse_indices = torch.nn.Parameter(
                sparse_mask.cuda().nonzero().T.cpu(), requires_grad=False
            )  # 2 x K
        self.sparse_shape = sparse_mask.shape

        weight = torch.zeros(out_channels, in_channels)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.bias = None

        # Initialize
        for out_idx in range(out_channels):
            # By default, self.weight is initialized with kaiming,
            # fan_in, linear default.
            # Here, the entire thing (even stuff that should be 0) are initialized,
            # only relevant stuff will be kept
            fan_in = sparse_mask[out_idx].sum()
            gain = torch.nn.init.calculate_gain("leaky_relu", math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            weight[out_idx].uniform_(-bound, bound)
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in)
                self.bias[out_idx : out_idx + 1].uniform_(-bound, bound)
        self.sparse_weight = torch.nn.Parameter(
            weight[self.sparse_indices[0], self.sparse_indices[1]]
        )
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias)

    def forward(self, x):
        curr_weight = torch.sparse_coo_tensor(
            self.sparse_indices, self.sparse_weight, self.sparse_shape
        )
        if self.bias is None:
            return (curr_weight @ x.T).T
        else:
            return (curr_weight @ x.T).T + self.bias

    def __repr__(self):
        return f"SparseLinear(in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias is not None})"
