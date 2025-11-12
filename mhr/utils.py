# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

    result = torch.stack(
        [
            cy * cz,
            -cx * sz + sx * sy * cz,
            sx * sz + cx * sy * cz,
            cy * sz,
            cx * cz + sx * sy * sz,
            -sx * cz + cx * sy * sz,
            -sy,
            sx * cy,
            cx * cy,
        ],
        dim=-1,
    ).reshape(list(r.shape[:-1]) + [3, 3])

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

        # Added the following line to cope with torch.jit.trace as a
        # temporary solution to build torchscript model.
        self.register_buffer(
            "dense_weight",
            torch.zeros(self.sparse_shape[0], self.sparse_shape[1]),
            persistent=False,
        )

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
        # We commented out the following lines because it conflicts with
        # torch.jit.trace. Currently we can't use torch.jit.script because
        # the current pymomentum does not support it. The use of sparse
        # matrix does save memory and computation. We use the dense weight
        # as a compromised solution.

        # curr_weight = torch.sparse_coo_tensor(
        #     self.sparse_indices, self.sparse_weight, self.sparse_shape
        # )
        # if self.bias is None:
        #     return (curr_weight @ x.T).T
        # else:
        #     return (curr_weight @ x.T).T + self.bias

        # Set elements in self.dense_weight at sparse_indices to sparse_weight
        self.dense_weight.zero_()  # Clear previous values
        self.dense_weight[self.sparse_indices[0], self.sparse_indices[1]] = (
            self.sparse_weight
        )
        if self.bias is None:
            return (self.dense_weight @ x.T).T
            # return torch.sparse.mm(curr_weight, x.T).T
        else:
            return (self.dense_weight @ x.T).T + self.bias
            # return torch.sparse.mm(curr_weight, x.T).T + self.bias

    def __repr__(self):
        return f"SparseLinear(in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias is not None})"
