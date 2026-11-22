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

import unittest
import warnings

import torch

from mhr.utils import SparseLinear


def _make_layer(bias: bool = False, frozen: bool = True) -> SparseLinear:
    torch.manual_seed(0)
    mask = torch.rand(8, 6) < 0.5
    layer = SparseLinear(6, 8, mask, bias=bias)
    if frozen:
        # mhr.io loads the pose correctives network with frozen weights.
        for parameter in layer.parameters():
            parameter.requires_grad = False
    return layer


def _dense_reference(layer: SparseLinear) -> torch.Tensor:
    weight = torch.zeros(layer.sparse_shape, dtype=layer.sparse_weight.dtype)
    weight[layer.sparse_indices[0], layer.sparse_indices[1]] = layer.sparse_weight
    return weight


class TestSparseLinear(unittest.TestCase):
    def test_matches_dense_matmul(self):
        for bias in (False, True):
            layer = _make_layer(bias=bias)
            x = torch.randn(3, 6)
            expected = x @ _dense_reference(layer).T
            if bias:
                expected = expected + layer.bias
            torch.testing.assert_close(layer(x), expected)

    def test_two_forwards_then_one_backward(self):
        """Two calls summed into one loss must backpropagate through both."""

        for frozen in (True, False):
            layer = _make_layer(frozen=frozen)
            x1 = torch.randn(2, 6, requires_grad=True)
            x2 = torch.randn(2, 6, requires_grad=True)
            loss = layer(x1).square().sum() + layer(x2).square().sum()
            loss.backward()

            weight = _dense_reference(layer).detach()
            g1, g2 = x1.grad, x2.grad
            x1, x2 = x1.detach(), x2.detach()
            torch.testing.assert_close(g1, 2 * (x1 @ weight.T) @ weight)
            torch.testing.assert_close(g2, 2 * (x2 @ weight.T) @ weight)
            if not frozen:
                # d(loss)/d(weight), summed over both calls, at the nonzeros.
                dense_grad = 2 * (x1 @ weight.T).T @ x1 + 2 * (x2 @ weight.T).T @ x2
                rows, cols = layer.sparse_indices
                torch.testing.assert_close(
                    layer.sparse_weight.grad, dense_grad[rows, cols]
                )

    def test_gradcheck(self):
        layer = _make_layer(bias=True, frozen=False).double()
        x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(layer, (x,)))
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda w: torch.func.functional_call(
                    layer, {"sparse_weight": w}, (x.detach(),)
                ),
                (layer.sparse_weight.detach().clone().requires_grad_(),),
            )
        )

    def test_jit_trace(self):
        layer = _make_layer(bias=True)
        x = torch.randn(4, 6)
        with warnings.catch_warnings():
            # Recent PyTorch versions mark torch.jit.trace as deprecated.
            warnings.simplefilter("ignore", FutureWarning)
            traced = torch.jit.trace(layer, (x,))
        y = torch.randn(5, 6)
        torch.testing.assert_close(traced(y), layer(y))


if __name__ == "__main__":
    unittest.main()
