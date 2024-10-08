"""
Tests for `kspace_filter` classes
"""

import pytest
import torch

from torchpme.lib import KSpaceFilter, KSpaceKernel, MeshInterpolator


class TestKernel:
    class DemoKernel(KSpaceKernel):
        def __init__(self, param: float):
            super().__init__()
            self.param = param

        @torch.jit.export
        def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
            return torch.exp(-k_sq / self.param)

    class NoopKernel(KSpaceKernel):
        def __init__(self):
            super().__init__()

        @torch.jit.export
        def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(k_sq)

    def test_kernel_subclassing(self):
        # check that one can define and use a kernel
        my_krn = self.DemoKernel(1.0)
        k_sq = torch.arange(0, 10, 0.01)
        my_krn.kernel_from_k_sq(k_sq)

    def test_kernel_jitting(self):
        # pytorch
        my_krn = self.DemoKernel(1.0)
        k_sq = torch.arange(0, 10, 0.01)
        filter = my_krn.kernel_from_k_sq(k_sq)

        # jitted
        jit_krn = torch.jit.script(my_krn)
        jit_filter = jit_krn.kernel_from_k_sq(k_sq)

        assert torch.allclose(filter, jit_filter)


class TestFilter:
    cell1 = torch.randn((3, 3))
    cell2 = torch.randn((3, 3))
    ns1 = torch.tensor([3, 4, 5])
    ns2 = torch.tensor([4, 2, 1])

    mykernel = TestKernel.DemoKernel(1.0)
    myfilter1 = KSpaceFilter(cell1, ns1, mykernel)
    myfilter2 = KSpaceFilter(cell2, ns2, mykernel)
    mymesh1 = MeshInterpolator(cell1, ns1, 3)
    mymesh2 = MeshInterpolator(cell2, ns2, 3)
    points = torch.tensor([[1.0, 2, 3], [0, 1, 1]])
    weights = torch.tensor([[-0.1], [0.4]])

    mykernel_noop = TestKernel.NoopKernel()
    myfilter_noop = KSpaceFilter(cell1, ns1, mykernel_noop)

    def test_meshes_consistent_size(self):
        # make sure we get conistent mesh sizes
        self.mymesh1.compute_weights(self.points)
        mesh = self.mymesh1.points_to_mesh(self.weights)
        # nb - the third value is different because of the real-valued FT
        assert mesh.shape[1:3] == self.myfilter1._kvectors.shape[:-2]

    def test_meshes_inconsistent_size(self):
        # make sure we get consistent mesh sizes
        self.mymesh1.compute_weights(self.points)
        mesh = self.mymesh1.points_to_mesh(self.weights)
        match = "The real-space mesh is inconsistent with the k-space grid."
        with pytest.raises(ValueError, match=match):
            self.myfilter2.compute(mesh)

    def test_kernel_noop(self):
        # make sure that a filter of ones recovers the initial mesh
        self.mymesh1.compute_weights(self.points)
        mesh = self.mymesh1.points_to_mesh(self.weights)
        mesh_transformed = self.myfilter_noop.compute(mesh)

        torch.allclose(mesh, mesh_transformed, atol=1e-6, rtol=0)

    def test_filter_linear(self):
        # checks that the filter (as well as the mesh interpolator) are linear
        self.mymesh1.compute_weights(self.points)
        mesh1 = self.mymesh1.points_to_mesh(self.weights)

        mesh2 = torch.exp(mesh1)

        tmesh1 = self.myfilter1.compute(mesh1)
        tmesh2 = self.myfilter1.compute(mesh2)
        tmesh12 = self.myfilter1.compute(mesh1 + 0.3 * mesh2)

        torch.allclose(tmesh12, tmesh1 + 0.3 * tmesh2)


def test_different_devices_ns_cell():
    ns = torch.tensor([2, 2, 2], device="cpu")
    cell = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device="meta")
    match = "`cell` and `ns_mesh` are on different devices, got meta and cpu"
    with pytest.raises(ValueError, match=match):
        KSpaceFilter(cell, ns, KSpaceKernel())


def test_ns_wrong_shape():
    ns = torch.tensor([2, 2])
    cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    match = "shape \\[2\\] of `ns_mesh` has to be \\(3,\\)"
    with pytest.raises(ValueError, match=match):
        KSpaceFilter(cell, ns, KSpaceKernel())


def test_cell_wrong_shape():
    ns = torch.tensor([2, 2, 2])
    cell = torch.tensor([[1.0, 0, 0], [0, 1, 0]])
    match = "cell of shape \\[2, 3\\] should be of shape \\(3, 3\\)"
    with pytest.raises(ValueError, match=match):
        KSpaceFilter(cell, ns, KSpaceKernel())


def test_fft_modes():
    ns = torch.tensor([2, 2, 2], device="cpu")
    cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], device="cpu")
    match = "Invalid option 'faster' for the `fft_norm` parameter."
    with pytest.raises(ValueError, match=match):
        KSpaceFilter(cell, ns, KSpaceKernel(), fft_norm="faster")
    match = "Invalid option 'faster' for the `ifft_norm` parameter."
    with pytest.raises(ValueError, match=match):
        KSpaceFilter(cell, ns, KSpaceKernel(), ifft_norm="faster")
