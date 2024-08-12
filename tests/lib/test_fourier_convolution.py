"""
Tests for Fourier space convolution class
"""

import pytest
import torch
from torch.testing import assert_close

from torchpme.lib import FourierSpaceConvolution


class TestKvectorGeneration:
    """
    Tests for the subroutine that generates all reciprocal space vectors.
    """

    # Generate random cells and mesh parameters
    cells = []
    ns_list = []
    for _i in range(6):
        L = torch.rand((1,)) * 20 + 1.0
        cells.append(torch.randn((3, 3)) * L)
        ns_list.append(torch.randint(1, 20, size=(3,)))

    @pytest.mark.parametrize("ns", ns_list)
    @pytest.mark.parametrize("cell", cells)
    def test_duality_of_kvectors(self, cell, ns):
        """
        If a_j for j=1,2,3 are the three basis vectors of a unit cell and
        b_j the corresponding basis vectors of the reciprocal cell, the inner product
        between them needs to satisfy a_j*b_l=2pi*delta_jl.
        """
        # ns = torch.tensor([3,4,7])
        nx, ny, nz = ns
        kvectors = FourierSpaceConvolution().generate_kvectors(ns=ns, cell=cell)

        # Define frequencies with the same convention as in FFT
        # This is essentially a manual implementation of torch.fft.fftfreq
        ix_refs = torch.arange(nx)
        ix_refs[ix_refs >= (nx + 1) // 2] -= nx
        iy_refs = torch.arange(ny)
        iy_refs[iy_refs >= (ny + 1) // 2] -= ny

        for ix in range(nx):
            for iy in range(ny):
                for iz in range((nz + 1) // 2):
                    inner_prods = (
                        torch.matmul(cell, kvectors[ix, iy, iz]) / 2 / torch.pi
                    )
                    inner_prods = torch.round(inner_prods)
                    inner_prods_ref = torch.tensor([ix_refs[ix], iy_refs[iy], iz]) * 1.0
                    assert_close(inner_prods, inner_prods_ref, atol=1e-15, rtol=0.0)

    @pytest.mark.parametrize("ns", ns_list)
    @pytest.mark.parametrize("cell", cells)
    def test_lenghts_of_kvectors(self, cell, ns):
        """
        Check that the lengths of the obtained kvectors satisfy the triangle
        inequality.
        """
        # Compute an upper bound for the norms of the kvectors
        # that should be obtained
        reciprocal_cell = 2 * torch.pi * cell.inverse().T
        norms_basisvecs = torch.linalg.norm(reciprocal_cell, dim=1)
        norm_bound = torch.sum(norms_basisvecs * ns)

        # Compute the norms of all kvectors and check that they satisfy the bound
        kvectors = FourierSpaceConvolution().generate_kvectors(ns=ns, cell=cell)
        norms_all = torch.linalg.norm(kvectors, dim=3).flatten()
        assert torch.all(norms_all < norm_bound)


class TestConvolution:
    """
    Test the subroutine that performs the actual convolution in reciprocal space
    """

    # Generate random cell and mesh parameters
    cells = []
    mesh_vals_list = []
    for _i in range(6):
        L = torch.rand((1,)) * 20 + 1.0
        cells.append(torch.randn((3, 3)) * L)
        ns = torch.randint(1, 20, size=(4,))
        n_channels, nx, ny, nz = ns
        nz *= 2  # for now, last dimension needs to be even
        mesh_vals_list.append(torch.randn(size=(n_channels, nx, ny, nz)))

    @pytest.mark.parametrize("mesh_vals", mesh_vals_list)
    @pytest.mark.parametrize("cell", cells)
    def test_convolution_for_delta(self, cell, mesh_vals):
        volume = cell.det()
        _, nx, ny, nz = mesh_vals.shape
        n_fft = nx * ny * nz
        mesh_vals_new = (
            FourierSpaceConvolution().compute(
                mesh_values=mesh_vals,
                cell=cell,
                potential_exponent=0,
                atomic_smearing=0.0,
            )
            * volume
            / n_fft
        )

        assert_close(mesh_vals, mesh_vals_new, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("mesh_vals", mesh_vals_list)
    @pytest.mark.parametrize("cell", cells)
    def test_caching(self, cell, mesh_vals):
        """Test that values for a second time calling compute (when cache is used) are
        the same.
        """
        fsc = FourierSpaceConvolution()

        compute_kwargs = dict(
            mesh_values=mesh_vals, cell=cell, potential_exponent=0, atomic_smearing=0.0
        )
        calculated = fsc.compute(**compute_kwargs)
        cached = fsc.compute(**compute_kwargs)

        assert_close(cached, calculated, rtol=0, atol=0)


def test_different_devices_ns_cell():
    fsc = FourierSpaceConvolution()
    ns = torch.tensor([2, 2, 2], device="cpu")
    cell = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device="meta")
    match = "`ns` and `cell` are not on the same device, got cpu and meta."
    with pytest.raises(ValueError, match=match):
        fsc.generate_kvectors(ns, cell)


def test_ns_wrong_shape():
    fsc = FourierSpaceConvolution()
    ns = torch.tensor([2, 2])
    cell = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    match = "ns of shape \\[2\\] should be of shape \\(3, \\)"
    with pytest.raises(ValueError, match=match):
        fsc.generate_kvectors(ns, cell)


def test_cell_wrong_shape():
    fsc = FourierSpaceConvolution()
    ns = torch.tensor([2, 2, 2])
    cell = torch.tensor([[1, 0, 0], [0, 1, 0]])
    match = "cell of shape \\[2, 3\\] should be of shape \\(3, 3\\)"
    with pytest.raises(ValueError, match=match):
        fsc.generate_kvectors(ns, cell)


def test_mesh_values_wrong_dim():
    fsc = FourierSpaceConvolution()
    mesh_values = torch.randn(size=(2, 2, 2))  # Missing one dimension
    cell = torch.eye(3)
    match = "`mesh_values` needs to be a 4 dimensional tensor, got 3"
    with pytest.raises(ValueError, match=match):
        fsc.compute(mesh_values=mesh_values, cell=cell)


def test_compute_cell_wrong_shape():
    fsc = FourierSpaceConvolution()
    mesh_values = torch.randn(size=(1, 2, 2, 2))
    cell = torch.randn(size=(2, 3))  # incorrect shape
    match = "cell of shape \\[2, 3\\] should be of shape \\(3, 3\\)"
    with pytest.raises(ValueError, match=match):
        fsc.compute(mesh_values=mesh_values, cell=cell)


def test_different_devices_mesh_values_cell():
    fsc = FourierSpaceConvolution()
    mesh_values = torch.randn(size=(1, 2, 2, 2), device="cpu")
    cell = torch.eye(3, device="meta")  # different device
    match = "`mesh_values` and `cell` are on different devices, got cpu and meta"
    with pytest.raises(ValueError, match=match):
        fsc.compute(mesh_values=mesh_values, cell=cell)


def test_unsupported_potential_exponent_kernel_func():
    fsc = FourierSpaceConvolution()
    ksq = torch.tensor([1.0])
    match = "Only potential exponents 0 and 1 are supported!"
    with pytest.raises(ValueError, match=match):
        fsc.kernel_func(ksq, potential_exponent=2)


def test_unsupported_potential_exponent_value_at_origin():
    fsc = FourierSpaceConvolution()
    match = "Only potential exponents 0 and 1 are supported"
    with pytest.raises(ValueError, match=match):
        fsc.value_at_origin(potential_exponent=2)
