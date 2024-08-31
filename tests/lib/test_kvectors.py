import pytest
import torch
from torch.testing import assert_close

from torchpme.lib import Kvectors


# Generate random cells and mesh parameters
cells = []
ns_list = []
for _i in range(6):
    L = torch.rand((1,)) * 20 + 1.0
    cells.append(torch.randn((3, 3)) * L)
    ns_list.append(torch.randint(1, 12, size=(3,)))


@pytest.mark.parametrize("ns", ns_list)
@pytest.mark.parametrize("cell", cells)
def test_duality_of_kvectors_mesh(ns, cell):
    """
    If a_j for j=1,2,3 are the three basis vectors of a unit cell and
    b_j the corresponding basis vectors of the reciprocal cell, the inner product
    between them needs to satisfy a_j*b_l=2pi*delta_jl.
    """
    nx, ny, nz = ns
    kvector_generator = Kvectors()
    kvectors = kvector_generator.compute(ns=ns, cell=cell)

    # Define frequencies with the same convention as in FFT
    # This is essentially a manual implementation of torch.fft.fftfreq
    ix_refs = torch.arange(nx)
    ix_refs[ix_refs >= (nx + 1) // 2] -= nx
    iy_refs = torch.arange(ny)
    iy_refs[iy_refs >= (ny + 1) // 2] -= ny

    for ix in range(nx):
        for iy in range(ny):
            for iz in range((nz + 1) // 2):
                inner_prods = torch.matmul(cell, kvectors[ix, iy, iz]) / 2 / torch.pi
                inner_prods = torch.round(inner_prods)
                inner_prods_ref = torch.tensor([ix_refs[ix], iy_refs[iy], iz]) * 1.0
                assert_close(inner_prods, inner_prods_ref, atol=1e-15, rtol=0.0)


@pytest.mark.parametrize("ns", ns_list)
@pytest.mark.parametrize("cell", cells)
def test_duality_of_kvectors_squeezed(ns, cell):
    """
    If a_j for j=1,2,3 are the three basis vectors of a unit cell and
    b_j the corresponding basis vectors of the reciprocal cell, the inner product
    between them needs to satisfy a_j*b_l=2pi*delta_jl.
    """
    nx, ny, nz = ns
    kvector_generator = Kvectors(for_ewald=True)
    kvectors = kvector_generator.compute(ns=ns, cell=cell)

    # Define frequencies with the same convention as in FFT
    # This is essentially a manual implementation of torch.fft.fftfreq
    ix_refs = torch.arange(nx)
    ix_refs[ix_refs >= (nx + 1) // 2] -= nx
    iy_refs = torch.arange(ny)
    iy_refs[iy_refs >= (ny + 1) // 2] -= ny
    iz_refs = torch.arange(nz)
    iz_refs[iz_refs >= (nz + 1) // 2] -= nz

    i_tot = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                inner_prods = torch.matmul(cell, kvectors[i_tot]) / 2 / torch.pi
                inner_prods = torch.round(inner_prods)
                inner_prods_ref = (
                    torch.tensor([ix_refs[ix], iy_refs[iy], iz_refs[iz]]) * 1.0
                )
                assert_close(inner_prods, inner_prods_ref, atol=1e-15, rtol=0.0)
                i_tot += 1


@pytest.mark.parametrize("ns", ns_list)
@pytest.mark.parametrize("cell", cells)
@pytest.mark.parametrize("for_ewald", [True, False])
def test_lenghts_of_kvectors(ns, cell, for_ewald):
    """
    Check that the lengths of the obtained kvectors satisfy the triangle
    inequality.
    """
    # Compute an upper bound for the norms of the kvectors
    # that should be obtained
    reciprocal_cell = 2 * torch.pi * cell.inverse().T
    norms_basisvecs = torch.linalg.norm(reciprocal_cell, dim=1)
    norm_bound = torch.sum(norms_basisvecs * ns)

    kvector_generator = Kvectors(for_ewald=for_ewald)
    kvectors = kvector_generator.compute(ns=ns, cell=cell)

    if for_ewald:
        norms_all = torch.linalg.norm(kvectors, dim=1).flatten()
    else:
        norms_all = torch.linalg.norm(kvectors, dim=3).flatten()

    assert torch.all(norms_all < norm_bound)


@pytest.mark.parametrize("ns", ns_list)
@pytest.mark.parametrize("cell", cells)
@pytest.mark.parametrize("for_ewald", [True, False])
def test_caching(ns, cell, for_ewald):
    """Test that values for a second time calling compute (when cache is used) are
    the same.
    """
    kvector_generator = Kvectors(for_ewald=for_ewald)
    calculated = kvector_generator.compute(ns=ns, cell=cell)
    cached = kvector_generator.compute(ns=ns, cell=cell)

    assert_close(cached, calculated, rtol=0, atol=0)


# Tests that errors are raised when the inputs are of the wrong shape or have
# inconsistent devices
@pytest.mark.parametrize("for_ewald", [True, False])
def test_ns_wrong_shape(for_ewald):
    ns = torch.tensor([2, 2])
    cell = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    kvector_generator = Kvectors(for_ewald=for_ewald)

    match = "ns of shape \\[2\\] should be of shape \\(3, \\)"
    with pytest.raises(ValueError, match=match):
        kvector_generator.compute(ns, cell)


@pytest.mark.parametrize("for_ewald", [True, False])
def test_cell_wrong_shape(for_ewald):
    ns = torch.tensor([2, 2, 2])
    cell = torch.tensor([[1, 0, 0], [0, 1, 0]])
    kvector_generator = Kvectors(for_ewald=for_ewald)

    match = "cell of shape \\[2, 3\\] should be of shape \\(3, 3\\)"
    with pytest.raises(ValueError, match=match):
        kvector_generator.compute(ns, cell)


@pytest.mark.parametrize("for_ewald", [True, False])
def test_different_devices_mesh_values_cell(for_ewald):
    ns = torch.tensor([2, 2, 2], device="cpu")
    cell = torch.eye(3, device="meta")  # different device
    kvector_generator = Kvectors(for_ewald=for_ewald)

    match = "`ns` and `cell` are not on the same device, got cpu and meta"
    with pytest.raises(ValueError, match=match):
        kvector_generator.compute(ns, cell)
