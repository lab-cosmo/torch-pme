import pytest
import torch
from torch.testing import assert_close

from torchpme.lib import generate_kvectors_for_ewald, generate_kvectors_for_mesh

# Generate random cells and mesh parameters
cells = []
ns_list = []
for _i in range(6):
    L = torch.rand((1,)) * 20 + 1.0
    cells.append(torch.randn((3, 3)) * L)
    ns_list.append(torch.randint(1, 12, size=(3,)))
kvec_generators = [generate_kvectors_for_mesh, generate_kvectors_for_ewald]


@pytest.mark.parametrize("ns", ns_list)
@pytest.mark.parametrize("cell", cells)
def test_duality_of_kvectors_mesh(cell, ns):
    """
    If a_j for j=1,2,3 are the three basis vectors of a unit cell and
    b_j the corresponding basis vectors of the reciprocal cell, the inner product
    between them needs to satisfy a_j*b_l=2pi*delta_jl.
    """
    nx, ny, nz = ns
    kvectors = generate_kvectors_for_mesh(ns=ns, cell=cell)

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
def test_duality_of_kvectors_squeezed(cell, ns):
    """
    If a_j for j=1,2,3 are the three basis vectors of a unit cell and
    b_j the corresponding basis vectors of the reciprocal cell, the inner product
    between them needs to satisfy a_j*b_l=2pi*delta_jl.
    """
    nx, ny, nz = ns
    kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell)

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
@pytest.mark.parametrize("kvec_type", ["fft", "ewald"])
def test_lenghts_of_kvectors(cell, ns, kvec_type):
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
    if kvec_type == "fft":
        kvectors = generate_kvectors_for_mesh(ns=ns, cell=cell)
        norms_all = torch.linalg.norm(kvectors, dim=3).flatten()
    elif kvec_type == "ewald":
        kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell)
        norms_all = torch.linalg.norm(kvectors, dim=1).flatten()

    assert torch.all(norms_all < norm_bound)


# Tests that errors are raised when the inputs are of the wrong shape or have
# inconsistent devices
@pytest.mark.parametrize("generate_kvectors", kvec_generators)
def test_ns_wrong_shape(generate_kvectors):
    ns = torch.tensor([2, 2])
    cell = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    match = "ns of shape \\[2\\] should be of shape \\(3, \\)"
    with pytest.raises(ValueError, match=match):
        generate_kvectors(ns, cell)


@pytest.mark.parametrize("generate_kvectors", kvec_generators)
def test_cell_wrong_shape(generate_kvectors):
    ns = torch.tensor([2, 2, 2])
    cell = torch.tensor([[1, 0, 0], [0, 1, 0]])
    match = "cell of shape \\[2, 3\\] should be of shape \\(3, 3\\)"
    with pytest.raises(ValueError, match=match):
        generate_kvectors(ns, cell)


@pytest.mark.parametrize("generate_kvectors", kvec_generators)
def test_different_devices_mesh_values_cell(generate_kvectors):
    ns = torch.tensor([2, 2, 2], device="cpu")
    cell = torch.eye(3, device="meta")  # different device
    match = "`ns` and `cell` are not on the same device, got cpu and meta"
    with pytest.raises(ValueError, match=match):
        generate_kvectors(ns, cell)
