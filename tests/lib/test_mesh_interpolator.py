"""
Tests for mesh interpolator class
"""

import pytest
import torch
from torch.testing import assert_close

from torchpme.lib import MeshInterpolator


class TestMeshInterpolatorForward:
    """
    Tests for the "points_to_mesh" function of the MeshInterpolator class
    """

    # Define parameters that are common to all tests
    order_P3M = [1, 2, 3, 4, 5]
    order_Lagrange = [2, 3, 4, 5, 6]

    @pytest.mark.parametrize(
        ("order", "method"),
        [(order, "P3M") for order in order_P3M]
        + [(order, "Lagrange") for order in order_Lagrange],
    )
    @pytest.mark.parametrize("n_mesh", torch.arange(19, 26))
    def test_charge_conservation_cubic(self, order, method, n_mesh):
        """
        Test that the total "charge" on the grid after the atomic_smearing the particles
        onto the mesh is conserved for a cubic cell.
        """
        # Define some basic parameteres for this test
        # While we could also loop over these, we are instead varying these
        # parameters across the various tests to reduce the number of calls
        n_particles = 8
        n_channels = 5
        L = torch.tensor(6.28318530717)  # tau

        # Generate inputs for interpolator class
        cell = torch.eye(3) * L
        positions = torch.rand((n_particles, 3)) * L
        particle_weights = 3 * torch.randn((n_particles, n_channels))
        ns_mesh = torch.tensor([n_mesh, n_mesh, n_mesh])

        # Run interpolation
        interpolator = MeshInterpolator(
            cell=cell, ns_mesh=ns_mesh, order=order, method=method
        )
        interpolator.compute_weights(positions)
        mesh_values = interpolator.points_to_mesh(particle_weights)

        # Compare total "weight (charge)" on the mesh with the sum of the particle
        # contributions
        total_weight_target = torch.sum(particle_weights, axis=0)
        total_weight = torch.sum(mesh_values, dim=(1, 2, 3))
        assert_close(total_weight, total_weight_target, rtol=3e-6, atol=3e-6)

    @pytest.mark.parametrize(
        ("order", "method"),
        [(order, "P3M") for order in order_P3M]
        + [(order, "Lagrange") for order in order_Lagrange],
    )
    def test_charge_conservation_general(self, order, method):
        """
        Test that the total "charge" on the grid after the atomic_smearing the particles
        onto the mesh is conserved for a generic triclinic cell.
        It is basically the same test as the previous one, but without the restriction
        to cubic cells.
        """
        # Define some basic parameteres for this test
        # While we could also loop over these, we are instead varying these
        # parameters across the various tests to reduce the number of calls
        n_particles = 11
        n_channels = 2
        L = torch.tensor(2.718281828)  # e

        # Generate inputs for interpolator class
        cell = torch.randn((3, 3)) * L
        positions = torch.rand((n_particles, 3)) * L
        particle_weights = 3 * torch.randn((n_particles, n_channels))
        ns_mesh = torch.randint(11, 18, size=(3,))

        # Run interpolation
        inetrpolator = MeshInterpolator(
            cell=cell, ns_mesh=ns_mesh, order=order, method=method
        )
        inetrpolator.compute_weights(positions)
        mesh_values = inetrpolator.points_to_mesh(particle_weights)

        # Compare total "weight (charge)" on the mesh with the sum of the particle
        # contributions
        total_weight_target = torch.sum(particle_weights, axis=0)
        total_weight = torch.sum(mesh_values, dim=(1, 2, 3))
        assert_close(total_weight, total_weight_target, rtol=3e-6, atol=3e-6)

    # Since the results of the next test fail if two randomly placed atoms are
    # too close to one another to share the identical nearest mesh point,
    # we fix the seed of the random number generator
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("n_mesh", torch.arange(7, 13))
    def test_exact_agreement(self, order, n_mesh):
        """
        Test that for interpolation order = 1, 2, if atoms start exactly on the mesh,
        their total mass matches the exact value.
        """
        torch.random.manual_seed(8794329)
        # Define some basic parameteres for this test
        # While we could also loop over these, we are instead varying these
        # parameters across the various tests to reduce the number of calls
        n_particles = 10
        n_channels = 3
        L = torch.tensor(0.28209478)  # 1/sqrt(4pi)

        # Define all relevant quantities using random numbers
        # The implementation also works if the particle positions
        # are not contained within the unit cell
        cell = torch.randn((3, 3)) * L
        indices = torch.randint(low=0, high=n_mesh, size=(3, n_particles))
        positions = torch.matmul(cell.T, indices / n_mesh).T
        particle_weights = 3 * torch.randn((n_particles, n_channels))
        ns_mesh = torch.tensor([n_mesh, n_mesh, n_mesh])

        # Perform interpolation
        inetrpolator = MeshInterpolator(cell=cell, ns_mesh=ns_mesh, order=order)
        inetrpolator.compute_weights(positions)
        mesh_values = inetrpolator.points_to_mesh(particle_weights)

        # Recover the interpolated values at the atomic positions
        indices_x = indices[0]
        indices_y = indices[1]
        indices_z = indices[2]
        recovered_weights = mesh_values[:, indices_x, indices_y, indices_z].T

        # !!! WARNING for debugging !!!
        # If two particles are so close to one another that
        # their closest mesh point is the same, this test will fail since these
        # two particles will essentially get merged into a single particle.
        # With the current seed of the random number generator, however,
        # this should not be an issue.
        assert_close(particle_weights, recovered_weights, rtol=4e-5, atol=1e-6)


class TestMeshInterpolatorBackward:
    """
    Tests for the "mesh_to_points" function of the MeshInterpolator class
    """

    # Define parameters that are common to all tests
    orders = [1, 2, 3, 4, 5]
    random_runs = torch.arange(10)

    torch.random.manual_seed(3482389)

    @pytest.mark.parametrize("random_runs", random_runs)
    def test_exact_invertibility_for_order_one(self, random_runs):
        """
        For interpolation order = 1, interpolating forwards and backwards with no
        changes should recover the original values.
        """
        # Define some basic parameteres for this test
        # While we could also loop over these, we are instead varying these
        # parameters across the various tests to reduce the number of calls
        n_particles = 7
        n_channels = 4
        L = torch.tensor(2.5066282)  # sqrt(tau)

        # Define all relevant quantities using random numbers
        # The implementation also works if the particle positions
        # are not contained within the unit cell
        cell = torch.randn((3, 3)) * L
        positions = torch.rand((n_particles, 3)) * L
        particle_weights = 3 * torch.randn((n_particles, n_channels))
        ns_mesh = torch.randint(17, 25, size=(3,))

        # Smear particles onto mesh and interpolate back onto
        # their own positions.
        interpolator = MeshInterpolator(cell=cell, ns_mesh=ns_mesh, order=1)
        interpolator.compute_weights(positions)
        mesh_values = interpolator.points_to_mesh(particle_weights)
        interpolated_values = interpolator.mesh_to_points(mesh_values)

        # !!! WARNING for debugging !!!
        # If two particles are so close to one another that
        # their closest mesh point is the same, this test will fail since these
        # two particles will essentially get merged into a single particle.
        # With the current seed of the random number generator, however,
        # this should not be an issue.
        assert_close(particle_weights, interpolated_values, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("n_mesh", torch.arange(18, 31))
    def test_exact_invertibility_for_order_two(self, n_mesh):
        """
        Test for interpolation order = 2
        """
        torch.random.manual_seed(3351285)
        # Define some basic parameteres for this test
        # While we could also loop over these, we are instead varying these
        # parameters across the various tests to reduce the number of calls
        n_particles = 5
        n_channels = 1
        L = torch.tensor(1.4142135)  # sqrt(2)

        # Define all relevant quantities using random numbers
        # The implementation also works if the particle positions
        # are not contained within the unit cell
        cell = torch.randn((3, 3)) * L
        indices = torch.randint(low=0, high=n_mesh, size=(3, n_particles))
        positions = torch.matmul(cell.T, indices / n_mesh).T
        particle_weights = 10 * torch.randn((n_particles, n_channels))
        ns_mesh = torch.tensor([n_mesh, n_mesh, n_mesh])

        # Smear particles onto mesh and interpolate back onto
        # their own positions.
        interpolator = MeshInterpolator(cell=cell, ns_mesh=ns_mesh, order=2)
        interpolator.compute_weights(positions)
        mesh_values = interpolator.points_to_mesh(particle_weights)
        interpolated_values = interpolator.mesh_to_points(mesh_values)

        # !!! WARNING for debugging !!!
        # If two particles are so close to one another that
        # their closest mesh point is the same, this test will fail since these
        # two particles will essentially get merged into a single particle.
        # With the current seed of the random number generator, however,
        # this should not be an issue.
        assert_close(particle_weights, interpolated_values, rtol=3e-4, atol=1e-6)

    @pytest.mark.parametrize("random_runs", random_runs)
    @pytest.mark.parametrize("order", orders)
    def test_total_mass(self, order, random_runs):
        """
        interpolate on all mesh points: should yield same total mass
        """
        # Define some basic parameteres for this test
        # While we could also loop over these, we are instead varying these
        # parameters across the various tests to reduce the number of calls
        n_channels = 3
        L = torch.tensor(1.7320508)  # sqrt(3)

        # Define random cell and its three basis vectors
        # The reshaping is to make more efficient use of
        # broadcasting
        cell = torch.randn((3, 3)) * L
        ax = cell[0].reshape((3, 1))
        ay = cell[1].reshape((3, 1))
        az = cell[2].reshape((3, 1))

        # Generate the vector positions of
        ns_mesh = torch.randint(11, 27, size=(3,))
        nx, ny, nz = ns_mesh
        nxs_1d = torch.arange(nx) / nx
        nys_1d = torch.arange(ny) / ny
        nzs_1d = torch.arange(nz) / nz
        nxs, nys, nzs = torch.meshgrid(nxs_1d, nys_1d, nzs_1d, indexing="ij")
        nxs = torch.flatten(nxs)
        nys = torch.flatten(nys)
        nzs = torch.flatten(nzs)
        positions = (ax * nxs + ay * nys + az * nzs).T

        # Generate mesh with random values and interpolate
        interpolator = MeshInterpolator(cell=cell, ns_mesh=ns_mesh, order=order)
        interpolator.compute_weights(positions)
        mesh_values = torch.randn(size=(n_channels, nx, ny, nz)) * 3.0 + 9.3
        interpolated_values = interpolator.mesh_to_points(mesh_values)

        # Sum and test
        weight_before = torch.sum(mesh_values, dim=(1, 2, 3))
        weight_after = torch.sum(interpolated_values, dim=0)
        torch.testing.assert_close(weight_before, weight_after, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("random_runs", random_runs)
    @pytest.mark.parametrize("order", [2, 3])
    def test_derivatives(self, order, random_runs):
        """
        check that derivatives on charges are all ones, and derivatives
        on cell and positions are zero (should be the case if the interpolation
        fulfills the sum rules)
        """
        # Define some basic parameteres for this test
        # While we could also loop over these, we are instead varying these
        # parameters across the various tests to reduce the number of calls
        n_channels = 3
        n_points = 12
        L = torch.tensor(1.7320508)

        # Generate random cell, positions and weights
        cell = torch.randn((3, 3)) * L
        ns_mesh = torch.randint(3, 5, size=(3,))
        positions = torch.matmul(torch.rand((n_points, 3)) * L, cell).detach()
        weights = torch.randn((n_points, n_channels))

        # Requires derivatives
        cell.requires_grad_(True)
        weights.requires_grad_(True)
        positions.requires_grad_(True)

        interpolator = MeshInterpolator(cell=cell, ns_mesh=ns_mesh, order=order)

        interpolator.compute_weights(positions)
        mesh_values = interpolator.points_to_mesh(weights)
        total_mass = mesh_values.sum()

        # Computes derivatives by backpropagation
        total_mass.backward()

        torch.testing.assert_close(
            cell.grad, torch.zeros_like(cell.grad), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            positions.grad, torch.zeros_like(positions.grad), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            weights.grad, torch.ones_like(weights.grad), rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("method", ["P3M", "Lagrange"])
def test_cell_wrong_shape(method):
    ns_mesh = torch.tensor([2, 2, 2])
    cell = torch.randn(size=(2, 3))  # incorrect shape
    order = 3
    match = "cell of shape \\[2, 3\\] should be of shape \\(3, 3\\)"

    with pytest.raises(ValueError, match=match):
        MeshInterpolator(cell, ns_mesh, order, method=method)


@pytest.mark.parametrize("method", ["P3M", "Lagrange"])
def test_ns_mesh_wrong_shape(method):
    cell = torch.eye(3)
    ns_mesh = torch.tensor([2, 2])  # incorrect shape
    order = 3
    match = "shape \\[2\\] of `ns_mesh` has to be \\(3,\\)"

    with pytest.raises(ValueError, match=match):
        MeshInterpolator(cell, ns_mesh, order, method=method)


def test_order_not_allowed():
    cell = torch.eye(3)
    ns_mesh = torch.tensor([2, 2, 2])
    order = 6  # not allowed
    match = "Only `order` from 1 to 5 are allowed"

    with pytest.raises(ValueError, match=match):
        MeshInterpolator(cell, ns_mesh, order)


def test_order_not_allowed_private():
    cell = torch.eye(3)
    ns_mesh = torch.tensor([2, 2, 2])

    interpolator = MeshInterpolator(cell, ns_mesh, order=5)
    interpolator.order = 6  # not allowed
    match = "Only `order` from 1 to 5 are allowed"

    with pytest.raises(ValueError, match=match):
        interpolator._compute_1d_weights(torch.tensor([0]))


@pytest.mark.parametrize("method", ["P3M", "Lagrange"])
def test_different_devices_cell_ns_mesh(method):
    cell = torch.eye(3, device="cpu")
    ns_mesh = torch.tensor([2, 2, 2], device="meta")  # different device
    order = 3
    match = "`cell` and `ns_mesh` are on different devices, got cpu and meta"

    with pytest.raises(ValueError, match=match):
        MeshInterpolator(cell, ns_mesh, order, method=method)


@pytest.fixture
def P3M_mesh_interpolator():
    cell = torch.eye(3)
    ns_mesh = torch.tensor([2, 2, 2])
    order = 3
    return MeshInterpolator(cell, ns_mesh, order, method="P3M")


@pytest.fixture
def Lagrange_mesh_interpolator():
    cell = torch.eye(3)
    ns_mesh = torch.tensor([2, 2, 2])
    order = 3
    return MeshInterpolator(cell, ns_mesh, order, method="Lagrange")


@pytest.mark.parametrize("method", ["P3M", "Lagrange"])
def test_mexh_xyz_edge(method):
    cell = torch.normal(mean=1, std=1, size=(3, 3))
    mesh_interpolator = MeshInterpolator(
        cell, torch.tensor([2, 2, 2]), 3, method=method
    )
    xyz = mesh_interpolator.get_mesh_xyz()

    torch.testing.assert_close(xyz[1, 1, 1], cell.sum(axis=0) / 2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "mesh_interpolator", ["P3M_mesh_interpolator", "Lagrange_mesh_interpolator"]
)
def test_mexh_xyz_shape(mesh_interpolator, request):
    mesh_interpolator = request.getfixturevalue(mesh_interpolator)
    xyz = mesh_interpolator.get_mesh_xyz()

    assert xyz.shape == (2, 2, 2, 3)


@pytest.mark.parametrize(
    "mesh_interpolator", ["P3M_mesh_interpolator", "Lagrange_mesh_interpolator"]
)
def test_positions_wrong_device(mesh_interpolator, request):
    mesh_interpolator = request.getfixturevalue(mesh_interpolator)
    positions = torch.randn(size=(10, 3), device="meta")  # different device
    match = "`positions` device meta is not the same as instance device cpu"

    with pytest.raises(ValueError, match=match):
        mesh_interpolator.compute_weights(positions)


@pytest.mark.parametrize(
    "mesh_interpolator", ["P3M_mesh_interpolator", "Lagrange_mesh_interpolator"]
)
def test_positions_wrong_shape(mesh_interpolator, request):
    mesh_interpolator = request.getfixturevalue(mesh_interpolator)
    positions = torch.randn(size=(10, 2))  # incorrect shape
    match = "shape \\[10, 2\\] of `positions` has to be \\(N, 3\\)"

    with pytest.raises(ValueError, match=match):
        mesh_interpolator.compute_weights(positions)


@pytest.mark.parametrize(
    "mesh_interpolator", ["P3M_mesh_interpolator", "Lagrange_mesh_interpolator"]
)
def test_particle_weights_wrong_device(mesh_interpolator, request):
    mesh_interpolator = request.getfixturevalue(mesh_interpolator)
    particle_weights = torch.randn(size=(10, 1), device="meta")  # different device
    match = "`particle_weights` device meta is not the same as instance device cpu"

    with pytest.raises(ValueError, match=match):
        mesh_interpolator.points_to_mesh(particle_weights)


@pytest.mark.parametrize(
    "mesh_interpolator", ["P3M_mesh_interpolator", "Lagrange_mesh_interpolator"]
)
def test_particle_weights_wrong_dim(mesh_interpolator, request):
    mesh_interpolator = request.getfixturevalue(mesh_interpolator)
    particle_weights = torch.randn(size=(10,))  # missing one dimension
    match = "`particle_weights` of dimension 1 has to be of dimension 2"

    with pytest.raises(ValueError, match=match):
        mesh_interpolator.points_to_mesh(particle_weights)


@pytest.mark.parametrize(
    "mesh_interpolator", ["P3M_mesh_interpolator", "Lagrange_mesh_interpolator"]
)
def test_mesh_to_points_wrong_dim(mesh_interpolator, request):
    mesh_interpolator = request.getfixturevalue(mesh_interpolator)
    mesh_vals = torch.randn(size=(10,))  # missing one dimension
    match = "`mesh_vals` of dimension 1 has to be of dimension 4"

    with pytest.raises(ValueError, match=match):
        mesh_interpolator.mesh_to_points(mesh_vals)
