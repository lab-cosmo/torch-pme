"""
Tests for mesh interpolator class
"""

import pytest
import torch
from torch.testing import assert_close

from meshlode.lib import MeshInterpolator


class TestMeshInterpolatorForward:
    """
    Tests for the "points_to_mesh" function of the MeshInterpolator class
    """

    # Define parameters that are common to all tests
    interpolation_order = [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("interpolation_order", interpolation_order)
    @pytest.mark.parametrize("n_mesh", torch.arange(19, 26))
    def test_charge_conservation_cubic(self, interpolation_order, n_mesh):
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
        MI = MeshInterpolator(
            cell=cell, ns_mesh=ns_mesh, interpolation_order=interpolation_order
        )
        MI.compute_interpolation_weights(positions)
        mesh_values = MI.points_to_mesh(particle_weights)

        # Compare total "weight (charge)" on the mesh with the sum of the particle
        # contributions
        total_weight_target = torch.sum(particle_weights, axis=0)
        total_weight = torch.sum(mesh_values, dim=(1, 2, 3))
        assert_close(total_weight, total_weight_target, rtol=3e-6, atol=3e-6)

    @pytest.mark.parametrize("interpolation_order", interpolation_order)
    def test_charge_conservation_general(self, interpolation_order):
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
        MI = MeshInterpolator(
            cell=cell, ns_mesh=ns_mesh, interpolation_order=interpolation_order
        )
        MI.compute_interpolation_weights(positions)
        mesh_values = MI.points_to_mesh(particle_weights)

        # Compare total "weight (charge)" on the mesh with the sum of the particle
        # contributions
        total_weight_target = torch.sum(particle_weights, axis=0)
        total_weight = torch.sum(mesh_values, dim=(1, 2, 3))
        assert_close(total_weight, total_weight_target, rtol=3e-6, atol=3e-6)

    # Since the results of the next test fail if two randomly placed atoms are
    # too close to one another to share the identical nearest mesh point,
    # we fix the seed of the random number generator
    @pytest.mark.parametrize("interpolation_order", [1, 2])
    @pytest.mark.parametrize("n_mesh", torch.arange(7, 13))
    def test_exact_agreement(self, interpolation_order, n_mesh):
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
        MI = MeshInterpolator(
            cell=cell, ns_mesh=ns_mesh, interpolation_order=interpolation_order
        )
        MI.compute_interpolation_weights(positions)
        mesh_values = MI.points_to_mesh(particle_weights)

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
    interpolation_orders = [1, 2, 3, 4, 5]
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
        MI = MeshInterpolator(cell=cell, ns_mesh=ns_mesh, interpolation_order=1)
        MI.compute_interpolation_weights(positions)
        mesh_values = MI.points_to_mesh(particle_weights)
        interpolated_values = MI.mesh_to_points(mesh_values)

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
        MI = MeshInterpolator(cell=cell, ns_mesh=ns_mesh, interpolation_order=2)
        MI.compute_interpolation_weights(positions)
        mesh_values = MI.points_to_mesh(particle_weights)
        interpolated_values = MI.mesh_to_points(mesh_values)

        # !!! WARNING for debugging !!!
        # If two particles are so close to one another that
        # their closest mesh point is the same, this test will fail since these
        # two particles will essentially get merged into a single particle.
        # With the current seed of the random number generator, however,
        # this should not be an issue.
        assert_close(particle_weights, interpolated_values, rtol=3e-4, atol=1e-6)

    @pytest.mark.parametrize("random_runs", random_runs)
    @pytest.mark.parametrize("interpolation_order", interpolation_orders)
    def test_total_mass(self, interpolation_order, random_runs):
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
        MI = MeshInterpolator(
            cell=cell, ns_mesh=ns_mesh, interpolation_order=interpolation_order
        )
        MI.compute_interpolation_weights(positions)
        mesh_values = torch.randn(size=(n_channels, nx, ny, nz)) * 3.0 + 9.3
        interpolated_values = MI.mesh_to_points(mesh_values)

        # Sum and test
        weight_before = torch.sum(mesh_values, dim=(1, 2, 3))
        weight_after = torch.sum(interpolated_values, dim=0)
        torch.testing.assert_close(weight_before, weight_after, rtol=1e-5, atol=1e-6)
