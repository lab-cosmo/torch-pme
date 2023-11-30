"""
Mesh Interpolator
=================
"""

import torch


class MeshInterpolator:
    """
    Class for handling all steps related to interpolations in the context of a mesh
    based Ewald summation.

    In particular, this includes two core functionalities: 1. "forwards"
    interpolation, in which the "charges" or more general "particle weights" of
    atoms are assigned to grid points of a mesh. This is done in the
    "points_to_mesh" function. 2. "backwards" interpolation, in which values defined
    on a mesh are interpolated to arbitrary positions typically lying between mesh
    points. This is done in the "mesh_to_points" function.

    Since the computation of the interpolation weights for both of the above types
    of calculations is identical, this is performed in a separate function called
    "compute_interpolation_weights".

    :param cell: torch.tensor of shape (3,3)
        cell[i] is the i-th basis vector of the unit cell
    :param ns_mesh: list of tuple of size 3
        Number of mesh points to use along each of the three axes
    :param interpolation_order: int
        The degree of the polynomials used for interpolation. A higher order leads
        to smoother interpolation, at a computational cost that grows cubically with
        the interpolation order (once one moves to the 3D case).
    """
    def __init__(
        self, cell: torch.Tensor, ns_mesh: torch.Tensor, interpolation_order: int
    ):

        self.cell = cell
        self.ns_mesh = ns_mesh
        self.interpolation_order = interpolation_order

        # Initialize the variables in which to store the intermediate
        # interpolation nodes and weights
        self.interpolation_weights: torch.Tensor = torch.tensor(0.)
        self.x_shifts: torch.Tensor = torch.tensor(0)
        self.y_shifts: torch.Tensor = torch.tensor(0)
        self.z_shifts: torch.Tensor = torch.tensor(0)
        self.x_indices: torch.Tensor = torch.tensor(0)
        self.y_indices: torch.Tensor = torch.tensor(0)
        self.z_indices: torch.Tensor = torch.tensor(0)
        
        

        
    def compute_1d_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate the smooth interpolation weights used to smear the particles onto a
        mesh.

        The details of the method are described in
        J. Chem. Phys. 109, 7678–7693 (1998)
        https://doi.org/10.1063/1.477414

        :param x: torch.tensor of shape (n,)
            Set of relative positions in the interval [-1/2, 1/2].

        :returns: torch.tensor of shape (interpolation_order, n)
            Interpolation weights
        """
        # Compute weights based on the given order
        if self.interpolation_order == 1:
            return torch.ones((1, x.shape[0], x.shape[1]))
        elif self.interpolation_order == 2:
            return torch.stack([0.5 * (1 - 2 * x), 0.5 * (1 + 2 * x)])
        elif self.interpolation_order == 3:
            x2 = x * x
            return torch.stack(
                [
                    1 / 8 * (1 - 4 * x + 4 * x2),
                    1 / 4 * (3 - 4 * x2),
                    1 / 8 * (1 + 4 * x + 4 * x2),
                ]
            )
        elif self.interpolation_order == 4:
            x2 = x * x
            x3 = x * x2
            return torch.stack(
                [
                    1 / 48 * (1 - 6 * x + 12 * x2 - 8 * x3),
                    1 / 48 * (23 - 30 * x - 12 * x2 + 24 * x3),
                    1 / 48 * (23 + 30 * x - 12 * x2 - 24 * x3),
                    1 / 48 * (1 + 6 * x + 12 * x2 + 8 * x3),
                ]
            )
        elif self.interpolation_order == 5:
            x2 = x * x
            x3 = x * x2
            x4 = x * x3
            return torch.stack(
                [
                    1 / 384 * (1 - 8 * x + 24 * x2 - 32 * x3 + 16 * x4),
                    1 / 96 * (19 - 44 * x + 24 * x2 + 16 * x3 - 16 * x4),
                    1 / 192 * (115 - 120 * x2 + 48 * x4),
                    1 / 96 * (19 + 44 * x + 24 * x2 - 16 * x3 - 16 * x4),
                    1 / 384 * (1 + 8 * x + 24 * x2 + 32 * x3 + 16 * x4),
                ]
            )
        else:
            raise ValueError("Only `interpolation_order` from 1 to 5 are allowed")

    def compute_interpolation_weights(self, positions: torch.Tensor):
        """
        Compute the interpolation weights of each atom for a given cell (specified
        during initialization of this class). The weights are not returned, but are used
        when calling the forward (points_to_mesh) and backward (mesh_to_points)
        interpolation functions.

        :param positions: torch.tensor of shape (N,3)
            Absolute positions of atoms in Cartesian coordinates
        """
        # Compute positions relative to the mesh basis vectors
        positions_rel = torch.linalg.solve(self.cell.T, positions.T).T
        positions_rel *= self.ns_mesh

        # Calculate positions and distances based on interpolation order
        if self.interpolation_order % 2 == 0:
            positions_rel_idx = torch.floor(positions_rel).long()
            offsets = positions_rel - (positions_rel_idx + 1 / 2)
        else:
            positions_rel_idx = torch.round(positions_rel).long()
            offsets = positions_rel - positions_rel_idx

        # Compute weights based on distances and interpolation order
        self.interpolation_weights = self.compute_1d_weights(offsets)

        # Calculate indices of mesh points on which
        # the particle weights are interpolated
        # For each particle, its weight is "smeared" onto
        # (interpolation_order)**3 mesh points,
        # which can be achived using meshgrid below.
        indices_to_interpolate = torch.stack(
            [
                (positions_rel_idx + i) % self.ns_mesh
                for i in range(
                    1 - (self.interpolation_order + 1) // 2,
                    1 + self.interpolation_order // 2,
                )
            ],
            dim=0,
        )

        # Generate shifts for x, y, z axes and flatten for indexing
        x_shifts, y_shifts, z_shifts = torch.meshgrid(
            torch.arange(self.interpolation_order),
            torch.arange(self.interpolation_order),
            torch.arange(self.interpolation_order),
            indexing="ij",
        )
        self.x_shifts = x_shifts.flatten()
        self.y_shifts = y_shifts.flatten()
        self.z_shifts = z_shifts.flatten()

        # Generate a flattened representation of all the indices
        # of the mesh points on which we wish to interpolate the
        # density.
        self.x_indices = indices_to_interpolate[self.x_shifts, :, 0]
        self.y_indices = indices_to_interpolate[self.y_shifts, :, 1]
        self.z_indices = indices_to_interpolate[self.z_shifts, :, 2]

    def points_to_mesh(self, particle_weights: torch.Tensor) -> torch.Tensor:
        """
        Generate a discretized density from interpolation weights. It assumes that
        "compute_interpolation_weights" has been called before to compute all the
        necessary weights and indices.

        :param particle_weights: torch.tensor of shape (n_atoms, n_channels)
            particle_weights[i,a] is the "weight" or "charge" that atom i has to
            generate the "a-th" potential. In practice, this can be used to compute e.g.
            the Na and Cl contributions to the potential separately by using a one-hot
            encoding of the species.

        :returns: torch.tensor of shape (n_channels, n_mesh, n_mesh, n_mesh)
            Discrete density
        """
        # Update mesh values by combining particle weights and interpolation weights
        n_channels = particle_weights.shape[1]
        nx = int(self.ns_mesh[0])
        ny = int(self.ns_mesh[1])
        nz = int(self.ns_mesh[2])
        rho_mesh = torch.zeros((n_channels,nx,ny,nz))
        for a in range(n_channels):
            rho_mesh[a].index_put_(
                (self.x_indices, self.y_indices, self.z_indices),
                (
                    particle_weights[:, a]
                    * self.interpolation_weights[self.x_shifts, :, 0]
                    * self.interpolation_weights[self.y_shifts, :, 1]
                    * self.interpolation_weights[self.z_shifts, :, 2]
                ),
                accumulate=True,
            )

        return rho_mesh

    def mesh_to_points(self, mesh_vals: torch.Tensor) -> torch.Tensor:
        """
        Take a function defined on a mesh and interpolate
        its values on arbitrary positions.

        :param mesh_vals: torch.tensor of shape (n_channels, nx, ny, nz)
            The tensor contains the values of a function evaluated on a
            three-dimensional mesh. (nx, ny, nz) are the number of points along each of
            the three directions, while n_channels provides the number of such functions
            that are treated simulateously for the present system.
        :param positions: torch.tensor of shape (n_points,3)
            Absolute positions of particles in Cartesian coordinates, onto whose
            locations we wish to interpolate the mesh values.

        :returns: interpolated_values: torch.tensor of shape (n_points, n_channels)
            Values of the interpolated function.
        """
        interpolated_values = (
            (
                mesh_vals[:, self.x_indices, self.y_indices, self.z_indices]
                * self.interpolation_weights[self.x_shifts, :, 0]
                * self.interpolation_weights[self.y_shifts, :, 1]
                * self.interpolation_weights[self.z_shifts, :, 2]
            )
            .sum(dim=1)
            .T
        )

        return interpolated_values
