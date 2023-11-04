from typing import Optional

import torch
from metatensor.torch import TensorBlock

from .system import System

class Mesh:
    """
    Minimal class to store a tensor on a 3D grid.    
    """
    def __init__(
            self,
            box: torch.tensor, 
            n_channels: int = 1,
            mesh_resolution: float = 0.1,
            mesh_style: str = "real_space", 
            dtype = None,
            device = None
            ):

        if device is None:
            device = box.device
        if dtype is None:
            dtype = box.dtype

        # Checks that the cell is cubic
        mesh_size = torch.trace(box)/3
        if (((box-torch.eye(3)*mesh_size)**2)).sum() > 1e-8:
            raise ValueError("The current implementation is restricted to cubic boxes. ")
        self.box_size = mesh_size

        # Computes mesh parameters
        # makes sure mesh size is even, torch.fft is very slow otherwise (possibly needs powers of 2...)
        n_mesh = 2*torch.round(mesh_size/(2*mesh_resolution)).long().item()
        self.n_mesh = n_mesh
        self.spacing = mesh_size / n_mesh
        
        self.n_channels = n_channels        
        
        self.mesh_style = mesh_style
        if self.mesh_style == "real_space":
            # real-space grid, same dimension on all axes
            self.grid_x = torch.linspace(0, mesh_size*(n_mesh-1)/n_mesh, n_mesh)
            self.grid_y = torch.linspace(0, mesh_size*(n_mesh-1)/n_mesh, n_mesh)
            self.grid_z = torch.linspace(0, mesh_size*(n_mesh-1)/n_mesh, n_mesh) 
            self.values = torch.zeros(size=(n_channels, n_mesh, n_mesh, n_mesh), device=device, dtype=dtype) 
        elif self.mesh_style == "fft":
            # full FFT grod
            self.grid_x = torch.fft.fftfreq(n_mesh)*mesh_size
            self.grid_y = torch.fft.fftfreq(n_mesh)*mesh_size
            self.grid_z = torch.fft.fftfreq(n_mesh)*mesh_size
            self.values = torch.zeros(size=(n_channels, n_mesh, n_mesh, n_mesh), device=device, dtype=dtype) 
        elif self.mesh_style == "rfft":
            # real-valued FFT grid (to store FT of a real-valued function)
            self.grid_x = torch.fft.fftfreq(n_mesh)*mesh_size
            self.grid_y = torch.fft.fftfreq(n_mesh)*mesh_size
            self.grid_z = torch.fft.rfftfreq(n_mesh)*mesh_size
            self.values = torch.zeros(size=(n_channels, n_mesh, n_mesh, len(self.grid_z)), device=device, dtype=dtype) 
        else: 
            raise ValueError(f"Invalid mesh style {mesh_style}")



class FieldBuilder(torch.nn.Module):
    """
    Takes a list of points and builds a representation as a density field on a mesh.
    """
    def __init__(self, 
                 mesh_resolution: float = 0.1,
                 mesh_interpolation_order: int =2,
                 ):
        
        super(FieldBuilder, self).__init__()
        self.mesh_resolution = mesh_resolution
        self.mesh_interpolation_order = mesh_interpolation_order
    
    def compute(self, 
                system : System,
                embeddings: Optional[torch.tensor] = None
               ) -> Mesh:

        device = system.positions.device 

        # If atom embeddings are not given, build them as one-hot encodings of the atom types
        if embeddings is None:
            all_species, species_indices = torch.unique(system.species, sorted=True, return_inverse=True)
            embeddings = torch.zeros(size=(len(system.species), len(all_species)) ,device=device)
            embeddings[range(len(embeddings)), species_indices] = 1.0
        
        if embeddings.shape[0] != len(system.species):
            raise ValueError(f"The atomic embeddings length {embeddings.shape[0]} does not match the number of atoms {len(system.species)}.")  

        n_channels =  embeddings.shape[1]       
        mesh = Mesh(system.cell, n_channels, self.mesh_resolution)

        positions_cell = torch.div(system.positions, mesh.spacing)

        def compute_weights(dist, order):
            # Compute weights based on the given order
            if order == 2:
                return torch.stack([0.5 * (1 - 2 * dist), 0.5 * (1 + 2 * dist)])
            elif order == 3:
                return torch.stack([1/8 * (1 - 4 * dist + 4 * dist * dist),
                                    1/4 * (3 - 4 * dist * dist),
                                    1/8 * (1 + 4 * dist + 4 * dist * dist)])
            elif order == 4:
                return torch.stack([1/48 * (1 - 6 * dist + 12 * dist * dist - 8 * dist * dist * dist),
                                    1/48 * (23 - 30 * dist - 12 * dist * dist + 24 * dist * dist * dist),
                                    1/48 * (23 + 30 * dist - 12 * dist * dist - 24 * dist * dist * dist),
                                    1/48 * (1 + 6 * dist + 12 * dist * dist + 8 * dist * dist * dist)])
            else:
                raise ValueError("Only `mesh_interpolation_order` 2, 3 or 4 is allowed")
        
        def interpolate(mesh, positions_cell, embeddings):
            # Validate interpolation order
            if self.mesh_interpolation_order not in [2, 3, 4]:
                raise ValueError("Only `mesh_interpolation_order` 2, 3 or 4 is allowed")
            
            # Calculate positions and distances based on interpolation order
            if self.mesh_interpolation_order % 2 == 0: 
                positions_cell_idx = torch.floor(positions_cell).long()
                dist = positions_cell - (positions_cell_idx + 1/2)
            else: 
                positions_cell_idx = torch.round(positions_cell).long()
                dist = positions_cell - positions_cell_idx
            
            # Compute weights based on distances and interpolation order
            weight = compute_weights(dist, self.mesh_interpolation_order)

            # Calculate shifts in each direction (x, y, z)
            rp_shift = torch.stack([(positions_cell_idx + i) % mesh.n_mesh 
                                    for i in range(1 - (self.mesh_interpolation_order + 1) // 2, 
                                                   1 + self.mesh_interpolation_order // 2)], dim=0)
            
            # Generate shifts for x, y, z axes and flatten for indexing
            x_shifts, y_shifts, z_shifts = torch.meshgrid(torch.arange(self.mesh_interpolation_order), 
                                                          torch.arange(self.mesh_interpolation_order), 
                                                          torch.arange(self.mesh_interpolation_order), indexing="ij")
            x_shifts, y_shifts, z_shifts = x_shifts.flatten(), y_shifts.flatten(), z_shifts.flatten()

            # Index shifts for x, y, z coordinates
            x_indices = rp_shift[x_shifts, :, 0]
            y_indices = rp_shift[y_shifts, :, 1]
            z_indices = rp_shift[z_shifts, :, 2]

            # Update mesh values by combining embeddings and computed weights
            for a in range(mesh.n_channels):
                mesh.values[a].index_put_(
                    (x_indices, y_indices, z_indices),
                    (weight[x_shifts, :, 0] * weight[y_shifts, :, 1] * weight[z_shifts, :, 2] * embeddings[:, a]),
                    accumulate=True
                )

            return mesh
        
        return interpolate(mesh, positions_cell, embeddings)
    
    def forward(
        self,
        system: System,
        embeddings: Optional[torch.tensor] = None
    ) -> Mesh:
    
        """forward just calls :py:meth:`FieldBuilder.compute`"""
        return self.compute(system=system, embeddings=embeddings)


class MeshInterpolator(torch.nn.Module):
    """
    Evaluates a function represented on a mesh at an arbitrary list of points.
    """
    def __init__(self,  
                mesh_interpolation_order: int =2,
                ):
        
        self.mesh_interpolation_order = mesh_interpolation_order
        super(MeshInterpolator, self).__init__()  
        # TODO perhaps this does not have to be a nn.Module 
    
    def compute(self, 
                mesh: Mesh, 
                points: torch.tensor
                ):
        
        n_points = points.shape[0]

        points_cell = torch.div(points, mesh.spacing)
        points_cell_idx = torch.round(points_cell).long()
        
        # TODO rewrite the code below to use the more descriptive variables
        rp = points_cell_idx

        rp_shift = torch.stack([(points_cell_idx - 1 + mesh.n_mesh) % mesh.n_mesh,
                    (points_cell_idx + 0) % mesh.n_mesh, 
                    (points_cell_idx + 1) % mesh.n_mesh], dim=0)
        """
        rp_0 = (points_cell_idx + 0) % mesh.n_mesh
        rp_p = (points_cell_idx + 1) % mesh.n_mesh
        rp_m = (points_cell_idx - 1 + mesh.n_mesh) % mesh.n_mesh
        """
        interpolated_values = torch.zeros((points.shape[0], mesh.n_channels), 
                                dtype=points.dtype, device=points.device)
        if self.mesh_interpolation_order == 3:
            # Find closest mesh point
            dist = points_cell - rp

            # Define auxilary functions
            " [m, 0, p] "
            f_shift = [ lambda x: ((x+x)-1)**2/8, lambda x: (3/4 - x*x), lambda x: ((x+x)+1)**2/8 ]

            # compute weights for the three shifts
            weight = torch.stack([f(dist) for f in f_shift], dim=0)

            # now compute the product of weights with the mesh points, using index unrolling to make it quick
            # this builds indices corresponding to three nested loops
            x_shifts, y_shifts, z_shifts = torch.meshgrid(torch.arange(3), torch.arange(3), torch.arange(3), indexing="ij")
            x_shifts, y_shifts, z_shifts = x_shifts.flatten(), y_shifts.flatten(), z_shifts.flatten()

            # get indices of mesh positions
            x_indices = rp_shift[x_shifts, :, 0]
            y_indices = rp_shift[y_shifts, :, 1]
            z_indices = rp_shift[z_shifts, :, 2]
            
            interpolated_values = (mesh.values[:, x_indices, y_indices, z_indices] *
                                    weight[x_shifts, :, 0] * weight[y_shifts, :, 1] * weight[z_shifts, :, 2]).sum(axis=1).T
            
        return interpolated_values
    
    def forward(self, 
                mesh: Mesh, 
                points: torch.tensor
                ):
        return self.compute(mesh, points)