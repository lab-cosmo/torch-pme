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
        n_mesh = torch.ceil(mesh_size/mesh_resolution).long().item()
        self.n_mesh = n_mesh
        self. spacing = mesh_size / n_mesh
        
        self.n_channels = n_channels
        self.values = torch.zeros(size=(n_channels, n_mesh, n_mesh, n_mesh), device=device, dtype=dtype) 
        
        self.grid_x = torch.linspace(0, mesh_size*(n_mesh-1)/n_mesh, n_mesh)
        self.grid_y = torch.linspace(0, mesh_size*(n_mesh-1)/n_mesh, n_mesh)
        self.grid_z = torch.linspace(0, mesh_size*(n_mesh-1)/n_mesh, n_mesh) 

class FieldBuilder(torch.nn.Module):
    """
    Takes a list of points and builds a representation as a density field on a mesh.
    """
    def __init__(self, 
                 mesh_resolution: float = 0.1,
                 mesh_interpolation_order: int =2,
                 ):
        
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

        # TODO - THIS IS COPIED AND JUST ADAPTED FROM M&k CODE. NEEDS CLEANUP AND COMMENTING (AS WELL AS COPYING OVER HIGHER P AND HANDLING OF PBC)
        positions_cell = torch.div(system.positions, mesh.spacing)
        positions_cell_idx = torch.ceil(positions_cell).long()
        
        if self.mesh_interpolation_order == 2:
            # TODO - CHECK IF THIS ACTUALLY WORKS, GETTING FISHY RESULTS
            l_dist = positions_cell - positions_cell_idx
            r_dist = 1 - l_dist
            w = mesh.values
            N_mesh = mesh.n_mesh
            
            frac_000 = l_dist[:, 0] * l_dist[:, 1] * l_dist[:, 2]
            frac_001 = l_dist[:, 0] * l_dist[:, 1] * r_dist[:, 2]
            frac_010 = l_dist[:, 0] * r_dist[:, 1] * l_dist[:, 2]
            frac_011 = l_dist[:, 0] * r_dist[:, 1] * r_dist[:, 2]
            frac_100 = r_dist[:, 0] * l_dist[:, 1] * l_dist[:, 2]
            frac_101 = r_dist[:, 0] * l_dist[:, 1] * r_dist[:, 2]
            frac_110 = r_dist[:, 0] * r_dist[:, 1] * l_dist[:, 2]
            frac_111 = r_dist[:, 0] * r_dist[:, 1] * r_dist[:, 2]        

            rp_a_species = positions_cell_idx
            print(rp_a_species.shape, embeddings.shape, frac_000.shape, w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+0) % N_mesh].shape)
            # Perform actual smearing on density grid. takes indices modulo N_mesh to handle PBC
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_000*embeddings.T
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_001*embeddings.T
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_010*embeddings.T
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_011*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_100*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_101*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_110*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_111*embeddings.T
        elif self.mesh_interpolation_order == 3:

            dist = positions_cell - positions_cell_idx
            w = mesh.values
            N_mesh = mesh.n_mesh
            
            frac_000 = 1/4 * (3 - 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2)
            frac_001 = 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2)
            frac_00m = 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2)
            
            frac_010 = 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2)
            frac_011 = 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2)
            frac_01m = 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2)
            
            frac_0m0 = 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2)
            frac_0m1 = 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2)
            frac_0mm = 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2)
            
            frac_100 = 1/4 * (3 - 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_101 = 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_10m = 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            
            frac_110 = 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_111 = 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_11m = 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            
            frac_1m0 = 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_1m1 = 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_1mm = 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            
            frac_m00 = 1/4 * (3 - 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_m01 = 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_m0m = 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            
            frac_m10 = 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_m11 = 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_m1m = 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            
            frac_mm0 = 1/4 * (3 - 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_mm1 = 1/8 * (1 + 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2)
            frac_mmm = 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2) * 1/8 * (1 - 4 * dist[:, 0] + 4 * dist[:, 0]**2)

            rp_a_species = positions_cell_idx
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_000*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_001*embeddings.T
            w[:, (rp_a_species[:,0]-1)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_00m*embeddings.T
            
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_010*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_011*embeddings.T
            w[:, (rp_a_species[:,0]-1)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_01m*embeddings.T
            
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]-1)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_0m0*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]-1)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_0m1*embeddings.T
            w[:, (rp_a_species[:,0]-1)% N_mesh, (rp_a_species[:,1]-1)% N_mesh, (rp_a_species[:,2]+0) % N_mesh] += frac_0mm*embeddings.T
            
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_100*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_101*embeddings.T
            w[:, (rp_a_species[:,0]-1)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_10m*embeddings.T
            
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_110*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_111*embeddings.T
            w[:, (rp_a_species[:,0]-1)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_11m*embeddings.T
            
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]-1)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_1m0*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]-1)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_1m1*embeddings.T
            w[:, (rp_a_species[:,0]-1)% N_mesh, (rp_a_species[:,1]-1)% N_mesh, (rp_a_species[:,2]+1) % N_mesh] += frac_1mm*embeddings.T

            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]-1) % N_mesh] += frac_m00*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]-1) % N_mesh] += frac_m01*embeddings.T
            w[:, (rp_a_species[:,0]-1)% N_mesh, (rp_a_species[:,1]+0)% N_mesh, (rp_a_species[:,2]-1) % N_mesh] += frac_m0m*embeddings.T
            
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]-1) % N_mesh] += frac_m10*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]-1) % N_mesh] += frac_m11*embeddings.T
            w[:, (rp_a_species[:,0]-1)% N_mesh, (rp_a_species[:,1]+1)% N_mesh, (rp_a_species[:,2]-1) % N_mesh] += frac_m1m*embeddings.T
            
            w[:, (rp_a_species[:,0]+0)% N_mesh, (rp_a_species[:,1]-1)% N_mesh, (rp_a_species[:,2]-1) % N_mesh] += frac_mm0*embeddings.T
            w[:, (rp_a_species[:,0]+1)% N_mesh, (rp_a_species[:,1]-1)% N_mesh, (rp_a_species[:,2]-1) % N_mesh] += frac_mm1*embeddings.T
            w[:, (rp_a_species[:,0]-1)% N_mesh, (rp_a_species[:,1]-1)% N_mesh, (rp_a_species[:,2]-1) % N_mesh] += frac_mmm*embeddings.T

        mesh.values /= mesh.spacing**3
        return mesh
    
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
    
    def compute(self, 
                mesh: Mesh, 
                points: torch.tensor
                ):
        
        n_points = points.shape[0]

        points_cell = torch.div(points, mesh.spacing)
        points_cell_idx = torch.ceil(points_cell).long()
        
        # TODO rewrite the code below to use the more descriptive variables
        rp = points_cell_idx
        rp_0 = (points_cell_idx + 0) % mesh.n_mesh
        rp_1 = (points_cell_idx + 1) % mesh.n_mesh
        rp_m = (points_cell_idx - 1 + mesh.n_mesh) % mesh.n_mesh

        interpolated_values = torch.zeros((points.shape[0], mesh.n_channels), 
                                dtype=points.dtype, device=points.device)
        if self.mesh_interpolation_order == 3:
            # Find closest mesh point
            dist = points_cell - rp

            # Define auxilary functions
            f_m = lambda x: (1-4*x+4*x**2)/8
            f_0 = lambda x: (3-4*x**2)/4
            f_1 = lambda x: (1+4*x+4*x**2)/8
            weight_m = f_m(dist)
            weight_0 = f_0(dist)
            weight_1 = f_1(dist)

            frac_mmm = weight_m[:,0] * weight_m[:,1] * weight_m[:,2]
            frac_mm0 = weight_m[:,0] * weight_m[:,1] * weight_0[:,2]
            frac_mm1 = weight_m[:,0] * weight_m[:,1] * weight_1[:,2]
            frac_m0m = weight_m[:,0] * weight_0[:,1] * weight_m[:,2]
            frac_m00 = weight_m[:,0] * weight_0[:,1] * weight_0[:,2]
            frac_m01 = weight_m[:,0] * weight_0[:,1] * weight_1[:,2]
            frac_m1m = weight_m[:,0] * weight_1[:,1] * weight_m[:,2]
            frac_m10 = weight_m[:,0] * weight_1[:,1] * weight_0[:,2]
            frac_m11 = weight_m[:,0] * weight_1[:,1] * weight_1[:,2]

            frac_0mm = weight_0[:,0] * weight_m[:,1] * weight_m[:,2]
            frac_0m0 = weight_0[:,0] * weight_m[:,1] * weight_0[:,2]
            frac_0m1 = weight_0[:,0] * weight_m[:,1] * weight_1[:,2]
            frac_00m = weight_0[:,0] * weight_0[:,1] * weight_m[:,2]
            frac_000 = weight_0[:,0] * weight_0[:,1] * weight_0[:,2]
            frac_001 = weight_0[:,0] * weight_0[:,1] * weight_1[:,2]
            frac_01m = weight_0[:,0] * weight_1[:,1] * weight_m[:,2]
            frac_010 = weight_0[:,0] * weight_1[:,1] * weight_0[:,2]
            frac_011 = weight_0[:,0] * weight_1[:,1] * weight_1[:,2]

            frac_1mm = weight_1[:,0] * weight_m[:,1] * weight_m[:,2]
            frac_1m0 = weight_1[:,0] * weight_m[:,1] * weight_0[:,2]
            frac_1m1 = weight_1[:,0] * weight_m[:,1] * weight_1[:,2]
            frac_10m = weight_1[:,0] * weight_0[:,1] * weight_m[:,2]
            frac_100 = weight_1[:,0] * weight_0[:,1] * weight_0[:,2]
            frac_101 = weight_1[:,0] * weight_0[:,1] * weight_1[:,2]
            frac_11m = weight_1[:,0] * weight_1[:,1] * weight_m[:,2]
            frac_110 = weight_1[:,0] * weight_1[:,1] * weight_0[:,2]
            frac_111 = weight_1[:,0] * weight_1[:,1] * weight_1[:,2]

            for a in range(mesh.n_channels):
                # TODO I think the calculation of the channels can be serialized
                # Add up contributions to the potential from 27 closest mesh poitns
                for x in ['m', '0', '1']:
                    for y in ['m', '0', '1']:
                        for z in ['m', '0', '1']:
                            # TODO write this out
                            command = f"""interpolated_values[:,a] += (
                            mesh.values[a, rp_{x}[:,0], rp_{y}[:,1], rp_{z}[:,2]] 
                            * frac_{x}{y}{z}).float()"""
                            exec(command)
        
        return interpolated_values
    
    def forward(self, 
                mesh: Mesh, 
                points: torch.tensor
                ):
        return self.compute(mesh, points)