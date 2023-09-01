from typing import Optional

import torch
from metatensor.torch import TensorBlock

from .system import System

class Mesh:
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
    def __init__(self, 
                 mesh_resolution: float = 0.1,
                 point_interpolation_order: int =2,
                 ):
        
        self.mesh_resolution = mesh_resolution
        self.point_interpolation_order = point_interpolation_order
    
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
        print(positions_cell_idx)
        print(embeddings)
        if self.point_interpolation_order == 2:
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
        elif self.point_interpolation_order == 3:

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
        return self.compute(systems=system, embeddings=embeddings)
    
class MeshInterpolate(torch.nn.Module):
    pass


class FieldProjector(torch.nn.Module):
    pass
