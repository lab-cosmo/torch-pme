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
            mesh_centering: str = "real", 
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
        self.values = torch.zeros(size=(n_channels, n_mesh, n_mesh, n_mesh), device=device, dtype=dtype) 
        
        self.mesh_centering = mesh_centering
        if self.mesh_centering == "real":
            self.grid_x = torch.linspace(0, mesh_size*(n_mesh-1)/n_mesh, n_mesh)
            self.grid_y = torch.linspace(0, mesh_size*(n_mesh-1)/n_mesh, n_mesh)
            self.grid_z = torch.linspace(0, mesh_size*(n_mesh-1)/n_mesh, n_mesh) 
        elif self.mesh_centering == "fft":
            self.grid_x = torch.fft.fftfreq(n_mesh)*mesh_size
            self.grid_y = torch.fft.fftfreq(n_mesh)*mesh_size
            self.grid_z = torch.fft.fftfreq(n_mesh)*mesh_size
        else: 
            raise ValueError(f"Invalid mesh centering mode {mesh_centering}")



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

        # TODO - THIS IS COPIED AND JUST ADAPTED FROM M&k CODE. NEEDS CLEANUP AND COMMENTING (AS WELL AS COPYING OVER HIGHER P AND HANDLING OF PBC)
        positions_cell = torch.div(system.positions, mesh.spacing)
        positions_cell_idx = torch.round(positions_cell).long()
        
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
            # Define auxilary functions
            f_m = lambda x: ((x+x)-1)**2/8
            f_0 = lambda x: (3/4 - x*x)
            f_p = lambda x: ((x+x)+1)**2/8
            weight_m = f_m(dist)
            weight_0 = f_0(dist)
            weight_p = f_p(dist)

            frac_mmm = weight_m[:,0] * weight_m[:,1] * weight_m[:,2]
            frac_mm0 = weight_m[:,0] * weight_m[:,1] * weight_0[:,2]
            frac_mmp = weight_m[:,0] * weight_m[:,1] * weight_p[:,2]
            frac_m0m = weight_m[:,0] * weight_0[:,1] * weight_m[:,2]
            frac_m00 = weight_m[:,0] * weight_0[:,1] * weight_0[:,2]
            frac_m0p = weight_m[:,0] * weight_0[:,1] * weight_p[:,2]
            frac_mpm = weight_m[:,0] * weight_p[:,1] * weight_m[:,2]
            frac_mp0 = weight_m[:,0] * weight_p[:,1] * weight_0[:,2]
            frac_mpp = weight_m[:,0] * weight_p[:,1] * weight_p[:,2]

            frac_0mm = weight_0[:,0] * weight_m[:,1] * weight_m[:,2]
            frac_0m0 = weight_0[:,0] * weight_m[:,1] * weight_0[:,2]
            frac_0mp = weight_0[:,0] * weight_m[:,1] * weight_p[:,2]
            frac_00m = weight_0[:,0] * weight_0[:,1] * weight_m[:,2]
            frac_000 = weight_0[:,0] * weight_0[:,1] * weight_0[:,2]
            frac_00p = weight_0[:,0] * weight_0[:,1] * weight_p[:,2]
            frac_0pm = weight_0[:,0] * weight_p[:,1] * weight_m[:,2]
            frac_0p0 = weight_0[:,0] * weight_p[:,1] * weight_0[:,2]
            frac_0pp = weight_0[:,0] * weight_p[:,1] * weight_p[:,2]

            frac_pmm = weight_p[:,0] * weight_m[:,1] * weight_m[:,2]
            frac_pm0 = weight_p[:,0] * weight_m[:,1] * weight_0[:,2]
            frac_pmp = weight_p[:,0] * weight_m[:,1] * weight_p[:,2]
            frac_p0m = weight_p[:,0] * weight_0[:,1] * weight_m[:,2]
            frac_p00 = weight_p[:,0] * weight_0[:,1] * weight_0[:,2]
            frac_p0p = weight_p[:,0] * weight_0[:,1] * weight_p[:,2]
            frac_ppm = weight_p[:,0] * weight_p[:,1] * weight_m[:,2]
            frac_pp0 = weight_p[:,0] * weight_p[:,1] * weight_0[:,2]
            frac_ppp = weight_p[:,0] * weight_p[:,1] * weight_p[:,2]            
   
            pci = positions_cell_idx
            w[:, (pci[:,0]+0)% N_mesh, (pci[:,1]+0)% N_mesh, (pci[:,2]+0) % N_mesh] += frac_000*embeddings.T
            w[:, (pci[:,0]+1)% N_mesh, (pci[:,1]+0)% N_mesh, (pci[:,2]+0) % N_mesh] += frac_p00*embeddings.T
            w[:, (pci[:,0]-1)% N_mesh, (pci[:,1]+0)% N_mesh, (pci[:,2]+0) % N_mesh] += frac_m00*embeddings.T
            
            w[:, (pci[:,0]+0)% N_mesh, (pci[:,1]+1)% N_mesh, (pci[:,2]+0) % N_mesh] += frac_0p0*embeddings.T
            w[:, (pci[:,0]+1)% N_mesh, (pci[:,1]+1)% N_mesh, (pci[:,2]+0) % N_mesh] += frac_pp0*embeddings.T
            w[:, (pci[:,0]-1)% N_mesh, (pci[:,1]+1)% N_mesh, (pci[:,2]+0) % N_mesh] += frac_mp0*embeddings.T
            
            w[:, (pci[:,0]+0)% N_mesh, (pci[:,1]-1)% N_mesh, (pci[:,2]+0) % N_mesh] += frac_0m0*embeddings.T
            w[:, (pci[:,0]+1)% N_mesh, (pci[:,1]-1)% N_mesh, (pci[:,2]+0) % N_mesh] += frac_pm0*embeddings.T
            w[:, (pci[:,0]-1)% N_mesh, (pci[:,1]-1)% N_mesh, (pci[:,2]+0) % N_mesh] += frac_mm0*embeddings.T
            
            w[:, (pci[:,0]+0)% N_mesh, (pci[:,1]+0)% N_mesh, (pci[:,2]+1) % N_mesh] += frac_00p*embeddings.T
            w[:, (pci[:,0]+1)% N_mesh, (pci[:,1]+0)% N_mesh, (pci[:,2]+1) % N_mesh] += frac_p0p*embeddings.T
            w[:, (pci[:,0]-1)% N_mesh, (pci[:,1]+0)% N_mesh, (pci[:,2]+1) % N_mesh] += frac_m0p*embeddings.T
            
            w[:, (pci[:,0]+0)% N_mesh, (pci[:,1]+1)% N_mesh, (pci[:,2]+1) % N_mesh] += frac_0pp*embeddings.T
            w[:, (pci[:,0]+1)% N_mesh, (pci[:,1]+1)% N_mesh, (pci[:,2]+1) % N_mesh] += frac_ppp*embeddings.T
            w[:, (pci[:,0]-1)% N_mesh, (pci[:,1]+1)% N_mesh, (pci[:,2]+1) % N_mesh] += frac_mpp*embeddings.T
            
            w[:, (pci[:,0]+0)% N_mesh, (pci[:,1]-1)% N_mesh, (pci[:,2]+1) % N_mesh] += frac_0mp*embeddings.T
            w[:, (pci[:,0]+1)% N_mesh, (pci[:,1]-1)% N_mesh, (pci[:,2]+1) % N_mesh] += frac_pmp*embeddings.T
            w[:, (pci[:,0]-1)% N_mesh, (pci[:,1]-1)% N_mesh, (pci[:,2]+1) % N_mesh] += frac_mmp*embeddings.T

            w[:, (pci[:,0]+0)% N_mesh, (pci[:,1]+0)% N_mesh, (pci[:,2]-1) % N_mesh] += frac_00m*embeddings.T
            w[:, (pci[:,0]+1)% N_mesh, (pci[:,1]+0)% N_mesh, (pci[:,2]-1) % N_mesh] += frac_p0m*embeddings.T
            w[:, (pci[:,0]-1)% N_mesh, (pci[:,1]+0)% N_mesh, (pci[:,2]-1) % N_mesh] += frac_m0m*embeddings.T
            
            w[:, (pci[:,0]+0)% N_mesh, (pci[:,1]+1)% N_mesh, (pci[:,2]-1) % N_mesh] += frac_0pm*embeddings.T
            w[:, (pci[:,0]+1)% N_mesh, (pci[:,1]+1)% N_mesh, (pci[:,2]-1) % N_mesh] += frac_ppm*embeddings.T
            w[:, (pci[:,0]-1)% N_mesh, (pci[:,1]+1)% N_mesh, (pci[:,2]-1) % N_mesh] += frac_mpm*embeddings.T
            
            w[:, (pci[:,0]+0)% N_mesh, (pci[:,1]-1)% N_mesh, (pci[:,2]-1) % N_mesh] += frac_0mm*embeddings.T
            w[:, (pci[:,0]+1)% N_mesh, (pci[:,1]-1)% N_mesh, (pci[:,2]-1) % N_mesh] += frac_pmm*embeddings.T
            w[:, (pci[:,0]-1)% N_mesh, (pci[:,1]-1)% N_mesh, (pci[:,2]-1) % N_mesh] += frac_mmm*embeddings.T


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
        rp_0 = (points_cell_idx + 0) % mesh.n_mesh
        rp_p = (points_cell_idx + 1) % mesh.n_mesh
        rp_m = (points_cell_idx - 1 + mesh.n_mesh) % mesh.n_mesh

        interpolated_values = torch.zeros((points.shape[0], mesh.n_channels), 
                                dtype=points.dtype, device=points.device)
        if self.mesh_interpolation_order == 3:
            # Find closest mesh point
            dist = points_cell - rp

            # Define auxilary functions
            f_m = lambda x: ((x+x)-1)**2/8
            f_0 = lambda x: (3/4 - x*x)
            f_p = lambda x: ((x+x)+1)**2/8
            weight_m = f_m(dist)
            weight_0 = f_0(dist)
            weight_p = f_p(dist)

            frac_mmm = weight_m[:,0] * weight_m[:,1] * weight_m[:,2]
            frac_mm0 = weight_m[:,0] * weight_m[:,1] * weight_0[:,2]
            frac_mmp = weight_m[:,0] * weight_m[:,1] * weight_p[:,2]
            frac_m0m = weight_m[:,0] * weight_0[:,1] * weight_m[:,2]
            frac_m00 = weight_m[:,0] * weight_0[:,1] * weight_0[:,2]
            frac_m0p = weight_m[:,0] * weight_0[:,1] * weight_p[:,2]
            frac_mpm = weight_m[:,0] * weight_p[:,1] * weight_m[:,2]
            frac_mp0 = weight_m[:,0] * weight_p[:,1] * weight_0[:,2]
            frac_mpp = weight_m[:,0] * weight_p[:,1] * weight_p[:,2]

            frac_0mm = weight_0[:,0] * weight_m[:,1] * weight_m[:,2]
            frac_0m0 = weight_0[:,0] * weight_m[:,1] * weight_0[:,2]
            frac_0mp = weight_0[:,0] * weight_m[:,1] * weight_p[:,2]
            frac_00m = weight_0[:,0] * weight_0[:,1] * weight_m[:,2]
            frac_000 = weight_0[:,0] * weight_0[:,1] * weight_0[:,2]
            frac_00p = weight_0[:,0] * weight_0[:,1] * weight_p[:,2]
            frac_0pm = weight_0[:,0] * weight_p[:,1] * weight_m[:,2]
            frac_0p0 = weight_0[:,0] * weight_p[:,1] * weight_0[:,2]
            frac_0pp = weight_0[:,0] * weight_p[:,1] * weight_p[:,2]

            frac_pmm = weight_p[:,0] * weight_m[:,1] * weight_m[:,2]
            frac_pm0 = weight_p[:,0] * weight_m[:,1] * weight_0[:,2]
            frac_pmp = weight_p[:,0] * weight_m[:,1] * weight_p[:,2]
            frac_p0m = weight_p[:,0] * weight_0[:,1] * weight_m[:,2]
            frac_p00 = weight_p[:,0] * weight_0[:,1] * weight_0[:,2]
            frac_p0p = weight_p[:,0] * weight_0[:,1] * weight_p[:,2]
            frac_ppm = weight_p[:,0] * weight_p[:,1] * weight_m[:,2]
            frac_pp0 = weight_p[:,0] * weight_p[:,1] * weight_0[:,2]
            frac_ppp = weight_p[:,0] * weight_p[:,1] * weight_p[:,2]

            for a in range(mesh.n_channels):
                # TODO I think the calculation of the channels can be serialized
                # Add up contributions to the potential from 27 closest mesh poitns
                for x in ['m', '0', 'p']:
                    for y in ['m', '0', 'p']:
                        for z in ['m', '0', 'p']:
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