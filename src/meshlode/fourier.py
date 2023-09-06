import torch
import math 

from typing import Optional
from metatensor.torch import TensorBlock
from .system import System

from .mesh import Mesh

from time import time

# TODO we don't really need to re-compute the Fourier mesh at each call. one could separate the construction of the grid and the update of the values
class FourierFilter(torch.nn.Module):
    def __init__(self, kspace_filter="coulomb", kzero_value=None):
        """
        The `kspace_filter` argument defines a R->R function that is applied to the squared norm of the k vectors
        """

        super(FourierFilter, self).__init__()
        self.kzero_value = kzero_value
        if kspace_filter == "coulomb":
            self.kspace_filter = torch.reciprocal
            self.kzero_value = 0.0
        else:
            self.kspace_filter = kspace_filter    

        self.timings=dict(n_eval=0, r2k=0, k2r=0, filter=0, 
                          filter_grid=0, filter_calc=0, filter_prod=0)    
        pass

    def compute_r2k(self, mesh: Mesh) -> Mesh:
        
        k_size = math.pi*2/mesh.box_size
        k_mesh = Mesh(torch.eye(3)*k_size, mesh.n_channels, k_size/mesh.n_mesh, dtype=torch.complex64)
        
        for i_channel in range(mesh.n_channels):
            k_mesh.values[i_channel] = torch.fft.fftn(mesh.values[i_channel])        
        
        return k_mesh
    
    def apply_filter(self, k_mesh: Mesh) -> Mesh:     
        self.timings["filter_grid"] -= time()   
        kxs, kys, kzs = torch.meshgrid(k_mesh.grid_x, k_mesh.grid_y, k_mesh.grid_z, 
                                       indexing="ij") 
        self.timings["filter_grid"] += time()

        self.timings["filter_calc"] -= time()
        k_norm2 = kxs**2 + kys**2 + kzs**2
        k_filter = self.kspace_filter(k_norm2)
        self.timings["filter_calc"] += time()

        self.timings["filter_prod"] -= time()
        k_mesh.values *= k_filter
        if self.kzero_value is not None:
            k_mesh.values[:,0,0,0] = self.kzero_value
        self.timings["filter_prod"] += time()

        pass

    def compute_k2r(self, k_mesh: Mesh) -> Mesh:

        box_size = math.pi*2/k_mesh.box_size
        mesh = Mesh(torch.eye(3)*box_size, k_mesh.n_channels, box_size/k_mesh.n_mesh, dtype=torch.float64)
        
        for i_channel in range(mesh.n_channels):
            mesh.values[i_channel] = torch.fft.ifftn(k_mesh.values[i_channel]).real
        
        return mesh
    
    def forward(self, mesh:Mesh) -> Mesh:

        self.timings["n_eval"]+=1
        self.timings["r2k"] -= time()
        k_mesh = self.compute_r2k(mesh)
        self.timings["r2k"] += time()

        self.timings["filter"] -= time()
        self.apply_filter(k_mesh)
        self.timings["filter"] += time()
        
        self.timings["k2r"] -= time()
        rval=self.compute_k2r(k_mesh)
        self.timings["k2r"] += time()

        return rval
    