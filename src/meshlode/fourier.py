import torch
import math 

from typing import Optional
from metatensor.torch import TensorBlock
from .system import System

from .mesh import Mesh


class FourierFilter(torch.nn.Module):
    def __init__(self):
        pass

    def compute_r2k(self, mesh: Mesh) -> Mesh:

        k_size = math.pi*2/mesh.box_size
        k_mesh = Mesh(torch.eye(3)*k_size, mesh.n_channels, k_size/mesh.n_mesh, dtype=torch.complex64)
        
        for i_channel in range(mesh.n_channels):
            k_mesh.values[i_channel] = torch.fft.fftn(mesh.values[i_channel])        
        
        return k_mesh
    
    def apply_filter(self, k_mesh: Mesh) -> Mesh:
        # TODO - general filter, possibly defined in __init__?
        kxs, kys, kzs = torch.meshgrid(k_mesh.grid_x, k_mesh.grid_y, k_mesh.grid_z) 

        k_norm2 = kxs**2 + kys**2 + kzs**2
        k_norm2[0,0,0] = 1.
        filter_coulomb = torch.reciprocal(k_norm2)

        k_mesh.values *= filter_coulomb
        pass

    def compute_k2r(self, k_mesh: Mesh) -> Mesh:

        box_size = math.pi*2/k_mesh.box_size
        mesh = Mesh(torch.eye(3)*box_size, k_mesh.n_channels, box_size/k_mesh.n_mesh, dtype=torch.float64)
        
        for i_channel in range(mesh.n_channels):
            mesh.values[i_channel] = torch.fft.ifftn(k_mesh.values[i_channel]).real
        
        return mesh
    
    def forward(self, mesh:Mesh) -> Mesh:

        k_mesh = self.compute_r2k(mesh)
        self.apply_filter(k_mesh)
        return self.compute_k2r(k_mesh)
    