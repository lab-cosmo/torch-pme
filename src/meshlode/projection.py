from typing import Optional

import torch

# TODO get rid of numpy dependence 
import numpy as np

from .system import System
from.mesh import Mesh, MeshInterpolator

import sphericart.torch as sph

from.radial import RadialBasis


def _radial_nodes_and_weights(a, b, num_nodes):
    """
    Define Gauss-Legendre quadrature nodes and weights on the interval [a,b].
    
    The nodes and weights are obtained using the Golub-Welsh algorithm.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to be used in Gauss-Legendre quadrature

    a, b : float
        The integral is over the interval [a,b]. The Gauss-Legendre
        nodes are defined on the interval [-1,1] by default, and will
        be rescaled to [a,b] before returning.


    Returns
    -------
    Gauss-Legendre integration nodes and weights
    
    """
    nodes = np.linspace(a, b, num_nodes)
    weights = np.ones_like(nodes)
    

    # Generate auxilary matrix A
    i = np.arange(1, num_nodes) # array([1,2,3,...,n-1])
    dd = i/np.sqrt(4*i**2-1.) # values of nonzero entries
    A = np.diag(dd,-1) + np.diag(dd,1) 

    # The optimal nodes are the eigenvalues of A
    nodes, evec = np.linalg.eigh(A)
    # The optimal weights are the squared first components of the normalized
    # eigenvectors. In this form, the sum of the weights is equal to one.
    # Since the nodes are on the interval [-1,1], we would need to multiply
    # by a factor of 2 (the length of the interval) to get the proper weights
    # on [-1,1].
    weights = evec[0,:]**2
    
    # Rescale nodes and weights to the interval [a,b]
    nodes = (nodes + 1) / 2
    nodes = nodes * (b-a) + a
    weights *= (b-a)

    return nodes, weights


def _angular_nodes_and_weights():
    """
    Define angular nodes and weights arising from Lebedev quadrature
    for an integration on the surface of the sphere. See the reference
    
    V.I. Lebedev "Values of the nodes and weights of ninth to seventeenth 
    order gauss-markov quadrature formulae invariant under the octahedron
    group with inversion" (1975)
    
    for details.

    Returns
    -------
    Nodes and weights for Lebedev cubature of degree n=9.

    """
    
    num_nodes = 38
    nodes = np.zeros((num_nodes,3))
    weights = np.zeros((num_nodes,))
    
    # Base coefficients
    A1 = 1/105 * 4*np.pi
    A3 = 9/280 * 4*np.pi
    C1 = 1/35 * 4*np.pi
    p = 0.888073833977
    q = np.sqrt(1-p**2)
    
    # Nodes of type a1: 6 points along [1,0,0] direction
    nodes[0,0] = 1
    nodes[1,0] = -1
    nodes[2,1] = 1
    nodes[3,1] = -1
    nodes[4,1] = 1
    nodes[5,1] = -1
    weights[:6] = A1
    
    # Nodes of type a2: 12 points along [1,1,0] direction
    # idx = 6
    # for j in [-1,1]:
    #     for k in [-1,1]:
    #         nodes[idx] = j, k, 0
    #         nodes[idx+4] = 0, j, k
    #         nodes[idx+8] = k, 0, j
    #         idx += 1
    # nodes[6:18] /= np.sqrt(2)
    # weights[6:18] = 1.

    # Nodes of type a3: 8 points along [1,1,1] direction
    idx = 6
    for j in [-1,1]:
        for k in [-1,1]:
            for l in [-1,1]:
                nodes[idx] = j,k,l
                idx += 1
    nodes[idx-8:idx] /= np.sqrt(3)
    weights[idx-8:idx] = A3
    
    # Nodes of type c1: 24 points
    for j in [-1,1]:
        for k in [-1,1]:
            nodes[idx] = j*p, k*q, 0
            nodes[idx+4] = j*q, k*p, 0
            nodes[idx+8] = 0, j*p, k*q
            nodes[idx+12] = 0, j*q, k*p
            nodes[idx+16] = j*p, 0, k*q
            nodes[idx+20] = j*q, 0, k*p
            idx += 1
    weights[14:] = C1
    
    return nodes, weights


class FieldProjector(torch.nn.Module):

    def __init__(self,        
        max_radial,
        max_angular,
        radial_basis_radius,
        radial_basis,
        n_radial_grid,
        n_lebdev=9,
        dtype=torch.float64,
        device="cpu"
    ):
        super(FieldProjector, self).__init__()
        # TODO have more lebdev grids implemented
        assert(n_lebdev==9) # this is the only one implemented
        rb = RadialBasis(max_radial, max_angular, radial_basis_radius, radial_basis)

        # computes radial basis
        grid_r, weights_r = _radial_nodes_and_weights(0, radial_basis_radius, n_radial_grid)
        values_r = rb.evaluate_radial_basis_functions(grid_r)

        self.grid_r = torch.tensor(grid_r, dtype=dtype, device=device)
        self.weights_r = torch.tensor(weights_r, dtype=dtype, device=device)
        self.values_r = torch.tensor(values_r, dtype=dtype, device=device)

        # computes lebdev grid
        grid_lebd, weights_lebd = _angular_nodes_and_weights()
        self.grid_lebd = torch.tensor(grid_lebd, dtype=dtype, device=device)
        self.weights_lebd = torch.tensor(weights_lebd, dtype=dtype, device=device)

        SH = sph.SphericalHarmonics(l_max = max_angular)
        self.values_lebd = SH.compute(self.grid_lebd) 

        # combines to make grid
        self.n_grid = len(self.grid_r)*len(self.grid_lebd)
        self.grid = torch.stack([
            r*rhat for r in self.grid_r for rhat in self.grid_lebd
        ])

        self.weights = torch.stack([
            w*what for w in self.weights_r for what in self.weights_lebd
        ]
        )

        self.values = torch.zeros(((max_angular+1)**2,max_radial,  
                                   self.n_grid), dtype=dtype, device=device)
        for l in range(max_angular+1):
            for n in range(max_radial):            
                self.values[l**2:(l+1)**2,n] = torch.einsum("i,jm->mij",
                    self.values_r[l,n], self.values_lebd[:,l**2:(l+1)**2]
                ).reshape((2*l+1,-1))                

    def compute(self, 
                mesh:Mesh, 
                system:System):

        mesh_interpolator = MeshInterpolator(mesh_interpolation_order=3)
        
        feats = []
        for position in system.positions:
            grid_i = self.grid + position
            values_i = mesh_interpolator.compute(mesh, grid_i)
            feats.append(torch.einsum("ga,kng,g->kan",values_i,self.values,self.weights))
        return torch.stack(feats)
    
    def forward(self,
                mesh, system):
        
        return self.compute(mesh, system)


