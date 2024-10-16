import torch
from torchpme.lib import KSpaceKernel, KSpaceFilter, MeshInterpolator
import numpy as np
import argparse
from time import time

class TestKernel(KSpaceKernel):
    def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        return k_sq*0.0+1.0


def allocate_grids(n:int, n_points:int, dtype:torch.dtype=torch.float32, device:torch.device=torch.device("cuda"), requires_grad:bool=True):    
    test_kernel = TestKernel()
    cell = torch.eye(3, dtype=dtype, device=device).requires_grad_(requires_grad)
    ns = torch.tensor([n, n, n], dtype=torch.int32, device=device)
    mesh = MeshInterpolator(cell, ns, interpolation_nodes=3)
    filter = KSpaceFilter(ns_mesh = ns,
                    cell=cell,
                    kernel=test_kernel 
                    )
    positions = torch.ones(size=[n_points,3], device=device, dtype=dtype).requires_grad_(requires_grad)
    charges = torch.ones(size=[n_points,2], device=device, dtype=dtype).requires_grad_(requires_grad)                   
    return mesh, filter, positions, charges, cell

def run_kspace(mesh_values:MeshInterpolator, kernel:torch.nn.Module, positions:torch.Tensor, charges:torch.Tensor):

    mesh.compute_weights(positions)
    mesh_values = mesh.points_to_mesh(charges)
    point_values = kernel.compute(mesh_values)
    point_values = mesh.mesh_to_points(point_values)
    return point_values

def parse_args():
    parser = argparse.ArgumentParser(description="Parser for mesh and run configurations.")
    
    parser.add_argument('--n_mesh', type=int, default=1024, help='Number of mesh points (default: 1024)')
    parser.add_argument('--n_points', type=int, default=64, help='Number of atoms (default: 64)')
    parser.add_argument('--n_runs', type=int, default=16, help='Number of runs (default: 16)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], default='float32', 
                        help='Data type (default: float32)')
    parser.add_argument('--jit', action='store_true', help='Enable JIT (default: False)')
    parser.add_argument('--backward', action='store_true', help='Enable backward pass (default: False)')

    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    args.dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    args.device = torch.device(args.device)

    return args


def run_benchmark(n_mesh:int, n_points:int, n_runs:int, device:torch.device, dtype:torch.dtype, backward:bool=False, jit:bool=False):

    mesh, kernel, positions, charges, cell = allocate_grids(n=n_mesh, n_points=n_points, device=device, dtype=dtype, requires_grad=backward)

    class RunModule(torch.nn.Module):
        def __init__(self, mesh:torch.nn.Module, filter: torch.nn.Module):
            super().__init__()
            self.mesh:torch.nn.Module = mesh
            self.kernel:torch.nn.Module = filter
    
        def forward(self, positions:torch.Tensor, charges:torch.Tensor, cell:torch.tensor, backward:bool):
            
            self.mesh.compute_weights(positions)
            mesh_values = self.mesh.points_to_mesh(charges)
            self.kernel.update_mesh(cell, mesh.ns_mesh)            
            filter_mesh = self.kernel.compute(mesh_values)
            point_values = self.mesh.mesh_to_points(filter_mesh)
            
            if backward:
                mesh_sum = point_values.sum()
                mesh_sum.backward(retain_graph=True)
            else:
                mesh_sum = torch.tensor(0.0)
            return point_values, mesh_sum
    
    run_module = RunModule(mesh, kernel)
    if jit:
        run_module = torch.jit.script(run_module)

    point_values, mesh_sum = run_module(positions, charges, cell, backward)
    if device=="cuda":
        torch.cuda.synchronize()        

    timings = []
    for i in range(n_runs):
        if backward:
            positions.grad.zero_()
            charges.grad.zero_()
            cell.grad.zero_()
            run_module.zero_grad()
        start = time()
        point_values, mesh_sum = run_module(positions, charges, cell, backward)
        if device==torch.device("cuda"):
            torch.cuda.synchronize()
        end = time()
        timings.append(end-start)
    
    return timings


args = parse_args()

print(f"""
Running tests with n_mesh={args.n_mesh}, n_points={args.n_points}, n_runs={args.n_runs}, device={args.device}, dtype={args.dtype}, backward={args.backward}, jit={args.jit}
""")

timings = run_benchmark(args.n_mesh, args.n_points, args.n_runs, args.device, args.dtype, args.backward, args.jit)

print(f"""
Average timing: {np.mean(timings)}  ,  std= {np.std(timings)}
""")