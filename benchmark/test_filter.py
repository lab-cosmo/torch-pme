import torch
from torchpme.lib import KSpaceKernel, KSpaceFilter
import numpy as np
import argparse
from time import time

class TestKernel(KSpaceKernel):
    def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        return k_sq*0.0+1.0


def allocate_grids(n:int, dtype:torch.dtype=torch.float32, device:torch.device=torch.device("cuda"), requires_grad:bool=True):    
    test_kernel = TestKernel()
    mesh = torch.ones(size=[2, n, n, n], dtype=dtype, device=device).requires_grad_(requires_grad)
    filter = KSpaceFilter(ns_mesh = torch.tensor([n, n, n], dtype=torch.int32, device=device),
                    cell=torch.eye(3, dtype=dtype, device=device).requires_grad_(requires_grad),
                    kernel=test_kernel 
                    )
    return mesh, filter


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for mesh and run configurations.")
    
    parser.add_argument('--n_mesh', type=int, default=1024, help='Number of mesh points (default: 1024)')
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


def run_benchmark(n_mesh:int, n_runs:int, device:torch.device, dtype:torch.dtype, backward:bool=False, jit:bool=False):

    mesh, kernel = allocate_grids(n=n_mesh, device=device, dtype=dtype, requires_grad=backward)
        
    class RunModule(torch.nn.Module):
        def __init__(self, filter: KSpaceFilter):
            super().__init__()
            self.kernel:KSpaceFilter = filter
    
        def forward(self, mesh:torch.Tensor, backward:bool):
            filter_mesh = self.kernel.compute(mesh)
            if backward:
                mesh_sum = filter_mesh.sum()
                mesh_sum.backward()
            else:
                mesh_sum = torch.tensor(0.0)
            return filter_mesh, mesh_sum
    
    run_module = RunModule(kernel)
    if jit:
        run_module = torch.jit.script(run_module)

    filter_mesh = run_module(mesh, backward)
    if device=="cuda":
        torch.cuda.synchronize()

    timings = []
    for i in range(n_runs):
        if backward:
            mesh.grad.zero_()
        start = time()
        run_module(mesh, backward)
        if device==torch.device("cuda"):
            torch.cuda.synchronize()
        end = time()
        timings.append(end-start)
    
    return timings


args = parse_args()

print(f"""
Running tests with n_mesh={args.n_mesh}, n_runs={args.n_runs}, device={args.device}, dtype={args.dtype}, backward={args.backward}, jit={args.jit}
""")

timings = run_benchmark(args.n_mesh, args.n_runs, args.device, args.dtype, args.backward, args.jit)

print(f"""
Average timing: {np.mean(timings)}  ,  std= {np.std(timings)}
""")