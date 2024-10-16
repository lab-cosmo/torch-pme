import torch
import numpy as np
import argparse
from time import time

def allocate_grids(n:int, dtype:torch.dtype=torch.float32, device:torch.device=torch.device("cuda"), requires_grad:bool=True):    
    return (
            torch.ones(size=[2, n, n, n], dtype=dtype, device=device).requires_grad_(requires_grad),
            torch.ones(size=[n, n, n//2+1], dtype=dtype, device=device).requires_grad_(requires_grad)
            )

def run_fftn(mesh_values, kernel):

    mesh_hat = torch.fft.rfftn(mesh_values, dim=[1,2,3])

    filter_hat = mesh_hat * kernel

    filter_mesh = torch.fft.irfftn(
        filter_hat,
        dim=(1,2,3),
        # NB: we must specify the size of the output
        # as for certain mesh sizes the inverse FT is not
        # well-defined
        s=mesh_values.shape[-3:],
    )    

    return filter_mesh

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
    
    filter_mesh = run_fftn(mesh, kernel)
    if device=="cuda":
        torch.cuda.synchronize()

    def run(mesh, kernel, backward:bool):
        filter_mesh = run_fftn(mesh, kernel)
        if backward:
            mesh_sum = filter_mesh.sum()
            mesh_sum.backward()
        else:
            mesh_sum = torch.tensor(0.0)
        return filter_mesh, mesh_sum
    
    if jit:
        run = torch.jit.script(run)

    timings = []
    for i in range(n_runs):
        if backward:
            mesh.grad.zero_()
            kernel.grad.zero_()
        start = time()
        run(mesh, kernel, backward)
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