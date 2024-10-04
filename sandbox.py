
# %%
# init 
import torchpme.lib.potential as potential
import torchpme.lib.kspace_filter as kspace_filter
import torchpme.calculators as calculators
import torch
import matplotlib.pyplot as plt
import vesin.torch

dtype=torch.float64
device="cpu"

cell = torch.eye(3, dtype=dtype, device=device) * 20.0
positions = torch.tensor([[1,0,0],[-1.,0,0]], dtype=dtype, device=device)
charges = torch.tensor([[1],[-1.]], dtype=dtype, device=device)

nl = vesin.torch.NeighborList(cutoff=5.0, full_list=False)
i, j, S, D, neighbor_distances = nl.compute(
    points=positions, box=cell, periodic=True, quantities="ijSDd"
)
neighbor_indices = torch.stack([i, j], dim=1)

do_jit = False
def jit(obj):
    return torch.jit.script(obj) if do_jit else obj


# %%
# Define calculators

mycalc = jit(calculators.calculatorpme.PMECalculator(
    potential=potential.InversePowerLawPotential(exponent=1.0, range_radius=1.5, cutoff_radius=None)
))

pots = mycalc(charges=charges, cell=cell, positions=positions,
               neighbor_distances=neighbor_distances, 
               neighbor_indices=neighbor_indices)

print(f"Here come the pots {pots}")

# %%
#  No cutoff

lrpot = jit(potential.InversePowerLawPotential(exponent=1.0, range_radius=1.5, cutoff_radius=None))

dist = torch.linspace(1,10,50)

full_v = lrpot.from_dist(dist)
lr_v = lrpot.lr_from_dist(dist)
sr_v = lrpot.sr_from_dist(dist)

plt.plot(dist, full_v, "gray", label="V")
plt.plot(dist, lr_v, "r:", label="V_LR")
plt.plot(dist, sr_v, "b:", label="V_SR")
plt.plot(dist, sr_v+lr_v, "k:", label="V_LR+V_SR")
plt.legend()
plt.show()
# %%
# cutoff

lrpot = jit(potential.InversePowerLawPotential(exponent=1.0, range_radius=1.5, cutoff_radius=4))

dist = torch.linspace(1,10,50)

full_v = lrpot.from_dist(dist)
lr_v = lrpot.lr_from_dist(dist)
sr_v = lrpot.sr_from_dist(dist)

plt.plot(dist, full_v, "gray", label="V")
plt.plot(dist, lr_v, "r:", label="V_LR")
plt.plot(dist, sr_v, "b:", label="V_SR")
plt.plot(dist, sr_v+lr_v, "g:", label="V_LR+V_SR")
plt.legend()
plt.show()

# %%
# KSpace filters from filter
mesh_size = (1,3,4,5)
mesh = torch.randn(size=mesh_size)
ns_mesh = torch.tensor(mesh_size[1:])
class MyKernel(kspace_filter.KSpaceKernel):
    def kernel_from_k_sq(self, k_sq:torch.Tensor) -> torch.Tensor:
        return 1.0/(1.0+k_sq)        

mykrn = MyKernel()
myfilter = jit(kspace_filter.KSpaceFilter(
    cell=cell,
    ns_mesh=ns_mesh,
    kernel=mykrn,
))

fmesh = myfilter.compute(mesh)

# %%
# plot

fig, ax = plt.subplots(1,2)
ax[0].imshow(mesh[0,0])
ax[1].imshow(fmesh[0,0])
plt.show()

# %%
# KSpace filters from potential
myfilter = jit(kspace_filter.KSpaceFilter(
    cell=cell,
    ns_mesh=ns_mesh,
    kernel=lrpot,
))

fmesh = myfilter.compute(mesh)

# %%
# plot

fig, ax = plt.subplots(1,2)
ax[0].imshow(mesh[0,0])
ax[1].imshow(fmesh[0,0])

# %%
