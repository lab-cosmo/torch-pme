
# %%
# init 
import torch
import matplotlib.pyplot as plt
import vesin.torch
import torchpme.lib.potentials as potentials
import torchpme.lib.kspace_filter as kspace_filter
import torchpme.calculators as calculators
from torchpme.metatensor import PMECalculator as MetaCalculator
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import System

dtype=torch.float64
device="cpu"

cell = torch.eye(3, dtype=dtype, device=device) * 20.0
positions = torch.tensor([[1,0,0],[-1.,0,0]], dtype=dtype, device=device)
charges = torch.tensor([[1],[-1.]], dtype=dtype, device=device)
types = torch.tensor([55, 17])

nl = vesin.torch.NeighborList(cutoff=5.0, full_list=False)
i, j, S, D, neighbor_distances = nl.compute(
    points=positions, box=cell, periodic=True, quantities="ijSDd"
)
neighbor_indices = torch.stack([i, j], dim=1)

system = System(types=types, positions=positions, cell=cell)
data = TensorBlock(
    values=charges,
    samples=Labels.range("atom", charges.shape[0]),
    components=[],
    properties=Labels.range("charge", charges.shape[1]),
)
system.add_data(name="charges", data=data)

sample_values = torch.hstack([neighbor_indices, S])
samples = Labels(
    names=[
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ],
    values=sample_values,
)
components = Labels(names=["xyz"], values=torch.tensor([[0, 1, 2]]).T)
properties = Labels(names=["distance"], values=torch.tensor([[0]]))
neighbors = TensorBlock(D.reshape(-1, 3, 1), samples, [components], properties)

do_jit = False
torch._dynamo.config.capture_scalar_outputs = True
def jit(obj):
    return torch.jit.script(obj) if do_jit else obj
    #return torch.compile(obj, fullgraph=True) if do_jit else obj

# %%
# Metatensor

mymeta = MetaCalculator(
    potential=potentials.InversePowerLawPotential(exponent=1.0, range_radius=1.5, cutoff_radius=None)
)
pots = mymeta.forward(system, neighbors)

print(f"Here come the pots (MTT) {pots.block(0).values}")

# %%
# Direct calculators

mycalc = jit(calculators.base.Calculator(
    potential=potentials.InversePowerLawPotential(exponent=1.0, range_radius=None, cutoff_radius=None)
))

pots = mycalc(charges=charges, cell=cell, positions=positions,
               neighbor_distances=neighbor_distances, 
               neighbor_indices=neighbor_indices)

print(f"Here come the pots (Direct) {pots}")


# %%
# PME calculators

mycalc = jit(calculators.pme.PMECalculator(
    potential=potentials.InversePowerLawPotential(exponent=1.0, range_radius=1.5, cutoff_radius=None)
))

pots = mycalc(charges=charges, cell=cell, positions=positions,
               neighbor_distances=neighbor_distances, 
               neighbor_indices=neighbor_indices)

print(f"Here come the pots (PME) {pots}")

# %%
# Ewald calculators

mycalc = jit(calculators.ewald.EwaldCalculator(
    potentials=potentials.InversePowerLawPotential(exponent=1.0, range_radius=1.5, cutoff_radius=None)
))

pots = mycalc(charges=charges, cell=cell, positions=positions,
               neighbor_distances=neighbor_distances, 
               neighbor_indices=neighbor_indices)

print(f"Here come the pots (EWALD) {pots}")


# %%
#  No cutoff

lrpot = jit(potentials.InversePowerLawPotential(exponent=1.0, range_radius=1.5, cutoff_radius=None))

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

lrpot = jit(potentials.InversePowerLawPotential(exponent=1.0, range_radius=1.5, cutoff_radius=4))

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
# KSpace filters from potentials
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
