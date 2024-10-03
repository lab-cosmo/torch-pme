
# %%
# init 
import torchpme.lib.potential as potential
import torch
import matplotlib.pyplot as plt

do_jit = True
def jit(obj):
    return torch.jit.script(obj) if do_jit else obj

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
