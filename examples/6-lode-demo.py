"""
.. _example-lode-demo:

Computing LODE descriptors
==========================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This notebook demonstrates the use of some advanced features of ``torch-pme``
to compute long-distance equivariant features, cf. GRISAFI PAPER
"""

# %%
%reload_ext autoreload
%autoreload 2

from time import time

import ase
import chemiscope
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib

import torchpme

device = "cpu"
dtype = torch.float64
matplotlib.use("widget")

# %%

x_grid = torch.linspace(0,10,100)
y_grid = torch.exp(-x_grid**2/2)

myspline = torchpme.lib.splines.CubicSpline(x_grid, y_grid)

fig, ax = plt.subplots(
    1, 1, figsize=(6, 4), sharey=True, sharex=True, constrained_layout=True
)
ax.plot(x_grid, myspline(x_grid), 'b-')
ax.plot(x_grid, y_grid, 'r:')

# %%

k64 = torch.linspace(0,10,400, dtype=torch.float64)
x64 = myspline.x_points.to(dtype=torch.float64)
y64 = myspline.y_points.to(dtype=torch.float64)
d2y64 = myspline.d2y_points.to(dtype=torch.float64)
krn64=[ torchpme.lib.splines.compute_spline_ft(k, x64, y64, d2y64)
     for k in k64]

k32 = torch.linspace(0.2,10,400, dtype=torch.float32)
x32 = myspline.x_points.to(dtype=torch.float32)
y32 = myspline.y_points.to(dtype=torch.float32)
d2y32 = myspline.d2y_points.to(dtype=torch.float32)
krn32=[ torchpme.lib.splines.compute_spline_ft(k, x32, y32, d2y32)
     for k in k32]
krn32_stable=[ compute_spline_ft32(k, x32, y32, d2y32)
     for k in k32]
#krn32_stable=[ compute_spline_ft32(k, x64, y64, d2y64) for k in k64]

fig, ax = plt.subplots(
    1, 1, figsize=(6, 4), sharey=True, sharex=True, constrained_layout=True
)
ax.plot(k32, krn32, 'g-')
ax.plot(k64, krn64, 'y-')
ax.plot(k32, krn32_stable, 'w--')
ax.plot(k64, 2*np.sqrt(2)*torch.exp(-0.5*k64**2)*np.pi**(3./2.),
         'r:')

# %%
# Coulomb potential

mypot = torchpme.CoulombPotential(smearing=2)

x_grid = torch.logspace(-3,2,100)
y_grid = mypot.lr_from_dist(x_grid)

myspline = torchpme.lib.splines.CubicSpline(x_grid, y_grid)

fig, ax = plt.subplots(
    1, 1, figsize=(6, 4), sharey=True, sharex=True, constrained_layout=True
)
ax.plot(x_grid, myspline(x_grid), 'b-')
ax.plot(x_grid, y_grid, 'r:')


# %%

k64 = torch.linspace(0,10,400, dtype=torch.float64)
x64 = myspline.x_points.to(dtype=torch.float64)
y64 = myspline.y_points.to(dtype=torch.float64)
d2y64 = myspline.d2y_points.to(dtype=torch.float64)
krn64=[ torchpme.lib.splines.compute_spline_ft(k, x64, y64, d2y64)
     for k in k64]

k32 = torch.linspace(0.2,10,400, dtype=torch.float32)
x32 = myspline.x_points.to(dtype=torch.float32)
y32 = myspline.y_points.to(dtype=torch.float32)
d2y32 = myspline.d2y_points.to(dtype=torch.float32)
krn32=[ torchpme.lib.splines.compute_spline_ft(k, x32, y32, d2y32)
     for k in k32]
krn32_stable=compute_spline_ft32(k32, x32, y32, d2y32)

fig, ax = plt.subplots(
    1, 1, figsize=(6, 4), sharey=True, sharex=True, constrained_layout=True
)
ax.loglog(k32, krn32, 'g-')
ax.plot(k64, krn64, 'y-')
ax.plot(k32, krn32_stable, 'w--')


ax.plot(k64, mypot.kernel_from_k_sq(k64**2), 'r:')

ax.set_ylim(1e-4,1e5)
# %%
krn32_stable

# %%
from torchpme.lib.splines import compute_second_derivatives
def compute_spline_ft32(k_points, x_points, y_points, d2y_points):
    r"""
    Computes the Fourier transform of a splined radial function.

    Evaluates the integral

    .. math::
        \hat{f}(k) =4\pi\int \mathrm{d}r \frac{\sin k r}{k} r f(r)
    
    where :math:`f(r)` is expressed as a cubic spline. The function
    also includes a tail correction to continue the integral beyond
    the last splined point, assuming that the function converges to 
    zero at infinity. 

    :param k_points:  Points on which the Fourier kernel should be 
        computed. It is a good idea to take them to be 
        :math:`2\pi/x` based on the real-space ``x_points``
    :param x_points:  Abscissas of the splining points for the real-space function
    :param y_points:  Ordinates of the splining points for the real-space function
    :param d2y_points:  Second derivatives for the spline points

    :return: The radial Fourier transform :math:`\hat{f}(k)` computed
        at the ``k_points`` provided.
    """

    # broadcast to compute at once on all k values.
    # all these are terms that enter the analytical integral.
    # might be possible to write this in a more concise way, but
    # this works and is reasonably numerically stable, so it will do
    k = k_points.reshape(-1,1)
    ri = x_points[torch.newaxis,:-1]
    yi = y_points[torch.newaxis,:-1]
    d2yi = d2y_points[torch.newaxis,:-1]
    dr = x_points[torch.newaxis,1:] - x_points[torch.newaxis,:-1] 
    dy = y_points[torch.newaxis,1:] - y_points[torch.newaxis,:-1] 
    dd2y = d2y_points[torch.newaxis,1:] - d2y_points[torch.newaxis,:-1] 
    coskx = torch.cos(k*ri)
    sinkx = torch.sin(k*ri)    
    #cos r+dr - cos r
    dcoskx = 2*torch.sin(k*dr/2)*torch.sin(k*(dr/2+ri))
    #sin r+dr - cos r
    dsinkx = -2*torch.sin(k*dr/2)*torch.cos(k*(dr/2+ri))

    # this monstruous expression computes, for each interval in the spline,
    # \int_{r_i}^{r_{i+1}} .... using the coefficients of the spline.
    # the expression here is also cast in a Horner form, and uses a few 
    # tricks to make it stabler, as a naive implementation is very noisy
    # in float32 for small k. for instance, the first term contains the difference 
    # of two cosines, but is computed with a trigonometric identity 
    # (see the definition of dcoskx) to avoid the 1-k^2 form of the bare cosines
    spline_integrals = 24*dcoskx*dd2y + 6*k*(dsinkx*(3*d2yi*dr + dd2y*(4*dr + ri)) - 4*dd2y*dr*sinkx + 
        k*(6*coskx*dr*(3*d2yi*dr + dd2y*(2*dr + ri)) - 
        2*dcoskx*(6*dy + dr*((6*d2yi + 5*dd2y)*dr + 3*(d2yi + dd2y)*ri)) + 
        k*(dr*(12*dy + 3*d2yi*dr*(dr + 2*ri) + dd2y*dr*(2*dr + 3*ri))*sinkx + 
        dsinkx*(-6*dy*ri - 3*d2yi*dr**2*(dr + ri) - 2*dd2y*dr**2*(dr + ri) - 
        6*dr*(2*dy + yi)) + k*
        (6*dcoskx*dr*(dr + ri)*(dy + yi) + coskx*(6*dr*ri*yi - 6*dr*(dr + ri)*(dy + yi))))))
    
    # especially for Coulomb-like integrals, no matter how far we push the splining
    # in real space, the tail matters, so we compute it separately. to do this
    # stably and acurately, we build the tail as a spline in 1/r (using the last two)
    # points of the spline) and use an analytical expression for the resulting
    # integral from the last point to infinity
    tail_d2y = compute_second_derivatives(
        torch.tensor([0, 1/x_points[-1], 1/x_points[-2]]),
        torch.tensor([0, y_points[-1], y_points[-2]])
        )

    r0 = x_points[-1]
    y0 = y_points[-1]
    d2y0 = tail_d2y[1] 
    # the expression contains the cosine integral special function, that
    # is only available in scipy
    try:
        from scipy.special import sici
    except ImportError:
        raise ImportError("Computing the Fourier-domain kernel based on a spline requires scipy")    
    tail = (-2*torch.pi*((d2y0 - 6*r0**2*y0)*torch.cos(k*r0) + 
                         d2y0*k*r0*(k*r0*sici(k*r0)[1] - torch.sin(k*r0))))/(3.*k**2*r0)

    ft = 2*torch.pi/3*torch.sum(spline_integrals/dr,axis=1).reshape(-1,1)/k**6+tail
    return ft.reshape(k_points.shape)

# %%
test = compute_spline_ft32(k64, x64, y64, d2y64)
test.shape
# %%
# Demonstrates the application of a k-space filter
# ------------------------------------------------
#
# We define a fairly rugged function on a mesh, and apply a
# smoothening filter. We start creating a grid
# (we use a :py:class:`MeshInterpolator` object for
# simplicity to generate the grid with the right shape) and computing a
# sharp Gaussian field in the :math:`xy` plane.

cell = torch.eye(3, dtype=dtype, device=device) * 6.0
ns_mesh = torch.tensor([9, 9, 9])
interpolator = torchpme.lib.MeshInterpolator(cell, ns_mesh, interpolation_nodes=2)
xyz_mesh = interpolator.get_mesh_xyz()

mesh_value = (
    np.exp(-4 * ((cell[0, 0] / 2 - xyz_mesh) ** 2)[..., :2].sum(axis=-1))
).reshape(1, *xyz_mesh.shape[:-1])

# %%



# %%
# To define and apply a Gaussian smearing filter,
# we first define the convolution kernel that must be applied
# in the Fourier domain, and then use it as a parameter of the
# filter class. The application of the filter requires simply
# a call to :py:func:`KSpaceKernel.compute`.


# This is the filter function. NB it is applied
# to the *squared k vector norm*
class GaussianSmearingKernel(torchpme.lib.KSpaceKernel):
    def __init__(self, sigma2: float):
        self._sigma2 = sigma2

    def kernel_from_k_sq(self, k2):
        return torch.exp(-k2 * self._sigma2 * 0.5)


# This is the filter
KF = torchpme.lib.KSpaceFilter(cell, ns_mesh, kernel=GaussianSmearingKernel(sigma2=1.0))

# Apply the filter to the mesh
mesh_filtered = KF.compute(mesh_value)

# %%
# Visualize the effect of the filter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The Gaussian smearing enacted by the filter can be
# easily visualized taking a slice of the mesh (we make
# it periodic for a cleaner view)

fig, ax = plt.subplots(
    1, 2, figsize=(8, 4), sharey=True, sharex=True, constrained_layout=True
)
mesh_extent = [
    0,
    cell[0, 0],
    0,
    cell[1, 1],
]

z_plot = mesh_value[0, :, :, 0].detach().numpy()
z_plot = np.vstack([z_plot, z_plot[0, :]])  # Add first row at the bottom
z_plot = np.hstack(
    [z_plot, z_plot[:, 0].reshape(-1, 1)]
)  # Add first column at the right

z_min, z_max = (z_plot.min(), z_plot.max())

cf = ax[0].imshow(
    z_plot,
    extent=mesh_extent,
    vmin=z_min,
    vmax=z_max,
    origin="lower",
    interpolation="bilinear",
)

z_plot = mesh_filtered[0, :, :, 0].detach().numpy()
z_plot = np.vstack([z_plot, z_plot[0, :]])  # Add first row at the bottom
z_plot = np.hstack(
    [z_plot, z_plot[:, 0].reshape(-1, 1)]
)  # Add first column at the right

cf_fine = ax[1].imshow(
    z_plot,
    extent=mesh_extent,
    vmin=z_min,
    vmax=z_max,
    origin="lower",
    interpolation="bilinear",
)
ax[0].set_xlabel("x / Å")
ax[1].set_xlabel("x / Å")
ax[0].set_ylabel("y / Å")
ax[0].set_title(r"input values")
ax[1].set_title(r"filtered")
fig.colorbar(cf_fine, label=r"density / e/Å$^3$")
fig.show()

# %%
# We can also show the filter in action in 3D, by creating
# a grid of dummy atoms corresponding to the mesh
# points, and colored according to the function
# value. Use the `chemiscope` option panel to
# switch between the sharp input and the filtered
# values.

dummy = ase.Atoms(
    positions=xyz_mesh.reshape(-1, 3),
    symbols="H" * len(xyz_mesh.reshape(-1, 3)),
    cell=cell,
)
chemiscope.show(
    frames=[dummy],
    properties={
        "input value": {
            "target": "atom",
            "values": mesh_value.detach().numpy().flatten(),
        },
        "filtered value": {
            "target": "atom",
            "values": mesh_filtered.detach().numpy().flatten(),
        },
    },
    mode="structure",
    settings=chemiscope.quick_settings(
        structure_settings={
            "unitCell": True,
            "bonds": False,
            "environments": {"activated": False},
            "color": {
                "property": "input value",
                "transform": "linear",
                "palette": "viridis",
            },
        }
    ),
    environments=chemiscope.all_atomic_environments([dummy]),
)

# %%
# Adjustable and multi-channel filters
# ------------------------------------
# :py:class:`KSpaceFilter <torchpme.lib.KSpaceFilter>` can
# also be applied for more complicated use cases. For
# instance, one can apply multiple filters to multiple
# real-space mesh channels, and use a
# :py:class:`torch.nn.Module`-derived class to define an
# adjustable kernel.

# %%
# We initialize a three-channel mesh, with identical
# patterns along the three Cartesian directions

multi_mesh = torch.stack(
    [
        np.exp(-4 * ((cell[0, 0] / 2 - xyz_mesh) ** 2)[..., :2].sum(axis=-1)),
        np.exp(-4 * ((cell[0, 0] / 2 - xyz_mesh) ** 2)[..., [0, 2]].sum(axis=-1)),
        np.exp(-4 * ((cell[0, 0] / 2 - xyz_mesh) ** 2)[..., 1:].sum(axis=-1)),
    ]
)

# %%
# We also define a filter with three smearing parameters, corresponding to
# three channels


class MultiKernel(torchpme.lib.KSpaceKernel):
    def __init__(self, sigma: torch.Tensor):
        super().__init__()
        self._sigma = sigma

    def kernel_from_k_sq(self, k2):
        return torch.stack([torch.exp(-k2 * s**2 / 2) for s in self._sigma])


# %%
# This can be used just as a simple filter

multi_kernel = MultiKernel(torch.tensor([0.25, 0.5, 1.0], dtype=dtype, device=device))
multi_KF = torchpme.lib.KSpaceFilter(cell, ns_mesh, kernel=multi_kernel)
multi_filtered = multi_KF.compute(multi_mesh)

# %%
# When the parameters of the kernel are modified, it is sufficient
# to call :py:func:`KSpaceFilter.update_filter` before applying the
# filter

multi_kernel._sigma = torch.tensor([1.0, 0.5, 0.25])
multi_KF.update_filter()
multi_filtered_2 = multi_KF.compute(multi_mesh)

# NB: when one needs to perform a full update, including the
# cell, it is possible to call the ``forward`` function of the
# class

multi_filtered_3 = multi_KF(cell, multi_mesh)

# %%
# Visualize the application of the filters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The first colum shows the input function (the
# three channels contain the same function as in
# the previous example, but with different orientations,
# which explains the different appearence when sliced
# along the :math:`xy` plane).
#
# The second and third columns show the same three
# channels with the Gaussian smearings defined above.

fig, ax = plt.subplots(
    3, 3, figsize=(9, 9), sharey=True, sharex=True, constrained_layout=True
)

# reuse the same mesh_extent given the mesh is cubic
mesh_extent = [
    0,
    cell[0, 0],
    0,
    cell[1, 1],
]

z_min, z_max = (mesh_value.flatten().min(), mesh_value.flatten().max())
cfs = []
for j, mesh_value in enumerate([multi_mesh, multi_filtered, multi_filtered_2]):
    for i in range(3):
        z_plot = mesh_value[i, :, :, 4].detach().numpy()
        z_plot = np.vstack([z_plot, z_plot[0, :]])  # Add first row at the bottom
        z_plot = np.hstack(
            [z_plot, z_plot[:, 0].reshape(-1, 1)]
        )  # Add first column at the right

        cf = ax[i, j].imshow(
            z_plot,
            extent=mesh_extent,
            vmin=z_min,
            vmax=z_max,
            origin="lower",
            interpolation="bilinear",
        )
        cfs.append(cf)
    ax[j, 0].set_ylabel("y / Å")
    ax[2, j].set_xlabel("x / Å")

ax[0, 0].set_title("input values")
ax[0, 1].set_title("filter 1")
ax[0, 2].set_title("filter 2")
cbar_ax = fig.add_axes(
    [1.0, 0.2, 0.03, 0.6]
)  # [left, bottom, width, height] in figure coordinates
# Add the colorbar to the new axes
fig.colorbar(cfs[0], cax=cbar_ax, orientation="vertical")

# %%
# Jit-ting of the k-space filter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The k-space filter can also be compiled to torch-script, for
# faster execution (the impact is marginal for this very simple case)

multi_filtered = multi_KF.compute(multi_mesh)
start = time()
for _i in range(100):
    multi_filtered = multi_KF.compute(multi_mesh)
time_python = (time() - start) * 1e6 / 100

jitted_KF = torch.jit.script(multi_KF)
jit_filtered = jitted_KF.compute(multi_mesh)
start = time()
for _i in range(100):
    jit_filtered = jitted_KF.compute(multi_mesh)
time_jit = (time() - start) * 1e6 / 100

print(f"Evaluation time:\nPytorch: {time_python}µs\nJitted:  {time_jit}µs")
