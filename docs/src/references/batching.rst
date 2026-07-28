.. _batched-tiling:

Batched evaluation with tiling
##############################

All calculators evaluate one system per :meth:`forward <torchpme.Calculator.forward>`
call. For training on datasets of many small, differently sized systems this is
inefficient: each call launches the same sequence of small kernels, and the natural
alternative — padding every system to a common size and using ``torch.vmap`` — wastes
compute and memory on padding, since every system is padded to the largest atom count
*and* the largest k-vector count in the batch.

The *tiled* batched evaluation, available for :class:`torchpme.EwaldCalculator`
through :meth:`forward_batched <torchpme.EwaldCalculator.forward_batched>`, avoids
both problems. It rests on two ideas, described below: a single parameter
:math:`\mathrm{num\_k}` that balances real- and reciprocal-space work for every system
at once, and a block-diagonal tiled evaluation of the reciprocal-space sum that never
materializes a padded rectangle.

The knob: ``num_k``
===================

The Ewald method splits the interaction into a short-range part, summed over a
neighbor list truncated at a cutoff :math:`r_c`, and a long-range part, summed over
reciprocal-space vectors. Both are controlled by the smearing :math:`\sigma`: a
smaller :math:`\sigma` shifts work from real space (smaller :math:`r_c`) to
reciprocal space (more k-vectors), and vice versa.

In the batched evaluation you do not pick :math:`\sigma` or :math:`r_c` per system.
Instead you pick one **target half-space k-vector count** :math:`\mathrm{num\_k}`,
shared by every system. Inverting the k-count of a cell with basis-vector lengths
:math:`L_1, L_2, L_3` gives a per-system reciprocal resolution, from which the
smearing and the real-space cutoff follow:

.. math::

    \lambda_b = \left(\frac{L_1 L_2 L_3}{2\,\mathrm{num\_k}}\right)^{1/3},
    \qquad
    \sigma_b = c_\sigma \lambda_b,
    \qquad
    r_c(b) = c_r \sigma_b

The two factors default to :math:`c_\sigma = 2` and :math:`c_r = 5`, conservative rules
of thumb that can be adjusted through the ``smearing_factor`` and ``cutoff_factor``
arguments: :math:`c_\sigma` controls how well the k-grid resolves the smeared charges
(the reciprocal-space error), :math:`c_r` the real-space truncation error
:math:`\propto \mathrm{erfc}(c_r/\sqrt{2})`. The default :math:`c_r = 5` is the same
ratio :class:`torchpme.EwaldCalculator` recommends for a single system (a smearing of
one fifth of the neighbor-list cutoff). Since ``smearing_factor`` determines both
the cutoff your neighbor lists are built at and the smearing the evaluation splits at,
the same value must be given to :func:`torchpme.lib.ewald_params_from_num_k` and to
:func:`torchpme.lib.prepare_tiled_batch`. For accuracy-driven parameters of a *single*
structure, use :func:`torchpme.tuning.tune_ewald` instead.

Every system thus realizes *roughly* :math:`\mathrm{num\_k}` k-vectors — the k-axis
of the batch has an (almost) fixed size, independent of how large the largest cell is
— while its real-space cutoff grows with the cell so that the overall accuracy stays
balanced. Increasing :math:`\mathrm{num\_k}` uniformly moves work from the neighbor
list to the reciprocal sum.

:func:`torchpme.lib.ewald_params_from_num_k` computes
:math:`(\lambda_b, \sigma_b, r_c(b))` from a cell (or a batch of cells). The
**neighbor list of each system must be built at its own** :math:`r_c(b)` (or any
larger radius) *before* collating the batch; since :math:`r_c(b)` only depends on the
cell and :math:`\mathrm{num\_k}`, this can be done once per dataset.

Block-diagonal tiled summation
==============================

The reciprocal-space part of the Ewald sum is, per system, an outer product between
atoms and k-vectors. Across a batch this work matrix is **block-diagonal**: atoms of
one system never interact with k-vectors of another. The tiled evaluation exploits
this directly:

- **Atoms** are laid out in one flat, concatenated array; each system's atom count is
  rounded up to a multiple of the atom-tile size ``block_atoms`` ("sum-padding" — the
  total padding is at most ``block_atoms`` per system, unlike rectangular padding to
  the largest system).
- **K-vectors** form a rectangular array of shape ``[n_systems, K_pad, 3]``, where
  ``K_pad`` is the common padded k-count (a multiple of the k-tile size
  ``block_kvecs``). Padded rows are zero vectors.
- A host-built **dispatch table** enumerates only the on-diagonal
  ``(system, atom_tile, k_tile)`` work blocks. The kernel runs two passes over these
  tiles: pass one accumulates each system's structure factors, pass two contracts
  them back into per-atom potentials. The dense
  ``[n_systems, max_atoms, max_kvectors]`` array of a padded batch is never formed.

Padding is inert by construction: padded atoms carry zero charge, and padded k-rows
are zero vectors, which the potentials map to a zero reciprocal-space kernel.

.. note::

    The tiled evaluation removes the *heterogeneity* waste (padding to the largest
    system). Its peak memory is still the block-diagonal product
    :math:`O(N_\mathrm{tot}\,K_\mathrm{pad})` — the phase matrix of all on-diagonal
    tiles — in both the forward and the backward pass.

Shape stability
---------------

Because the per-axis k-grid extents are rounded up, each system realizes slightly
more than :math:`\mathrm{num\_k}` k-vectors. The k-axis is therefore padded to a
window of ``(1 + k_pad_fraction) * num_k`` (default 10 %), so that every batch
prepared with the same ``num_k`` gets the *same* ``K_pad`` — useful under
``torch.compile``, where stable shapes avoid recompilation. If a cell overshoots the
window, the batch is still valid (``K_pad`` grows), and a warning suggests increasing
``k_pad_fraction``.

Mixed periodicities
===================

A single tiled batch may mix:

- **3D-periodic** systems — the standard Ewald evaluation.
- **2D-periodic slabs** — the vacuum axis is internally replaced by an *effective*
  shrunk axis of length ``thickness + 1.5 * L_max`` (see
  :func:`torchpme.lib.shrink_2d_cell`; a zero vacuum row, as used by ``metatomic``
  for non-periodic directions, is supported), which keeps the derived parameters and
  k-grid physical, and a slab (Yeh–Berkowitz) correction is applied. The slab
  correction is implemented for :class:`torchpme.CoulombPotential` (and the
  equivalent :class:`torchpme.InversePowerLawPotential` with ``exponent=1``); for
  other potentials the atoms of 2D systems evaluate to ``NaN`` — loudly, rather than
  silently wrong.
- **Non-periodic clusters** — no reciprocal-space part at all; their pair list is
  summed with the *bare* (non-range-separated) potential.

1D-periodic systems are not supported and are rejected at collation time.

Usage: pure torch
=================

The evaluation is split into three steps:

#. Build the neighbor list of each system at the cutoff its cell implies for the
   chosen ``num_k``, with :func:`torchpme.lib.ewald_params_from_num_k`.
#. Collate the systems with :func:`torchpme.lib.prepare_tiled_batch`, which returns a
   ``batch`` dictionary holding the concatenated inputs and a ``tiling`` dictionary
   holding the static index tensors. This step runs on the host, once per batch, and
   nothing in it is differentiable.

   Because it decides the *shapes* of the tiling — the k-grid extents, the padded
   k-count, the length of the flat atom layout — it needs a handful of values as host
   integers, and copies the per-system metadata from the device in one go: a batch of
   3D-periodic and non-periodic systems synchronizes once, however many systems it
   holds. Only 2D slabs add to that, through the scalars
   :func:`torchpme.lib.shrink_2d_cell` reads back per slab. Collating on CPU tensors —
   in a dataloader worker, say — and moving the batch to the device afterwards avoids
   the transfers altogether.
#. Evaluate the batch with :meth:`forward_batched
   <torchpme.EwaldCalculator.forward_batched>`, passing the entries of ``batch``
   together with ``tiling``. The per-system smearing derived from ``num_k`` overrides
   the ``smearing`` of the calculator's potential.

Both dictionaries come back on the device of the inputs. The result is the
concatenation of the per-atom potentials; split it per system with
``batch["system_index"]``. For **forces**, recompute the pair distances inside the
computational graph (from positions, neighbor indices and cell shifts) and pass those
instead of ``batch["neighbor_distances"]`` — a precomputed distance array is a
constant to autograd.

The docstring of :func:`torchpme.lib.prepare_tiled_batch` shows a minimal collate and
evaluate; for a runnable end-to-end version with real neighbor lists and forces, see
the :ref:`batched example <sphx_glr_examples_13-batched-ewald.py>`.

Usage: metatensor interface
===========================

The :ref:`metatensor bindings <metatensor>` mirror the same split.
:func:`torchpme.metatensor.prepare_tiled_batch` builds the tiling directly from
:class:`metatomic.torch.System` objects (with attached ``"charge"`` data) and their
neighbor-list :class:`TensorBlock <metatensor.torch.TensorBlock>` objects, and
:meth:`forward_batched <torchpme.metatensor.Calculator.forward_batched>` evaluates
the batch, returning a :class:`metatensor.torch.TensorMap` whose samples label each
atom by ``("system", "atom")``. Since the systems already carry their cells,
periodicities and charges, the collation takes only the systems, their neighbor blocks
and ``num_k``, and there is no separate ``batch`` dictionary to unpack — the second
half of the :ref:`batched example <sphx_glr_examples_13-batched-ewald.py>` runs the
same systems through this interface.

Pair distances are recomputed internally from the neighbor blocks' distance
*vectors*, so forces flow to the positions those vectors were computed from. Both the
calculator (including ``forward_batched``) and the two ``prepare_tiled_batch``
front-ends are TorchScript-compatible.

Practical notes
===============

- Only :class:`torchpme.EwaldCalculator` implements the tiled evaluation; the
  mesh-based calculators (:class:`torchpme.PMECalculator`,
  :class:`torchpme.P3MCalculator`) raise :class:`NotImplementedError`.
- ``float64`` is recommended, as for all Ewald evaluations.
- On CUDA, the scatter reductions (``index_add_``) used by the kernel are
  non-deterministic by default; call ``torch.use_deterministic_algorithms(True)``
  for bit-reproducible results and gradients.
- The physical result is independent of the tile sizes ``block_atoms`` /
  ``block_kvecs`` (up to floating-point summation order); they only affect
  performance.
- The tiling depends only on cells, periodicities, atom counts and pair counts — it
  must be rebuilt when those change (e.g. a new batch), not when positions move. The
  one exception is a 2D slab, whose vacuum-shrunk cell follows the extent of its atoms
  along the vacuum axis.
- Nothing in the tiling carries a gradient, smearings included: it is static data, and
  the smearing is a convergence parameter rather than a function of the geometry.
  Gradients flow through the ``batch`` entries, which are the caller's own tensors.

API reference
=============

.. autofunction:: torchpme.lib.ewald_params_from_num_k

.. autofunction:: torchpme.lib.shrink_2d_cell

.. autofunction:: torchpme.lib.prepare_tiled_batch

.. autofunction:: torchpme.metatensor.prepare_tiled_batch

The evaluation entry points are documented with their calculators:
:meth:`torchpme.EwaldCalculator.forward_batched` and
:meth:`torchpme.metatensor.Calculator.forward_batched`.
