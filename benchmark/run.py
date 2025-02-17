#!/usr/bin/env python

import datetime
import os
import platform
import subprocess
from pathlib import Path
from time import monotonic
from pathlib import Path

import metatensor.torch as mt
import numpy as np
import torch
import yaml
from ase.io import read
from vesin import NeighborList

import torchpme

primitive = read(Path(__file__).parent / "geometry.in")

# -- settings --
multipliers = [4, 8]
repeats = 16

# -- setup --
devices = []
if torch.cuda.is_available():
    devices.append("cuda")

# run CUDA first!
devices.append("cpu")

systems = {}
for multiplier in multipliers:
    atoms = primitive * [multiplier, multiplier, multiplier]
    N = len(atoms)

    systems[N] = atoms

# -- hypers --
all_hypers = {}

for N, atoms in systems.items():
    cell = atoms.get_cell().array

    cell_dimensions = np.linalg.norm(cell, axis=1)
    half_cell = np.min(cell_dimensions) / 2 - 1e-6
    atomic_smearing = half_cell / 5.0

    # tight settings
    cutoff_tight = half_cell
    mesh_spacing_tight = atomic_smearing / 8.0
    lr_wavelength_tight = atomic_smearing / 2.0

    # light settings, roughly 4 times less computational costs
    cutoff_light = cutoff_tight / 2.0
    mesh_spacing_light = mesh_spacing_tight * 2.0
    lr_wavelength_light = lr_wavelength_tight * 2.0

    all_hypers[N] = {
        "tight": {
            "cutoff": cutoff_tight,
            "mesh_spacing": mesh_spacing_tight,
            "lr_wavelength": lr_wavelength_tight,
            "atomic_smearing": atomic_smearing,
        },
        "light": {
            "cutoff": cutoff_light,
            "mesh_spacing": mesh_spacing_light,
            "lr_wavelength": lr_wavelength_light,
            "atomic_smearing": atomic_smearing,
        },
    }

# -- system info --
system = {
    "platform": platform.platform(),
    "cpu": platform.processor(),
    "node": platform.node(),
}

if "cuda" in devices:
    system["gpu"] = torch.cuda.get_device_name(0)


# -- version info --
cwd = os.path.dirname(os.path.abspath(__file__))
try:
    torch_pme_commit = subprocess.check_output(
        ["git", "log", "--oneline", "-1"], cwd=cwd
    ).decode("utf-8")
    torch_pme_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=cwd
    ).decode("utf-8")
except subprocess.CalledProcessError:
    torch_pme_commit = "not found"
    torch_pme_status = "not found"

version = {
    "torch": str(torch.__version__),
    "torch-pme-commit": torch_pme_commit,
    "torch-pme-status": torch_pme_status,
    "torch-pme-version": torchpme.__version__,
}


# -- what do we have to do? --
runs = {}

for N, atoms in systems.items():
    for h, hypers in all_hypers[N].items():
        for d in devices:
            if h == "tight" and d == "cpu":
                # avoid very expensive work on CPU
                continue

            name = f"{N}_{h}_{d}"

            runs[name] = {
                "N": N,
                "device": d,
                "hypers": hypers,
                "atoms": atoms.copy(),
            }


def compute_distances(positions, neighbor_indices, cell=None, neighbor_shifts=None):
    """Compute pairwise distances."""
    atom_is = neighbor_indices[:, 0]
    atom_js = neighbor_indices[:, 1]

    pos_is = positions[atom_is]
    pos_js = positions[atom_js]

    distance_vectors = pos_js - pos_is

    if cell is not None and neighbor_shifts is not None:
        shifts = neighbor_shifts.type(cell.dtype)
        distance_vectors += shifts @ cell
    elif cell is not None and neighbor_shifts is None:
        raise ValueError("Provided `cell` but no `neighbor_shifts`.")
    elif cell is None and neighbor_shifts is not None:
        raise ValueError("Provided `neighbor_shifts` but no `cell`.")

    return torch.linalg.norm(distance_vectors, dim=1)


def atoms_to_inputs(atoms, device, cutoff):
    """Convert atoms to inputs for calculate function."""
    charges = torch.tensor([-1.0, 1.0]).type(torch.float32).to(device)
    charges = charges.repeat(len(atoms) // 2).reshape(-1, 1)
    positions = torch.from_numpy(atoms.positions).type(torch.float32).to(device)
    cell = torch.from_numpy(atoms.cell.array).type(torch.float32).to(device)

    # nl = NeighborList(cutoff=cutoff, full_list=True)
    nl = NeighborList(cutoff=cutoff, full_list=False)

    i, j, S = nl.compute(
        points=positions.cpu(), box=cell.cpu(), periodic=True, quantities="ijS"
    )

    i = torch.from_numpy(i.astype(int)).to(device)
    j = torch.from_numpy(j.astype(int)).to(device)
    neighbor_indices = torch.stack([i, j], dim=1).to(device)
    neighbor_shifts = torch.from_numpy(S.astype(int)).to(device)

    return charges, positions, cell, neighbor_indices, neighbor_shifts


def atoms_to_metatensor_inputs(atoms, device, cutoff):
    """Convert atoms to metatensor inputs for calculate functions."""
    from metatensor.torch.atomistic.ase_calculator import _compute_ase_neighbors

    system = mt.atomistic.systems_to_torch(
        atoms, device=device, positions_requires_grad=True
    )
    nl_options = mt.atomistic.NeighborListOptions(cutoff=cutoff, full_list=False)

    charges = torch.tensor([-1.0, 1.0]).type(torch.float32).to(device)
    charges = charges.repeat(len(atoms) // 2).reshape(-1, 1)
    data = mt.TensorBlock(
        values=charges,
        samples=mt.Labels.range("atom", charges.shape[0]).to(device),
        components=[],
        properties=mt.Labels.range("charge", charges.shape[1]).to(device),
    )

    system.add_data(name="charges", data=data)
    neighbors = _compute_ase_neighbors(
        atoms, nl_options, dtype=torch.float32, device=device
    )
    mt.atomistic.register_autograd_neighbors(system, neighbors)

    return (system, neighbors)


def get_calculate_fn(calculator):
    """Define a calculate function for a given calculator."""

    def calculate(charges, positions, cell, neighbor_indices, neighbor_shifts):
        positions.requires_grad = True
<<<<<<< Updated upstream
        distances = compute_distances(
            positions, neighbor_indices, cell, neighbor_shifts
        )
        potentials = calculator(charges, cell, positions, neighbor_indices, distances)
=======
        distances = compute_distances(positions, neighbor_indices, cell, neighbor_shifts)
        # potentials = potential(positions, charges, cell, neighbor_indices, distances)
        potentials = potential(charges, cell, positions, neighbor_indices, distances)
>>>>>>> Stashed changes
        energy = potentials.sum()  # we don't benchmark multiplying with charges
        forces = -torch.autograd.grad(energy, positions)[0]
        return energy, forces

    return calculate


def get_metatensor_calculate_fn(calculator):
    """Get a metatensor calculate function from calculator."""

    def calculate(system, neighbors):
        potentials = calculator(system, neighbors)
        energy = (potentials.block_by_id(0).values).sum()
        forces = -torch.autograd.grad(energy, system.positions)[0]
        return energy, forces

    return calculate


def timed_run(calculate_fn, inputs, repeats, device, warmup=True):
    """Helper for repeated and timed running of a function with warmup."""
    if warmup:
        for i in range(repeats):
            calculate_fn(*inputs[i])
        if device == "cuda":
            torch.cuda.synchronize()

    start = monotonic()
    for i in range(repeats):
        calculate_fn(*inputs[i])

    # we allow parallel execution of the repeats -- just like in real training!
    if device == "cuda":
        torch.cuda.synchronize()
    end = monotonic()
    duration = end - start
    return duration / repeats


def execute(run):
    """Do a benchmark for given settings."""
    device = run["device"]
    atoms = run["atoms"]
    hypers = run["hypers"]
    run["N"]

    cutoff = hypers["cutoff"]
    mesh_spacing = hypers["mesh_spacing"]
    lr_wavelength = hypers["lr_wavelength"]
    atomic_smearing = hypers["atomic_smearing"]

    rng = np.random.default_rng(42)

    results = {}

    inputs = []
    atomss = []
    metatensor_inputs = []
    for _ in range(repeats):
        a = atoms.copy()
        a.rattle(stdev=0.01, rng=rng)
        inputs.append(atoms_to_inputs(atoms, device, cutoff))
        atomss.append(a)

    potential = torchpme.CoulombPotential(smearing=atomic_smearing)

    # PME, no metatensor
    calculator = torchpme.PMECalculator(
        potential,
        mesh_spacing=mesh_spacing,
        full_neighbor_list=False,
    )
    calculator = torch.jit.script(calculator)
    calculator = calculator.to(device)
    calculate = get_calculate_fn(calculator)
    time_per_calculation = timed_run(calculate, inputs, repeats, device)
    results["PME"] = time_per_calculation

    # PME, metatensor
    metatensor_inputs = [
        atoms_to_metatensor_inputs(atoms, device, cutoff) for atoms in atomss
    ]
    warmup_metatensor_inputs = [
        atoms_to_metatensor_inputs(atoms, device, cutoff) for atoms in atomss
    ]
    calculator = torchpme.metatensor.PMECalculator(
        potential,
        mesh_spacing=mesh_spacing,
        full_neighbor_list=False,
    )
    calculator = torch.jit.script(calculator)
    calculator = calculator.to(device)
    calculate = get_metatensor_calculate_fn(calculator)

    # metatensor doesn't reset grads, so we can't reuse inputs for warmup
    timed_run(calculate, warmup_metatensor_inputs, repeats, device, warmup=False)
    time_per_calculation = timed_run(
        calculate, metatensor_inputs, repeats, device, warmup=False
    )
    results["PME_MT"] = time_per_calculation

    # P3M, no metatensor
    calculator = torchpme.P3MCalculator(
        potential,
        mesh_spacing=mesh_spacing,
        full_neighbor_list=False,
    )
    calculator = torch.jit.script(calculator)
    calculator = calculator.to(device)
    calculate = get_calculate_fn(calculator)
    time_per_calculation = timed_run(calculate, inputs, repeats, device)
    results["P3M"] = time_per_calculation

    # P3M, metatensor
    metatensor_inputs = [
        atoms_to_metatensor_inputs(atoms, device, cutoff) for atoms in atomss
    ]
    warmup_metatensor_inputs = [
        atoms_to_metatensor_inputs(atoms, device, cutoff) for atoms in atomss
    ]
    calculator = torchpme.metatensor.P3MCalculator(
        potential,
        mesh_spacing=mesh_spacing,
        full_neighbor_list=False,
    )
    calculator = torch.jit.script(calculator)
    calculator = calculator.to(device)
    calculate = get_metatensor_calculate_fn(calculator)

    # metatensor doesn't reset grads, so we can't reuse inputs for warmup
    timed_run(calculate, warmup_metatensor_inputs, repeats, device, warmup=False)
    time_per_calculation = timed_run(
        calculate, metatensor_inputs, repeats, device, warmup=False
    )
    results["P3M_MT"] = time_per_calculation

    # Ewald, no metatensor
    calculator = torchpme.EwaldCalculator(
        potential,
        lr_wavelength=lr_wavelength,
        full_neighbor_list=False,
    )
    calculator = torch.jit.script(calculator)
    calculator = calculator.to(device)
    calculate = get_calculate_fn(calculator)
    time_per_calculation = timed_run(calculate, inputs, repeats, device)
    results["E"] = time_per_calculation

    # Ewald, metatensor
    metatensor_inputs = [
        atoms_to_metatensor_inputs(atoms, device, cutoff) for atoms in atomss
    ]
    warmup_metatensor_inputs = [
        atoms_to_metatensor_inputs(atoms, device, cutoff) for atoms in atomss
    ]
    calculator = torchpme.metatensor.EwaldCalculator(
        potential,
        lr_wavelength=lr_wavelength,
        full_neighbor_list=False,
    )
    calculator = torch.jit.script(calculator)
    calculator = calculator.to(device)
    calculate = get_metatensor_calculate_fn(calculator)

    # metatensor doesn't reset grads, so we can't reuse inputs for warmup
    timed_run(calculate, warmup_metatensor_inputs, repeats, device, warmup=False)
    time_per_calculation = timed_run(
        calculate, metatensor_inputs, repeats, device, warmup=False
    )
    results["E_MT"] = time_per_calculation

    return results


results = {}
for name, run in runs.items():
    print(f"Running {name}...")
    r = execute(run)
    print(r)
    print()
    results[name] = r

timestamp = datetime.datetime.now().isoformat()
out = {"timestamp": timestamp, "system": system, "version": version, "results": results}

with open(f"{timestamp}.yaml", "w") as file:
    yaml.dump(out, file)

print("Have a nice day.")
