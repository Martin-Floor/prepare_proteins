"""Single-point MM-GBSA on an Amber prmtop+inpcrd pair.

The full topology is assumed to be a parameterized protein+ligand complex
in vacuum (no waters/ions). Receptor and ligand subsystems are obtained
by ParmEd strip masks based on the ligand residue name.

dG_bind = E_complex - E_receptor - E_ligand

This is the standard 'single-trajectory MM-GBSA' approximation: subsystem
geometries are inherited from the complex (no separate apo state). Each
subsystem can optionally be minimised before scoring (default: minimise
each subsystem; the complex is left untouched so the input pose is honoured).

Usage
-----
    python openmm_mmgbsa.py complex.prmtop complex.inpcrd \\
        --ligand_resname LIG --gb_model OBC2 --output mmgbsa.csv

Dependencies
------------
- OpenMM 7.7+
- ParmEd 4+
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    import openmm
    import openmm.app as app
    import openmm.unit as u
except ImportError:  # pragma: no cover
    from simtk import openmm
    from simtk.openmm import app
    from simtk import unit as u

import parmed

GB_MODELS = {
    "HCT": app.HCT,
    "OBC1": app.OBC1,
    "OBC2": app.OBC2,
    "GBn": app.GBn,
    "GBn2": app.GBn2,
}


def build_system(parm, gb_model):
    """Build an OpenMM System with implicit solvent for the given parm."""
    return parm.createSystem(
        implicitSolvent=GB_MODELS[gb_model],
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )


def single_point_energy(parm, gb_model, minimize_steps=0,
                        minimize_tolerance_kj_per_nm=10.0, platform_name=None):
    """Build a System, optionally minimise, return potential energy in kcal/mol."""
    system = build_system(parm, gb_model)
    integrator = openmm.LangevinIntegrator(
        300 * u.kelvin, 1.0 / u.picosecond, 0.002 * u.picoseconds
    )
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)
        context = openmm.Context(system, integrator, platform)
    else:
        context = openmm.Context(system, integrator)
    context.setPositions(parm.positions)
    if minimize_steps and minimize_steps > 0:
        openmm.LocalEnergyMinimizer.minimize(
            context,
            minimize_tolerance_kj_per_nm * u.kilojoules_per_mole / u.nanometer,
            minimize_steps,
        )
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
    del context
    del integrator
    return energy


def main():
    ap = argparse.ArgumentParser(
        description="Single-point MM-GBSA on a protein+ligand prmtop/inpcrd pair."
    )
    ap.add_argument("prmtop")
    ap.add_argument("inpcrd")
    ap.add_argument("--ligand_resname", default="LIG",
                    help="Ligand residue name (ParmEd mask). Default: LIG.")
    ap.add_argument("--gb_model", default="OBC2", choices=list(GB_MODELS),
                    help="OpenMM implicit-solvent model. Default: OBC2.")
    ap.add_argument("--minimize", default="subsystems",
                    choices=["none", "complex", "subsystems", "all"],
                    help="What to minimise before scoring. Default: subsystems.")
    ap.add_argument("--minimize_steps", type=int, default=200)
    ap.add_argument("--minimize_tolerance", type=float, default=10.0,
                    help="Energy tolerance (kJ/mol/nm) for LocalEnergyMinimizer.")
    ap.add_argument("--platform", default=None,
                    help="OpenMM platform: CPU, CUDA, OpenCL, Reference. "
                         "Default: auto-pick.")
    ap.add_argument("--output", default="mmgbsa.csv")
    ap.add_argument("--label", default=None,
                    help="Label written into the result row. Default: prmtop basename.")
    args = ap.parse_args()

    label = args.label or Path(args.prmtop).stem

    print(f"[mmgbsa] loading {args.prmtop} + {args.inpcrd}", flush=True)
    complex_parm = parmed.load_file(args.prmtop, xyz=args.inpcrd)

    ligand_mask = f":{args.ligand_resname}"
    n_total = sum(1 for _ in complex_parm.atoms)
    n_lig = sum(1 for atom in complex_parm.view[ligand_mask].atoms)
    if n_lig == 0:
        raise RuntimeError(
            f"No atoms matched ligand mask {ligand_mask!r}. "
            f"Available residue names: {sorted(set(r.name for r in complex_parm.residues))}"
        )
    print(f"[mmgbsa] complex atoms={n_total}  ligand atoms={n_lig}  "
          f"protein atoms={n_total - n_lig}", flush=True)

    do_minc = args.minimize in ("complex", "all")
    do_mins = args.minimize in ("subsystems", "all")

    t0 = time.time()
    print(f"[mmgbsa] complex single-point (minimise={do_minc})", flush=True)
    e_complex = single_point_energy(
        complex_parm, args.gb_model,
        minimize_steps=args.minimize_steps if do_minc else 0,
        minimize_tolerance_kj_per_nm=args.minimize_tolerance,
        platform_name=args.platform,
    )

    receptor_parm = parmed.load_file(args.prmtop, xyz=args.inpcrd)
    receptor_parm.strip(ligand_mask)
    print(f"[mmgbsa] receptor single-point (minimise={do_mins})", flush=True)
    e_receptor = single_point_energy(
        receptor_parm, args.gb_model,
        minimize_steps=args.minimize_steps if do_mins else 0,
        minimize_tolerance_kj_per_nm=args.minimize_tolerance,
        platform_name=args.platform,
    )

    ligand_parm = parmed.load_file(args.prmtop, xyz=args.inpcrd)
    ligand_parm.strip(f"!{ligand_mask}")
    print(f"[mmgbsa] ligand single-point (minimise={do_mins})", flush=True)
    e_ligand = single_point_energy(
        ligand_parm, args.gb_model,
        minimize_steps=args.minimize_steps if do_mins else 0,
        minimize_tolerance_kj_per_nm=args.minimize_tolerance,
        platform_name=args.platform,
    )

    dg = e_complex - e_receptor - e_ligand
    elapsed = time.time() - t0
    print(f"[mmgbsa] dG_bind = {dg:.3f} kcal/mol  "
          f"(E_c={e_complex:.3f}, E_r={e_receptor:.3f}, E_l={e_ligand:.3f}, "
          f"{elapsed:.1f}s)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "gb_model", "minimize",
                    "E_complex_kcal", "E_receptor_kcal", "E_ligand_kcal",
                    "dG_bind_kcal", "n_atoms_complex", "n_atoms_ligand",
                    "minimize_steps", "elapsed_s"])
        w.writerow([label, args.gb_model, args.minimize,
                    f"{e_complex:.4f}", f"{e_receptor:.4f}", f"{e_ligand:.4f}",
                    f"{dg:.4f}", n_total, n_lig,
                    args.minimize_steps, f"{elapsed:.1f}"])

    print(f"[mmgbsa] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
