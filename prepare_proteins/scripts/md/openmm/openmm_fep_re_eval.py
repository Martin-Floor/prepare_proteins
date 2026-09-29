"""Re-evaluate u_kn for a saved FEP window using a (possibly different)
λ schedule, without re-running MD.

Reads
-----
   <window_dir>/traj_NNN.dcd     (saved by openmm_fep_window.py)
   <window_dir>/meta_NNN.json    (window's metadata, including original schedule)
   <prmtop>                      (parameterised system, complex or solvent leg)

Computes
--------
   u_kn[l, n] = U(x_n; λ_l) / kT  for every frame n in the trajectory and
   every λ_l in the supplied schedule.

Use cases
---------
- Try a finer/different λ schedule without re-running MD (interpolating windows
  is illegal — you must re-run those — but you can change the existing λ values).
- Re-evaluate with a different precision setting.
- Recover from a corrupted u_kn_NNN.npy.
- Sanity-check by re-doing the inline u_kn eval and comparing.

Usage
-----
    python openmm_fep_re_eval.py <window_dir> --prmtop <prmtop_path> \\
        [--schedule_lj 1.0,...] [--schedule_ele 1.0,...] \\
        [--ligand_resname LIG] [--platform CUDA --precision mixed]

If --schedule_* is omitted, the original schedule from meta_NNN.json is used
(useful for re-evaluating with a different platform/precision).

Dependencies
------------
- openmm 8+, openmmtools 0.25+, mdtraj, numpy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u
import parmed
import mdtraj
from openmmtools import alchemy

KB = 1.987204259e-3  # kcal / (mol K)


def find_ligand_atoms(parm, resname):
    idx = [a.idx for a in parm.atoms if a.residue.name == resname]
    if not idx:
        names = sorted({a.residue.name for a in parm.atoms})
        raise RuntimeError(f"No atoms with residue name {resname!r}. Available: {names}")
    return idx


def build_alchemical_system(parm, ligand_atoms, cutoff_nm, use_periodic):
    """Same configuration as the runner — must match for u_kn to be comparable."""
    if use_periodic:
        system = parm.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=cutoff_nm * u.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
            removeCMMotion=False,
            hydrogenMass=3.0 * u.amu,
        )
    else:
        system = parm.createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=True,
            removeCMMotion=False,
            hydrogenMass=3.0 * u.amu,
        )

    factory = alchemy.AbsoluteAlchemicalFactory(
        consistent_exceptions=False,
        alchemical_pme_treatment="exact",
        disable_alchemical_dispersion_correction=True,
    )
    region = alchemy.AlchemicalRegion(
        alchemical_atoms=ligand_atoms,
        annihilate_electrostatics=True,
        annihilate_sterics=False,
        softcore_alpha=0.5,
        softcore_a=1, softcore_b=1, softcore_c=6,
    )
    return factory.create_alchemical_system(system, region)


def main():
    ap = argparse.ArgumentParser(
        description="Re-evaluate u_kn from a saved trajectory."
    )
    ap.add_argument("window_dir",
                    help="Folder containing traj_NNN.dcd + meta_NNN.json.")
    ap.add_argument("--prmtop", required=True,
                    help="prmtop used for the original simulation.")
    ap.add_argument("--ligand_resname", default="LIG")
    ap.add_argument("--schedule_lj", default=None,
                    help="Override λ_LJ schedule (comma-separated). "
                         "Default: use the one in meta_NNN.json.")
    ap.add_argument("--schedule_ele", default=None,
                    help="Override λ_ele schedule. Default: meta_NNN.json.")
    ap.add_argument("--cutoff_nm", type=float, default=1.0)
    ap.add_argument("--nonperiodic", action="store_true")
    ap.add_argument("--platform", default=None,
                    choices=["CPU", "CUDA", "OpenCL", "Reference"])
    ap.add_argument("--precision", default="mixed",
                    choices=["single", "mixed", "double"])
    ap.add_argument("--output", default=None,
                    help="Output u_kn npy path. Default: <window_dir>/u_kn_<NNN>_re.npy")
    args = ap.parse_args()

    win_dir = Path(args.window_dir)
    if not win_dir.is_dir():
        raise FileNotFoundError(win_dir)

    # Locate the meta + dcd by NNN
    metas = sorted(win_dir.glob("meta_*.json"))
    if not metas:
        raise FileNotFoundError(f"no meta_*.json in {win_dir}")
    if len(metas) > 1:
        raise RuntimeError(f"multiple meta_*.json in {win_dir}; ambiguous")
    meta_path = metas[0]
    idx = int(meta_path.stem.split("_")[-1])
    dcd_path = win_dir / f"traj_{idx:03d}.dcd"
    if not dcd_path.exists():
        raise FileNotFoundError(f"missing trajectory: {dcd_path}")
    meta = json.loads(meta_path.read_text())

    # Schedule: override or fall back to meta
    if args.schedule_lj and args.schedule_ele:
        sched_lj = np.array([float(x) for x in args.schedule_lj.split(",")])
        sched_ele = np.array([float(x) for x in args.schedule_ele.split(",")])
    else:
        sched_lj = np.array(meta["schedule_lj"])
        sched_ele = np.array(meta["schedule_ele"])
    if sched_lj.shape != sched_ele.shape:
        raise ValueError("schedule_lj and schedule_ele must have the same length.")
    K = len(sched_lj)
    print(f"[re-eval] window {idx}  λ_lj={meta['lambda_lj']:.3f}  "
          f"λ_ele={meta['lambda_ele']:.3f}", flush=True)
    print(f"[re-eval] schedule: K={K}, lj={sched_lj.tolist()}", flush=True)

    print(f"[re-eval] loading prmtop {args.prmtop}", flush=True)
    parm = parmed.load_file(args.prmtop, xyz=None)
    ligand_atoms = find_ligand_atoms(parm, args.ligand_resname)

    use_periodic = (not args.nonperiodic) and (parm.box_vectors is not None or
                                                meta.get("periodic", False))
    alch = build_alchemical_system(parm, ligand_atoms, args.cutoff_nm, use_periodic)

    # Build a context (no integrator stepping; we just compute energies)
    integrator = openmm.VerletIntegrator(0.001 * u.picoseconds)
    if args.platform:
        platform = openmm.Platform.getPlatformByName(args.platform)
        props = {}
        if args.platform == "CUDA":
            props = {"Precision": args.precision}
        elif args.platform == "OpenCL":
            props = {"Precision": args.precision}
        ctx = openmm.Context(alch, integrator, platform, props)
    else:
        ctx = openmm.Context(alch, integrator)

    print(f"[re-eval] loading trajectory {dcd_path}", flush=True)
    traj = mdtraj.load_dcd(str(dcd_path), top=args.prmtop)
    n_frames = traj.n_frames
    print(f"[re-eval] {n_frames} frames; evaluating u_kn at K={K}", flush=True)

    kT_kcal = meta["kT_kcal_per_mol"]
    u_kn = np.zeros((K, n_frames), dtype=np.float64)
    t0 = time.time()
    for n in range(n_frames):
        ctx.setPositions(traj.xyz[n] * u.nanometer)
        if traj.unitcell_vectors is not None and use_periodic:
            v = traj.unitcell_vectors[n]
            ctx.setPeriodicBoxVectors(
                v[0] * u.nanometer, v[1] * u.nanometer, v[2] * u.nanometer)
        for l in range(K):
            ctx.setParameter("lambda_sterics", float(sched_lj[l]))
            ctx.setParameter("lambda_electrostatics", float(sched_ele[l]))
            e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                u.kilocalories_per_mole)
            u_kn[l, n] = e / kT_kcal
    elapsed = time.time() - t0
    print(f"[re-eval] {n_frames * K} evals in {elapsed:.1f}s "
          f"({n_frames * K / max(elapsed, 1e-9):.0f} evals/s)", flush=True)

    out_path = (Path(args.output) if args.output
                else win_dir / f"u_kn_{idx:03d}_re.npy")
    np.save(out_path, u_kn)
    print(f"[re-eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
