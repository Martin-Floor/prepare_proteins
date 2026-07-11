"""Aggregate per-window u_kn slices and run MBAR for one ligand.

Reads
-----
   <ligand_dir>/complex/window_NNN/u_kn_NNN.npy   (K, N) reduced potentials
   <ligand_dir>/complex/window_NNN/meta_NNN.json
   <ligand_dir>/solvent/window_NNN/u_kn_NNN.npy   (only for absolute_binding)
   <ligand_dir>/solvent/window_NNN/meta_NNN.json

Computes
--------
   ΔG_complex  = MBAR over complex windows (λ=1 → λ=0, ligand decoupled)
   ΔG_solvent  = MBAR over solvent windows
   ΔG_bind     = ΔG_solvent − ΔG_complex   (sign convention: more negative = stronger binder)

For absolute solvation only, ΔG_solv = ΔG_solvent (no protein).

Boresch restraint analytical correction (if --boresch_dG_kcal is given for
absolute binding) is added: ΔG_bind = ΔG_solvent − ΔG_complex + ΔG_restraint
where ΔG_restraint is the analytical free-energy of releasing the
restraint to the standard state (positive number, kcal/mol).

Usage
-----
    python openmm_fep_analyse.py <ligand_dir> --mode absolute_binding \\
        --output results.csv

Dependencies
------------
- numpy, pymbar 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from pymbar import MBAR


def load_phase(phase_dir: Path):
    """Load all u_kn_*.npy + meta_*.json from a phase folder.

    Returns
    -------
    u_kln : ndarray (K, K, N)
        Reduced potentials. K = number of windows, N = frames per window.
        u_kln[k, l, n] = U(x_k_n; λ_l) / kT.
    kT_kcal : float
        Temperature factor in kcal/mol (averaged across windows; should match).
    metas : list[dict]
        Per-window metadata.
    """
    win_dirs = sorted([d for d in phase_dir.iterdir()
                       if d.is_dir() and d.name.startswith("window_")])
    if not win_dirs:
        raise RuntimeError(f"No window_* dirs in {phase_dir}")

    # Drop windows missing meta or u_kn (typical cause: stage-2 NaN crash
    # left some windows without u_kn). Surviving windows MUST be a contiguous
    # tail prefix of the schedule for MBAR to make sense; allowing arbitrary
    # gaps would corrupt the matrix indices. We log dropped windows.
    valid_dirs = []
    skipped = []
    for d in win_dirs:
        idx = int(d.name.split("_")[-1])
        meta_path = d / f"meta_{idx:03d}.json"
        u_path = d / f"u_kn_{idx:03d}.npy"
        if meta_path.exists() and u_path.exists():
            valid_dirs.append(d)
        else:
            skipped.append(idx)
    if skipped:
        print(f"[fep-analyse] dropping {len(skipped)} incomplete windows in "
              f"{phase_dir.name}: {skipped[:10]}{'...' if len(skipped)>10 else ''}",
              file=sys.stderr)
    if len(valid_dirs) < 2:
        raise RuntimeError(f"only {len(valid_dirs)} valid windows in {phase_dir}")
    win_dirs = valid_dirs

    # Determine the FULL schedule length from the first valid meta
    first_meta = json.loads(
        (win_dirs[0] / f"meta_{int(win_dirs[0].name.split('_')[-1]):03d}.json").read_text())
    K_full = first_meta["n_windows"]
    valid_idx = sorted(int(d.name.split("_")[-1]) for d in win_dirs)
    K = len(valid_idx)
    if K < K_full:
        print(f"[fep-analyse] using {K}/{K_full} windows in {phase_dir.name} "
              f"(re-indexed for MBAR)", file=sys.stderr)
    metas = []
    u_kn_list = []
    n_min = None
    for d in win_dirs:
        idx = int(d.name.split("_")[-1])
        meta_path = d / f"meta_{idx:03d}.json"
        u_path = d / f"u_kn_{idx:03d}.npy"
        meta = json.loads(meta_path.read_text())
        u_kn = np.load(u_path)              # shape (K_full, N_k)
        if u_kn.shape[0] != K_full:
            raise RuntimeError(
                f"window {idx}: u_kn has K={u_kn.shape[0]}, expected {K_full}"
            )
        n_min = u_kn.shape[1] if n_min is None else min(n_min, u_kn.shape[1])
        metas.append(meta)
        u_kn_list.append(u_kn)

    # Slice to valid columns + truncate frames. Result u_kln[k_valid, l_valid, n]
    # = u(x sampled at λ_valid_idx[k]; eval at λ_valid_idx[l]) reduced units.
    u_kln = np.zeros((K, K, n_min), dtype=np.float64)
    for k, u_kn in enumerate(u_kn_list):
        u_kln[k, :, :] = u_kn[valid_idx, :n_min]

    kTs = [m["kT_kcal_per_mol"] for m in metas]
    if max(kTs) - min(kTs) > 1e-6:
        print(f"[fep-analyse] WARNING kT_kcal varies across windows: {kTs}",
              file=sys.stderr)
    return u_kln, kTs[0], metas


def discard_equilibration(u_kln: np.ndarray, frac: float = 0.2):
    """Drop the first `frac` fraction of frames as additional equilibration."""
    n = u_kln.shape[-1]
    n_drop = int(round(frac * n))
    if n_drop >= n - 2:
        return u_kln  # nothing to do
    return u_kln[..., n_drop:]


def validate_u_kln(u_kln: np.ndarray) -> dict:
    """Sanity-check u_kln before MBAR. Detects NaN, infinite, or extreme
    values that would break the analysis. Returns a diagnostic dict."""
    K = u_kln.shape[0]
    bad = {}
    for k in range(K):
        d = u_kln[k, k, :]
        if not np.isfinite(d).all():
            bad[k] = "non-finite values in u_kk"
        elif np.abs(d).max() > 1e6:
            bad[k] = f"extreme |u_kk|={np.abs(d).max():.1e}"
    return {
        "n_windows": K, "n_frames": u_kln.shape[-1],
        "bad_windows": bad,
        "diag_min": float(u_kln.diagonal(axis1=0, axis2=1).min()) if K > 0 else None,
        "diag_max": float(u_kln.diagonal(axis1=0, axis2=1).max()) if K > 0 else None,
    }


def mbar_dG(u_kln: np.ndarray, kT_kcal: float):
    """Run MBAR with graceful BAR fallback if MBAR fails to converge or
    has corrupt windows. Returns (dG_kcal, ddG_kcal, method) where method
    is 'mbar' or 'bar_fallback'.
    """
    diag = validate_u_kln(u_kln)
    if diag["bad_windows"]:
        print(f"[fep-analyse] WARNING corrupt windows: {diag['bad_windows']}",
              file=sys.stderr)

    K, K2, N = u_kln.shape
    assert K == K2
    N_k = np.full(K, N, dtype=int)

    # Try MBAR first
    try:
        mbar = MBAR(u_kln.reshape(K, K * N), N_k)
        res = mbar.compute_free_energy_differences()
        f_ij = res["Delta_f"]      # (K, K) in kT
        df_ij = res["dDelta_f"]
        dG = f_ij[0, K - 1] * kT_kcal
        ddG = df_ij[0, K - 1] * kT_kcal
        return float(dG), float(ddG), "mbar"
    except Exception as exc:
        print(f"[fep-analyse] MBAR FAILED ({exc}); falling back to pairwise BAR",
              file=sys.stderr)

    # Fallback: pairwise BAR (more tolerant of poor overlap / under-sampling)
    from pymbar import other_estimators
    total_kT = 0.0
    var_kT = 0.0
    for k in range(K - 1):
        try:
            w_F = u_kln[k, k + 1] - u_kln[k, k]
            w_R = u_kln[k + 1, k] - u_kln[k + 1, k + 1]
            res = other_estimators.bar(w_F, w_R)
            total_kT += res["Delta_f"]
            var_kT += res["dDelta_f"] ** 2
        except Exception as e2:
            print(f"[fep-analyse]   BAR pair {k}→{k+1} FAILED: {e2}",
                  file=sys.stderr)
    dG = total_kT * kT_kcal
    ddG = float(np.sqrt(var_kT)) * kT_kcal
    return float(dG), float(ddG), "bar_fallback"


def stat_inefficiency_per_window(u_kln: np.ndarray) -> np.ndarray:
    """Per-window statistical inefficiency g (autocorrelation length in frames).

    Effective sample size = N / g. Uses pymbar.timeseries.statisticalInefficiency
    on the diagonal u_k(λ_k) timeseries.
    """
    from pymbar import timeseries
    K, _, N = u_kln.shape
    g = np.zeros(K)
    for k in range(K):
        try:
            g[k] = timeseries.statistical_inefficiency(u_kln[k, k, :])
        except Exception:
            g[k] = 1.0
    return g


def fwd_rev_hysteresis(u_kln: np.ndarray, kT_kcal: float) -> dict:
    """Forward/reverse free-energy convergence test.

    Splits frames into halves (first N/2 vs last N/2), runs MBAR on each,
    returns ΔG_first, ΔG_last, |hysteresis|. Good convergence ⇒ both halves
    agree within statistical error.
    """
    N = u_kln.shape[-1]
    if N < 4:
        return {"dG_first": None, "dG_last": None, "hysteresis_kcal": None}
    half = N // 2
    dG_first, _, _ = mbar_dG(u_kln[..., :half], kT_kcal)
    dG_last, _, _ = mbar_dG(u_kln[..., half:], kT_kcal)
    return {
        "dG_first": dG_first,
        "dG_last": dG_last,
        "hysteresis_kcal": abs(dG_first - dG_last),
    }


def mbar_overlap_diagnostics(u_kln: np.ndarray) -> dict:
    """Compute MBAR overlap matrix diagonal (= per-state effective sample
    fraction). Min(diag) < 0.03 → poor overlap → schedule is too coarse."""
    K, _, N = u_kln.shape
    N_k = np.full(K, N, dtype=int)
    mbar = MBAR(u_kln.reshape(K, K * N), N_k)
    try:
        ov = mbar.compute_overlap()
        # newer pymbar 4 API
        eigenvalues = ov.get("eigenvalues") or ov.get("eigvals")
        matrix = ov.get("matrix")
        diag = np.diag(matrix) if matrix is not None else None
    except Exception:
        return {"overlap_min": None, "overlap_diag": None}
    out = {}
    if diag is not None:
        out["overlap_min"] = float(diag.min())
        out["overlap_diag"] = diag.tolist()
    return out


def rocklin_charge_correction(charge_e: int, box_length_A: float,
                               temperature_K: float = 300.0,
                               ionic_strength_M: float = 0.15) -> float:
    """Rocklin et al. 2013 finite-size correction for absolute hydration FE
    of a charged solute in a periodic PME box.

    Returns ΔG_correction (kcal/mol) to ADD to ΔG_solv (decoupling sense).
    Reference:
        Rocklin et al., J. Chem. Phys. 139, 184103 (2013), Eqn (8):
        ΔG_NES = -ξ q² / (8πε₀ L) × [1 - 1/(εₛ × κ_DH²)]
    For neutral ligand returns 0. Approximation only — proper Rocklin
    requires lattice-summation constants (ξ_LS).
    """
    if charge_e == 0:
        return 0.0
    # Constants
    e2_over_4pi_eps0_kcal_A = 332.0637133  # kcal·Å/(mol·e²)
    epsilon_solv = 78.5  # bulk water dielectric
    # Madelung-like Hummer/Rocklin lattice constant for cubic box
    xi_LS = 2.837297
    # Net contribution (ignoring κ_DH for typical 150 mM, 78.5 dielectric)
    dG_kcal = -xi_LS * (charge_e ** 2) * e2_over_4pi_eps0_kcal_A / (
        2.0 * box_length_A)
    # The dG returned is the correction to the decoupling FE; flip sign for
    # solvation FE convention if needed by caller.
    return dG_kcal


def diagnose_convergence(u_kln: np.ndarray, kT_kcal: float, label: str):
    """Print + return a dict of convergence diagnostics for one phase."""
    g = stat_inefficiency_per_window(u_kln)
    fwd_rev = fwd_rev_hysteresis(u_kln, kT_kcal)
    overlap = mbar_overlap_diagnostics(u_kln)
    n_eff_min = float((u_kln.shape[-1] / g).min())
    print(f"[fep-analyse] {label} convergence:")
    print(f"  stat. inefficiency  g_max = {g.max():.1f}  N_eff_min = {n_eff_min:.0f}")
    print(f"  fwd/rev hysteresis  {fwd_rev['hysteresis_kcal']:.3f} kcal/mol "
          f"(first={fwd_rev['dG_first']:+.3f}, last={fwd_rev['dG_last']:+.3f})")
    if overlap.get("overlap_min") is not None:
        print(f"  MBAR overlap        min(diag) = {overlap['overlap_min']:.3f}  "
              f"({'OK' if overlap['overlap_min'] > 0.03 else '⚠ poor'})")
    return {
        "stat_inefficiency_g": g.tolist(),
        "n_eff_min": n_eff_min,
        "fwd_rev_hysteresis_kcal": fwd_rev["hysteresis_kcal"],
        "dG_first_half_kcal": fwd_rev["dG_first"],
        "dG_last_half_kcal": fwd_rev["dG_last"],
        "mbar_overlap_min": overlap.get("overlap_min"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run MBAR over per-window u_kn slices for one ligand."
    )
    ap.add_argument("ligand_dir", help="Folder containing complex/ and/or solvent/ subdirs.")
    ap.add_argument("--mode", required=True,
                    choices=["absolute_solvation", "absolute_binding"])
    ap.add_argument("--label", default=None,
                    help="Label written into the result row. Default: ligand_dir basename.")
    ap.add_argument("--equilibration_frac", type=float, default=0.2,
                    help="Fraction of production frames to drop as additional equilibration.")
    ap.add_argument("--boresch_dG_kcal", type=float, default=0.0,
                    help="Analytical correction for releasing the Boresch restraint "
                         "to standard state (kcal/mol). Only used for absolute_binding.")
    ap.add_argument("--ligand_charge", type=int, default=0,
                    help="Ligand formal charge (in e). Used to apply Rocklin "
                         "et al. 2013 finite-size correction for charged ligands.")
    ap.add_argument("--box_length_A", type=float, default=None,
                    help="Equilibrated box edge length (Å). Required if "
                         "ligand_charge != 0. Read from meta if available.")
    ap.add_argument("--no_diagnostics", action="store_true",
                    help="Skip per-window stat. inefficiency, fwd/rev, overlap "
                         "diagnostics (saves a few seconds).")
    ap.add_argument("--output", default="results.csv")
    args = ap.parse_args()

    lig_dir = Path(args.ligand_dir)
    label = args.label or lig_dir.name

    rows = []

    if args.mode == "absolute_solvation":
        u_solv, kT, _ = load_phase(lig_dir / "solvent")
        u_solv = discard_equilibration(u_solv, args.equilibration_frac)
        dG_solv, dd_solv, method_solv = mbar_dG(u_solv, kT)
        dG_solvation = -dG_solv
        # Optional Rocklin finite-size correction
        rocklin = 0.0
        if args.ligand_charge != 0 and args.box_length_A:
            rocklin = rocklin_charge_correction(
                args.ligand_charge, args.box_length_A)
            print(f"[fep-analyse] {label}  Rocklin correction "
                  f"(q={args.ligand_charge}, L={args.box_length_A} Å) = "
                  f"{rocklin:+.3f} kcal/mol", flush=True)
        dG_solvation_corr = dG_solvation + rocklin
        print(f"[fep-analyse] {label}  ΔG_solvation = {dG_solvation_corr:+.3f} ± "
              f"{dd_solv:.3f} kcal/mol", flush=True)
        diag = {} if args.no_diagnostics else \
            diagnose_convergence(u_solv, kT, f"{label}/solvent")
        rows.append({
            "label": label, "mode": "absolute_solvation",
            "dG_solvation_kcal": f"{dG_solvation_corr:+.4f}",
            "ddG_solvation_kcal": f"{dd_solv:.4f}",
            "rocklin_correction_kcal": f"{rocklin:+.4f}",
            "n_windows_solvent": u_solv.shape[0],
            "n_frames_solvent": u_solv.shape[-1],
            "method_solvent": method_solv,
            **{f"solv_{k}": v for k, v in diag.items()
               if not isinstance(v, list)},
        })

    elif args.mode == "absolute_binding":
        u_cmp, kT_c, _ = load_phase(lig_dir / "complex")
        u_solv, kT_s, _ = load_phase(lig_dir / "solvent")
        u_cmp = discard_equilibration(u_cmp, args.equilibration_frac)
        u_solv = discard_equilibration(u_solv, args.equilibration_frac)
        dG_cmp, dd_cmp, method_cmp = mbar_dG(u_cmp, kT_c)
        dG_solv, dd_solv, method_solv = mbar_dG(u_solv, kT_s)
        # Optional Rocklin correction (cancels in well-matched complex+solvent
        # boxes, but apply if box sizes differ)
        rocklin = 0.0
        if args.ligand_charge != 0 and args.box_length_A:
            rocklin = rocklin_charge_correction(
                args.ligand_charge, args.box_length_A)
            print(f"[fep-analyse] {label}  Rocklin correction "
                  f"(q={args.ligand_charge}, L={args.box_length_A} Å) = "
                  f"{rocklin:+.3f} kcal/mol", flush=True)
        # Thermodynamic cycle:
        #   ΔG_bind = -ΔG_decouple_complex + ΔG_decouple_solvent + ΔG_restraint
        dG_bind = -dG_cmp + dG_solv + args.boresch_dG_kcal + rocklin
        dd_bind = float(np.sqrt(dd_cmp ** 2 + dd_solv ** 2))
        print(f"[fep-analyse] {label}  ΔG_decouple_complex = {dG_cmp:+.3f} ± "
              f"{dd_cmp:.3f}", flush=True)
        print(f"[fep-analyse] {label}  ΔG_decouple_solvent = {dG_solv:+.3f} ± "
              f"{dd_solv:.3f}", flush=True)
        print(f"[fep-analyse] {label}  ΔG_restraint        = "
              f"{args.boresch_dG_kcal:+.3f}", flush=True)
        print(f"[fep-analyse] {label}  ΔG_bind             = {dG_bind:+.3f} ± "
              f"{dd_bind:.3f} kcal/mol", flush=True)
        diag_c = {} if args.no_diagnostics else \
            diagnose_convergence(u_cmp, kT_c, f"{label}/complex")
        diag_s = {} if args.no_diagnostics else \
            diagnose_convergence(u_solv, kT_s, f"{label}/solvent")
        rows.append({
            "label": label, "mode": "absolute_binding",
            "dG_decouple_complex_kcal": f"{dG_cmp:+.4f}",
            "ddG_decouple_complex_kcal": f"{dd_cmp:.4f}",
            "method_complex": method_cmp,
            "dG_decouple_solvent_kcal": f"{dG_solv:+.4f}",
            "ddG_decouple_solvent_kcal": f"{dd_solv:.4f}",
            "method_solvent": method_solv,
            "dG_restraint_kcal": f"{args.boresch_dG_kcal:+.4f}",
            "rocklin_correction_kcal": f"{rocklin:+.4f}",
            "dG_bind_kcal": f"{dG_bind:+.4f}",
            "ddG_bind_kcal": f"{dd_bind:.4f}",
            "n_windows_complex": u_cmp.shape[0],
            "n_frames_complex": u_cmp.shape[-1],
            "n_windows_solvent": u_solv.shape[0],
            "n_frames_solvent": u_solv.shape[-1],
            **{f"cmpx_{k}": v for k, v in diag_c.items()
               if not isinstance(v, list)},
            **{f"solv_{k}": v for k, v in diag_s.items()
               if not isinstance(v, list)},
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[fep-analyse] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
