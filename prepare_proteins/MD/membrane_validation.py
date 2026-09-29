"""Tier-1 pre-MD validation for MARTINI3 + INSANE membrane-embedded systems.

Four checks before committing expensive CG-MD compute:

  1. **Topology cross-check** — DeepTMHMM (Hallgren et al. 2022) predicts
     per-residue topology (signal peptide / TM helix / inside / outside)
     from the sequence. Each residue's *embedded* z-position in the
     INSANE-built system is compared against the prediction; disagreements
     are flagged.

  2. **Net charge** — sums charges across all included MARTINI3 .itp files
     weighted by molecule count from the [molecules] block. Should be 0
     after INSANE adds counter-ions.

  3. **Periodic-image safety** — minimum protein-image distance along x, y,
     z must clear the non-bonded cutoff (1.1 nm default) by ≥ 1 nm to
     avoid spurious self-interactions.

  4. **Bilayer leaflet detection** — k-means split of POPC PO4 z-positions
     into upper/lower leaflets, asymmetry, peak-to-peak thickness.

Public API::

    from prepare_proteins.MD import validate_membrane_embedding

    report = validate_membrane_embedding(
        gro_path, top_path,
        subunits=[("A", "SEC11_KPH", fasta_A),
                  ("B", "SPC1_KPH",  fasta_B), ...],
        deeptmhmm=True,            # set False to skip TM prediction
    )
    print(report.summary)          # one-row pandas Series
    print(report.topology)         # per-residue DataFrame
    if not report.ok:
        raise RuntimeError(report.failures)

The MARTINI3-focused convention (v1):
  - Lipid bead naming: ``resname POPC and name PO4`` for leaflet detection
  - Protein backbone bead: ``name BB``
  - Water: ``resname W``

Designed for the workflow ``martini_md``: integrates as
``martini_md.validateEmbedding(...)`` with the workspace's
``system.gro`` and ``system.top`` automatically wired in. Auto-hooks
into ``martini_md.prepareJobs(validate=True)`` (default) so production
runs cannot be submitted on a system that fails the checks.

Optional dependencies (graceful fallback if missing):
  - ``MDAnalysis`` — needed for any check; raises ``ImportError`` if absent
  - ``scikit-learn`` — for k-means leaflet detection
  - ``pybiolib`` — only needed when ``deeptmhmm=True``
"""
from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


_MDA_IMPORT_ERROR: Optional[Exception] = None
try:
    import MDAnalysis as mda  # type: ignore
    _MDA_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    mda = None  # type: ignore
    _MDA_AVAILABLE = False
    _MDA_IMPORT_ERROR = exc


def _ensure_mda() -> None:
    if not _MDA_AVAILABLE:
        raise ImportError(
            "MDAnalysis is required for membrane-embedding validation. "
            "Install with `pip install MDAnalysis`."
        ) from _MDA_IMPORT_ERROR


# DeepTMHMM single-letter labels.
TMHMM_LABEL_NAMES = {"I": "inside", "O": "outside", "M": "TM_helix",
                     "S": "signal_peptide", "B": "TM_beta"}
TMHMM_TM_LABELS = {"M", "B"}


@dataclass
class MembraneValidationReport:
    """Result of :func:`validate_membrane_embedding`.

    Attributes
    ----------
    summary
        Single-row dict with the headline metrics (bilayer thickness,
        net charge, image safety, topology agreement).
    topology
        Per-residue pandas DataFrame: chain, resid, aa, predicted_label,
        predicted_TM, z_nm, in_bilayer, agreement.
    bilayer
        Dict with leaflet z-positions, thickness, leaflet counts.
    charge
        Dict with net charge and per-moleculetype breakdown.
    image_safety
        Dict with min image distances and pass/fail.
    failures
        List of human-readable strings describing any failed checks.
    ok
        True if no checks failed.
    """
    summary: dict
    topology: "object"   # pandas DataFrame
    bilayer: dict
    charge: dict
    image_safety: dict
    failures: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# 1. DeepTMHMM topology prediction
# ---------------------------------------------------------------------------

def _default_cache_dir() -> Path:
    return Path(os.environ.get("PREPARE_PROTEINS_CACHE",
                                Path.home() / ".cache" / "prepare_proteins")
                ) / "deeptmhmm"


def predict_topology_deeptmhmm(
    sequences: Sequence[Tuple[str, str]],
    cache_dir: Optional[os.PathLike] = None,
) -> dict:
    """Run DeepTMHMM on each (label, sequence) pair via the BioLib API.

    Results are cached on disk keyed by sequence string, so repeated calls
    on the same sequences hit the cache (DeepTMHMM is deterministic per
    sequence). Returns ``{label: per-residue topology string}``.

    Default cache dir: ``$PREPARE_PROTEINS_CACHE/deeptmhmm/`` or
    ``~/.cache/prepare_proteins/deeptmhmm/``.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "deeptmhmm_cache.json"
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    needed = [(label, seq) for label, seq in sequences if seq not in cache]
    if needed:
        try:
            import biolib  # pybiolib
        except ImportError:
            warnings.warn("pybiolib not installed — DeepTMHMM skipped. "
                          "Install with `pip install pybiolib`.")
            return {label: cache.get(seq, "?" * len(seq))
                    for label, seq in sequences}

        fasta_str = "".join(f">{label}\n{seq}\n" for label, seq in needed)
        in_path = cache_dir / "_deeptmhmm_input.fasta"
        in_path.write_text(fasta_str)

        try:
            tmhmm_app = biolib.load("DTU/DeepTMHMM")
            job = tmhmm_app.cli(args=f"--fasta {in_path}")
            out_dir = cache_dir / "_deeptmhmm_out"
            if out_dir.exists():
                import shutil
                shutil.rmtree(out_dir)
            job.save_files(str(out_dir))
            three_line = out_dir / "predicted_topologies.3line"
            if not three_line.exists():
                cands = list(out_dir.rglob("predicted_topologies.3line"))
                if cands:
                    three_line = cands[0]
            if three_line.exists():
                lines = three_line.read_text().splitlines()
                for i in range(0, len(lines), 3):
                    if i + 2 >= len(lines):
                        break
                    seq = lines[i + 1].strip()
                    topo = lines[i + 2].strip()
                    if len(seq) == len(topo):
                        cache[seq] = topo
        except Exception as exc:
            warnings.warn(f"DeepTMHMM call failed: {exc}")

        cache_path.write_text(json.dumps(cache, indent=2) + "\n")

    return {label: cache.get(seq, "?" * len(seq))
            for label, seq in sequences}


# ---------------------------------------------------------------------------
# 2. Net charge from GROMACS topology
# ---------------------------------------------------------------------------

def _parse_itp_charges(itp: Path) -> dict:
    """Returns {moleculetype: per-molecule total charge} for one .itp."""
    out: dict = {}
    if not itp.exists():
        return out
    moleculetype = None
    in_atoms = False
    per_mol_charge = 0.0
    for line in itp.read_text().splitlines():
        line = line.split(";", 1)[0].rstrip()
        if not line.strip():
            continue
        sec = re.match(r"\[\s*(\w+)\s*\]", line)
        if sec:
            if moleculetype and in_atoms:
                out[moleculetype] = per_mol_charge
            section = sec.group(1).lower()
            if section == "moleculetype":
                moleculetype = None
                in_atoms = False
                per_mol_charge = 0.0
            elif section == "atoms":
                in_atoms = True
                per_mol_charge = 0.0
            else:
                in_atoms = False
            continue
        if moleculetype is None and re.match(r"^\s*\S+\s+\d+\s*$", line):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                moleculetype = parts[0]
                continue
        if in_atoms:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    per_mol_charge += float(parts[6])
                except ValueError:
                    pass
    if moleculetype and in_atoms:
        out[moleculetype] = per_mol_charge
    return out


def compute_net_charge(top_path: os.PathLike) -> Tuple[float, dict]:
    """Sum charges across the system. Walks #include'd .itp files relative
    to the .top file's directory. Returns ``(net_charge, breakdown_dict)``."""
    top_path = Path(top_path)
    base = top_path.parent
    text = top_path.read_text()
    includes = re.findall(r'#include\s+"([^"]+)"', text)
    moltype_charges: dict = {}
    for inc in includes:
        moltype_charges.update(_parse_itp_charges(base / inc))

    counts: List[Tuple[str, int]] = []
    in_molecules = False
    for line in text.splitlines():
        ll = line.split(";", 1)[0].strip()
        if not ll:
            continue
        sec = re.match(r"\[\s*(\w+)\s*\]", ll)
        if sec:
            in_molecules = sec.group(1).lower() == "molecules"
            continue
        if in_molecules:
            parts = ll.split()
            if len(parts) >= 2 and parts[1].isdigit():
                counts.append((parts[0], int(parts[1])))

    breakdown: dict = {}
    net = 0.0
    for mol, n in counts:
        per = moltype_charges.get(mol, 0.0)
        breakdown[f"{mol}×{n}"] = per * n
        net += per * n
    return net, breakdown


# ---------------------------------------------------------------------------
# 3. Periodic-image safety
# ---------------------------------------------------------------------------

def check_periodic_image_safety(universe, cutoff_nm: float = 1.1,
                                  buffer_nm: float = 1.0) -> dict:
    """Min protein-image distance along x, y, z. Returns dict including
    ``safe`` boolean (True if all axes clear ``cutoff_nm + buffer_nm``).

    The ``buffer_nm`` accounts for thermal fluctuations during MD; 1 nm
    is conservative for typical MARTINI3 systems."""
    bb = universe.select_atoms("name BB")
    if len(bb) == 0:
        return {"error": "no BB beads found"}
    box_nm = universe.dimensions[:3] / 10.0
    bb_pos_nm = bb.positions / 10.0

    def axis_min_image_distance(axis: int) -> float:
        coords = bb_pos_nm[:, axis]
        return float(box_nm[axis] - (coords.max() - coords.min()))

    dx = axis_min_image_distance(0)
    dy = axis_min_image_distance(1)
    dz = axis_min_image_distance(2)
    threshold = cutoff_nm + buffer_nm
    safe = dx > threshold and dy > threshold and dz > threshold
    return {
        "box_nm": tuple(float(b) for b in box_nm),
        "min_image_dx_nm": dx,
        "min_image_dy_nm": dy,
        "min_image_dz_nm": dz,
        "cutoff_nm": cutoff_nm,
        "buffer_nm": buffer_nm,
        "threshold_nm": threshold,
        "safe": safe,
    }


# ---------------------------------------------------------------------------
# 4. Bilayer leaflet detection
# ---------------------------------------------------------------------------

def find_leaflet_peaks(po4_z_nm) -> Tuple[float, float]:
    """Return (lower_z_nm, upper_z_nm) by 2-cluster k-means on PO4 z-values."""
    import numpy as np
    from sklearn.cluster import KMeans
    arr = np.asarray(po4_z_nm).reshape(-1, 1)
    if len(arr) < 4:
        raise ValueError("need at least 4 PO4 atoms to detect leaflets")
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(arr)
    centers = sorted(km.cluster_centers_.flatten())
    return float(centers[0]), float(centers[1])


def detect_bilayer(universe,
                    lipid_resname: str = "POPC",
                    head_atom: str = "PO4") -> dict:
    """Detect bilayer geometry from a lipid headgroup atom."""
    head = universe.select_atoms(f"resname {lipid_resname} and name {head_atom}")
    if len(head) == 0:
        return {"error": f"no {lipid_resname}@{head_atom} atoms"}
    z_nm = head.positions[:, 2] / 10.0
    lower, upper = find_leaflet_peaks(z_nm)
    centre = (lower + upper) / 2
    n_lower = int((z_nm < centre).sum())
    n_upper = int((z_nm >= centre).sum())
    return {
        "lipid_resname": lipid_resname,
        "head_atom": head_atom,
        "lower_z_nm": lower,
        "upper_z_nm": upper,
        "centre_z_nm": centre,
        "thickness_nm": upper - lower,
        "n_lipids_lower": n_lower,
        "n_lipids_upper": n_upper,
        "asymmetry_pct": 100 * (n_upper - n_lower) / (n_upper + n_lower),
    }


# ---------------------------------------------------------------------------
# 5. Topology cross-check
# ---------------------------------------------------------------------------

def cross_check_topology(
    universe, sequences: Sequence[Tuple[str, str, str]],
    bilayer: dict, predictions: dict,
    hydrophobic_core_half_thickness_nm: float = 1.3,
):
    """Compare each residue's predicted topology vs its embedded z-position.

    The "in_bilayer" criterion uses the *hydrophobic core* (default
    ±1.3 nm around the bilayer centre, ≈ POPC carbonyl-to-carbonyl span)
    rather than the full peak-to-peak PO4 distance. This avoids
    classifying interface residues just inside the phosphate plane as
    "in bilayer" when DeepTMHMM correctly labels them as luminal/cytosolic.
    The 1.3 nm half-thickness matches OPM's TM definition for POPC.

    Parameters
    ----------
    universe
        MDAnalysis Universe of the embedded system.
    sequences
        List of ``(chain_id, label, sequence)`` for the protein subunits in
        the order they appear in the BB-bead stream of the universe.
    bilayer
        Output of :func:`detect_bilayer`.
    predictions
        ``{label: per-residue topology string}`` from
        :func:`predict_topology_deeptmhmm`.
    hydrophobic_core_half_thickness_nm
        Half-thickness of the hydrophobic-core slab used to declare
        "in_bilayer". Default 1.3 nm; OPM uses ~1.5 nm for general
        proteins. Decrease (1.0 nm) to be more conservative about
        interface-zone agreement; increase (1.5 nm) to be more inclusive.

    Returns
    -------
    pandas.DataFrame
        Per-residue: chain, subunit, resid, aa, predicted_label,
        predicted_TM, z_nm, in_bilayer, agreement.
    """
    import pandas as pd
    bb = universe.select_atoms("name BB")
    bb_z_nm = bb.positions[:, 2] / 10.0
    centre = bilayer["centre_z_nm"]
    half = hydrophobic_core_half_thickness_nm
    lo = centre - half
    hi = centre + half

    rows = []
    cursor = 0
    for chain_id, label, seq in sequences:
        topo = predictions.get(label, "?" * len(seq))
        if len(topo) != len(seq):
            topo = "?" * len(seq)
        for i, (aa, t) in enumerate(zip(seq, topo)):
            global_idx = cursor + i
            z = float(bb_z_nm[global_idx]) if global_idx < len(bb_z_nm) else float("nan")
            in_bilayer = (lo <= z <= hi) if not (z != z) else False
            tm_pred = t in TMHMM_TM_LABELS
            rows.append({
                "chain": chain_id, "subunit": label,
                "resid": i + 1, "aa": aa,
                "predicted_label": TMHMM_LABEL_NAMES.get(t, "unknown"),
                "predicted_TM": tm_pred,
                "z_nm": z if z == z else None,
                "in_bilayer": bool(in_bilayer),
                "agreement": bool(in_bilayer == tm_pred),
            })
        cursor += len(seq)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _read_fasta_single(path: os.PathLike) -> Tuple[str, str]:
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    return lines[0][1:].split()[0], "".join(lines[1:])


def validate_membrane_embedding(
    gro_path: os.PathLike,
    top_path: os.PathLike,
    *,
    subunits: Sequence[Tuple[str, str, os.PathLike]] = (),
    cutoff_nm: float = 1.1,
    image_buffer_nm: float = 1.0,
    lipid_resname: str = "POPC",
    head_atom: str = "PO4",
    deeptmhmm: bool = True,
    cache_dir: Optional[os.PathLike] = None,
) -> MembraneValidationReport:
    """Run Tier-1 pre-MD validation on a MARTINI3 + INSANE-built system.

    Parameters
    ----------
    gro_path, top_path
        GROMACS .gro and .top of the assembled system (after martinize2 +
        INSANE + topology merge).
    subunits
        ``[(chain_id, label, fasta_path), ...]`` for each protein subunit
        in the order they appear in the BB-bead stream. The first subunit
        starts at BB index 0; concatenated lengths must match the total
        number of BB beads.
    cutoff_nm
        Non-bonded cutoff for the periodic-image-safety check (default
        MARTINI3 reaction-field cutoff = 1.1 nm).
    image_buffer_nm
        Extra clearance beyond cutoff_nm to absorb MD thermal fluctuations.
    lipid_resname, head_atom
        Lipid resname and head-atom name for leaflet detection. Default
        ``POPC`` / ``PO4`` covers MARTINI3 phospholipids; for ergosterol
        or CHOL replace ``head_atom`` accordingly.
    deeptmhmm
        Whether to call DeepTMHMM for per-residue topology prediction.
        Set ``False`` to skip (e.g. on offline runs); the topology
        cross-check then degrades to "unknown" labels.
    cache_dir
        Override DeepTMHMM cache location. Defaults to
        ``~/.cache/prepare_proteins/deeptmhmm/``.

    Returns
    -------
    MembraneValidationReport
    """
    _ensure_mda()
    import pandas as pd

    u = mda.Universe(str(gro_path))

    # Bilayer
    bilayer = detect_bilayer(u, lipid_resname=lipid_resname,
                              head_atom=head_atom)

    # Charge
    net_charge, breakdown = compute_net_charge(top_path)
    charge = {"net_charge": net_charge, "breakdown": breakdown}

    # Image safety
    image = check_periodic_image_safety(u, cutoff_nm=cutoff_nm,
                                         buffer_nm=image_buffer_nm)

    # Topology cross-check
    topology_df = pd.DataFrame()
    sequences: List[Tuple[str, str, str]] = []
    if subunits:
        seqs_for_query: List[Tuple[str, str]] = []
        for chain_id, label, fasta in subunits:
            _, seq = _read_fasta_single(fasta)
            sequences.append((chain_id, label, seq))
            seqs_for_query.append((label, seq))
        if deeptmhmm:
            predictions = predict_topology_deeptmhmm(seqs_for_query, cache_dir)
        else:
            predictions = {label: "?" * len(seq) for label, seq in seqs_for_query}
        topology_df = cross_check_topology(u, sequences, bilayer, predictions)

    # Failure flags
    failures: List[str] = []
    if "error" in bilayer:
        failures.append(f"bilayer detection error: {bilayer['error']}")
    elif bilayer["thickness_nm"] < 2.5 or bilayer["thickness_nm"] > 5.0:
        failures.append(f"bilayer thickness {bilayer['thickness_nm']:.2f} nm "
                        f"outside plausible 2.5–5.0 nm range")
    if abs(net_charge) > 0.01:
        failures.append(f"system not electroneutral: net charge {net_charge:+.3f}")
    if not image.get("safe", False):
        failures.append(
            f"periodic image distance too small: "
            f"dx={image.get('min_image_dx_nm'):.2f} nm, "
            f"dy={image.get('min_image_dy_nm'):.2f} nm, "
            f"dz={image.get('min_image_dz_nm'):.2f} nm "
            f"(threshold {image.get('threshold_nm'):.2f} nm)"
        )
    if subunits and not topology_df.empty:
        n_known = (topology_df["predicted_label"] != "unknown").sum()
        if n_known > 0:
            n_disagree = ((topology_df["predicted_label"] != "unknown")
                          & ~topology_df["agreement"]).sum()
            pct_disagree = 100 * n_disagree / n_known
            # Default threshold = 50%. Interface-zone residues (between
            # bilayer hydrophobic core and leaflet PO4 plane) are
            # frequently ambiguous, so a moderate disagreement is not
            # diagnostic. We only fail on >50% disagreement, which would
            # indicate a serious structural mismatch (e.g. AF3 predicted
            # the wrong topology, or INSANE embedded the protein flipped).
            if pct_disagree > 50:
                failures.append(
                    f"topology vs embedded: {pct_disagree:.1f}% of "
                    f"DeepTMHMM-labelled residues disagree with their "
                    f"embedded z-position (threshold 50%); inspect "
                    f"per-chain breakdown — likely AF3-misoriented subunit"
                )

    summary = {
        "bilayer_thickness_nm": round(bilayer.get("thickness_nm", float("nan")), 2),
        "bilayer_centre_z_nm": round(bilayer.get("centre_z_nm", float("nan")), 2),
        "net_charge": round(net_charge, 3),
        "image_safe": image.get("safe", False),
        "min_image_dxyz_nm": (round(image.get("min_image_dx_nm", float("nan")), 2),
                               round(image.get("min_image_dy_nm", float("nan")), 2),
                               round(image.get("min_image_dz_nm", float("nan")), 2)),
        "topology_n_residues": len(topology_df),
        "topology_pct_agreement": (
            round(100 * topology_df["agreement"].sum() / len(topology_df), 1)
            if len(topology_df) else None),
        "ok": not failures,
    }

    return MembraneValidationReport(
        summary=summary,
        topology=topology_df,
        bilayer=bilayer,
        charge=charge,
        image_safety=image,
        failures=failures,
    )
