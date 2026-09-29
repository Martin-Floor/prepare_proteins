"""PPM 3.0 wrapper for membrane-protein orientation.

PPM 3.0 (Lomize et al., OPM, U Michigan) finds the optimal orientation of
a membrane protein with respect to a planar bilayer by minimising the
transfer energy of hydrophobic side chains into the membrane core. The
public web server at https://opm.phar.umich.edu/ppm_server3 is
intermittent; this module installs PPM 3.0 from its public source bucket
(``gs://opm-assets/ppm3_code/``) and runs it locally.

Public API::

    from prepare_proteins.MD import orient_with_ppm

    result = orient_with_ppm(
        input_pdb=Path("af3_apo.pdb"),
        output_dir=Path("orient_out"),
        membrane_type="ERf",            # ER (fungi)
        n_terminal_topology="out",       # lumen-facing N-terminus
        planar=True,                     # force flat bilayer
    )
    print(result["oriented_pdb"])        # bilayer-centered PDB at z=0
    print(result["tm_segments"])          # per-chain TM helices
    print(result["hydrophobic_thickness_A"], result["transfer_energy_kcal_mol"])

The orientation step plugs into ``martini_md`` as
``martini_md.orientWithPPM(...)`` which updates the workspace's input
PDB so subsequent ``setUpCG`` / ``buildMembrane`` steps work on the
oriented coordinates. ``buildMembrane`` can then be called with
``orient=False`` (auto-set when PPM has already run).

Membrane-type codes (set ``membrane_type=...``):
  ============== =====
  Plasma membrane (mammalian)              PMm
  Plasma membrane (plants)                 PMp
  Plasma membrane (Fungi)                  PMf
  ER (fungi)                               ERf
  ER (mammalian)                           ERm
  Golgi                                    GOL
  Lysosome                                 LYS
  Endosome                                 END
  Vacuole                                  VAC
  Outer mitochondrial                      MOM
  Inner mitochondrial                      MIM
  Thylakoid (plants / bacteria)            THp / THb
  Gram(-) outer / inner                    GnO / GnI
  Gram(+) inner                            GpI
  Archaebacteria cell                      ARC
  ============== =====

(Mind the case — ``ERf`` not ``Erf``; the docx is wrong, the source is right.)

Required system dependencies:
  - ``gfortran`` (any version >= 9 — only used to compile PPM 3.0 once)
  - ``curl`` (for the one-time source fetch)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# Files in the GCS public bucket that we need to compile PPM 3.0.
_PPM_SOURCE_FILES = [
    "1membrane.inp", "2membranes.inp", "Makefile",
    "deftm.f", "find_segm.f", "hbcor.f", "locate.f", "min.f", "opm.f",
    "orient.f", "profile.f", "read_small.f", "readpdb.f", "res.lib",
    "rmsd.f", "solva.f", "tilting.f", "watface.f",
]
_PPM_BUCKET_URL = "https://storage.googleapis.com/opm-assets/ppm3_code"
_PPM_BINARY_NAME = "immers"

# Membrane types known to PPM 3.0 (validates user input early).
MEMBRANE_TYPE_CODES = frozenset({
    "PMm", "PMp", "PMf", "ERf", "ERm", "GOL", "LYS", "END", "VAC",
    "MOM", "MIM", "THp", "THb", "GnO", "GnI", "GpI", "ARC",
    "MIC", "LPC", "MPC", "OPC", "EPC",
})


# ---------------------------------------------------------------------------
# Install / locate PPM 3.0 binary
# ---------------------------------------------------------------------------

def _default_cache_dir() -> Path:
    return Path(os.environ.get("PREPARE_PROTEINS_CACHE",
                                Path.home() / ".cache" / "prepare_proteins")
                ) / "ppm3"


def install_ppm3(cache_dir: Optional[os.PathLike] = None,
                  force: bool = False) -> Path:
    """Download PPM 3.0 source, compile with gfortran, return path to binary.

    Idempotent — if the binary already exists in the cache and ``force`` is
    False, returns it directly. Otherwise fetches the public source from
    ``gs://opm-assets/ppm3_code/`` and compiles via the bundled Makefile.

    Returns the path to the ``immers`` executable.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    src_dir = cache_dir / "src"
    binary = src_dir / _PPM_BINARY_NAME
    if binary.is_file() and not force:
        return binary

    if shutil.which("gfortran") is None:
        raise RuntimeError(
            "gfortran is required to build PPM 3.0; install it (e.g. "
            "`apt install gfortran`) and retry."
        )
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required to download PPM 3.0 source.")

    src_dir.mkdir(parents=True, exist_ok=True)
    for fname in _PPM_SOURCE_FILES:
        out = src_dir / fname
        if out.is_file() and out.stat().st_size > 0:
            continue
        url = f"{_PPM_BUCKET_URL}/{fname}"
        subprocess.run(["curl", "-sSLfo", str(out), url],
                        check=True)

    res = subprocess.run(["make"], cwd=str(src_dir), capture_output=True,
                          text=True)
    # gfortran spits warnings; only fail if no binary produced
    if not binary.is_file():
        raise RuntimeError(
            f"PPM 3.0 build failed.\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    return binary


# ---------------------------------------------------------------------------
# PDB cleanup before PPM
# ---------------------------------------------------------------------------

def _strip_non_terminal_oxt(pdb_in: Path, pdb_out: Path,
                              drop_chains: Sequence[str] = ()) -> None:
    """Remove non-terminal OXT atoms (CIF→PDB conversion artefact) and
    optionally drop entire chains. Writes a clean PDB file."""
    from openmm.app import PDBxFile, PDBFile, Modeller
    suffix = pdb_in.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        src = PDBxFile(str(pdb_in))
    else:
        src = PDBFile(str(pdb_in))
    modeller = Modeller(src.topology, src.positions)

    drop_set = set(drop_chains or ())
    if drop_set:
        drop_atoms = [a for a in modeller.topology.atoms()
                       if a.residue.chain.id in drop_set]
        if drop_atoms:
            modeller.delete(drop_atoms)

    last_residue_per_chain: Dict[str, int] = {}
    for chain in modeller.topology.chains():
        last = None
        for res in chain.residues():
            last = res
        if last is not None:
            last_residue_per_chain[chain.id] = last.index
    bad_oxt = [a for a in modeller.topology.atoms()
                if a.name == "OXT"
                and a.residue.index != last_residue_per_chain.get(
                    a.residue.chain.id, -1)]
    if bad_oxt:
        modeller.delete(bad_oxt)

    pdb_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pdb_out, "w") as fh:
        PDBFile.writeFile(modeller.topology, modeller.positions, fh,
                          keepIds=True)


# ---------------------------------------------------------------------------
# Run PPM + parse outputs
# ---------------------------------------------------------------------------

def _parse_datasub1(path: Path) -> Dict[str, List[Tuple[int, int, float]]]:
    """Returns {pdb_basename: {chain: [(tm_start, tm_end, tilt_deg), ...]}}."""
    if not path.is_file():
        return {}
    out: Dict[str, Dict[str, List[Tuple[int, int, float]]]] = {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(";") if p.strip()]
        if len(parts) < 3:
            continue
        pdb, chain, tilt_str = parts[0], parts[1], parts[2]
        try:
            tilt = float(tilt_str)
        except ValueError:
            continue
        # Each ";"-separated chunk after the chain+tilt may contain multiple
        # TM segments separated by commas, e.g. "1(   3-  30), 2( 153- 169)".
        # Use findall to grab every "(start-end)" pair.
        segs: List[Tuple[int, int, float]] = []
        rest = ";".join(parts[3:])
        for m in re.finditer(r"\(\s*(\d+)\s*-\s*(\d+)\s*\)", rest):
            segs.append((int(m.group(1)), int(m.group(2)), tilt))
        out.setdefault(pdb, {})[chain] = segs
    return out  # type: ignore[return-value]


def _parse_datapar1(path: Path) -> Dict[str, dict]:
    """datapar1 (planar): pdb_basename → dict(thickness, tilt, energy, ...)."""
    if not path.is_file():
        return {}
    out: Dict[str, dict] = {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(";") if p.strip()]
        if len(parts) < 5:
            continue
        try:
            out[parts[0]] = {
                "hydrophobic_thickness_A": float(parts[1]),
                "thickness_uncertainty_A": float(parts[2]),
                "tilt_deg": float(parts[3]),
                "transfer_energy_kcal_mol": float(parts[5]),
            }
        except (ValueError, IndexError):
            continue
    return out


def _bilayer_centre_from_dum(pdb_with_dum: Path) -> Tuple[float, float, float]:
    """Return (centre_z_A, lower_z_A, upper_z_A) from DUM atom z-distribution."""
    import numpy as np
    z = []
    for line in pdb_with_dum.read_text().splitlines():
        if line.startswith("HETATM") and "DUM" in line[12:20]:
            z.append(float(line[46:54]))
    if not z:
        raise RuntimeError(f"no DUM atoms in {pdb_with_dum}; PPM did not "
                            f"identify a bilayer plane")
    z_arr = np.asarray(z)
    return float(z_arr.mean()), float(z_arr.min()), float(z_arr.max())


def _write_centered_pdb(pdb_with_dum: Path, output_pdb: Path,
                         z_shift: float,
                         half_thickness_A: float) -> None:
    """Write protein-only PDB with z shifted so bilayer centre = 0."""
    out_lines = [
        "REMARK   1 PPM 3.0 oriented + bilayer-centered (z=0 = bilayer middle)",
        f"REMARK   2 hydrophobic half-thickness {half_thickness_A:.2f} A",
    ]
    for line in pdb_with_dum.read_text().splitlines():
        if line.startswith("ATOM"):
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            new_z = z + z_shift
            out_lines.append(f"{line[:30]}{x:8.3f}{y:8.3f}{new_z:8.3f}{line[54:]}")
    out_lines.append("END")
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(out_lines) + "\n")


def orient_with_ppm(
    input_pdb: os.PathLike,
    output_dir: os.PathLike,
    *,
    membrane_type: str = "ERf",
    n_terminal_topology: str = "out",
    include_heteroatoms: bool = False,
    planar: bool = True,
    chains: Optional[Sequence[str]] = None,
    drop_chains: Sequence[str] = (),
    binary_path: Optional[os.PathLike] = None,
    cache_dir: Optional[os.PathLike] = None,
    force_rebuild: bool = False,
) -> dict:
    """Run PPM 3.0 on a structure, return a centered + oriented PDB.

    Parameters
    ----------
    input_pdb
        Path to atomistic .pdb / .cif. CIF inputs are auto-converted via
        OpenMM's PDBxFile reader.
    output_dir
        Where to write PPM artefacts (input copy, raw PPM output, centered
        output, parsed metadata).
    membrane_type
        3-letter code from :py:data:`MEMBRANE_TYPE_CODES`. Default ``ERf``
        (fungal ER).
    n_terminal_topology
        ``"in"`` (cytoplasmic side) or ``"out"`` (lumen / extracellular)
        for the N-terminus of the first chain in the PDB. For SPC-like
        complexes where SEC11's catalytic domain is luminal, use
        ``"out"``.
    include_heteroatoms
        Whether to keep heteroatoms (non-standard residues) for membrane
        positioning. Solvent is always excluded by PPM.
    planar
        If True, force a planar bilayer fit (``2membranes.inp`` format
        with ``planar`` keyword). If False, PPM auto-selects between
        flat and curved.
    chains
        Subset of chain IDs to consider for orientation. Default: all.
    drop_chains
        Chains to remove from the input before PPM (e.g. small substrate
        chain that shouldn't influence bilayer fit).
    binary_path
        Path to a pre-built ``immers`` binary. Default: install via
        :func:`install_ppm3`.
    cache_dir
        Override the PPM source/binary cache location.
    force_rebuild
        Re-download + recompile PPM source.

    Returns
    -------
    dict
        ``{
            'oriented_pdb': Path,          # protein only, bilayer at z=0
            'oriented_pdb_with_dum': Path, # raw PPM output incl. DUM atoms
            'cleaned_input_pdb': Path,
            'tm_segments': {chain: [(start, end, tilt_deg), ...]},
            'hydrophobic_thickness_A': float,
            'transfer_energy_kcal_mol': float,
            'tilt_deg': float,
            'bilayer_centre_z_A': float,    # in input frame
            'membrane_type': str,
            'topology_n_term': str,
        }``

    Raises
    ------
    ValueError
        Invalid ``membrane_type`` or ``n_terminal_topology``.
    RuntimeError
        PPM compilation failed, no DUM atoms in output, or PPM exited with
        an error.
    """
    if membrane_type not in MEMBRANE_TYPE_CODES:
        raise ValueError(
            f"unknown membrane_type {membrane_type!r}; valid codes: "
            f"{sorted(MEMBRANE_TYPE_CODES)}"
        )
    if n_terminal_topology not in {"in", "out"}:
        raise ValueError("n_terminal_topology must be 'in' or 'out'.")

    binary = (Path(binary_path) if binary_path else
              install_ppm3(cache_dir=cache_dir, force=force_rebuild))
    if not binary.is_file():
        raise RuntimeError(f"PPM 3.0 binary not found at {binary}")
    src_dir = binary.parent  # res.lib lives next to the binary

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_pdb = Path(input_pdb).resolve()

    # Clean input → PPM working copy (kept as a record alongside outputs)
    cleaned_pdb = output_dir / f"{input_pdb.stem}.pdb"
    _strip_non_terminal_oxt(input_pdb, cleaned_pdb, drop_chains=drop_chains)

    # Stage in a working dir alongside the binary (PPM expects res.lib in cwd).
    # PPM 3.0 has a Fortran fixed-width filename buffer; long names overflow
    # silently and cause "Peripheral protein" misclassification. Use a short
    # canonical name (`input.pdb`) inside the work dir to be safe.
    workdir = output_dir / "_ppm_run"
    workdir.mkdir(parents=True, exist_ok=True)
    for required in (binary, src_dir / "res.lib"):
        if required.is_file():
            shutil.copy2(required, workdir / required.name)
    work_pdb_name = "input.pdb"
    shutil.copy2(cleaned_pdb, workdir / work_pdb_name)
    (workdir / binary.name).chmod(0o755)

    # Auto-detect chain list from the cleaned PDB if user didn't provide one;
    # PPM does not accept "all" — needs an explicit comma-separated list.
    if chains:
        detected_chains = list(chains)
    else:
        detected_chains = []
        seen: set = set()
        for line in cleaned_pdb.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                cid = line[21]
                if cid != " " and cid not in seen:
                    seen.add(cid)
                    detected_chains.append(cid)
        if not detected_chains:
            raise RuntimeError(
                f"could not detect chain IDs in {cleaned_pdb.name}; "
                "explicit `chains=` argument required."
            )

    # Build input file (uses the short work_pdb_name).
    if planar:
        chain_list = ",".join(detected_chains)
        het = "yes" if include_heteroatoms else "no"
        input_inp = (
            "2\n"
            f"{het}\n"
            f"{work_pdb_name}\n"
            "1\n"
            f"{membrane_type}\n"
            "planar\n"
            f"{n_terminal_topology}\n"
            f"{chain_list}\n"
        )
    else:
        het = "1" if include_heteroatoms else "0"
        input_inp = (
            "1\n"
            f" {het} {membrane_type} {n_terminal_topology} {work_pdb_name}\n"
        )
    (workdir / "input.inp").write_text(input_inp)

    # Run PPM
    log_path = output_dir / "ppm_run.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            [f"./{binary.name}"], stdin=open(workdir / "input.inp"),
            cwd=str(workdir), stdout=logf, stderr=subprocess.STDOUT,
        )

    raw_out = workdir / "inputout.pdb"
    if not raw_out.is_file():
        raise RuntimeError(
            f"PPM did not produce {raw_out.name}; check {log_path}"
        )

    # Parse — PPM truncates names to ~14 chars in the data files, so for the
    # `input.pdb` working copy the key is `input` (or `input.pdb` literal).
    datasub1 = workdir / "datasub1"
    datapar1 = workdir / "datapar1"
    datasub_data = _parse_datasub1(datasub1)
    datapar_data = _parse_datapar1(datapar1)
    # Pick the only / first key in each table.
    sub_key = next(iter(datasub_data), None)
    par_key = next(iter(datapar_data), None)
    tm_segments = datasub_data.get(sub_key, {}) if sub_key else {}
    summary = datapar_data.get(par_key, {}) if par_key else {}

    # Center on bilayer
    centre_z, lower_z, upper_z = _bilayer_centre_from_dum(raw_out)
    half_thickness = (upper_z - lower_z) / 2
    centered = output_dir / f"{input_pdb.stem}_oriented.pdb"
    _write_centered_pdb(raw_out, centered, z_shift=-centre_z,
                         half_thickness_A=half_thickness)

    # Persist outputs (rename the raw output to use the input stem so it's
    # easy to associate with multiple PPM runs in the same directory).
    persisted_raw = output_dir / f"{input_pdb.stem}_with_dum.pdb"
    shutil.copy2(raw_out, persisted_raw)
    if datasub1.is_file():
        shutil.copy2(datasub1, output_dir / "datasub1")
    if datapar1.is_file():
        shutil.copy2(datapar1, output_dir / "datapar1")

    result = {
        "oriented_pdb": centered,
        "oriented_pdb_with_dum": persisted_raw,
        "cleaned_input_pdb": cleaned_pdb,
        "tm_segments": tm_segments,
        "hydrophobic_thickness_A": summary.get(
            "hydrophobic_thickness_A", float("nan")),
        "transfer_energy_kcal_mol": summary.get(
            "transfer_energy_kcal_mol", float("nan")),
        "tilt_deg": summary.get("tilt_deg", float("nan")),
        "bilayer_centre_z_A": centre_z,
        "bilayer_half_thickness_A": half_thickness,
        "membrane_type": membrane_type,
        "topology_n_term": n_terminal_topology,
        "log_path": log_path,
    }
    (output_dir / "ppm_summary.json").write_text(
        json.dumps(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in result.items()},
            indent=2, default=str,
        ) + "\n"
    )
    return result
