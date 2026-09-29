"""MARTINI3 coarse-grained MD setup driving OpenMM via `martini_openmm`.

Mirrors the API shape of ``mlcg_setup.mlcg_md`` and ``openmm_setup.openmm_md``:
a per-system class with build steps that produce GROMACS-format CG topology +
coordinates (via vermouth's ``martinize2`` CLI and the ``insane`` membrane
builder), then writes a self-contained OpenMM run script that loads the
GROMACS topology through ``martini_openmm.MartiniTopFile`` and runs minimisation
→ equilibration → production on a single GPU. Per-replica job directories
are produced via :py:meth:`martini_md.prepareJobs`, returning a command list
suitable for ``bsc_calculations.mn5.jobArrays(jobs, program='openmm', ...)``.

External CLI tools required (run on the cluster, not at setup time):
  * ``martinize2`` (from the ``vermouth`` Python package)
  * ``insane`` (Python tool; ``pip install insane`` or bundled with vermouth)
  * ``dssp``  (for secondary-structure annotation; optional, default-on)

Python runtime (cluster, in the ``openmm_cuda`` conda env):
  * ``openmm >= 8.0``
  * ``martini_openmm`` (https://github.com/maccallumlab/martini_openmm)
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_MARTINI_OPENMM_IMPORT_ERROR: Optional[Exception] = None
try:  # optional dep: only used by the run-time helper, not by setup
    import martini_openmm  # type: ignore  # noqa: F401
    _MARTINI_OPENMM_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    martini_openmm = None  # type: ignore[assignment]
    _MARTINI_OPENMM_AVAILABLE = False
    _MARTINI_OPENMM_IMPORT_ERROR = exc


def _ensure_external(tool: str, hint: str = "") -> None:
    """Raise FileNotFoundError if ``tool`` is not on $PATH at setup time."""
    if shutil.which(tool) is None:
        msg = f"External tool {tool!r} not found on $PATH."
        if hint:
            msg += f" {hint}"
        raise FileNotFoundError(msg)


def _run_command(command: Sequence[str], cwd: Optional[os.PathLike] = None,
                 log_path: Optional[os.PathLike] = None,
                 check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with stdout+stderr captured; optionally tee to a log file."""
    cwd_str = str(cwd) if cwd is not None else None
    res = subprocess.run(
        list(command), cwd=cwd_str, capture_output=True, text=True, check=False,
    )
    if log_path is not None:
        with open(log_path, "a") as fh:
            fh.write(f"$ {' '.join(shlex.quote(c) for c in command)}\n")
            fh.write(res.stdout or "")
            if res.stderr:
                fh.write("\n--- STDERR ---\n")
                fh.write(res.stderr)
            fh.write(f"\n--- exit {res.returncode} ---\n\n")
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {res.returncode}): "
            f"{' '.join(shlex.quote(c) for c in command)}\n"
            f"stderr:\n{res.stderr or '(empty)'}"
        )
    return res


# --------------------------------------------------------------------------
# Default lipid mixes for ER-mimicking bilayers (mole fractions, must sum ≈1)
# Composition follows Casares et al. 2019 yeast ER-membrane CG models;
# users can override via buildMembrane(lipid_composition=...).
# --------------------------------------------------------------------------
ER_LIKE_BILAYER = {
    "POPC": 0.50,
    "POPE": 0.30,
    "POPI": 0.10,
    "ERG":  0.10,   # ergosterol
}


def _validate_lipid_mix(mix: Mapping[str, float]) -> Dict[str, float]:
    if not mix:
        raise ValueError("lipid_composition must contain at least one lipid type.")
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("lipid mole fractions must be positive.")
    return {k: float(v) / total for k, v in mix.items()}


class martini_md:
    """Set up a MARTINI3 + OpenMM CG-MD job for a single atomistic input PDB.

    Typical flow::

        m = martini_md("input.pdb")
        m.setUpCG("workspace")
        m.buildMembrane("workspace", lipid_composition={"POPC": 1.0})
        m.setUpOpenMMSimulation("workspace", prod_ns=5000)
        jobs = m.prepareJobs("workspace/jobs", n_replicas=3)
        # → list of `python run_martini.py --replica-dir <…>` commands;
        #   pass to bsc_calculations.mn5.jobArrays(jobs, program="openmm")

    All artefacts (CG PDB, topology .top + .itp, membrane-embedded system,
    run script) are written under ``workspace/``. The class is *stateless
    across calls* in the sense that each method writes files; intermediate
    state is reloaded from disk by the next step. This matches the existing
    ``openmm_md``/``mlcg_md`` pattern.
    """

    # ----------------------------------------------------------------------
    # Construction & basic state
    # ----------------------------------------------------------------------
    def __init__(self, input_pdb: os.PathLike) -> None:
        input_pdb = Path(input_pdb).resolve()
        if not input_pdb.is_file():
            raise FileNotFoundError(input_pdb)
        if input_pdb.suffix.lower() not in {".pdb"}:
            raise ValueError(
                f"martini_md expects an atomistic .pdb file, got {input_pdb.suffix!r}. "
                "Convert .cif → .pdb first (e.g. with `gmx editconf` or pymol)."
            )
        self.input_pdb: Path = input_pdb
        self.pdb_name: str = input_pdb.stem
        self.command_log: List[Dict[str, str]] = []

        # Filled in by setUpCG / buildMembrane / setUpOpenMMSimulation
        self.cg_pdb: Optional[Path] = None
        self.cg_top: Optional[Path] = None
        self.cg_itp_files: List[Path] = []
        self.system_gro: Optional[Path] = None
        self.system_top: Optional[Path] = None
        self.run_script: Optional[Path] = None
        self.simulation_config: Optional[Dict[str, Any]] = None
        # Flag set by orientWithPPM(); buildMembrane uses it to
        # auto-disable INSANE's -orient (which would re-rotate the
        # already-oriented protein and undo PPM's work).
        self._ppm_oriented: bool = False

    # ----------------------------------------------------------------------
    # Step 1 — atomistic → CG (martinize2)
    # ----------------------------------------------------------------------
    def setUpCG(self, output_dir: os.PathLike, *,
                ff: str = "martini3001",
                elastic_network: bool = True,
                elastic_force_kJmol_nm2: float = 700.0,
                elastic_lower_nm: float = 0.5,
                elastic_upper_nm: float = 0.9,
                elastic_unit: str = "molecule",
                elastic_regions: Optional[Sequence[str]] = None,
                ss_source: str = "dssp",
                merge_chains: Optional[Sequence[str]] = None,
                neutral_termini: bool = False,
                command_log_name: str = "martinize2.log",
                extra_args: Optional[Sequence[str]] = None,
                ) -> Dict[str, Path]:
        """Run ``martinize2`` to convert the atomistic input PDB to MARTINI3 CG.

        Parameters
        ----------
        output_dir
            Directory to write CG artefacts (created if missing).
        ff
            MARTINI force field identifier (vermouth name). MARTINI3 main
            release is ``"martini3001"``.
        elastic_network
            Apply an elastic network model to maintain protein folds.
        elastic_force_kJmol_nm2, elastic_lower_nm, elastic_upper_nm
            ENM force constant and bond-distance window. The MARTINI3
            recommendation for membrane proteins is k=700 kJ/mol/nm², bonds
            for Cα-pairs with 0.5 nm < d < 0.9 nm.
        elastic_unit
            ``"molecule"`` (default; ENM within each merged molecule),
            ``"chain"`` (each chain gets its own ENM), or ``"all"`` (ENM
            across all atoms). Maps to martinize2's ``-eunit``.
        elastic_regions
            Optional list of region selectors of the form
            ``"<chain>-<start_resid>:<end_resid>"`` passed verbatim to
            ``-eunit``. ENM bonds will only form within each region.
            Use this to keep a specific chain (e.g. the substrate) free
            of ENM by listing only the receptor regions.
        ss_source
            ``"dssp"`` (run DSSP on the input PDB), ``"random_coil"``
            (force every residue to RC, useful for unfolded substrates),
            or a per-residue string the same length as the input sequence.
        merge_chains
            If provided, merge these chains into a single martinize2 molecule
            (otherwise each chain becomes its own .itp file).
        neutral_termini
            Use neutral N/C termini (``-nter none -cter none``).
        extra_args
            Additional CLI arguments to pass through to ``martinize2``.
        """
        _ensure_external("martinize2",
                         hint="Install via `pip install vermouth` (provides martinize2 CLI).")
        if ss_source.lower() == "dssp":
            _ensure_external("mkdssp", hint="Install DSSP (mkdssp) or pass ss_source='random_coil'.")

        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / command_log_name

        cg_pdb = out / f"{self.pdb_name}_cg.pdb"
        cg_top = out / "topol.top"

        cmd: List[str] = [
            "martinize2",
            "-f", str(self.input_pdb),
            "-x", str(cg_pdb),
            "-o", str(cg_top),
            "-ff", ff,
        ]
        if ss_source.lower() == "dssp":
            cmd.extend(["-dssp", "mkdssp"])
        elif ss_source.lower() == "random_coil":
            cmd.extend(["-ss", "C"])  # martinize2 accepts 'C' to force RC
        else:
            cmd.extend(["-ss", str(ss_source)])

        if elastic_network:
            cmd.extend([
                "-elastic",
                "-ef", str(elastic_force_kJmol_nm2),
                "-el", str(elastic_lower_nm),
                "-eu", str(elastic_upper_nm),
            ])
            if elastic_regions:
                cmd.append("-eunit")
                cmd.extend(str(r) for r in elastic_regions)
            elif elastic_unit and elastic_unit != "molecule":
                cmd.extend(["-eunit", str(elastic_unit)])
        if merge_chains:
            cmd.extend(["-merge", ",".join(merge_chains)])
        if neutral_termini:
            cmd.extend(["-nter", "none", "-cter", "none"])
        if extra_args:
            cmd.extend(list(extra_args))

        _run_command(cmd, cwd=out, log_path=log_path, check=True)
        self.command_log.append({"step": "setUpCG", "log": str(log_path)})

        # martinize2 writes molecule_*.itp next to the .top
        itp_files = sorted(out.glob("molecule_*.itp"))
        # Plus an elastic-network header file is typically embedded in the .top.

        self.cg_pdb = cg_pdb
        self.cg_top = cg_top
        self.cg_itp_files = itp_files
        return {"cg_pdb": cg_pdb, "cg_top": cg_top, "itp_files": itp_files}

    # ----------------------------------------------------------------------
    # Step 2 — embed in membrane + solvate (insane)
    # ----------------------------------------------------------------------
    def buildMembrane(self, output_dir: os.PathLike, *,
                      lipid_composition: Optional[Mapping[str, float]] = None,
                      box_xy_nm: float = 15.0,
                      box_z_nm: float = 12.0,
                      orient: Optional[bool] = None,
                      sol: str = "W",
                      salt_M: float = 0.15,
                      command_log_name: str = "insane.log",
                      extra_args: Optional[Sequence[str]] = None,
                      ) -> Dict[str, Path]:
        """Run ``insane`` to embed the CG protein in a bilayer + solvate.

        Requires :py:meth:`setUpCG` to have run first.
        Default lipid mix (``ER_LIKE_BILAYER``): 50 % POPC + 30 % POPE +
        10 % POPI + 10 % ergosterol — a yeast-ER-style composition.
        """
        if self.cg_pdb is None or self.cg_top is None:
            raise RuntimeError("Call setUpCG(...) before buildMembrane(...).")
        _ensure_external("insane", hint="Install via `pip install insane`.")

        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / command_log_name

        mix = _validate_lipid_mix(
            lipid_composition if lipid_composition is not None else ER_LIKE_BILAYER
        )

        system_gro = out / f"{self.pdb_name}_system.gro"
        system_top = out / "system.top"

        cmd: List[str] = [
            "insane",
            "-f", str(self.cg_pdb),
            "-o", str(system_gro),
            "-p", str(system_top),
            "-x", str(box_xy_nm), "-y", str(box_xy_nm), "-z", str(box_z_nm),
            "-pbc", "rectangular",
            "-sol", sol,
            "-salt", str(salt_M),
        ]
        # If orient=None, auto-disable when PPM has already oriented the
        # protein (otherwise INSANE's -orient would re-rotate it and undo
        # PPM's choice). Default for non-PPM workspaces: orient=True.
        if orient is None:
            orient = not self._ppm_oriented
        if orient:
            cmd.append("-orient")  # auto-orient TM helices along z
        # insane lipid syntax: `-l NAME[:#]` (relative abundance). For a
        # single-lipid bilayer the `:#` suffix is omitted; for mixes we pass
        # mole fractions normalised to sum 1 (insane handles relative units).
        if len(mix) == 1:
            (lipid,) = mix.keys()
            cmd.extend(["-l", lipid])
        else:
            for lipid, frac in mix.items():
                cmd.extend(["-l", f"{lipid}:{frac:.4f}"])
        if extra_args:
            cmd.extend(list(extra_args))

        _run_command(cmd, cwd=out, log_path=log_path, check=True)
        self.command_log.append({"step": "buildMembrane", "log": str(log_path)})

        # insane writes a .top that #includes "martini.itp" and lists the
        # lipid/solvent/ion molecules but does NOT include the protein .itp
        # files. Rewrite the top to merge in the FF and protein includes so
        # martini_openmm can load it directly.
        self._merge_protein_into_membrane_top(system_top)

        # insane's -salt computes ion counts from concentration × volume,
        # which can leave the system off-neutral by ±1 due to integer
        # rounding. Detect and fix.
        self._neutralize_system(system_gro, system_top)

        self.system_gro = system_gro
        self.system_top = system_top
        return {"system_gro": system_gro, "system_top": system_top}

    def _merge_protein_into_membrane_top(self, system_top: Path) -> None:
        """Replace insane's ``#include "martini.itp"`` with explicit MARTINI3
        force-field includes and the protein .itp files generated by
        martinize2, and prepend the protein molecule entries to the
        ``[ molecules ]`` block (in the order they appear in the .gro file)."""
        if not self.cg_top or not self.cg_itp_files:
            return
        text = system_top.read_text()

        # 1. Build the include block we want at the top.
        ff_includes = [
            "martini_v3.0.0.itp",
            "martini_v3.0.0_phospholipids_v1.itp",
            "martini_v3.0.0_solvents_v1.itp",
            "martini_v3.0.0_ions_v1.itp",
        ]
        protein_includes = [p.name for p in self.cg_itp_files]
        new_includes = "\n".join(
            f'#include "{f}"' for f in ff_includes + protein_includes
        )
        # 2. Replace the first `#include "martini.itp"` line; if not found,
        #    prepend.
        if '#include "martini.itp"' in text:
            text = text.replace('#include "martini.itp"', new_includes, 1)
        else:
            text = new_includes + "\n\n" + text

        # 3. Replace insane's placeholder protein entry with the actual
        #    martinize2 protein molecule entries. insane writes a single
        #    `Protein   1` line that doesn't match any .itp [moleculetype].
        protein_molnames = [p.stem for p in self.cg_itp_files]
        mol_lines = "\n".join(f"{name}    1" for name in protein_molnames)

        import re
        # Remove insane's placeholder line (any single-protein placeholder
        # name that isn't one of our martinize2 names).
        text = re.sub(
            r"^\s*Protein\s+\d+\s*\n",
            "",
            text, count=1, flags=re.MULTILINE,
        )
        # Inject our protein molecule lines right after [ molecules ] header
        text = re.sub(
            r"(\[\s*molecules\s*\][^\n]*\n(?:;[^\n]*\n)?)",
            r"\1" + mol_lines + "\n",
            text, count=1,
        )
        system_top.write_text(text)

    def _neutralize_system(self, system_gro: Path, system_top: Path,
                            cation_name: str = "NA",
                            anion_name: str = "CL") -> None:
        """Adjust ion counts so the system is electroneutral.

        insane's `-salt` flag uses concentration × volume which can leave
        the system off-neutral by ±1 due to integer rounding. We compute
        the net charge from system.top and convert excess water beads
        into the appropriate counter-ion to balance.
        """
        from .membrane_validation import compute_net_charge
        net, _ = compute_net_charge(system_top)
        delta = int(round(net))
        if delta == 0:
            return
        ion_name = anion_name if delta > 0 else cation_name
        n_to_add = abs(delta)

        # 1. Edit system.gro: convert n_to_add W beads → ion beads.
        gro_text = system_gro.read_text().splitlines()
        n_atoms = int(gro_text[1])
        new_lines = list(gro_text)
        flipped = 0
        for i in range(2 + n_atoms - 1, 1, -1):
            line = new_lines[i]
            if len(line) < 15:
                continue
            resname = line[5:10].strip()
            atomname = line[10:15].strip()
            if resname == "W" and atomname == "W":
                new_line = (line[:5]
                            + f"{ion_name:<5s}"
                            + f"{ion_name:>5s}"
                            + line[15:])
                new_lines[i] = new_line
                flipped += 1
                if flipped == n_to_add:
                    break
        if flipped < n_to_add:
            raise RuntimeError(
                f"could not find {n_to_add} W beads to convert; only "
                f"{flipped} found in {system_gro.name}")
        system_gro.write_text("\n".join(new_lines) + "\n")

        # 2. Edit system.top: decrement W count, increment ion count
        #    in the [molecules] block.
        import re
        text = system_top.read_text()
        m_w = re.search(r"^\s*W\s+(\d+)\s*$", text, flags=re.MULTILINE)
        if not m_w:
            raise RuntimeError("[molecules] block has no W entry")
        old_w = int(m_w.group(1))
        new_w = old_w - n_to_add
        text = text[:m_w.start()] + f"W {new_w}\n" + text[m_w.end() + 1:]
        m_ion = re.search(rf"^\s*{ion_name}\s+(\d+)\s*$", text,
                           flags=re.MULTILINE)
        if m_ion:
            old_ion = int(m_ion.group(1))
            new_ion = old_ion + n_to_add
            text = (text[:m_ion.start()]
                    + f"{ion_name} {new_ion}\n"
                    + text[m_ion.end() + 1:])
        else:
            text = text.rstrip() + f"\n{ion_name} {n_to_add}\n"
        system_top.write_text(text)

    # ----------------------------------------------------------------------
    # Step 3 — write OpenMM run script
    # ----------------------------------------------------------------------
    def setUpOpenMMSimulation(self, output_dir: os.PathLike, *,
                               dt_fs: int = 20,
                               equilibration_ns: float = 20.0,
                               prod_ns: float = 5000.0,
                               chunk_size_ns: float = 1000.0,
                               temperature_K: float = 310.0,
                               pressure_bar: float = 1.0,
                               nb_cutoff_nm: float = 1.1,
                               epsilon_r: float = 15.0,
                               save_xtc_every_ns: float = 0.5,
                               save_log_every_ns: float = 0.1,
                               minimize_max_iter: int = 10000,
                               script_name: str = "run_martini.py",
                               restrained_minimize_kJmol_nm2: float = 1000.0,
                               equilibrate_with_restraints: bool = True,
                               ) -> Dict[str, Any]:
        """Write a self-contained ``run_martini.py`` that runs minimise →
        equilibrate → produce in OpenMM via ``martini_openmm.MartiniTopFile``.

        The script is identical for every replica; per-replica state lives
        in the replica directory passed at runtime via ``--replica-dir``.

        Defaults follow MARTINI3 standard simulation protocol:
          - dt = 20 fs
          - LangevinMiddleIntegrator at 310 K, friction 1 ps⁻¹
          - MonteCarloMembraneBarostat (1 bar, semiisotropic, ZFree)
          - non-bonded cutoff 1.1 nm, reaction-field with ε_r = 15

        Production is split into chunks of ``chunk_size_ns`` (default 1 µs).
        Each chunk produces ``production_<NN>.{xtc,log,chk}`` artefacts.
        Re-running the same script in the same replica directory auto-resumes
        from the latest chunk checkpoint — re-submit until ``prod_ns`` is
        reached. This mirrors the chunk pattern in
        ``prepare_proteins.scripts.md.openmm.openmm_simulation.py``.
        """
        if self.system_gro is None or self.system_top is None:
            raise RuntimeError("Call buildMembrane(...) before setUpOpenMMSimulation(...).")
        if chunk_size_ns <= 0:
            raise ValueError("chunk_size_ns must be positive.")
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        run_path = out / script_name

        cfg: Dict[str, Any] = {
            "dt_fs": int(dt_fs),
            "equilibration_ns": float(equilibration_ns),
            "prod_ns": float(prod_ns),
            "chunk_size_ns": float(chunk_size_ns),
            "temperature_K": float(temperature_K),
            "pressure_bar": float(pressure_bar),
            "nb_cutoff_nm": float(nb_cutoff_nm),
            "epsilon_r": float(epsilon_r),
            "save_xtc_every_ns": float(save_xtc_every_ns),
            "save_log_every_ns": float(save_log_every_ns),
            "minimize_max_iter": int(minimize_max_iter),
            "restrained_minimize_kJmol_nm2": float(restrained_minimize_kJmol_nm2),
            "equilibrate_with_restraints": bool(equilibrate_with_restraints),
            "system_gro_basename": self.system_gro.name,
            "system_top_basename": self.system_top.name,
        }
        run_path.write_text(_RUN_SCRIPT_TEMPLATE.format(**cfg))
        run_path.chmod(0o755)

        cfg_path = out / "simulation_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        self.run_script = run_path
        self.simulation_config = cfg
        return {"run_script": run_path, "config": cfg, "config_path": cfg_path}

    # ----------------------------------------------------------------------
    # PPM 3.0 orientation hook — replace input PDB with bilayer-centered one
    # ----------------------------------------------------------------------
    def orientWithPPM(self, *,
                       output_dir: Optional[os.PathLike] = None,
                       membrane_type: str = "ERf",
                       n_terminal_topology: str = "out",
                       planar: bool = True,
                       include_heteroatoms: bool = False,
                       chains: Optional[Sequence[str]] = None,
                       drop_chains: Sequence[str] = (),
                       cache_dir: Optional[os.PathLike] = None,
                       force_rebuild: bool = False,
                       ) -> dict:
        """Run PPM 3.0 on the workspace's input PDB; replace it with the
        bilayer-centered output.

        After this returns, ``self.input_pdb`` points to the PPM-oriented
        PDB (membrane plane at z=0). Subsequent ``setUpCG`` /
        ``buildMembrane`` will operate on it. ``buildMembrane`` should
        be called with ``orient=False`` to avoid re-orientation by
        INSANE; this is set automatically when the embedded
        ``self._ppm_oriented`` flag is True.

        Parameters: see :func:`prepare_proteins.MD.ppm.orient_with_ppm`.

        Returns the result dict from
        :func:`prepare_proteins.MD.ppm.orient_with_ppm`.
        """
        from openmm.app import PDBFile, Modeller
        from .ppm import orient_with_ppm

        if output_dir is None:
            output_dir = self.input_pdb.parent / "ppm_orient"
        result = orient_with_ppm(
            self.input_pdb, output_dir,
            membrane_type=membrane_type,
            n_terminal_topology=n_terminal_topology,
            planar=planar,
            include_heteroatoms=include_heteroatoms,
            chains=chains, drop_chains=drop_chains,
            cache_dir=cache_dir, force_rebuild=force_rebuild,
        )
        # Re-init from oriented PDB
        self.input_pdb = result["oriented_pdb"]
        self.pdb_name = self.input_pdb.stem
        self._ppm_oriented = True
        self.command_log.append({"step": "orientWithPPM",
                                  "result_dir": str(output_dir)})
        return result

    # ----------------------------------------------------------------------
    # Validation hook — call directly or via prepareJobs(validate=True)
    # ----------------------------------------------------------------------
    def validateEmbedding(self, *,
                           subunits: Optional[Sequence[Tuple[str, str, os.PathLike]]] = None,
                           cutoff_nm: float = 1.1,
                           image_buffer_nm: float = 1.0,
                           lipid_resname: str = "POPC",
                           head_atom: str = "PO4",
                           deeptmhmm: bool = True,
                           cache_dir: Optional[os.PathLike] = None,
                           ) -> "MembraneValidationReport":
        """Run Tier-1 pre-MD validation on the workspace's system.gro/.top.

        Wraps :func:`prepare_proteins.MD.membrane_validation.validate_membrane_embedding`
        with the workspace state already on the class.

        Parameters
        ----------
        subunits
            ``[(chain_id, label, fasta_path), ...]`` for protein topology
            cross-check. If None, only bilayer / charge / image-safety
            checks run.
        Other parameters: see :func:`validate_membrane_embedding`.

        Returns
        -------
        MembraneValidationReport
            Has ``.ok`` attribute and ``.failures`` list.
        """
        if self.system_gro is None or self.system_top is None:
            raise RuntimeError("Call buildMembrane(...) before validateEmbedding(...).")
        from .membrane_validation import validate_membrane_embedding
        return validate_membrane_embedding(
            self.system_gro, self.system_top,
            subunits=subunits or (),
            cutoff_nm=cutoff_nm, image_buffer_nm=image_buffer_nm,
            lipid_resname=lipid_resname, head_atom=head_atom,
            deeptmhmm=deeptmhmm, cache_dir=cache_dir,
        )

    # ----------------------------------------------------------------------
    # Step 4 — per-replica directory layout, returns job-command list
    # ----------------------------------------------------------------------
    def prepareJobs(self, jobs_folder: os.PathLike, *,
                    n_replicas: int = 3,
                    seed_offsets: Optional[Sequence[int]] = None,
                    use_symlinks: bool = True,
                    validate: bool = True,
                    validate_subunits: Optional[Sequence[Tuple[str, str, os.PathLike]]] = None,
                    validate_kwargs: Optional[dict] = None,
                    ) -> List[str]:
        """Create per-replica subdirectories with all files needed at runtime
        and return the list of commands (one per replica) suitable for
        :py:func:`bsc_calculations.mn5.jobArrays`.

        Each replica directory contains:
          * symlink (or copy) of system.gro and system.top
          * symlink (or copy) of all .itp files referenced by the .top
          * a copy of run_martini.py
          * a replica.json with {replica_index, random_seed}

        If ``validate=True`` (default), runs Tier-1 pre-MD checks via
        :py:meth:`validateEmbedding` first; raises ``RuntimeError`` if any
        check fails. Pass ``validate_subunits=[(chain, label, fasta), ...]``
        to enable the DeepTMHMM topology cross-check; otherwise only the
        bilayer / charge / image-safety checks run. Pass
        ``validate=False`` to skip validation entirely.
        """
        if self.run_script is None or self.system_gro is None or self.system_top is None:
            raise RuntimeError("Call setUpOpenMMSimulation(...) before prepareJobs(...).")

        if validate:
            extra_kwargs = validate_kwargs or {}
            try:
                report = self.validateEmbedding(
                    subunits=validate_subunits, **extra_kwargs)
            except ImportError as exc:
                raise RuntimeError(
                    f"Cannot validate membrane embedding: {exc}. "
                    "Either install MDAnalysis (and pybiolib for DeepTMHMM) "
                    "or call prepareJobs(validate=False) to skip."
                ) from exc
            if not report.ok:
                raise RuntimeError(
                    "Membrane-embedding validation FAILED. "
                    "Inspect the system before submitting MD:\n  - "
                    + "\n  - ".join(report.failures)
                    + "\nPass prepareJobs(validate=False) to override."
                )

        jobs_folder = Path(jobs_folder).resolve()
        jobs_folder.mkdir(parents=True, exist_ok=True)
        if seed_offsets is None:
            seed_offsets = list(range(1, n_replicas + 1))
        if len(seed_offsets) != n_replicas:
            raise ValueError("seed_offsets must have length n_replicas.")

        # Collect all .itp / topol_* / molecule_* files referenced by the .top.
        # (Conservative: copy *every* .itp in system.top's directory.)
        src_dir = self.system_top.parent
        itp_files = list(src_dir.glob("*.itp"))

        commands: List[str] = []
        for i, seed in zip(range(n_replicas), seed_offsets):
            rdir = jobs_folder / f"replica_{i + 1:02d}"
            rdir.mkdir(parents=True, exist_ok=True)

            for src in [self.system_gro, self.system_top, self.run_script, *itp_files]:
                dst = rdir / src.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if use_symlinks:
                    # Relative symlink so it resolves wherever the workspace
                    # ends up (including after rsync to a remote cluster).
                    rel = os.path.relpath(src, start=dst.parent)
                    dst.symlink_to(rel)
                else:
                    shutil.copy2(src, dst)

            (rdir / "replica.json").write_text(json.dumps(
                {"replica_index": i + 1, "random_seed": int(seed),
                 "model_name": self.pdb_name}, indent=2) + "\n")

            # The SLURM script runs from `jobs_folder.parent` (the workspace
            # root, where bsc_calculations.mn5.jobArrays is invoked), so the
            # --replica-dir path must be relative to that, not just `rdir.name`
            # (which would resolve to <workspace>/replica_NN, the wrong level).
            rel_rdir = os.path.relpath(rdir, start=jobs_folder.parent)
            commands.append(
                f"python {self.run_script.name} --replica-dir {rel_rdir} "
                f"--random-seed {int(seed)}"
            )
        return commands


# --------------------------------------------------------------------------
# Run-script template, filled by setUpOpenMMSimulation. Triple-braces are
# escaped to literal single braces in the output via str.format.
# --------------------------------------------------------------------------
_RUN_SCRIPT_TEMPLATE = textwrap.dedent('''\
    #!/usr/bin/env python
    """Auto-generated MARTINI3 + OpenMM production script.

    Run as:
        python run_martini.py --replica-dir replica_01 --random-seed 1

    The replica-dir contains the system.gro/.top/.itp + this script.
    """
    from __future__ import annotations
    import argparse, gc, json, os, sys, time
    from pathlib import Path

    from openmm import (
        Platform, MonteCarloMembraneBarostat, LangevinMiddleIntegrator,
    )
    from openmm.app import (
        Simulation, GromacsGroFile, StateDataReporter,
        DCDReporter, CheckpointReporter,
    )
    from openmm.unit import (
        nanometer, picosecond, femtosecond, kelvin, bar, kilojoule_per_mole,
    )

    try:
        from openmm.app import XTCReporter
        _HAS_XTC = True
    except ImportError:
        _HAS_XTC = False

    from martini_openmm import MartiniTopFile

    DT_FS = {dt_fs}
    EQUIL_NS = {equilibration_ns}
    PROD_NS = {prod_ns}
    CHUNK_NS = {chunk_size_ns}
    TEMP_K = {temperature_K}
    PRESSURE_BAR = {pressure_bar}
    NB_CUTOFF_NM = {nb_cutoff_nm}
    EPSILON_R = {epsilon_r}
    SAVE_XTC_EVERY_NS = {save_xtc_every_ns}
    SAVE_LOG_EVERY_NS = {save_log_every_ns}
    # Mid-chunk checkpoint save interval (ns). The CheckpointReporter only
    # writes a binary state file, so a small value is cheap. This bounds
    # data loss on a SIGKILL/timeout to at most SAVE_CHK_EVERY_NS of trajectory.
    SAVE_CHK_EVERY_NS = 50.0
    MINIMIZE_MAX_ITER = {minimize_max_iter}
    RESTR_K = {restrained_minimize_kJmol_nm2}
    EQUIL_WITH_RESTRAINTS = {equilibrate_with_restraints}
    SYSTEM_GRO = "{system_gro_basename}"
    SYSTEM_TOP = "{system_top_basename}"

    EQUIL_CHK = "equilibration.chk"


    def ns_to_steps(ns: float) -> int:
        return int(round(ns * 1.0e6 / DT_FS))


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--replica-dir", type=Path, default=Path("."),
                         help="Working directory containing system.{{gro,top}}")
        ap.add_argument("--random-seed", type=int, default=1)
        ap.add_argument("--total-ns", type=float, default=PROD_NS,
                         help="Total production length (ns); default {{PROD_NS}}.")
        ap.add_argument("--chunk-ns", type=float, default=CHUNK_NS,
                         help="Chunk length (ns); default {{CHUNK_NS}}.")
        args = ap.parse_args()
        rdir = args.replica_dir.resolve()

        gro_path = rdir / SYSTEM_GRO
        top_path = rdir / SYSTEM_TOP
        if not gro_path.is_file() or not top_path.is_file():
            print(f"missing {{gro_path}} or {{top_path}}", file=sys.stderr)
            sys.exit(2)

        gro = GromacsGroFile(str(gro_path))
        box = gro.getPeriodicBoxVectors()

        top = MartiniTopFile(
            str(top_path),
            periodicBoxVectors=box,
            epsilon_r=EPSILON_R,
        )
        system = top.create_system(nonbonded_cutoff=NB_CUTOFF_NM * nanometer)

        # Membrane barostat: pressure-coupled in xy and z separately.
        barostat = MonteCarloMembraneBarostat(
            PRESSURE_BAR * bar,
            0.0 * bar * nanometer,
            TEMP_K * kelvin,
            MonteCarloMembraneBarostat.XYIsotropic,
            MonteCarloMembraneBarostat.ZFree,
            25,
        )
        system.addForce(barostat)

        integrator = LangevinMiddleIntegrator(
            TEMP_K * kelvin, 1.0 / picosecond, DT_FS * femtosecond,
        )
        integrator.setRandomNumberSeed(int(args.random_seed))

        try:
            platform = Platform.getPlatformByName("CUDA")
            properties = {{"Precision": "mixed"}}
        except Exception:
            platform = Platform.getPlatformByName("CPU")
            properties = {{}}

        sim = Simulation(top.topology, system, integrator, platform, properties)
        sim.context.setPositions(gro.positions)
        sim.context.setPeriodicBoxVectors(*box)

        # ------------------------------------------------------------------
        # Chunk strategy mirrors prepare_proteins openmm_simulation.py:
        # * production_<NN>.{{xtc,log,chk}} per chunk (zfilled NN)
        # * Latest chunk's checkpoint is loaded on resume.
        # * Equilibration runs once on a fresh start; equilibration.chk
        #   anchors the first production chunk.
        # ------------------------------------------------------------------
        chunk_steps = ns_to_steps(args.chunk_ns)
        total_steps = ns_to_steps(args.total_ns)
        num_chunks = max(1, (total_steps + chunk_steps - 1) // chunk_steps)
        zfill = max(2, len(str(num_chunks)))

        def fsync_path(p):
            """Best-effort fsync of a file *or* directory by its path. Required
            on GPFS / NFS where Python f.flush() only commits to OS page cache
            and a SIGKILL would lose the dirty data; fsync forces a write to
            persistent storage."""
            try:
                fd = os.open(str(p), os.O_RDONLY)
            except OSError:
                return
            try:
                os.fsync(fd)
            except OSError:
                pass
            finally:
                os.close(fd)

        # ------------------------------------------------------------------
        # Find latest existing chunk checkpoint and pick up *exactly* where
        # it left off. Earlier versions of this script assumed a chunk's .chk
        # was only written on chunk completion. With the mid-chunk
        # CheckpointReporter added below, a .chk now exists even mid-chunk,
        # so we use the loaded checkpoint's *current step* (not the file name)
        # as the authoritative resume point.
        # ------------------------------------------------------------------
        loaded_step = 0
        last_done = 0
        for n in range(num_chunks, 0, -1):
            chk = rdir / f"production_{{n:0{{zfill}}d}}.chk"
            if chk.exists():
                print(f"resume: loading {{chk.name}}", flush=True)
                sim.loadCheckpoint(str(chk))
                loaded_step = sim.context.getStepCount()
                chunk_end_for_n = min(n * chunk_steps, total_steps)
                # +equilibration_steps offset is baked into currentStep because
                # the equilibration sim.step() ran on the same context.
                # If we're mid-chunk-n, re-enter chunk n; otherwise n is done.
                if loaded_step < chunk_end_for_n + ns_to_steps(EQUIL_NS):
                    last_done = n - 1
                else:
                    last_done = n
                print(f"  loaded step {{loaded_step}}  -> last_done={{last_done}}",
                      flush=True)
                break

        if last_done == 0 and loaded_step == 0:
            equil_chk = rdir / EQUIL_CHK
            if equil_chk.exists():
                print(f"resume: loading {{EQUIL_CHK}} (equilibration done)", flush=True)
                sim.loadCheckpoint(str(equil_chk))
            else:
                # Fresh start: minimize → equilibrate → save EQUIL_CHK
                t0 = time.time()
                sim.minimizeEnergy(
                    tolerance=10 * kilojoule_per_mole / nanometer,
                    maxIterations=MINIMIZE_MAX_ITER,
                )
                sim.context.setVelocitiesToTemperature(TEMP_K * kelvin, int(args.random_seed))
                print(f"minimisation: {{time.time() - t0:.1f}} s", flush=True)

                if EQUIL_NS > 0:
                    sim.reporters.append(StateDataReporter(
                        str(rdir / "equilibration.log"),
                        ns_to_steps(SAVE_LOG_EVERY_NS),
                        step=True, potentialEnergy=True, temperature=True,
                        density=True, speed=True, separator="\\t",
                    ))
                    t0 = time.time()
                    sim.step(ns_to_steps(EQUIL_NS))
                    sim.reporters.clear()
                    print(f"equilibration {{EQUIL_NS:.1f}} ns: {{time.time() - t0:.1f}} s",
                          flush=True)
                sim.saveCheckpoint(str(equil_chk))
                fsync_path(equil_chk)

        # ------------------------------------------------------------------
        # Production chunk loop. Each chunk:
        #   * runs (chunk_end - currentStep) more steps (which may be < the
        #     full chunk if we resumed mid-chunk),
        #   * appends to existing log+xtc if resuming mid-chunk, else truncates,
        #   * writes a CheckpointReporter every SAVE_CHK_EVERY_NS ns so a
        #     SIGKILL leaves at most that much data unrecoverable,
        #   * after sim.step() returns: explicitly destroys the XTCReporter so
        #     XTCFile's C++ destructor flushes pending frames, then fsync the
        #     log, xtc, chunk_chk and the replica dir to commit to disk.
        # ------------------------------------------------------------------
        for n in range(last_done + 1, num_chunks + 1):
            tag = f"{{n:0{{zfill}}d}}"
            chunk_log = rdir / f"production_{{tag}}.log"
            chunk_traj = rdir / (f"production_{{tag}}.xtc" if _HAS_XTC
                                 else f"production_{{tag}}.dcd")
            chunk_chk = rdir / f"production_{{tag}}.chk"

            chunk_start = (n - 1) * chunk_steps + ns_to_steps(EQUIL_NS)
            chunk_end = min(n * chunk_steps, total_steps) + ns_to_steps(EQUIL_NS)
            current_step = sim.context.getStepCount()
            if current_step >= chunk_end:
                print(f"chunk {{tag}}: already complete (step {{current_step}} ≥ {{chunk_end}})",
                      flush=True)
                continue
            append = current_step > chunk_start
            steps_to_run = chunk_end - current_step

            sim.reporters.clear()
            log_reporter = StateDataReporter(
                str(chunk_log), ns_to_steps(SAVE_LOG_EVERY_NS),
                step=True, potentialEnergy=True, kineticEnergy=True,
                temperature=True, density=True, volume=True, speed=True,
                separator="\\t", append=append,
            )
            sim.reporters.append(log_reporter)

            Reporter = XTCReporter if _HAS_XTC else DCDReporter
            traj_reporter = Reporter(str(chunk_traj),
                                      ns_to_steps(SAVE_XTC_EVERY_NS),
                                      append=append)
            sim.reporters.append(traj_reporter)

            # Mid-chunk checkpoint: bounds data loss on SIGKILL to ≤ SAVE_CHK_EVERY_NS
            sim.reporters.append(CheckpointReporter(
                str(chunk_chk), ns_to_steps(SAVE_CHK_EVERY_NS),
            ))

            print(f"chunk {{tag}}: running {{steps_to_run}} steps "
                  f"({{current_step}} -> {{chunk_end}}, append={{append}})",
                  flush=True)
            t0 = time.time()
            sim.step(steps_to_run)
            elapsed = time.time() - t0

            # Explicit close + fsync so a subsequent SIGKILL can't lose data
            # buffered in OS page cache. The XTC writer is a C++ object that
            # only flushes on destruction; del + gc.collect() force that.
            try:
                log_reporter._out.flush()
                os.fsync(log_reporter._out.fileno())
            except (AttributeError, OSError) as e:
                print(f"  warn: log fsync skipped: {{e}}", flush=True)
            try:
                if hasattr(traj_reporter, "_xtc"):
                    traj_reporter._xtc = None
            except Exception as e:
                print(f"  warn: xtc reset skipped: {{e}}", flush=True)
            sim.reporters.clear()
            gc.collect()
            fsync_path(chunk_traj)
            fsync_path(chunk_log)

            # End-of-chunk checkpoint (overwrites the mid-chunk one with the
            # final state at chunk_end), plus a "rolling" production.chk.
            sim.saveCheckpoint(str(chunk_chk))
            sim.saveCheckpoint(str(rdir / "production.chk"))
            fsync_path(chunk_chk)
            fsync_path(rdir / "production.chk")
            fsync_path(rdir)   # commit directory entry

            ns_done = (sim.context.getStepCount() - ns_to_steps(EQUIL_NS)) * DT_FS / 1.0e6
            ns_per_day = (steps_to_run * DT_FS / 1.0e6) / max(elapsed / 86400, 1e-9)
            print(f"chunk {{tag}} done: ran {{steps_to_run}} steps, "
                  f"total {{ns_done:.0f}} ns / {{args.total_ns:.0f}} ns, "
                  f"{{ns_per_day:.1f}} ns/day  ({{elapsed:.1f}}s)", flush=True)

        sim.saveState(str(rdir / "final_state.xml"))
        fsync_path(rdir / "final_state.xml")
        print("all chunks done", flush=True)


    if __name__ == "__main__":
        main()
''')
