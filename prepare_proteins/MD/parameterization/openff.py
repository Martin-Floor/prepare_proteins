from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional
import warnings

import numpy as np

from ..openmm_setup import aa3 as _AMBER_AA3
from .base import ParameterizationBackend, ParameterizationResult, register_backend
from .utils import (
    DEFAULT_PARAMETERIZATION_SKIP_RESIDUES,
    extract_residue_subsystem,
    write_residue_pdb,
)

try:
    from openmm.app import PDBFile
except ImportError:  # pragma: no cover - openmm required at runtime
    PDBFile = None

from openff.toolkit.topology import Molecule
from openff.toolkit.utils import ToolkitRegistry, RDKitToolkitWrapper, AmberToolsToolkitWrapper

@register_backend
class OpenFFBackend(ParameterizationBackend):
    """Generate AMBER-format inputs using the OpenFF toolkit and OpenMM."""

    name = "openff"
    input_format = "amber"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        defaults = {
            "smirnoff_forcefield": "openff-2.2.0.offxml",
            "padding_nm": 1.2,
            "ionic_strength_m": 0.0,
            "nonbonded_cutoff_nm": 1.0,
            "water_model": "tip3p",
            "positive_ion": "Na+",
            "negative_ion": "Cl-",
            "neutralize": True,
            "solvate": True,
            "forcefield_files": (
                "amber14/protein.ff14SB.xml",
                "amber14/tip3p.xml",
            ),
        }
        for key, value in defaults.items():
            self.options.setdefault(key, value)

    def prepare_model(self, openmm_md, parameters_folder: str, **kwargs: Any) -> ParameterizationResult:
        """
        Build an OpenMM system using Amber14SB for protein, TIP3P for water, and OpenFF (Sage 2.2)
        for ligands coming from SDF and/or SMILES. For each ligand residue detected in the PDB,
        we:
        - load from SDF (preferred) or assemble from SMILES (fallback),
        - assign AM1-BCC charges (AmberTools),
        - verify atom counts vs. the PDB residue snapshot,
        - replace the residue with a templated small-molecule,
        - register SMIRNOFF parameters, solvate, and export prmtop/inpcrd.

        Required options (typically passed by setUpOpenMMSimulations):
        - ligand_sdf_files: {RESNAME: path or {path, index?, charge_method?}}
        - ligand_smiles:    {RESNAME: smiles}
        - ligand_charges:   {RESNAME: total_charge}  # used for validation only

        Notes:
        * If the user-provided total charge for a residue differs from the molecule formal charge
            seen by RDKit/OpenFF, we raise a ValueError with guidance to fix the protonation.
        """
        # ---- Options and imports
        options = dict(self.options)
        options.update(kwargs)
        strict_atom_name_check = bool(options.get("strict_atom_name_check", True))

        # We currently only prepare full systems in this backend
        if options.get("build_full_system") is False:
            raise NotImplementedError("OpenFF backend currently prepares full protein+ligand systems only.")

        try:
            from openmm import unit as u
            from openmm.app import ForceField, HBonds, Modeller, PME
        except ImportError as exc:
            raise RuntimeError("openmm is required for the OpenFF parameterization backend.") from exc
        try:
            from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        except ImportError as exc:
            raise RuntimeError("openmmforcefields is required to register SMIRNOFF templates.") from exc
        try:
            from openff.toolkit.utils.exceptions import InvalidConformerError
        except ImportError as exc:
            raise RuntimeError("openff-toolkit is required to build ligand templates.") from exc
        try:
            import parmed as pmd
        except ImportError as exc:
            raise RuntimeError("parmed is required to export prmtop/inpcrd files.") from exc

        # ---- Normalize ligand inputs
        ligand_smiles_map = self._normalize_ligand_smiles(options.get("ligand_smiles"))
        ligand_sdf_map = self._normalize_ligand_sdfs(options.get("ligand_sdf_files"))
        default_charge_method = options.get("ligand_charge_method", "am1bcc")
        skip_charge = bool(options.get("skip_ligand_charge_computation", False))
        provided_charges = options.get("charges")  # {RESNAME: total_charge}

        # Skip list and filters
        skip_ligands = {res.upper() for res in (options.get("skip_ligands") or [])}
        skip_ligands.update(DEFAULT_PARAMETERIZATION_SKIP_RESIDUES)
        only_residues_opt = options.get("only_residues")
        if only_residues_opt:
            if isinstance(only_residues_opt, str):
                only_residues = {only_residues_opt.strip().upper()} if only_residues_opt.strip() else None
            else:
                only_residues = {str(res).strip().upper() for res in only_residues_opt if str(res).strip()} or None
        else:
            only_residues = None

        # ---- Identify ligand residues in the modeller
        protein_residues = {name.upper() for name in _AMBER_AA3}
        for entry in options.get("non_standard_residues", []) or []:
            if isinstance(entry, str) and entry.strip():
                protein_residues.add(entry.strip().upper())

        modeller = Modeller(openmm_md.modeller.topology, openmm_md.modeller.positions)
        self._remove_skip_residues(modeller, DEFAULT_PARAMETERIZATION_SKIP_RESIDUES)
        ligand_residues = self._collect_ligand_residues(
            modeller.topology.residues(),
            protein_residues,
            skip_ligands=skip_ligands,
            only_residues=only_residues,
        )

        if not ligand_residues:
            warnings.warn(
                "[prepare_proteins] No ligand residues found to parameterize after filters. "
                "Continuing with protein/solvent only.",
                RuntimeWarning,
            )

        # ---- Toolkit registry (RDKit + AmberTools)
        rdkit = RDKitToolkitWrapper()
        ambertools = AmberToolsToolkitWrapper()
        toolkit_registry = ToolkitRegistry()
        toolkit_registry.register_toolkit(rdkit)
        toolkit_registry.register_toolkit(ambertools)

        # ---- Process each ligand residue group
        ligand_molecules: Dict[str, Molecule] = {}
        for resname, residue_specs in ligand_residues.items():
            # Require a user-provided total charge for clarity and validation
            residue_charge = self._find_residue_charge(resname, provided_charges)
            if residue_charge is None:
                raise ValueError(
                    f"No total charge provided for ligand residue '{resname}'. "
                    "Specify it via the 'ligand_charges' option when calling setUpOpenMMSimulations."
                )

            # Determine source (prefer SDF; use SMILES otherwise)
            sdf_entry = ligand_sdf_map.get(resname)
            smiles_entry = ligand_smiles_map.get(resname)
            smiles = smiles_entry["smiles"] if smiles_entry else None
            if sdf_entry is None and smiles is None:
                raise ValueError(
                    f"No ligand definition provided for residue '{resname}'. "
                    "Supply either a SMILES string or an SDF entry."
                )

            # Per-residue charge method (optional override in SDF map)
            entry_charge_method = default_charge_method
            if sdf_entry is not None and sdf_entry.get("charge_method"):
                entry_charge_method = str(sdf_entry["charge_method"]).strip() or default_charge_method
            elif smiles_entry is not None and smiles_entry.get("charge_method"):
                entry_charge_method = str(smiles_entry["charge_method"]).strip() or default_charge_method

            # Output pack folder for this residue
            pack_dir = Path(parameters_folder) / f"{resname}_parameters"
            pack_dir.mkdir(parents=True, exist_ok=True)

            # Snapshot first occurrence of this residue to align atom counts & positions
            pose_path = pack_dir / f"{resname}.pdb"
            first_residue = self._resolve_residue(modeller, residue_specs[0])
            first_topology, first_positions = write_residue_pdb(modeller, first_residue, str(pose_path))
            # Cache
            offmol_path = pack_dir / f"{resname}.offmol.json"
            base_ligand: Optional[Molecule] = None
            if offmol_path.exists():
                try:
                    cached_ligand = Molecule.from_json(offmol_path.read_text())
                except Exception as exc:
                    cached_ligand = None
                    warnings.warn(f"[prepare_proteins] Ignoring corrupt cache {offmol_path.name}: {exc}", RuntimeWarning)
                if cached_ligand is not None and cached_ligand.n_atoms == first_topology.getNumAtoms():
                    base_ligand = cached_ligand
                elif cached_ligand is not None:
                    (pack_dir / f"{resname}_cached_mismatch.offmol.json").write_text(cached_ligand.to_json())
                    base_ligand = None

            # Load ligand (SDF preferred; fallback to SMILES). Do NOT assign charges here.
            if base_ligand is None:
                if sdf_entry is not None:
                    # SDF path: load via RDKit with *controlled* sanitization to avoid adding H
                    from rdkit import Chem
                    idx = int(sdf_entry.get("index", 0)) if sdf_entry.get("index") is not None else 0

                    # 1) Read *without* sanitization and *without* removing explicit Hs
                    suppl = Chem.SDMolSupplier(sdf_entry["path"], removeHs=False, sanitize=False)
                    mols = [m for m in suppl if m is not None]
                    if not mols:
                        raise ValueError(f"No valid molecules found in SDF: {sdf_entry['path']}")
                    if idx < 0 or idx >= len(mols):
                        raise IndexError(
                            f"SDF index {idx} out of range (0..{len(mols)-1}) for {sdf_entry['path']}"
                        )
                    rdmol = Chem.Mol(mols[idx])

                    # 2) Sanitize but *disable* ADJUSTHS so RDKit won’t add/remove Hs
                    sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
                    Chem.SanitizeMol(rdmol, sanitizeOps=sanitize_flags)

                    # 3) Convert to OpenFF, declaring hydrogens are explicit
                    lig = Molecule.from_rdkit(
                        rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True
                    )

                    # 4) Assert atom-count invariance (RDKit -> OFF)
                    if lig.n_atoms != rdmol.GetNumAtoms():
                        raise RuntimeError(
                            f"RDKit->OpenFF atom count changed ({rdmol.GetNumAtoms()} -> {lig.n_atoms}) "
                            f"for {resname}. This indicates implicit H materialization—please report."
                        )

                    base_ligand = lig
                    shutil.copyfile(sdf_entry["path"], pack_dir / f"{resname}_input.sdf")

                    if not base_ligand.conformers:
                        raise ValueError(
                            f"Ligand {resname} from SDF has no 3D conformer coordinates. "
                            "Please provide 3D SDF or switch to SMILES with conformer generation."
                        )
                else:
                    # SMILES path: create 3D conformer
                    lig = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
                    if not lig.conformers:
                        try:
                            lig.generate_conformers(n_conformers=1, toolkit_registry=toolkit_registry)
                        except TypeError:
                            lig.generate_conformers(n_conformers=1)
                    base_ligand = lig

            # Persist cache
            offmol_path.write_text(base_ligand.to_json())
            base_ligand.properties.setdefault("residue_name", resname)

            # ---- Validate user-provided total charge vs formal charge from the molecule
            try:
                rdm = base_ligand.to_rdkit()
                formal_charge = int(sum(a.GetFormalCharge() for a in rdm.GetAtoms()))
            except Exception:
                formal_charge = None

            if (formal_charge is not None) and (int(residue_charge) != formal_charge):
                src = (
                    f"SDF: {sdf_entry['path']}"
                    if sdf_entry is not None
                    else f"SMILES: {smiles[:60]}{'...' if smiles and len(smiles) > 60 else ''}"
                )
                if skip_charge:
                    warnings.warn(
                        f"[prepare_proteins] Ligand {resname}: provided total charge {int(residue_charge)} "
                        f"differs from molecule formal charge {formal_charge} ({src}). "
                        "Continuing with skip_ligand_charge_computation=True; "
                        "assigning dummy charges that sum to the formal charge.",
                        RuntimeWarning,
                    )
                else:
                    raise ValueError(
                        f"Ligand {resname}: provided total charge {int(residue_charge)} "
                        f"does not match the molecule's formal charge {formal_charge} "
                        f"(source: {src}). Please supply the ligand in the desired protonation state."
                    )

            # --- Assign charges now ---
            if skip_charge:
                # Fast placeholder: uniform charges summing to the **formal** charge
                import numpy as _np
                n = base_ligand.n_atoms
                target_sum = float(formal_charge if formal_charge is not None else 0.0)
                arr = _np.zeros(n, dtype=float)
                if n > 0:
                    arr[:] = target_sum / n
                try:
                    from openff.units import unit as _u
                except Exception:
                    from openmm import unit as _u
                base_ligand.partial_charges = arr * _u.elementary_charge
            else:
                base_ligand.assign_partial_charges(entry_charge_method, toolkit_registry=toolkit_registry)

            # ---- Write a template SDF for inspection
            self._write_ligand_file(pack_dir / f"{resname}_template.sdf", base_ligand, rdkit)

            # ---- Replace each occurrence of the residue with the templated ligand (atom-count check)
            residues_to_delete = []
            ligand_entries: List[tuple[Molecule, Any, List[str], Any]] = []

            for idx, residue_spec in enumerate(residue_specs):
                residue = self._resolve_residue(modeller, residue_spec)
                if idx == 0:
                    ligand_positions = first_positions
                else:
                    _, ligand_positions = extract_residue_subsystem(modeller, residue)

                pdb_order_names = [a.name for a in residue.atoms()]

                coords_angstrom = np.asarray(ligand_positions.value_in_unit(u.angstrom), dtype=float)
                if coords_angstrom.shape[0] != base_ligand.n_atoms:
                    # dump debug artifacts and fail fast
                    self._write_ligand_file(pack_dir / f"{resname}_current.sdf", base_ligand, rdkit)
                    self._write_snapshot(pack_dir / f"{resname}_mismatch_{idx}.pdb", modeller)
                    (pack_dir / f"{resname}_atom_counts.txt").write_text(
                        f"Ligand atoms: {base_ligand.n_atoms}\nPDB atoms: {coords_angstrom.shape[0]}\n"
                    )
                    raise InvalidConformerError(
                        f"atom/position count mismatch for residue {resname}: "
                        f"{coords_angstrom.shape[0]} positions vs {base_ligand.n_atoms} atoms"
                    )

                ligand_instance = Molecule(base_ligand)
                ligand_instance.name = resname
                ligand_instance._conformers = []  # reset
                ligand_instance.add_conformer(coords_angstrom * u.angstrom)
                ligand_instance.partial_charges = base_ligand.partial_charges

                residues_to_delete.append(residue)
                ligand_entries.append((ligand_instance, ligand_positions, pdb_order_names, residue))

            if residues_to_delete:
                modeller.delete(residues_to_delete)
                for ligand_instance, ligand_positions, pdb_order_names, residue in ligand_entries:
                    if len(pdb_order_names) != ligand_instance.n_atoms:
                        raise ValueError(
                            f"[naming] Atom count mismatch for {resname}: "
                            f"{len(pdb_order_names)} names vs {ligand_instance.n_atoms} atoms."
                        )

                    off_atomic_numbers = [getattr(a, "atomic_number", None) for a in ligand_instance.atoms]
                    pdb_atomic_numbers = [
                        a.element.atomic_number if getattr(a, "element", None) else None for a in residue.atoms()
                    ]

                    mismatch_indices = [
                        i
                        for i, (z1, z2) in enumerate(zip(off_atomic_numbers, pdb_atomic_numbers))
                        if (z1 is not None and z2 is not None and z1 != z2)
                    ]
                    if mismatch_indices:
                        if strict_atom_name_check:
                            raise ValueError(
                                f"[naming] Element mismatch in OFF vs PDB order for {resname}. "
                                "Refuse to assign names to avoid incorrect mapping."
                            )
                        warnings.warn(
                            f"[prepare_proteins] OFF/PDB element mismatch for {resname} at indices {mismatch_indices}; "
                            "continuing with supplied atom names because strict_atom_name_check=False.",
                            RuntimeWarning,
                        )

                    atom_names_off_order = list(pdb_order_names)

                    lig_top_omm = self._build_named_openmm_topology(
                        ligand_instance,
                        resname=resname,
                        chain_id=residue.chain.id,
                        residue_id=str(residue.id),
                        atom_names_off_order=atom_names_off_order,
                    )
                    modeller.add(lig_top_omm, ligand_positions)
                # snapshot the modified modeller for inspection
                self._write_snapshot(pack_dir / f"{resname}_rebuilt.pdb", modeller)

            # optional processed dump (same as template in this streamlined flow)
            self._write_ligand_file(pack_dir / f"{resname}_processed.sdf", base_ligand, rdkit)
            ligand_molecules[resname] = Molecule(base_ligand)

        # ---- Register SMIRNOFF templates for all ligands
        forcefield_files = tuple(options.get("forcefield_files") or ())
        forcefield = ForceField(*forcefield_files)
        if ligand_molecules:
            generator = SMIRNOFFTemplateGenerator(
                forcefield=options.get("smirnoff_forcefield", "openff-2.2.0.offxml"),
                molecules=list(ligand_molecules.values()),
            )
            forcefield.registerTemplateGenerator(generator.generator)

        # ---- Solvate / ions
        if options.get("solvate", True):
            modeller.addSolvent(
                forcefield,
                model=str(options.get("water_model", "tip3p")),
                padding=float(options.get("padding_nm", 1.2)) * u.nanometer,
                ionicStrength=float(options.get("ionic_strength_m", 0.0)) * u.molar,
                positiveIon=str(options.get("positive_ion", "Na+")),
                negativeIon=str(options.get("negative_ion", "Cl-")),
                neutralize=bool(options.get("neutralize", True)),
            )

        # ---- Build two systems: one "export" (no constraints) and one runtime (HBonds)
        cutoff = float(options.get("nonbonded_cutoff_nm", 1.0)) * u.nanometer
        export_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=cutoff,
            constraints=None,
            rigidWater=False,
            removeCMMotion=False,
        )

        # ---- Export AMBER files
        structure = pmd.openmm.load_topology(modeller.topology, export_system, xyz=modeller.positions)
        output_root = Path(parameters_folder)
        output_root.mkdir(parents=True, exist_ok=True)
        prmtop_path = output_root / f"{openmm_md.pdb_name}.prmtop"
        inpcrd_path = output_root / f"{openmm_md.pdb_name}.inpcrd"
        structure.save(str(prmtop_path), overwrite=True)
        structure.save(str(inpcrd_path), overwrite=True)

        # ---- Update openmm_md + return result
        openmm_md.modeller = modeller
        openmm_md.positions = np.array([pos.value_in_unit(u.nanometer) for pos in modeller.positions])
        openmm_md.prmtop_file = str(prmtop_path)
        openmm_md.inpcrd_file = str(inpcrd_path)

        result = ParameterizationResult(input_format="amber")
        result.with_file("prmtop", str(prmtop_path))
        result.with_file("inpcrd", str(inpcrd_path))
        result.metadata["ligand_packs"] = {
            resname: str((Path(parameters_folder) / f"{resname}_parameters").resolve())
            for resname in ligand_molecules
        }
        if ligand_molecules:
            ligand_sources: Dict[str, Dict[str, Any]] = {}
            for rname in ligand_molecules:
                entry = ligand_sdf_map.get(rname)
                if entry:
                    ligand_sources[rname] = {"type": "sdf", "path": entry["path"], "index": entry.get("index")}
                else:
                    ligand_sources[rname] = {"type": "smiles", **ligand_smiles_map.get(rname, {})}
            result.metadata["ligand_sources"] = ligand_sources
        return result

    def _build_named_openmm_topology(
        self,
        off_mol: Molecule,
        *,
        resname: str,
        chain_id: str,
        residue_id: str,
        atom_names_off_order,
    ):
        """
        Create a minimal OpenMM Topology for a ligand, in OFF atom order,
        assigning the provided atom names 1:1 to indices (no reordering).
        """
        from openmm.app import Topology, Element  # local import to avoid top-level dependency

        top = Topology()
        chain = top.addChain(id=str(chain_id) if chain_id is not None else None)
        res = top.addResidue(str(resname), chain, id=str(residue_id) if residue_id is not None else None)

        omm_atoms = []
        for i, off_atom in enumerate(off_mol.atoms):
            elem = Element.getByAtomicNumber(off_atom.atomic_number)
            symbol = None
            off_elem = getattr(off_atom, "element", None)
            if off_elem is not None and hasattr(off_elem, "symbol"):
                symbol = off_elem.symbol
            elif elem is not None:
                symbol = elem.symbol
            fallback_symbol = symbol if symbol else "X"
            nm = (
                atom_names_off_order[i]
                if i < len(atom_names_off_order) and atom_names_off_order[i]
                else f"{fallback_symbol}{i+1}"
            )
            omm_atoms.append(top.addAtom(nm, elem, res))

        for b in off_mol.bonds:
            top.addBond(omm_atoms[b.atom1_index], omm_atoms[b.atom2_index])

        return top

    @staticmethod
    def _remove_skip_residues(modeller: "Modeller", skip_names: Iterable[str]) -> None:
        """Remove residues such as water/ions that should not be parameterized."""
        skip_set = {name.strip().upper() for name in skip_names}
        residues_to_delete = [
            residue
            for residue in modeller.topology.residues()
            if residue.name.strip().upper() in skip_set
        ]
        if residues_to_delete:
            modeller.delete(residues_to_delete)

    @staticmethod
    def _normalize_ligand_smiles(smiles: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Accepts:
        ligand_smiles = {
            "HCA": "O=C([O-])c1oc(cc1)C(O)O,
            "XYZ": {"smiles": "...", "charge_method": "am1bccelf10"}
        }
        Returns: {RESNAME: {"smiles": str, "charge_method": Optional[str]}}
        """
        if not smiles:
            return {}
        if not isinstance(smiles, Mapping):
            raise TypeError("ligand_smiles must be a mapping of residue name -> SMILES or options.")
        out: Dict[str, Dict[str, Any]] = {}
        for key, val in smiles.items():
            if not isinstance(key, str):
                raise TypeError("ligand_smiles keys must be strings.")
            name = key.strip().upper()
            if not name:
                raise ValueError("Empty residue key found in ligand_smiles.")
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    raise ValueError(f"Empty SMILES for residue {name!r}.")
                out[name] = {"smiles": s}
            elif isinstance(val, Mapping):
                if "smiles" not in val:
                    raise ValueError(f"ligand_smiles entry for {name!r} is missing 'smiles'.")
                s = str(val["smiles"]).strip()
                if not s:
                    raise ValueError(f"Empty SMILES for residue {name!r}.")
                entry: Dict[str, Any] = {"smiles": s}
                if "charge_method" in val and val["charge_method"]:
                    entry["charge_method"] = str(val["charge_method"]).strip()
                out[name] = entry
            else:
                raise TypeError(f"ligand_smiles entry for {name!r} must be a string or mapping.")
        return out

    @staticmethod
    def _normalize_ligand_sdfs(entries: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not entries:
            return {}
        if not isinstance(entries, Mapping):
            raise TypeError("ligand_sdf_files must be a mapping of residue name -> path or options.")
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in entries.items():
            if not isinstance(key, str):
                raise TypeError("ligand_sdf_files keys must be strings.")
            residue = key.strip().upper()
            if not residue:
                raise ValueError("Empty residue key found in ligand_sdf_files.")
            if isinstance(value, str):
                path = Path(value).expanduser().resolve()
                normalized[residue] = {"path": str(path)}
            elif isinstance(value, Mapping):
                if "path" not in value:
                    raise ValueError(f"ligand_sdf_files entry for {residue!r} is missing a 'path'.")
                entry_path = Path(value["path"]).expanduser().resolve()
                entry: Dict[str, Any] = {"path": str(entry_path)}
                if "charge_method" in value and value["charge_method"]:
                    entry["charge_method"] = str(value["charge_method"]).strip()
                normalized[residue] = entry
            else:
                raise TypeError(
                    f"ligand_sdf_files entry for {residue!r} must be a path string or a mapping with options."
                )
        return normalized

    @staticmethod
    def _write_snapshot(path: Path, modeller) -> None:
        if PDBFile is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            PDBFile.writeFile(modeller.topology, modeller.getPositions(), handle)

    @staticmethod
    def _write_ligand_file(path: Path, ligand: Molecule, toolkit: RDKitToolkitWrapper) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ligand.to_file(str(path), file_format="SDF", toolkit_registry=toolkit)

    @classmethod
    def _find_residue_charge(cls, resname: str, charges: Any) -> Optional[float]:
        if charges is None:
            return None
        target = resname.strip().upper()
        if isinstance(charges, Mapping):
            for key, value in charges.items():
                if isinstance(key, str) and key.strip().upper() == target and not isinstance(value, Mapping):
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
            for value in charges.values():
                if isinstance(value, Mapping):
                    nested = cls._find_residue_charge(target, value)
                    if nested is not None:
                        return nested
        elif isinstance(charges, (list, tuple, set)):
            for item in charges:
                nested = cls._find_residue_charge(target, item)
                if nested is not None:
                    return nested
        return None

    @staticmethod
    def _collect_ligand_residues(
        residues: Iterable[Any],
        protein_residues: Iterable[str],
        skip_ligands: Optional[Iterable[str]] = None,
        only_residues: Optional[Iterable[str]] = None,
    ) -> Dict[str, list]:
        protein_set = {res.upper() for res in protein_residues}
        skip_set = {res.upper() for res in (skip_ligands or [])}
        skip_set.update(DEFAULT_PARAMETERIZATION_SKIP_RESIDUES)
        only_set = {res.upper() for res in only_residues} if only_residues else None
        ligand_map: Dict[str, list] = {}
        for residue in residues:
            resname = residue.name.strip().upper()
            if not resname or resname in protein_set or resname == "HOH":
                continue
            if resname in skip_set:
                continue
            if only_set and resname not in only_set:
                continue
            ligand_map.setdefault(resname, []).append(
                {
                    "chain_id": residue.chain.id,
                    "residue_id": str(residue.id),
                    "residue_name": resname,
                }
            )
        return ligand_map

    @staticmethod
    def _resolve_residue(modeller: "Modeller", spec: Mapping[str, Any]):
        chain_id = str(spec.get("chain_id", "")).strip()
        residue_id = str(spec.get("residue_id", "")).strip()
        spec_name = str(spec.get("residue_name", "")).strip().upper()
        if not chain_id or not residue_id:
            raise ValueError("Ligand residue specification is missing chain_id or residue_id.")
        for residue in modeller.topology.residues():
            if residue.chain.id == chain_id and str(residue.id) == residue_id:
                if spec_name and residue.name.strip().upper() != spec_name:
                    continue
                return residue
        raise ValueError(f"Residue with chain '{chain_id}' and id '{residue_id}' not found in modeller.")
