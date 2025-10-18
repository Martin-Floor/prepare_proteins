"""
Helpers for building 3D ligand geometries (SDF) from SMILES using RDKit.

RDKit is treated as an optional dependency: importing this module succeeds even
if RDKit is unavailable, but calling the public helpers will raise an explicit
ImportError when RDKit cannot be imported.
"""

from __future__ import annotations

import base64
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Union

__all__ = ["smiles_to_sdf", "smiles_dict_to_sdf"]


def _load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdPartialCharges
    except ImportError as exc:
        raise ImportError(
            "smiles_to_sdf requires RDKit. Install it in the active environment "
            "(e.g. `conda install -c conda-forge rdkit`)."
        ) from exc
    return Chem, AllChem, rdPartialCharges


def smiles_to_sdf(
    smiles: str,
    out_sdf: Union[str, Path],
    *,
    name: str = "LIG",
    add_h: bool = True,
    n_confs: int = 10,
    optimize: bool = True,
    max_opt_iters: int = 200,
    assign_gasteiger: bool = True,
    expected_charge: int | None = None,
    preserve_aromatic_bonds: bool = False,
    require_aromatic_bonds: bool = False,
) -> str:
    """
    Build an SDF with a single conformer generated from SMILES.

    Parameters
    ----------
    smiles : str
        Input SMILES string (should encode protonation/tautomer states explicitly).
    out_sdf : str | Path
        Destination file path; parent directories are created automatically.
    name : str
        Title recorded in the SDF.
    add_h : bool
        If True, add explicit hydrogens before 3D embedding.
    n_confs : int
        Number of conformers to attempt with ETKDG; the lowest-energy is kept.
    optimize : bool
        Run MMFF/UFF optimizations on each conformer before selecting the minimum.
    max_opt_iters : int
        Maximum optimization iterations for MMFF/UFF.
    assign_gasteiger : bool
        If True, compute Gasteiger charges and store them in `_PartialCharge`.
    expected_charge : int | None
        Optional net charge check; raises if the optimized molecule's formal charge
        does not match.
    preserve_aromatic_bonds : bool
        If True, set `_MolFileKekulize=0` so RDKit keeps aromatic flags in the SDF,
        which is useful for downstream GAFF/antechamber tooling.
    require_aromatic_bonds : bool
        Require at least one aromatic bond after sanitization (raises otherwise).

    Returns
    -------
    str
        Absolute path to the written SDF file.
    """
    Chem, AllChem, rdPartialCharges = _load_rdkit()

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError("RDKit failed to parse SMILES.")
    Chem.SanitizeMol(mol)
    if add_h:
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.useBasicKnowledge = True
    params.useSmallRingTorsions = True
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=max(1, n_confs), params=params)
    if not cids:
        raise RuntimeError("Failed to embed 3D conformers.")

    def mmff_props(molecule):
        try:
            if AllChem.MMFFHasAllMoleculeParams(molecule):
                return AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
        except Exception:
            try:
                if AllChem.MMFFHasAllMoleculeParams(molecule):
                    return AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94")
            except Exception:
                return None
        return None

    mp = mmff_props(mol)
    energies = []
    for cid in cids:
        energy = _calc_mmff_energy(AllChem, mol, mp, cid, max_iters=max_opt_iters, optimize=optimize)
        if energy is None:
            energy = _calc_uff_energy(AllChem, mol, cid, max_iters=max_opt_iters, optimize=optimize)
        if energy is None:
            raise RuntimeError("Failed to compute energy for RDKit conformer; check RDKit installation.")
        energies.append((energy, cid))
    energies.sort(key=lambda x: x[0])
    best_cid = energies[0][1]

    mol_best = Chem.Mol(mol)
    mol_best.RemoveAllConformers()
    mol_best.AddConformer(mol.GetConformer(best_cid), assignId=True)
    mol_best.SetProp("_Name", name)

    if expected_charge is not None:
        formal_charge = Chem.GetFormalCharge(mol_best)
        if formal_charge != expected_charge:
            raise ValueError(
                f"Formal charge {formal_charge} differs from expected {expected_charge}. "
                "Adjust the SMILES protonation/tautomer to match the desired net charge."
            )

    if require_aromatic_bonds and not any(b.GetIsAromatic() for b in mol_best.GetBonds()):
        raise ValueError(
            "No aromatic bonds detected after sanitization. Ensure the SMILES encodes aromaticity "
            "(lowercase atoms) if you intend to preserve aromatic rings for GAFF workflows."
        )

    if assign_gasteiger:
        rdPartialCharges.ComputeGasteigerCharges(mol_best)
        for atom in mol_best.GetAtoms():
            q = atom.GetDoubleProp("_GasteigerCharge")
            atom.SetDoubleProp("_PartialCharge", float(q))

    if preserve_aromatic_bonds:
        mol_best.SetProp("_MolFileKekulize", "0")

    out_path = Path(out_sdf).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(out_path))
    writer.write(mol_best)
    writer.close()

    return str(out_path)


def _calc_mmff_energy(AllChem, mol, props, conf_id, *, max_iters, optimize) -> float | None:
    if props is None:
        return None

    def _run_with_props():
        try:
            if optimize:
                AllChem.MMFFOptimizeMolecule(mol, props, confId=conf_id, maxIters=max_iters)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            return ff.CalcEnergy()
        except TypeError:
            return None

    energy = _run_with_props()
    if energy is not None:
        return energy

    for variant in ("MMFF94s", "MMFF94"):
        def _run_named():
            try:
                if optimize:
                    AllChem.MMFFOptimizeMolecule(
                        mol,
                        mmffVariant=variant,
                        maxIters=max_iters,
                        confId=conf_id,
                    )
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol,
                    mmffVariant=variant,
                    confId=conf_id,
                )
                return ff.CalcEnergy()
            except TypeError:
                return None

        def _run_positional():
            try:
                if optimize:
                    AllChem.MMFFOptimizeMolecule(mol, variant, max_iters, 100.0, conf_id)
                ff = AllChem.MMFFGetMoleculeForceField(mol, variant, 100.0, conf_id)
                return ff.CalcEnergy()
            except TypeError:
                return None

        energy = _run_named()
        if energy is not None:
            return energy

        energy = _run_positional()
        if energy is not None:
            return energy

    return None


def _calc_uff_energy(AllChem, mol, conf_id, *, max_iters, optimize) -> float | None:
    def _run_named():
        try:
            if optimize:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            return ff.CalcEnergy()
        except TypeError:
            return None
        except Exception:
            return None

    def _run_positional():
        try:
            if optimize:
                AllChem.UFFOptimizeMolecule(mol, max_iters, confId=conf_id)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            return ff.CalcEnergy()
        except Exception:
            return None

    energy = _run_named()
    if energy is not None:
        return energy
    return _run_positional()


def smiles_dict_to_sdf(
    smiles_map: Mapping[str, Union[str, Mapping[str, Any]]],
    output_dir: Union[str, Path],
    *,
    common_kwargs: Mapping[str, Any] | None = None,
    conda_env: str | None = None,
    python_executable: Union[str, Path, None] = None,
) -> Dict[str, str]:
    """
    Convert a mapping of names to SMILES (or per-entry dicts) into SDF files.

    Returns a dict mapping each key to the absolute path of the generated SDF.
    """
    if not smiles_map:
        raise ValueError("smiles_dict_to_sdf received an empty mapping.")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if conda_env or python_executable:
        return _smiles_dict_to_sdf_external(
            smiles_map,
            out_dir,
            common_kwargs=common_kwargs or {},
            conda_env=conda_env,
            python_executable=python_executable,
        )

    return _smiles_dict_to_sdf_local(smiles_map, out_dir, common_kwargs or {})


def _smiles_dict_to_sdf_local(
    smiles_map: Mapping[str, Union[str, Mapping[str, Any]]],
    out_dir: Path,
    common_kwargs: Mapping[str, Any],
) -> Dict[str, str]:
    defaults: MutableMapping[str, Any] = dict(common_kwargs)
    results: Dict[str, str] = {}

    for key, entry in smiles_map.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Invalid dictionary key {key!r}; keys must be non-empty strings.")

        if isinstance(entry, str):
            smiles_value = entry
            kwargs = dict(defaults)
        elif isinstance(entry, Mapping):
            if "smiles" not in entry:
                raise ValueError(f"Entry for key {key!r} is missing a 'smiles' field.")
            entry_dict = dict(entry)
            smiles_value = entry_dict.pop("smiles")
            kwargs = dict(defaults)
            kwargs.update(entry_dict)
        else:
            raise TypeError(
                f"Entry for key {key!r} must be a SMILES string or a mapping with a 'smiles' field."
            )

        out_file = out_dir / f"{key}.sdf"
        results[key] = smiles_to_sdf(smiles_value, out_file, **kwargs)

    return results


def _smiles_dict_to_sdf_external(
    smiles_map: Mapping[str, Union[str, Mapping[str, Any]]],
    out_dir: Path,
    *,
    common_kwargs: Mapping[str, Any],
    conda_env: str | None,
    python_executable: Union[str, Path, None],
) -> Dict[str, str]:
    payload = {
        "smiles_map": smiles_map,
        "output_dir": str(out_dir),
        "common_kwargs": dict(common_kwargs),
        "module_path": str(Path(__file__).resolve()),
    }

    cmd = _build_external_command(conda_env=conda_env, python_executable=python_executable)
    script = _bootstrap_script()
    full_cmd = cmd + ["-c", script]

    token = base64.b64encode(pickle.dumps(payload)).decode("ascii")

    try:
        proc = subprocess.run(
            full_cmd,
            check=True,
            input=token,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "External execution failed because the specified Python/conda executable was not found."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "External RDKit execution failed.\n"
            f"Command: {' '.join(full_cmd)}\n"
            f"Stdout:\n{exc.stdout}\n"
            f"Stderr:\n{exc.stderr}"
        ) from exc

    try:
        result_bytes = base64.b64decode((proc.stdout or "").strip().encode("ascii"))
        result_map = pickle.loads(result_bytes)
    except Exception as exc:
        raise RuntimeError(
            "Failed to parse output from external RDKit execution.\n"
            f"Stdout was:\n{proc.stdout}\nStderr:\n{proc.stderr}"
        ) from exc

    return {key: str(path) for key, path in result_map.items()}


def _build_external_command(
    *, conda_env: str | None, python_executable: Union[str, Path, None]
) -> list[str]:
    if conda_env:
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env]
        if python_executable:
            cmd.append(str(python_executable))
        else:
            cmd.append("python")
        return cmd

    if python_executable:
        return [str(python_executable)]

    return [sys.executable]


def _bootstrap_script() -> str:
    return (
        "import base64, importlib.util, pickle, sys\n"
        "from pathlib import Path\n"
        "payload_token = sys.stdin.read()\n"
        "payload = pickle.loads(base64.b64decode(payload_token.encode('ascii')))\n"
        "spec = importlib.util.spec_from_file_location('external_ligand_builders', str(payload['module_path']))\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "result = mod._smiles_dict_to_sdf_local(\n"
        "    payload['smiles_map'],\n"
        "    Path(payload['output_dir']),\n"
        "    payload.get('common_kwargs', {}),\n"
        ")\n"
        "sys.stdout.write(base64.b64encode(pickle.dumps(result)).decode('ascii'))\n"
    )
