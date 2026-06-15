"""AutoDock Vina docking helpers for prepare_proteins.

Self-contained pipeline: rigid-receptor PDBQT (Open Babel), ligand PDBQT (meeko's
``mk_prepare_ligand``), search-box definition, running AutoDock Vina (python ``vina``
package), and pose parsing. The ``proteinModels.setUpVinaDocking`` method wraps these
to dock ligands into every loaded model.

Caveat for highly-charged ligands: Vina's scoring is hydrophobic/H-bond biased and
under-weights strong salt-bridge electrostatics, so a di-anion may dock in an extended
pose rather than a compact salt-bridged one. Use ``center``/``box_size`` to focus the
search, and refine the top pose with MD if a specific reactive conformation is needed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def prepare_receptor_pdbqt(pdb_path, out_pdbqt, obabel="obabel"):
    """Rigid-receptor PDBQT from a protein PDB via Open Babel (Gasteiger charges)."""
    if shutil.which(obabel) is None:
        raise RuntimeError("Open Babel ('obabel') not found on PATH.")
    subprocess.run([obabel, str(pdb_path), "-O", str(out_pdbqt), "-xr"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_pdbqt)


def prepare_ligand_pdbqt(ligand_path, out_pdbqt, mk_prepare="mk_prepare_ligand.py"):
    """Ligand PDBQT (rotatable torsions + charges) via meeko's mk_prepare_ligand.

    Accepts SDF/MOL2/PDB input. Using the CLI avoids the meeko<->RDKit version coupling
    of the python API.
    """
    exe = shutil.which(mk_prepare) or mk_prepare
    subprocess.run([exe, "-i", str(ligand_path), "-o", str(out_pdbqt)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_pdbqt)


def box_center_from_pdb(pdb_path, selections):
    """Geometric center of selected atoms, for placing the search box on a site.

    ``selections``: list of ``(chain_id, residue_id, atom_name)``; ``atom_name=None``
    selects all atoms of that residue.
    """
    from Bio.PDB import PDBParser
    model = PDBParser(QUIET=True).get_structure("x", str(pdb_path))[0]
    pts = []
    want = {(str(c), int(r), a) for c, r, a in selections}
    for chain in model:
        for res in chain:
            for atom in res:
                if ((str(chain.id), int(res.id[1]), atom.name) in want
                        or (str(chain.id), int(res.id[1]), None) in want):
                    pts.append(atom.coord)
    if not pts:
        raise ValueError("No atoms matched the given selections.")
    return [float(x) for x in np.mean(pts, axis=0)]


def run_vina(receptor_pdbqt, ligand_pdbqt, center, box_size=(22.0, 22.0, 22.0),
             exhaustiveness=16, n_poses=10, out_pdbqt=None, cpu=0, seed=0):
    """Run AutoDock Vina. Returns ``{'scores': [...], 'poses_pdbqt': path|None}``."""
    from vina import Vina
    v = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_file(str(ligand_pdbqt))
    v.compute_vina_maps(center=list(center), box_size=list(box_size))
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    scores = [float(e[0]) for e in v.energies(n_poses=n_poses)]
    if out_pdbqt is not None:
        v.write_poses(str(out_pdbqt), n_poses=n_poses, overwrite=True)
    return {"scores": scores, "poses_pdbqt": str(out_pdbqt) if out_pdbqt is not None else None}


def parse_pdbqt_models(pdbqt_path):
    """Read a multi-model Vina output PDBQT -> list of (atom_names, coords[N,3])."""
    models, names, coords = [], [], []
    for line in open(pdbqt_path):
        if line.startswith(("ATOM", "HETATM")):
            names.append(line[12:16].strip())
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ENDMDL"):
            models.append((names, np.array(coords))); names, coords = [], []
    if coords:
        models.append((names, np.array(coords)))
    return models


def dock(receptor_pdb, ligand_path, center, *, output_folder, box_size=(22.0, 22.0, 22.0),
         exhaustiveness=16, n_poses=10, name="dock"):
    """Full pipeline: prep receptor + ligand PDBQT, run Vina, write poses.

    Returns the run_vina result augmented with ``receptor_pdbqt``/``ligand_pdbqt`` paths.
    """
    out = Path(output_folder); out.mkdir(parents=True, exist_ok=True)
    rec = prepare_receptor_pdbqt(receptor_pdb, out / f"{name}_receptor.pdbqt")
    lig = prepare_ligand_pdbqt(ligand_path, out / f"{name}_ligand.pdbqt")
    res = run_vina(rec, lig, center, box_size, exhaustiveness, n_poses,
                   out / f"{name}_docked.pdbqt")
    res.update({"receptor_pdbqt": rec, "ligand_pdbqt": lig})
    return res
