#!/usr/bin/env python
"""
Convert Maestro (.mae/.maegz) files that contain one protein CT and several ligand
poses into individual PDB files that combine the protein with each ligand.

This script is intentionally placed inside the scripts/ folder so the core library
does not need to import Schrodinger's Python API directly.  The ProteinModels
class can invoke this script as an external process when it is handed a Maestro
file instead of a folder of PDB models.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from schrodinger import structure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "maestro_file",
        help="Path to the Maestro (.mae or .maegz) file that contains the protein and ligand poses.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the generated PDB models will be written.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Base name for the generated PDB models (defaults to the Maestro filename stem).",
    )
    parser.add_argument(
        "--protein-ct",
        type=int,
        default=None,
        help="1-based index of the CT that contains the protein (defaults to auto-detection).",
    )
    parser.add_argument(
        "--ligand-chain",
        default="L",
        help="Chain identifier assigned to the ligand residues (default: L).",
    )
    parser.add_argument(
        "--ligand-resnum",
        type=int,
        default=1,
        help="Residue number assigned to the ligand residues (default: 1).",
    )
    parser.add_argument(
        "--keep-original-ligand-ids",
        action="store_true",
        help="Keep the ligand chain/residue identifiers as-is instead of assigning --ligand-chain/--ligand-resnum.",
    )
    parser.add_argument(
        "--separator",
        default="-",
        help="Separator used when constructing output model names (default: '-').",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to a JSON manifest summarizing the generated models.",
    )
    return parser.parse_args()


def _load_cts(maestro_path: Path) -> List[structure.Structure]:
    return [st for st in structure.StructureReader(str(maestro_path))]


def _count_atoms(st: structure.Structure) -> int:
    return sum(1 for _ in st.atom)


def _guess_protein_ct(
    cts: Sequence[structure.Structure],
) -> Tuple[int, structure.Structure]:
    """
    Heuristically select the CT that corresponds to the protein by preferring CTs
    without docking score properties and, as a fallback, the CT with the largest
    atom count.
    """
    if not cts:
        raise ValueError("No CTs found in the Maestro file.")

    best_idx = 0
    best_atoms = -1
    for idx, st in enumerate(cts):
        props = getattr(st, "property", {}) or {}
        atom_count = _count_atoms(st)
        if atom_count > best_atoms:
            best_idx = idx
            best_atoms = atom_count
        if "r_i_glide_gscore" not in props:
            return idx, st
    return best_idx, cts[best_idx]


def _sanitize_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^A-Za-z0-9_.-]", "_", token)
    return token or "component"


def _derive_ligand_label(st: structure.Structure) -> Optional[str]:
    props = getattr(st, "property", {}) or {}
    for key in (
        "s_m_title",
        "s_glide_title",
        "s_pdb_PDBCODE",
        "s_protein_title",
    ):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    title = getattr(st, "title", None)
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _normalize_ligand_ids(
    ligand: structure.Structure, chain: str, resnum: int
) -> None:
    for residue in ligand.residue:
        residue.chain = chain
        residue.resnum = resnum


def _build_model_name(
    prefix: str, ligand_label: Optional[str], pose_index: int, separator: str
) -> str:
    components = [prefix]
    if ligand_label:
        components.append(ligand_label)
    components.append(f"pose{pose_index:03d}")
    return separator.join(_sanitize_token(part) for part in components if part)


def _write_manifest(manifest_path: Path, records: List[Dict[str, object]]) -> None:
    payload = {"models": records}
    with open(manifest_path, "w") as mf:
        json.dump(payload, mf, indent=2)


def main() -> None:
    args = _parse_args()

    maestro_path = Path(args.maestro_file).expanduser().resolve()
    if not maestro_path.exists():
        raise FileNotFoundError(f"Maestro file not found: {maestro_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)
    prefix = args.prefix or maestro_path.stem

    cts = _load_cts(maestro_path)
    if not cts:
        raise ValueError(f"No structures were found in {maestro_path}")

    if args.protein_ct:
        protein_index = args.protein_ct - 1
        if protein_index < 0 or protein_index >= len(cts):
            raise IndexError(
                f"Protein CT index {args.protein_ct} is out of bounds for {len(cts)} CTs."
            )
        protein_ct = cts[protein_index]
    else:
        protein_index, protein_ct = _guess_protein_ct(cts)

    ligand_cts = [
        (idx, st) for idx, st in enumerate(cts) if idx != protein_index
    ]
    if not ligand_cts:
        raise ValueError(
            "No ligand CTs were found after extracting the protein CT. "
            "Ensure the Maestro file contains at least one ligand pose."
        )

    manifest_records: List[Dict[str, object]] = []
    for pose_counter, (ct_index, ligand_ct) in enumerate(ligand_cts, start=1):
        ligand_label = _derive_ligand_label(ligand_ct)
        if not args.keep_original_ligand_ids:
            _normalize_ligand_ids(ligand_ct, args.ligand_chain, args.ligand_resnum)

        combined_structure = protein_ct.copy()
        combined_structure.extend(ligand_ct)

        model_name = _build_model_name(
            prefix=prefix,
            ligand_label=ligand_label,
            pose_index=pose_counter,
            separator=args.separator,
        )
        output_path = output_dir / f"{model_name}.pdb"
        combined_structure.write(str(output_path))

        manifest_records.append(
            {
                "model_name": model_name,
                "pdb_path": str(output_path),
                "ligand_label": ligand_label,
                "pose_index": pose_counter,
                "ct_index": ct_index + 1,
                "maestro_file": str(maestro_path),
            }
        )
        print(f"Generated {output_path}")

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        _write_manifest(manifest_path, manifest_records)
        print(f"Wrote manifest to {manifest_path}")

    print(
        f"Exported {len(manifest_records)} protein-ligand complexes to {output_dir} "
        f"(protein CT #{protein_index + 1})."
    )


if __name__ == "__main__":
    main()
