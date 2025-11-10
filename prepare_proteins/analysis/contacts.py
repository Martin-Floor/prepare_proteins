"""
Neighbour contact analysis helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

# --- simple chemistry heuristics
METAL_ELEMENTS: Set[str] = {"ZN", "FE", "MG", "MN", "CU", "CO", "NI", "CD", "CA", "K", "NA", "SR", "YB"}
WATER_RESNAMES: Set[str] = {"HOH", "WAT", "H2O"}
COFACTOR_RESNAMES: Set[str] = {"NAD", "NADP", "FAD", "FMN", "HEM", "HEME", "ZNB", "COA", "SAM", "SAH", "F5S"}
BACKBONE_NAMES: Set[str] = {"N", "CA", "C", "O", "OXT"}


_KDTREE_CLS = None
_KDTREE_CHECKED = False


def _get_kdtree_cls():
    global _KDTREE_CLS, _KDTREE_CHECKED
    if not _KDTREE_CHECKED:
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except Exception:
            _KDTREE_CLS = None
        else:
            _KDTREE_CLS = cKDTree
        _KDTREE_CHECKED = True
    return _KDTREE_CLS


@dataclass(frozen=True)
class AtomRec:
    idx: int
    xyz: np.ndarray  # (3,)
    chain: str
    resname: str
    resseq: int
    icode: str
    atom: str
    element: str
    is_protein: bool
    is_water: bool
    is_metal: bool
    is_ion: bool
    is_cofactor: bool
    is_organic: bool
    is_heavy: bool
    is_backbone: bool
    is_sidechain: bool


def _element_of(atom) -> str:
    e = atom.element.strip().upper() if atom.element else atom.get_name()[0].upper()
    return "ZN" if e == "ZN2" else e


def _is_protein_resname(resname: str) -> bool:
    r = resname.upper()
    return r.isalpha() and len(r) == 3 and r not in COFACTOR_RESNAMES and r not in WATER_RESNAMES


def _classify_atom(resname: str, element: str, hetflag: bool):
    r, e = resname.upper(), element.upper()
    is_water = r in WATER_RESNAMES
    is_metal = (e in METAL_ELEMENTS) or (r in METAL_ELEMENTS)
    is_ion = (e in {"CL", "BR", "I"} or r in {"CL", "BR", "IOD", "K", "NA"}) and not is_metal
    is_cofactor = r in COFACTOR_RESNAMES
    is_protein = (not hetflag) and _is_protein_resname(r)
    is_organic = hetflag and not (is_water or is_metal or is_ion or is_cofactor)
    return is_protein, is_water, is_metal, is_ion, is_cofactor, is_organic


def _iter_atomrecs(pdb_path: str, altloc_mode: Literal["first", "best_occ"] = "first") -> List[AtomRec]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    records: List[AtomRec] = []
    idx = 0

    best_occ: Dict[Tuple[str, Tuple, str], float] = {}
    chosen_alt: Dict[Tuple[str, Tuple, str], str] = {}

    if altloc_mode == "best_occ":
        for atom in structure.get_atoms():
            res = atom.get_parent()
            chain = res.get_parent().id
            hetflag, resseq, icode = res.id
            key = (chain, (hetflag, f"{resseq}{icode or ' '}"), atom.get_name())
            occ = float(atom.get_occupancy() or 1.0)
            if key not in best_occ or occ > best_occ[key]:
                best_occ[key] = occ
                chosen_alt[key] = atom.get_altloc() or " "

    for atom in structure.get_atoms():
        res = atom.get_parent()
        chain = res.get_parent().id
        hetflag, resseq, icode = res.id
        altloc = atom.get_altloc() or " "
        name = atom.get_name().strip()
        if altloc_mode == "first":
            if altloc not in (" ", "A"):
                continue
        else:
            key = (chain, (hetflag, f"{resseq}{icode or ' '}"), name)
            if chosen_alt.get(key, altloc) != altloc:
                continue

        element = _element_of(atom)
        het = hetflag != " "
        is_protein, is_water, is_metal, is_ion, is_cofactor, is_organic = _classify_atom(res.resname, element, het)

        rec = AtomRec(
            idx=idx,
            xyz=atom.get_vector().get_array().astype(float),
            chain=chain,
            resname=res.resname.upper(),
            resseq=int(resseq),
            icode=(icode or " ").strip() or " ",
            atom=name,
            element=element,
            is_protein=is_protein,
            is_water=is_water,
            is_metal=is_metal,
            is_ion=is_ion,
            is_cofactor=is_cofactor,
            is_organic=is_organic,
            is_heavy=(element != "H"),
            is_backbone=(name in BACKBONE_NAMES),
            is_sidechain=(name not in BACKBONE_NAMES and is_protein),
        )
        records.append(rec)
        idx += 1
    return records


def _mask_from_scope(arr: np.ndarray, scope: Literal["all", "heavy", "backbone", "sidechain"]) -> np.ndarray:
    return {
        "all": np.ones(arr.shape[0], bool),
        "heavy": arr["is_heavy"],
        "backbone": arr["is_backbone"],
        "sidechain": arr["is_sidechain"],
    }[scope]


def _pairs_within(q_xyz: np.ndarray, t_xyz: np.ndarray, cutoff: float) -> List[Tuple[int, int, float]]:
    if q_xyz.size == 0 or t_xyz.size == 0:
        return []

    tree_cls = _get_kdtree_cls()
    if tree_cls is not None:
        tree = tree_cls(t_xyz)
        out: List[Tuple[int, int, float]] = []
        for qi, pt in enumerate(q_xyz):
            idxs = tree.query_ball_point(pt, r=cutoff)
            if idxs:
                d = np.linalg.norm(t_xyz[idxs] - pt, axis=1)
                out.extend((qi, ti, float(dd)) for ti, dd in zip(idxs, d))
        return out

    diff = q_xyz[:, None, :] - t_xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    hits = np.where(dist <= cutoff)
    return [(int(qi), int(ti), float(dist[qi, ti])) for qi, ti in zip(*hits)]


def _contact_class(q_is_bb: bool, q_is_sc: bool, t_is_bb: bool, t_is_sc: bool) -> str:
    if q_is_bb and t_is_bb:
        return "BB"
    if q_is_sc and t_is_sc:
        return "SS"
    return "BS"


def _contact_class_extended(q_rec: np.void, t_rec: np.void, classify_bb_sc: bool) -> str:
    """Return contact class including metal-specific classes.

    - If ``t_rec`` is a metal atom not belonging to a cofactor residue, return:
        - "MB" when the query atom is backbone
        - "MS" when the query atom is sidechain (or neither explicitly BB/SC)
    - If ``t_rec`` is any non-protein non-metal ligand atom (water, ion, organic, cofactor, or cofactor-bound metal),
      classify as:
        - "LB" when the query atom is backbone
        - "LS" otherwise
    - If both sides are protein, fall back to BB/SS/BS when ``classify_bb_sc`` is True, or "NA" if False.
    - Cofactor-bound metal atoms are treated as cofactors (i.e., follow LB/LS, not MB/MS).
    """
    if not classify_bb_sc:
        return "NA"

    # If target is not protein, classify as ligand-specific classes
    if not t_rec["is_protein"]:
        # Metals not belonging to cofactors → MB/MS
        if t_rec["is_metal"] and (not t_rec["is_cofactor"]):
            return "MB" if q_rec["is_backbone"] else "MS"
        # All other ligands (including cofactors and cofactor-bound metals) → LB/LS
        return "LB" if q_rec["is_backbone"] else "LS"

    # Protein–protein contacts → BB/SS/BS
    return _contact_class(q_rec["is_backbone"], q_rec["is_sidechain"], t_rec["is_backbone"], t_rec["is_sidechain"])


def find_neighbours_in_pdb(
    pdb_path: str,
    query_chains: Iterable[str],
    *,
    mode: Literal["chains", "ligands", "both"] = "both",
    cutoff: float = 5.0,
    second_shell: Optional[float] = None,
    atom_scope: Literal["all", "heavy", "backbone", "sidechain"] = "heavy",
    group_by: Literal["residue", "atom", "chain"] = "residue",
    exclude_query_intra: bool = True,
    include: Optional[Dict[str, Any]] = None,
    ligand_filters: Optional[Dict[str, Any]] = None,
    chain_filters: Optional[Dict[str, Any]] = None,
    altloc_mode: Literal["first", "best_occ"] = "first",
    print_chain_summary: bool = False,
    only_residue_dict: bool = False,
    exclude_bb_contacts: bool = False,
    classify_bb_sc: bool = True,
    filter_contact_classes: Optional[Set[str]] = None,
    split_counts_by_class: bool = False,
) -> Dict[str, Any] | Dict[str, List[Tuple[int, str, str]]]:
    """
    Compute neighbours for the specified query chains in a single PDB.

    If only_residue_dict=True:
        returns { chain: [(resseq, icode, resname), ...], ... }
        If exclude_bb_contacts=True, residues whose contacts are all backbone-backbone (BB) are removed.
    Else:
        returns { "table": DataFrame, "contact_pairs": {...}, "residue_dict": {...} }

    Notes on contact_class when ``classify_bb_sc=True``:
    - Protein–protein contacts: "BB", "SS", or "BS" depending on backbone/sidechain.
    - Protein–metal contacts (neighbor_kind == "metal"): "MB" (metal–backbone) or "MS" (metal–sidechain).
    - Protein–ligand contacts (non-metals: water, ions, organics, cofactors, and cofactor-bound metals):
      "LB" (ligand–backbone) if the protein atom is backbone, else "LS" (ligand–sidechain).
    - Metal atoms that belong to cofactor residues are treated as cofactors (thus LB/LS, not MB/MS).
    """
    include = include or {"waters": False, "ions": True, "metals": True, "cofactors": True, "organics": True}
    ligand_filters = ligand_filters or {"resnames_in": [], "resnames_ex": [], "min_heavy_atoms": 1}
    chain_filters = chain_filters or {"include_chains": None, "exclude_chains": []}

    recs = _iter_atomrecs(pdb_path, altloc_mode=altloc_mode)
    arr = np.array(
        [
            (
                r.idx,
                *r.xyz,
                r.chain,
                r.resname,
                r.resseq,
                r.icode,
                r.atom,
                r.element,
                r.is_protein,
                r.is_water,
                r.is_metal,
                r.is_ion,
                r.is_cofactor,
                r.is_organic,
                r.is_heavy,
                r.is_backbone,
                r.is_sidechain,
            )
            for r in recs
        ],
        dtype=[
            ("idx", int),
            ("x", float),
            ("y", float),
            ("z", float),
            ("chain", "U4"),
            ("resname", "U8"),
            ("resseq", int),
            ("icode", "U1"),
            ("atom", "U6"),
            ("element", "U4"),
            ("is_protein", bool),
            ("is_water", bool),
            ("is_metal", bool),
            ("is_ion", bool),
            ("is_cofactor", bool),
            ("is_organic", bool),
            ("is_heavy", bool),
            ("is_backbone", bool),
            ("is_sidechain", bool),
        ],
    )
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)

    res_uid = np.core.defchararray.add(np.core.defchararray.add(arr["chain"], ":"), np.char.add(np.char.mod("%d", arr["resseq"]), arr["icode"]))
    resid_counts_heavy = pd.Series(arr["is_heavy"]).groupby(res_uid).sum().to_dict()

    def build_target_mask_for_ligands():
        m = ~arr["is_protein"]

        # Treat metals that belong to cofactor residues as cofactors (not metals)
        is_metal_effective = arr["is_metal"] & ~arr["is_cofactor"]

        if not include.get("waters", False):
            m &= ~arr["is_water"]
        if not include.get("ions", True):
            m &= ~arr["is_ion"]
        if not include.get("metals", True):
            m &= ~is_metal_effective
        if not include.get("cofactors", True):
            m &= ~arr["is_cofactor"]
        if not include.get("organics", True):
            m &= ~arr["is_organic"]
        rin = set(x.upper() for x in (ligand_filters.get("resnames_in") or []))
        rex = set(x.upper() for x in (ligand_filters.get("resnames_ex") or []))
        if rin:
            m &= np.isin(arr["resname"], np.array(list(rin)))
        if rex:
            m &= ~np.isin(arr["resname"], np.array(list(rex)))
        if ligand_filters.get("min_heavy_atoms", 1) > 1:
            keep = np.array([resid_counts_heavy[u] >= ligand_filters["min_heavy_atoms"] for u in res_uid])
            m &= keep
        return m

    ligand_mask_all = build_target_mask_for_ligands()
    protein_chains_all = np.unique(arr["chain"][arr["is_protein"]])

    rows: List[Dict[str, Any]] = []
    contact_pairs: Dict[str, List[Tuple[int, int, float]]] = {}
    residue_dict: Dict[str, List[Tuple[int, str, str]]] = {}

    for qch in map(str, query_chains):
        q_mask = (arr["chain"] == qch) & arr["is_protein"]
        q_mask &= _mask_from_scope(arr, atom_scope)
        if not np.any(q_mask):
            contact_pairs[qch] = []
            residue_dict[qch] = []
            continue

        target_masks = []
        if mode in ("chains", "both"):
            allowed = [c for c in protein_chains_all if c != qch]
            inc = chain_filters.get("include_chains")
            if inc is not None:
                allowed = [c for c in allowed if c in inc]
            exc = set(chain_filters.get("exclude_chains", []))
            allowed = [c for c in allowed if c not in exc]
            if allowed:
                m = arr["is_protein"] & np.isin(arr["chain"], np.array(allowed))
                m &= _mask_from_scope(arr, atom_scope)
                target_masks.append(m)
        if mode in ("ligands", "both"):
            m = ligand_mask_all & _mask_from_scope(arr, atom_scope)
            target_masks.append(m)

        if not target_masks:
            contact_pairs[qch] = []
            residue_dict[qch] = []
            continue

        t_mask = np.logical_or.reduce(target_masks)
        if not np.any(t_mask):
            contact_pairs[qch] = []
            residue_dict[qch] = []
            continue

        q_idx = np.where(q_mask)[0]
        t_idx = np.where(t_mask)[0]
        q_xyz, t_xyz = xyz[q_idx], xyz[t_idx]

        first_pairs_abs = [(int(q_idx[qi]), int(t_idx[ti]), float(d)) for (qi, ti, d) in _pairs_within(q_xyz, t_xyz, cutoff)]
        contact_pairs[qch] = first_pairs_abs

        second_pairs_abs: List[Tuple[int, int, float]] = []
        if second_shell and second_shell > cutoff:
            tree_cls = _get_kdtree_cls()
            if tree_cls is not None:
                tree = tree_cls(t_xyz)
                for qi, pt in enumerate(q_xyz):
                    idxs = tree.query_ball_point(pt, r=second_shell)
                    for ti in idxs:
                        d = float(np.linalg.norm(t_xyz[ti] - pt))
                        if cutoff < d <= second_shell:
                            second_pairs_abs.append((int(q_idx[qi]), int(t_idx[ti]), d))
            else:
                diff = q_xyz[:, None, :] - t_xyz[None, :, :]
                dist = np.linalg.norm(diff, axis=2)
                hits = np.where((dist > cutoff) & (dist <= second_shell))
                second_pairs_abs = [(int(q_idx[qi]), int(t_idx[ti]), float(dist[qi, ti])) for qi, ti in zip(*hits)]

        chain_rows: List[Dict[str, Any]] = []

        def _pack_row(qi, ti, dist, shell: str):
            q, t = arr[qi], arr[ti]
            if not t["is_protein"]:
                # Prioritize cofactor classification over metal when both apply
                if t["is_cofactor"]:
                    kind = "cofactor"
                elif t["is_metal"]:
                    kind = "metal"
                elif t["is_ion"]:
                    kind = "ion"
                elif t["is_cofactor"]:
                    kind = "cofactor"
                elif t["is_water"]:
                    kind = "water"
                else:
                    kind = "organic"
            else:
                kind = "chain"
            # Compute contact class with metal-specific MB/MS, treating cofactor-bound metals as cofactors
            cls = _contact_class_extended(q, t, classify_bb_sc)
            return {
                "query_chain": q["chain"],
                "query_resseq": int(q["resseq"]),
                "query_icode": q["icode"],
                "query_resname": q["resname"],
                "query_atom": q["atom"],
                "neighbor_kind": kind,
                "neighbor_chain": t["chain"] if t["is_protein"] else "-",
                "neighbor_resseq": int(t["resseq"]) if t["is_protein"] else -1,
                "neighbor_icode": t["icode"] if t["is_protein"] else " ",
                "neighbor_resname": t["resname"],
                "neighbor_atom": t["atom"],
                "min_distance": float(dist),
                "shell": shell,
                "contact_class": cls,
            }

        chain_rows.extend(_pack_row(qi, ti, d, "first") for qi, ti, d in first_pairs_abs)
        chain_rows.extend(_pack_row(qi, ti, d, "second") for qi, ti, d in second_pairs_abs)

        if classify_bb_sc and filter_contact_classes:
            keep = set(filter_contact_classes)
            chain_rows = [r for r in chain_rows if r["contact_class"] in keep]

        rows.extend(chain_rows)

        if only_residue_dict:
            atom_classes: Dict[int, Set[str]] = {}
            for qi, ti, _ in (first_pairs_abs + second_pairs_abs):
                cls = _contact_class_extended(arr[qi], arr[ti], classify_bb_sc)
                atom_classes.setdefault(qi, set()).add(cls)

            res_keys: Set[Tuple[int, str, str]] = set()
            for qi, classes in atom_classes.items():
                if exclude_bb_contacts and classes == {"BB"}:
                    continue
                res_keys.add((int(arr[qi]["resseq"]), arr[qi]["icode"], arr[qi]["resname"]))
            residue_dict[qch] = sorted(res_keys, key=lambda x: (x[0], x[1]))
        else:
            contact_q_atoms = {qi for qi, _, _ in first_pairs_abs} | {qi for qi, _, _ in second_pairs_abs}
            res_keys = {(int(arr[qi]["resseq"]), arr[qi]["icode"], arr[qi]["resname"]) for qi in contact_q_atoms}
            residue_dict[qch] = sorted(res_keys, key=lambda x: (x[0], x[1]))

    if only_residue_dict:
        return residue_dict

    cols = (
        [
            "query_chain",
            "query_resseq",
            "query_icode",
            "query_resname",
            "query_atom",
            "neighbor_kind",
            "neighbor_chain",
            "neighbor_resseq",
            "neighbor_icode",
            "neighbor_resname",
            "neighbor_atom",
            "min_distance",
            "shell",
            "contact_class",
        ]
        if classify_bb_sc
        else [
            "query_chain",
            "query_resseq",
            "query_icode",
            "query_resname",
            "query_atom",
            "neighbor_kind",
            "neighbor_chain",
            "neighbor_resseq",
            "neighbor_icode",
            "neighbor_resname",
            "neighbor_atom",
            "min_distance",
            "shell",
        ]
    )
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    if not df.empty:
        if group_by == "residue":
            base_grp = [
                "query_chain",
                "query_resseq",
                "query_icode",
                "query_resname",
                "neighbor_kind",
                "neighbor_chain",
                "neighbor_resseq",
                "neighbor_icode",
                "neighbor_resname",
                "shell",
            ]
            if classify_bb_sc and not split_counts_by_class:
                grp = base_grp + ["contact_class"]
                df = (
                    df.assign(n_atom_contacts=1)
                    .groupby(grp, as_index=False)
                    .agg(min_distance=("min_distance", "min"), n_atom_contacts=("n_atom_contacts", "sum"))
                )
            elif classify_bb_sc and split_counts_by_class:
                tmp = (
                    df.assign(n_atom_contacts=1)
                    .groupby(base_grp + ["contact_class"], as_index=False)
                    .agg(min_distance=("min_distance", "min"), n_atom_contacts=("n_atom_contacts", "sum"))
                )
                wide = (
                    tmp.pivot_table(index=base_grp, columns="contact_class", values="n_atom_contacts", fill_value=0).reset_index()
                )
                wide.columns = [c if isinstance(c, str) else f"counts_{c}" for c in wide.columns]
                md = tmp.groupby(base_grp, as_index=False)["min_distance"].min()
                df = wide.merge(md, on=base_grp, how="left")
            else:
                df = (
                    df.assign(n_atom_contacts=1)
                    .groupby(base_grp, as_index=False)
                    .agg(min_distance=("min_distance", "min"), n_atom_contacts=("n_atom_contacts", "sum"))
                )
        elif group_by == "chain":
            base_grp = ["query_chain", "neighbor_kind", "neighbor_chain", "shell"]
            if classify_bb_sc and not split_counts_by_class:
                grp = base_grp + ["contact_class"]
                df = (
                    df.assign(n_atom_contacts=1)
                    .groupby(grp, as_index=False)
                    .agg(min_distance=("min_distance", "min"), n_atom_contacts=("n_atom_contacts", "sum"))
                )
            elif classify_bb_sc and split_counts_by_class:
                tmp = (
                    df.assign(n_atom_contacts=1)
                    .groupby(base_grp + ["contact_class"], as_index=False)
                    .agg(min_distance=("min_distance", "min"), n_atom_contacts=("n_atom_contacts", "sum"))
                )
                wide = (
                    tmp.pivot_table(index=base_grp, columns="contact_class", values="n_atom_contacts", fill_value=0).reset_index()
                )
                wide.columns = [c if isinstance(c, str) else f"counts_{c}" for c in wide.columns]
                md = tmp.groupby(base_grp, as_index=False)["min_distance"].min()
                df = wide.merge(md, on=base_grp, how="left")
            else:
                df = (
                    df.assign(n_atom_contacts=1)
                    .groupby(base_grp, as_index=False)
                    .agg(min_distance=("min_distance", "min"), n_atom_contacts=("n_atom_contacts", "sum"))
                )
        elif group_by == "atom":
            pass
        else:
            raise ValueError(f"group_by={group_by}")

        sort_cols = [c for c in ["query_chain", "neighbor_kind", "neighbor_chain", "min_distance"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    if print_chain_summary and not df.empty:
        df_ch = df[df["neighbor_kind"] == "chain"].copy()
        if not df_ch.empty:
            if classify_bb_sc and "contact_class" in df_ch.columns and not split_counts_by_class:
                summary = (
                    df_ch.groupby(["query_chain", "neighbor_chain", "contact_class"], as_index=False)
                    .agg(
                        min_distance=("min_distance", "min"),
                        contacts=("n_atom_contacts", "sum" if "n_atom_contacts" in df_ch.columns else "size"),
                    )
                )
            else:
                summary = (
                    df_ch.groupby(["query_chain", "neighbor_chain"], as_index=False)
                    .agg(
                        min_distance=("min_distance", "min"),
                        contacts=("n_atom_contacts", "sum" if "n_atom_contacts" in df_ch.columns else "size"),
                    )
                )
            print("\n[Neighbour chain summary]")
            print(summary.to_string(index=False))
        else:
            print("\n[Neighbour chain summary] No chain–chain contacts within thresholds.")

    return {"table": df, "contact_pairs": contact_pairs, "residue_dict": residue_dict}
