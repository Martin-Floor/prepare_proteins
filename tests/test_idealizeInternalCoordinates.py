"""Tests for prepare_proteins.PDB.idealizeInternalCoordinates."""
import io
import os
import tempfile

import numpy as np
import pytest

from Bio.PDB import PDBParser

from prepare_proteins.PDB import idealizeInternalCoordinates


def _esmfold2_like_phe_pdb(path):
    """Write a tiny PDB with one Phe sidechain that has deformed ring geometry
    (mimicking ESMFold2 raw output: aromatic bonds 0.5-2.0 Å instead of 1.39)."""
    text = """\
ATOM      1  N   PHE A   1      10.000   0.000   0.000  1.00 95.0           N
ATOM      2  CA  PHE A   1      11.460   0.000   0.000  1.00 95.0           C
ATOM      3  C   PHE A   1      11.972   1.430   0.300  1.00 95.0           C
ATOM      4  O   PHE A   1      11.460   2.382  -0.298  1.00 95.0           O
ATOM      5  CB  PHE A   1      11.972  -0.760   1.218  1.00 95.0           C
ATOM      6  CG  PHE A   1      13.422  -0.760   1.218  1.00 95.0           C
ATOM      7  CD1 PHE A   1      14.122  -1.870   1.610  1.00 95.0           C
ATOM      8  CD2 PHE A   1      13.422   0.350   2.060  1.00 95.0           C
ATOM      9  CE1 PHE A   1      15.940  -1.870   1.610  1.00 95.0           C
ATOM     10  CE2 PHE A   1      14.890   0.350   2.580  1.00 95.0           C
ATOM     11  CZ  PHE A   1      15.940  -0.700   2.080  1.00 95.0           C
ATOM     12  N   GLY A   2      13.286   1.567   1.027  1.00 95.0           N
ATOM     13  CA  GLY A   2      13.857   2.882   1.227  1.00 95.0           C
ATOM     14  C   GLY A   2      15.380   2.802   1.227  1.00 95.0           C
ATOM     15  O   GLY A   2      15.992   3.866   1.227  1.00 95.0           O
TER
END
"""
    with open(path, "w") as fh:
        fh.write(text)


def _phe_ring_geometry(struct):
    """Return ring bonds (list of 6 lengths) + max non-planarity (Å)."""
    res = struct[0]["A"][1]
    ring = ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]
    bonds = []
    for i, a in enumerate(ring):
        b = ring[(i + 1) % len(ring)]
        bonds.append(float(np.linalg.norm(res[a].coord - res[b].coord)))
    pts = np.array([res[n].coord for n in ring])
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    plan = float(np.max(np.abs(centered @ vh[-1])))
    return bonds, plan


def _backbone_coords(struct):
    out = {}
    for chain in struct[0]:
        for res in chain:
            if res.id[0] != " ":
                continue
            for an in ("N", "CA", "C", "O"):
                if an in res:
                    out[(res.id[1], an)] = res[an].coord.copy()
    return out


def test_idealize_fixes_phe_ring(tmp_path):
    in_pdb = tmp_path / "deformed.pdb"
    out_pdb = tmp_path / "idealized.pdb"
    _esmfold2_like_phe_pdb(str(in_pdb))

    diag = idealizeInternalCoordinates(str(in_pdb), str(out_pdb))
    assert diag["n_hedra_reset"] > 0
    assert out_pdb.is_file()

    s_in = PDBParser(QUIET=True).get_structure("in", str(in_pdb))
    s_out = PDBParser(QUIET=True).get_structure("out", str(out_pdb))

    bonds_in, plan_in = _phe_ring_geometry(s_in)
    bonds_out, plan_out = _phe_ring_geometry(s_out)

    # Input ring is deformed
    assert max(abs(b - 1.39) for b in bonds_in) > 0.1
    assert plan_in > 0.1

    # Output ring is essentially canonical
    assert max(abs(b - 1.39) for b in bonds_out) < 0.1, bonds_out
    assert plan_out < 0.05, f"ring not planar: {plan_out}"


def test_idealize_preserves_backbone(tmp_path):
    in_pdb = tmp_path / "deformed.pdb"
    out_pdb = tmp_path / "idealized.pdb"
    _esmfold2_like_phe_pdb(str(in_pdb))

    idealizeInternalCoordinates(str(in_pdb), str(out_pdb))

    s_in = PDBParser(QUIET=True).get_structure("in", str(in_pdb))
    s_out = PDBParser(QUIET=True).get_structure("out", str(out_pdb))

    bb_in = _backbone_coords(s_in)
    bb_out = _backbone_coords(s_out)
    for key, coord_in in bb_in.items():
        coord_out = bb_out[key]
        # Every backbone atom must be at the same Cartesian position
        assert np.allclose(coord_in, coord_out, atol=1e-3), (key, coord_in, coord_out)


def _chi1(res):
    n, ca, cb, cg = (res[a].coord for a in ("N", "CA", "CB", "CG"))
    b1 = ca - n
    b2 = cb - ca
    b3 = cg - cb
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.degrees(np.arctan2(np.dot(m, n2), np.dot(n1, n2))))


def test_idealize_preserves_chi1(tmp_path):
    in_pdb = tmp_path / "deformed.pdb"
    out_pdb = tmp_path / "idealized.pdb"
    _esmfold2_like_phe_pdb(str(in_pdb))

    s_in = PDBParser(QUIET=True).get_structure("in", str(in_pdb))
    chi1_in = _chi1(s_in[0]["A"][1])

    idealizeInternalCoordinates(str(in_pdb), str(out_pdb))

    s_out = PDBParser(QUIET=True).get_structure("out", str(out_pdb))
    chi1_out = _chi1(s_out[0]["A"][1])

    # χ1 must be preserved to within a degree
    assert abs(chi1_in - chi1_out) < 1.0, (chi1_in, chi1_out)
