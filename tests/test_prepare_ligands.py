"""Tests for the Schrodinger-free ligand preparation (`proteinModels.prepareLigands`).

`prepareLigands` embeds 3D conformers from SMILES with RDKit (ETKDG+MMFF) and
writes per-ligand SD files for rDock/rxDock. Skipped where RDKit is unavailable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
prepare_proteins = pytest.importorskip("prepare_proteins")
from rdkit import Chem


def _empty_models(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return prepare_proteins.proteinModels(str(d))


def test_embeds_3d_and_skips_bad_smiles(tmp_path):
    m = _empty_models(tmp_path)
    res = m.prepareLigands(
        {"good": "CC(=O)Nc1ccc(O)cc1", "bad": "this_is_not_smiles"},
        str(tmp_path / "out"),
        properties={"good": {"ref_score": -5.2}},
    )
    # bad SMILES dropped, good one prepared
    assert "good" in res and "bad" not in res
    mol = next(Chem.SDMolSupplier(res["good"]))
    assert mol is not None
    # has a real 3D conformer
    conf = mol.GetConformer()
    assert any(abs(conf.GetAtomPosition(i).z) > 1e-3
               for i in range(mol.GetNumAtoms()))
    # property is carried through into the SD
    assert mol.GetProp("ref_score") == "-5.2"


def test_accepts_iterable_and_writes_combined(tmp_path):
    m = _empty_models(tmp_path)
    combined = tmp_path / "all.sd"
    res = m.prepareLigands(
        [("a", "CCO"), ("b", "c1ccccc1")],
        str(tmp_path / "out2"),
        combined_sdf=str(combined),
    )
    assert set(res) == {"a", "b"}
    assert sum(1 for _ in Chem.SDMolSupplier(str(combined))) == 2
