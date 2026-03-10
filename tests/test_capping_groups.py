from pathlib import Path

import pytest

import prepare_proteins
import prepare_proteins._protein_models as protein_models_module


MODEL_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
ATOM      5  N   GLY A   2      14.220  11.690   3.050  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.830  10.370   3.330  1.00 20.00           C
ATOM      7  C   GLY A   2      15.412   9.850   2.010  1.00 20.00           C
ATOM      8  OXT GLY A   2      15.962   8.743   1.970  1.00 20.00           O
TER
ATOM      9  N   GLY B   1      15.500  12.400   4.400  1.00 20.00           N
ATOM     10  CA  GLY B   1      16.040  11.110   4.880  1.00 20.00           C
ATOM     11  C   GLY B   1      15.220  10.050   4.140  1.00 20.00           C
ATOM     12  O   GLY B   1      15.620   8.890   4.030  1.00 20.00           O
TER
END
"""


def _write_model(models_dir: Path, model_name: str = "modelA") -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.pdb"
    model_path.write_text(MODEL_PDB)
    return model_path


def _load_models(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    return prepare_proteins.proteinModels(str(models_dir), conect_update=False)


def test_add_capping_groups_internal_rosetta_caps_selected_chain_and_saves(tmp_path, monkeypatch):
    models = _load_models(tmp_path)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Internal Rosetta capping should not invoke subprocess.run.")

    monkeypatch.setattr(protein_models_module.subprocess, "run", _fail_if_called)

    models.addCappingGroups(
        style="rosetta",
        backend="internal",
        models=["modelA"],
        chains="B",
    )

    structure = models.structures["modelA"]
    model = next(structure.get_models())
    chain_a = model["A"]
    chain_b = model["B"]

    first_a = next(residue for residue in chain_a if residue.id[0] == " ")
    first_b = next(residue for residue in chain_b if residue.id[0] == " ")
    assert not any(atom_name in first_a for atom_name in ("CP2", "CO", "OP1"))
    assert {"CP2", "CO", "OP1", "1HP2", "2HP2", "3HP2"}.issubset(first_b.child_dict)

    residues_b = list(chain_b)
    assert residues_b[1].resname == "NMA"
    assert {"N", "C", "HN2", "H1", "H2", "H3"} == {atom.name for atom in residues_b[1]}

    output_dir = tmp_path / "saved"
    models.saveModels(str(output_dir))
    saved_pdb = (output_dir / "modelA.pdb").read_text()
    assert " CP2" in saved_pdb
    assert " NMA B   2" in saved_pdb


def test_add_capping_groups_rosetta_bool_flag_uses_internal_backend(tmp_path, monkeypatch):
    models = _load_models(tmp_path)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Rosetta auto backend should not invoke subprocess.run.")

    monkeypatch.setattr(protein_models_module.subprocess, "run", _fail_if_called)

    models.addCappingGroups(
        rosetta_style_caps=True,
        backend="auto",
        chains="B",
    )

    chain_b = next(models.structures["modelA"].get_chains())
    # First chain returned is A, second is B.
    chain_b = list(models.structures["modelA"].get_chains())[1]
    first_b = next(residue for residue in chain_b if residue.id[0] == " ")
    assert {"CP2", "CO", "OP1", "1HP2", "2HP2", "3HP2"}.issubset(first_b.child_dict)


def test_remove_caps_removes_merged_rosetta_ace_and_nma(tmp_path):
    models = _load_models(tmp_path)
    models.addCappingGroups(style="rosetta", backend="internal", chains="B")

    models.removeCaps(models=["modelA"])

    model = next(models.structures["modelA"].get_models())
    chain_b = model["B"]
    first_b = next(residue for residue in chain_b if residue.id[0] == " ")
    assert not any(atom_name in first_b for atom_name in ("CP2", "CO", "OP1"))
    assert not any(residue.resname == "NMA" for residue in chain_b)


def test_add_capping_groups_internal_openmm_caps_selected_chain_and_saves(tmp_path, monkeypatch):
    models = _load_models(tmp_path)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Internal OpenMM capping should not invoke subprocess.run.")

    monkeypatch.setattr(protein_models_module.subprocess, "run", _fail_if_called)

    models.addCappingGroups(
        style="openmm",
        backend="internal",
        models=["modelA"],
        chains="B",
    )

    structure = models.structures["modelA"]
    model = next(structure.get_models())
    chain_a = model["A"]
    chain_b = model["B"]

    residues_a = list(chain_a)
    residues_b = list(chain_b)
    assert residues_a[0].resname == "ALA"
    assert residues_b[0].resname == "ACE"
    assert {"CH3", "C", "O", "HH31", "HH32", "HH33"} == {atom.name for atom in residues_b[0]}
    assert residues_b[1].resname == "GLY"
    assert residues_b[2].resname == "NME"
    assert {"N", "H", "CH3", "HH31", "HH32", "HH33"} == {atom.name for atom in residues_b[2]}

    output_dir = tmp_path / "saved_openmm"
    models.saveModels(str(output_dir))
    saved_pdb = (output_dir / "modelA.pdb").read_text()
    assert " ACE B   0" in saved_pdb
    assert " NME B   2" in saved_pdb


def test_add_capping_groups_openmm_bool_flag_uses_internal_backend(tmp_path, monkeypatch):
    models = _load_models(tmp_path)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("OpenMM auto backend should not invoke subprocess.run.")

    monkeypatch.setattr(protein_models_module.subprocess, "run", _fail_if_called)

    models.addCappingGroups(
        openmm_style_caps=True,
        backend="auto",
        chains="B",
    )

    chain_b = list(models.structures["modelA"].get_chains())[1]
    residues_b = list(chain_b)
    assert residues_b[0].resname == "ACE"
    assert residues_b[2].resname == "NME"
    assert {"CH3", "C", "O", "HH31", "HH32", "HH33"} == {atom.name for atom in residues_b[0]}
    assert {"N", "H", "CH3", "HH31", "HH32", "HH33"} == {atom.name for atom in residues_b[2]}


def test_remove_caps_removes_openmm_ace_and_nme(tmp_path):
    models = _load_models(tmp_path)
    models.addCappingGroups(style="openmm", backend="internal", chains="B")

    models.removeCaps(models=["modelA"])

    model = next(models.structures["modelA"].get_models())
    chain_b = model["B"]
    residues_b = list(chain_b)
    assert residues_b[0].resname == "GLY"
    assert not any(residue.resname in {"ACE", "NME"} for residue in chain_b)


def test_add_capping_groups_internal_rejects_prepwizard_style(tmp_path):
    models = _load_models(tmp_path)

    with pytest.raises(NotImplementedError):
        models.addCappingGroups(style="prepwizard", backend="internal")
