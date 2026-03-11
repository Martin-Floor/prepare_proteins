from pathlib import Path

import pytest

import prepare_proteins


pytest.importorskip("rdkit")


def _pdb_line(
    record,
    serial,
    atom_name,
    resname,
    chain,
    residue_id,
    x,
    y,
    z,
    element,
    charge="  ",
):
    return (
        f"{record:<6}{serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}{residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00{'':10s}{element:>2s}{charge:>2s}\n"
    )


def _write_models(tmp_path, model_pdbs, *, conect_update=True):
    models_dir = Path(tmp_path) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_name, pdb_text in model_pdbs.items():
        (models_dir / f"{model_name}.pdb").write_text(pdb_text)
    return prepare_proteins.proteinModels(str(models_dir), conect_update=conect_update)


def test_write_ligand_mol_files_writes_mol_and_preserves_formal_charge(tmp_path):
    from rdkit import Chem

    pdb_text = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "CAR", "A", 101, 0.0, 1.0, 0.0, "C")
        + _pdb_line("HETATM", 3, "O1", "CAR", "A", 101, 1.2, 1.0, 0.0, "O", "1-")
        + _pdb_line("HETATM", 4, "O2", "CAR", "A", 101, -1.2, 1.0, 0.0, "O")
        + "CONECT    2    3    4\n"
        + "END\n"
    )
    models = _write_models(tmp_path, {"modelA": pdb_text})

    paths, report = models.writeLigandMolFiles(tmp_path / "mol_files", return_report=True)

    assert set(paths) == {"CAR"}
    output_path = Path(paths["CAR"])
    assert output_path.exists()

    mol = Chem.MolFromMolFile(str(output_path), removeHs=False)
    assert mol is not None
    assert Chem.GetFormalCharge(mol) == -1

    assert report.loc[0, "resname"] == "CAR"
    assert report.loc[0, "output_path"] == str(output_path.resolve())
    assert report.loc[0, "formal_charge"] == -1
    assert report.loc[0, "block_source"] == "source"


def test_write_ligand_mol_files_raises_for_conflicting_reused_names_with_same_charge(tmp_path):
    model_a = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "LIG", "A", 101, 0.0, 1.0, 0.0, "C")
        + _pdb_line("HETATM", 3, "H1", "LIG", "A", 101, 0.9, 1.0, 0.0, "H")
        + _pdb_line("HETATM", 4, "H2", "LIG", "A", 101, -0.3, 1.8, 0.0, "H")
        + _pdb_line("HETATM", 5, "H3", "LIG", "A", 101, -0.3, 0.2, 0.0, "H")
        + _pdb_line("HETATM", 6, "H4", "LIG", "A", 101, 0.0, 1.0, 0.9, "H")
        + "CONECT    2    3    4    5    6\n"
        + "END\n"
    )
    model_b = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "O1", "LIG", "A", 201, 0.0, 1.0, 0.0, "O")
        + _pdb_line("HETATM", 3, "H1", "LIG", "A", 201, 0.8, 1.3, 0.0, "H")
        + _pdb_line("HETATM", 4, "H2", "LIG", "A", 201, -0.8, 1.3, 0.0, "H")
        + "CONECT    2    3    4\n"
        + "END\n"
    )
    models = _write_models(tmp_path, {"modelA": model_a, "modelB": model_b})

    with pytest.raises(ValueError, match="makeLigandNamesUnique"):
        models.writeLigandMolFiles(tmp_path / "mol_files")


def test_write_ligand_mol_files_aligns_with_unique_ligand_names(tmp_path):
    model_a = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "N1", "LIG", "A", 101, 0.0, 1.0, 0.0, "N")
        + _pdb_line("HETATM", 3, "H1", "LIG", "A", 101, 0.9, 1.0, 0.0, "H")
        + _pdb_line("HETATM", 4, "H2", "LIG", "A", 101, -0.3, 1.8, 0.0, "H")
        + _pdb_line("HETATM", 5, "H3", "LIG", "A", 101, -0.3, 0.2, 0.0, "H")
        + _pdb_line("HETATM", 6, "H4", "LIG", "A", 101, 0.0, 1.0, 0.9, "H")
        + "CONECT    2    3    4    5    6\n"
        + "END\n"
    )
    model_b = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "N1", "LIG", "A", 201, 0.0, 1.0, 0.0, "N")
        + _pdb_line("HETATM", 3, "H1", "LIG", "A", 201, 0.9, 1.0, 0.0, "H")
        + _pdb_line("HETATM", 4, "H2", "LIG", "A", 201, -0.3, 1.8, 0.0, "H")
        + _pdb_line("HETATM", 5, "H3", "LIG", "A", 201, -0.3, 0.2, 0.0, "H")
        + "CONECT    2    3    4    5\n"
        + "END\n"
    )
    models = _write_models(tmp_path, {"modelA": model_a, "modelB": model_b})
    models.makeLigandNamesUnique(target_names="LIG")

    paths = models.writeLigandMolFiles(tmp_path / "mol_files")

    assert set(paths) == {"L01", "LIG"}
    assert Path(paths["LIG"]).exists()
    assert Path(paths["L01"]).exists()


def test_write_ligand_mol_files_target_names_can_force_skipped_cofactor_names(tmp_path):
    pdb_text = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "N1", "NAD", "A", 101, 0.0, 1.0, 0.0, "N")
        + _pdb_line("HETATM", 3, "H1", "NAD", "A", 101, 0.9, 1.0, 0.0, "H")
        + _pdb_line("HETATM", 4, "H2", "NAD", "A", 101, -0.3, 1.8, 0.0, "H")
        + _pdb_line("HETATM", 5, "H3", "NAD", "A", 101, -0.3, 0.2, 0.0, "H")
        + _pdb_line("HETATM", 6, "H4", "NAD", "A", 101, 0.0, 1.0, 0.9, "H")
        + "CONECT    2    3    4    5    6\n"
        + "END\n"
    )
    models = _write_models(tmp_path, {"modelA": pdb_text})

    assert models.writeLigandMolFiles(tmp_path / "default_mols") == {}

    paths = models.writeLigandMolFiles(tmp_path / "targeted_mols", target_names="NAD")

    assert set(paths) == {"NAD"}
    assert Path(paths["NAD"]).exists()
