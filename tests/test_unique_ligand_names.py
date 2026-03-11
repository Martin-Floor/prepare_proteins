from pathlib import Path

import prepare_proteins


def _pdb_line(record, serial, atom_name, resname, chain, residue_id, x, y, z, element):
    return (
        f"{record:<6}{serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}{residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}\n"
    )


def _write_models(tmp_path, model_pdbs):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_name, pdb_text in model_pdbs.items():
        (models_dir / f"{model_name}.pdb").write_text(pdb_text)
    return prepare_proteins.proteinModels(str(models_dir), conect_update=False)


def _residue_names(models, model_name):
    return [
        residue.resname.strip()
        for residue in models.structures[model_name].get_residues()
    ]


def test_make_ligand_names_unique_targets_only_selected_names(tmp_path):
    pdb_text = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "LIG", "A", 101, 1.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 3, "C1", "ABC", "A", 102, 2.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 4, "C1", "LIG", "A", 103, 3.0, 0.0, 0.0, "C")
        + "END\n"
    )
    models = _write_models(tmp_path, {"modelA": pdb_text})

    report = models.makeLigandNamesUnique(target_names="LIG")

    assert _residue_names(models, "modelA") == ["ALA", "LIG", "ABC", "L01"]
    assert report.to_dict("records") == [
        {
            "model": "modelA",
            "chain": "A",
            "resid": 103,
            "icode": "",
            "old_name": "LIG",
            "new_name": "L01",
            "occurrence": 2,
            "reason": "target_name_repeat",
        }
    ]


def test_make_ligand_names_unique_tracks_history_across_models(tmp_path):
    model_a = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "LIG", "A", 101, 1.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 3, "C1", "ABC", "A", 102, 2.0, 0.0, 0.0, "C")
        + "END\n"
    )
    model_b = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 1.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "LIG", "A", 201, 1.0, 1.0, 0.0, "C")
        + _pdb_line("HETATM", 3, "C1", "ABC", "A", 202, 2.0, 1.0, 0.0, "C")
        + _pdb_line("HETATM", 4, "C1", "XYZ", "A", 203, 3.0, 1.0, 0.0, "C")
        + "END\n"
    )
    models = _write_models(tmp_path, {"modelA": model_a, "modelB": model_b})

    report = models.makeLigandNamesUnique()

    assert _residue_names(models, "modelA") == ["ALA", "LIG", "ABC"]
    assert _residue_names(models, "modelB") == ["ALA", "L01", "A01", "XYZ"]
    assert report[["model", "old_name", "new_name"]].to_dict("records") == [
        {"model": "modelB", "old_name": "LIG", "new_name": "L01"},
        {"model": "modelB", "old_name": "ABC", "new_name": "A01"},
    ]


def test_make_ligand_names_unique_skips_default_cofactors_but_targets_can_force_them(tmp_path):
    model_a = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "NAD", "A", 101, 1.0, 0.0, 0.0, "C")
        + "END\n"
    )
    model_b = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 1.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "NAD", "A", 201, 1.0, 1.0, 0.0, "C")
        + "END\n"
    )

    default_models = _write_models(tmp_path / "default_case", {"modelA": model_a, "modelB": model_b})
    default_report = default_models.makeLigandNamesUnique()
    assert default_report.empty
    assert _residue_names(default_models, "modelA") == ["ALA", "NAD"]
    assert _residue_names(default_models, "modelB") == ["ALA", "NAD"]

    targeted_models = _write_models(tmp_path / "targeted_case", {"modelA": model_a, "modelB": model_b})
    targeted_report = targeted_models.makeLigandNamesUnique(target_names="NAD")
    assert _residue_names(targeted_models, "modelA") == ["ALA", "NAD"]
    assert _residue_names(targeted_models, "modelB") == ["ALA", "N01"]
    assert targeted_report.iloc[0]["old_name"] == "NAD"
    assert targeted_report.iloc[0]["new_name"] == "N01"


def test_make_ligand_names_unique_skip_names_override_targets_and_collisions(tmp_path):
    skip_model_a = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "LIG", "A", 101, 1.0, 0.0, 0.0, "C")
        + "END\n"
    )
    skip_model_b = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 1.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "LIG", "A", 201, 1.0, 1.0, 0.0, "C")
        + "END\n"
    )
    skip_models = _write_models(tmp_path / "skip_case", {"modelA": skip_model_a, "modelB": skip_model_b})

    skip_report = skip_models.makeLigandNamesUnique(target_names="LIG", skip_names="LIG")
    assert skip_report.empty
    assert _residue_names(skip_models, "modelA") == ["ALA", "LIG"]
    assert _residue_names(skip_models, "modelB") == ["ALA", "LIG"]

    collision_model_a = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "LIG", "A", 101, 1.0, 0.0, 0.0, "C")
        + _pdb_line("HETATM", 3, "C1", "L01", "A", 102, 2.0, 0.0, 0.0, "C")
        + "END\n"
    )
    collision_model_b = (
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 1.0, 0.0, "C")
        + _pdb_line("HETATM", 2, "C1", "LIG", "A", 201, 1.0, 1.0, 0.0, "C")
        + "END\n"
    )
    collision_models = _write_models(
        tmp_path / "collision_case",
        {"modelA": collision_model_a, "modelB": collision_model_b},
    )

    collision_report = collision_models.makeLigandNamesUnique()
    assert _residue_names(collision_models, "modelB") == ["ALA", "L02"]
    assert collision_report.iloc[0]["new_name"] == "L02"
