from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import prepare_proteins
import prepare_proteins._protein_models as protein_models_module


def _pdb_line(serial, atom_name, resname, chain, residue_id, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}{residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}\n"
    )


def _write_model(models_dir: Path, model_name: str, pdb_text: str) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.pdb"
    model_path.write_text(pdb_text)
    return model_path


def _load_models(tmp_path, pdb_text: str, model_name: str = "modelA"):
    models_dir = tmp_path / "models"
    _write_model(models_dir, model_name, pdb_text)
    return prepare_proteins.proteinModels(str(models_dir), conect_update=False)


def _fake_propka_runner_factory(pka_lines):
    def _fake_propka_runner(command, cwd=None, capture_output=None, text=None):
        pdb_name = command[-1]
        stem = Path(pdb_name).stem
        pka_path = Path(cwd) / f"{stem}.pka"
        pka_path.write_text(
            "HEADER\n"
            "SUMMARY OF THIS PREDICTION\n"
            "Group pKa model-pKa\n"
            + "\n".join(pka_lines)
            + "\nWriting file\n"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return _fake_propka_runner


def test_propka_auto_histidine_tautomer_prefers_hie(tmp_path, monkeypatch):
    pdb_text = (
        _pdb_line(1, "ND1", "HIS", "A", 1, 0.000, 0.000, 0.000, "N")
        + _pdb_line(2, "NE2", "HIS", "A", 1, 4.000, 0.000, 0.000, "N")
        + _pdb_line(3, "N", "ALA", "A", 2, -2.700, 0.000, 0.000, "N")
        + _pdb_line(4, "O", "ALA", "A", 3, 6.700, 0.000, 0.000, "O")
        + "TER\nEND\n"
    )
    models = _load_models(tmp_path, pdb_text)

    monkeypatch.setattr(
        protein_models_module.subprocess,
        "run",
        _fake_propka_runner_factory(["HIS 1 A 5.00 5.00"]),
    )

    pka_df, residue_names = models.getModelsPropkaProtonationStates(
        pH=7.0,
        return_mode="both",
        propka_executable=["fake-propka"],
    )

    histidine = pka_df.loc[("modelA", "A", 1)]
    assert histidine["predicted_state"] == "HIE"
    assert histidine["histidine_tautomer"] == "HIE"
    assert bool(histidine["histidine_tautomer_ambiguous"]) is False
    assert residue_names["modelA"][("A", 1)] == "HIE"


def test_propka_auto_histidine_tautomer_marks_ambiguous_and_falls_back(tmp_path, monkeypatch):
    pdb_text = (
        _pdb_line(1, "ND1", "HIS", "A", 1, 0.000, 0.000, 0.000, "N")
        + _pdb_line(2, "NE2", "HIS", "A", 1, 4.000, 0.000, 0.000, "N")
        + "TER\nEND\n"
    )
    models = _load_models(tmp_path, pdb_text)

    monkeypatch.setattr(
        protein_models_module.subprocess,
        "run",
        _fake_propka_runner_factory(["HIS 1 A 5.00 5.00"]),
    )

    pka_df, residue_names = models.getModelsPropkaProtonationStates(
        pH=7.0,
        return_mode="both",
        propka_executable=["fake-propka"],
        his_neutral_name="HID",
    )

    histidine = pka_df.loc[("modelA", "A", 1)]
    assert pd.isna(histidine["histidine_tautomer"])
    assert bool(histidine["histidine_tautomer_ambiguous"]) is True
    assert bool(histidine["ambiguous"]) is True
    assert residue_names["modelA"][("A", 1)] == "HID"
