import json
import os
import shlex

import prepare_proteins
import pytest


def test_set_up_alphafold3_unpacked_seeds_use_relative_cd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "af3 jobs"

    jobs = models.setUpAlphaFold3(
        str(job_folder),
        model_seeds=[1, 2],
        unpack_model_seeds=True,
    )

    expected_model_dir = os.path.relpath(job_folder / "modelA", start=tmp_path)
    expected_lines = [
        'AF3_START_DIR="$(pwd)"',
        f"cd {shlex.quote(expected_model_dir)}",
        "bsc_alphafold seed_1/input seed_1/output $WEIGHTS",
        "sbatch runner seed_1/input",
        'cd "$AF3_START_DIR"',
    ]

    assert jobs[0].splitlines() == expected_lines
    assert str(job_folder / "modelA") not in jobs[0]


def test_set_up_alphafold3_combined_seeds_use_relative_cd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "af3 jobs"

    jobs = models.setUpAlphaFold3(
        str(job_folder),
        model_seeds=[1, 2],
        unpack_model_seeds=False,
    )

    expected_model_dir = os.path.relpath(job_folder / "modelA", start=tmp_path)
    expected_lines = [
        'AF3_START_DIR="$(pwd)"',
        f"cd {shlex.quote(expected_model_dir)}",
        "bsc_alphafold input output $WEIGHTS",
        "sbatch runner input",
        'cd "$AF3_START_DIR"',
    ]

    assert jobs[0].splitlines() == expected_lines
    assert str(job_folder / "modelA") not in jobs[0]


def test_set_up_alphafold3_writes_glycan_payload_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": {"A": "NNST", "B": "ACDE"}})
    job_folder = tmp_path / "af3 jobs"
    user_ccd_path = tmp_path / "glycan_components.cif"
    user_ccd_path.write_text("data_GLYCAN\n#\n")

    bonded_atom_pairs = [
        [["A", 1, "ND2"], ["G", 1, "C1"]],
        [["G", 1, "O4"], ["G", 2, "C1"]],
    ]

    models.setUpAlphaFold3(
        str(job_folder),
        model_seeds=[7],
        ligands={"modelA": [{"id": "G", "ccdCodes": ["NAG", "NAG"]}]},
        bonded_atom_pairs={"modelA": bonded_atom_pairs},
        user_ccd_path={"modelA": user_ccd_path},
    )

    payload_path = job_folder / "modelA" / "input" / "modelA.json"
    payload = json.loads(payload_path.read_text())

    assert payload["bondedAtomPairs"] == bonded_atom_pairs
    assert payload["userCCDPath"] == os.fspath(user_ccd_path)
    assert payload["sequences"][-1] == {"ligand": {"id": "G", "ccdCodes": ["NAG", "NAG"]}}


def test_set_up_alphafold3_rejects_user_ccd_and_user_ccd_path_for_same_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "af3 jobs"

    with pytest.raises(ValueError, match="cannot define both user_ccd and user_ccd_path"):
        models.setUpAlphaFold3(
            str(job_folder),
            user_ccd={"modelA": "data_CUSTOM\n#\n"},
            user_ccd_path={"modelA": tmp_path / "custom_components.cif"},
        )
