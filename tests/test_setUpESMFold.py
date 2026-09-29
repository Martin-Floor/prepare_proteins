"""Tests for setUpESMFoldCalculation."""
import os
import json

import prepare_proteins


def _make_dimer_models():
    return prepare_proteins.sequenceModels(
        {"complex1": {"A": "MKLVQRGTALAFP", "B": "PEPTID"}}
    )


def _make_monomer_models():
    return prepare_proteins.sequenceModels({"monomer1": "MKLVQRGTALAFP"})


def test_esmfold_writes_helper(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    job_folder = tmp_path / "esm_run"
    models.setUpESMFoldCalculation(str(job_folder))
    helper = job_folder / "_esmfold_run.py"
    assert helper.is_file()
    src = helper.read_text()
    assert "EsmForProteinFolding" in src
    assert "atom14_to_atom37" in src
    assert "_split_pdb_into_chains" in src
    # Defaults baked into the helper
    assert "facebook" not in src  # we should use local path, not HF hub id
    assert "/gpfs/projects/bsc72/Programs/models/esmfold_v1" in src
    assert "LINKER_LEN = 25" in src


def test_esmfold_dimer_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    job_folder = tmp_path / "esm_run"
    jobs = models.setUpESMFoldCalculation(str(job_folder))
    assert len(jobs) == 1
    cmd = jobs[0]
    # Calls the helper with --seqs JSON containing both chains
    assert "python " in cmd
    assert "_esmfold_run.py" in cmd
    assert "--seqs" in cmd
    assert "MKLVQRGTALAFP" in cmd
    assert "PEPTID" in cmd
    assert "--out-pdb" in cmd and "complex1.pdb" in cmd
    assert "--out-plddt" in cmd and "complex1_plddt.txt" in cmd
    # Sentinel guard
    assert cmd.startswith("[ -f ")


def test_esmfold_monomer_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_monomer_models()
    job_folder = tmp_path / "esm_run"
    jobs = models.setUpESMFoldCalculation(str(job_folder))
    cmd = jobs[0]
    # Monomer: single-chain JSON list
    assert '["MKLVQRGTALAFP"]' in cmd


def test_esmfold_custom_model_path_propagates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    custom = "/some/local/path/esmfold"
    models.setUpESMFoldCalculation(str(tmp_path / "esm_run"), model_path=custom)
    helper = (tmp_path / "esm_run" / "_esmfold_run.py").read_text()
    assert custom in helper


def test_esmfold_linker_length_propagates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    models.setUpESMFoldCalculation(str(tmp_path / "esm_run"), linker_length=40)
    helper = (tmp_path / "esm_run" / "_esmfold_run.py").read_text()
    assert "LINKER_LEN = 40" in helper


def test_esmfold_skip_finished_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    jobs = models.setUpESMFoldCalculation(
        str(tmp_path / "esm_run"), skip_finished=False
    )
    cmd = jobs[0]
    assert not cmd.startswith("[ -f ")
