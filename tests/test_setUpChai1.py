"""Tests for setUpChai1Calculation."""
import os

import prepare_proteins
import pytest


def _make_dimer_models():
    return prepare_proteins.sequenceModels(
        {"complex1": {"A": "MKLVQRGTALAFP", "B": "PEPTID"}}
    )


def test_chai1_default_seeds_and_fasta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    job_folder = tmp_path / "chai_run"

    jobs = models.setUpChai1Calculation(str(job_folder))

    # Default 5 seeds × 5 samples = 25 poses → 5 commands per model (one per seed)
    assert len(jobs) == 5

    # FASTA written with both chains
    fasta_path = job_folder / "inputs" / "complex1.fasta"
    assert fasta_path.is_file()
    text = fasta_path.read_text()
    assert ">protein|name=A" in text
    assert ">protein|name=B" in text
    assert "MKLVQRGTALAFP" in text
    assert "PEPTID" in text

    # Each command invokes chai-lab fold with seed and num-diffn-samples
    for i, cmd in enumerate(jobs, start=1):
        assert "chai-lab fold" in cmd
        assert f"--seed {i}" in cmd
        assert "--num-diffn-samples 5" in cmd


def test_chai1_skip_finished_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    jobs = models.setUpChai1Calculation(str(tmp_path / "chai_run"))
    # Each command wrapped in [ -f <last_cif> ] || ( ... )
    for cmd in jobs:
        assert cmd.startswith("[ -f ")
        assert "pred.model_idx_4.cif" in cmd  # samples_per_seed=5 → idx 0..4


def test_chai1_with_msa_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()

    jobs = models.setUpChai1Calculation(
        str(tmp_path / "chai_run"), msa_directory=str(msa_dir)
    )
    for cmd in jobs:
        assert f"--msa-directory {msa_dir}" in cmd


def test_chai1_no_msa_directory_omits_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    jobs = models.setUpChai1Calculation(str(tmp_path / "chai_run"))
    for cmd in jobs:
        assert "--msa-directory" not in cmd


def test_chai1_custom_seeds_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    jobs = models.setUpChai1Calculation(
        str(tmp_path / "chai_run"),
        seeds=list(range(1, 21)),
        samples_per_seed=5,
    )
    # 20 seeds × 5 samples = 100 poses → 20 commands per model
    assert len(jobs) == 20


def test_chai1_empty_seeds_errors(tmp_path):
    models = _make_dimer_models()
    with pytest.raises(ValueError, match="seeds"):
        models.setUpChai1Calculation(str(tmp_path / "x"), seeds=[])
