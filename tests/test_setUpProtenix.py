"""Tests for setUpProtenixCalculation."""
import json
import os
import re

import prepare_proteins
import pytest


def _make_dimer_models():
    """Build a sequenceModels with a receptor + peptide dimer."""
    return prepare_proteins.sequenceModels(
        {"complex1": {"A": "MKLVQRGTALAFP", "B": "PEPTID"}}
    )


def test_protenix_default_seeds_and_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    job_folder = tmp_path / "protenix_run"

    jobs = models.setUpProtenixCalculation(str(job_folder))

    # One command per model
    assert len(jobs) == 1

    # Default seeds 1..20, sample 5 → 100 poses
    cmd = jobs[0]
    assert "--seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" in cmd
    assert "--sample 5" in cmd
    assert "--cycle 4" in cmd
    assert "--step 200" in cmd
    assert "--use_msa False" in cmd
    assert "--use_default_params True" in cmd

    # JSON written
    json_path = job_folder / "inputs" / "complex1.json"
    assert json_path.is_file()
    spec = json.loads(json_path.read_text())
    assert isinstance(spec, list) and len(spec) == 1
    sequences = spec[0]["sequences"]
    # Two protein chains (multi-chain default)
    assert len(sequences) == 2
    chain_seqs = {entry["proteinChain"]["sequence"] for entry in sequences}
    assert chain_seqs == {"MKLVQRGTALAFP", "PEPTID"}
    assert spec[0]["name"] == "complex1"


def test_protenix_skip_finished_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    job_folder = tmp_path / "protenix_run"

    jobs = models.setUpProtenixCalculation(str(job_folder))
    # Default skip_finished=True → command wrapped in [ -f <last_cif> ] || ( ... )
    assert jobs[0].startswith("[ -f ")
    last_cif_match = re.search(
        r"complex1/seed_20/predictions/complex1_seed_20_sample_4\.cif", jobs[0]
    )
    assert last_cif_match is not None


def test_protenix_with_msa_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    job_folder = tmp_path / "protenix_run"
    msa_root = tmp_path / "msas"
    msa_root.mkdir()

    jobs = models.setUpProtenixCalculation(
        str(job_folder), use_msa=True, msa_dir=str(msa_root)
    )
    assert "--use_msa True" in jobs[0]
    spec = json.loads((job_folder / "inputs" / "complex1.json").read_text())
    for entry in spec[0]["sequences"]:
        msa = entry["proteinChain"].get("msa", {})
        assert "precomputed_msa_dir" in msa
        # Per-chain MSA directory under msa_dir/<model>/<chain>
        assert msa["precomputed_msa_dir"].startswith(str(msa_root))


def test_protenix_use_msa_without_dir_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    with pytest.raises(ValueError, match="msa_dir"):
        models.setUpProtenixCalculation(
            str(tmp_path / "x"), use_msa=True
        )


def test_protenix_custom_seeds_and_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    jobs = models.setUpProtenixCalculation(
        str(tmp_path / "protenix_run"),
        seeds=[42, 43, 44],
        sample=10,
    )
    assert "--seeds 42,43,44" in jobs[0]
    assert "--sample 10" in jobs[0]
    # last_cif now references seed_44_sample_9
    assert "seed_44/predictions/complex1_seed_44_sample_9.cif" in jobs[0]
