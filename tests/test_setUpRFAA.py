"""Tests for setUpRFAACalculation."""
import os

import prepare_proteins
import pytest


def _make_dimer_models():
    return prepare_proteins.sequenceModels(
        {"complex1": {"A": "MKLVQRGTALAFP", "B": "PEPTID"}}
    )


def _make_msa_dir(tmp_path):
    msa_root = tmp_path / "msas"
    msa_root.mkdir()
    for cid, seq in (("A", "MKLVQRGTALAFP"), ("B", "PEPTID")):
        (msa_root / f"complex1_{cid}.a3m").write_text(f">query\n{seq}\n")
    return msa_root


def test_rfaa_requires_msa(tmp_path, monkeypatch):
    """Without precomputed_msa_dir the function must refuse to set up."""
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    with pytest.raises(ValueError, match="precomputed_msa_dir"):
        models.setUpRFAACalculation(str(tmp_path / "rfaa_run"))


def test_rfaa_yaml_and_fastas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    msa_root = _make_msa_dir(tmp_path)
    job_folder = tmp_path / "rfaa_run"

    jobs = models.setUpRFAACalculation(
        str(job_folder), precomputed_msa_dir=str(msa_root)
    )

    # One command per model
    assert len(jobs) == 1

    # Per-chain FASTAs
    for cid in ("A", "B"):
        fasta = job_folder / "complex1" / "input" / f"complex1_{cid}.fasta"
        assert fasta.is_file()
        text = fasta.read_text()
        assert text.startswith(f">{cid}")

    # Hydra YAML config
    cfg = job_folder / "complex1" / "complex1.yaml"
    assert cfg.is_file()
    text = cfg.read_text()
    assert "defaults:" in text
    assert "- base" in text
    assert "job_name: complex1" in text
    assert "protein_inputs:" in text
    assert "  A:" in text
    assert "  B:" in text
    assert "fasta_file: input/complex1_A.fasta" in text
    assert "fasta_file: input/complex1_B.fasta" in text


def test_rfaa_command_runs_hhsearch_then_inference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    msa_root = _make_msa_dir(tmp_path)
    jobs = models.setUpRFAACalculation(
        str(tmp_path / "rfaa_run"), precomputed_msa_dir=str(msa_root)
    )
    cmd = jobs[0]
    # PYTHONPATH must include the RFAA repo root
    assert "export PYTHONPATH=" in cmd
    assert "RoseTTAFold-All-Atom" in cmd
    # `set -e` so any step's failure aborts the pipeline
    assert "set -e" in cmd
    # hhsearch invoked per chain (A and B) BEFORE rf2aa.run_inference
    assert cmd.count("hhsearch") >= 2
    hhsearch_pos = cmd.find("hhsearch")
    rf_pos = cmd.find("rf2aa.run_inference")
    assert 0 <= hhsearch_pos < rf_pos, "hhsearch must run before inference"
    # hhsearch outputs t000_.hhr and t000_.atab
    assert "t000_.hhr" in cmd
    assert "t000_.atab" in cmd
    # hhsearch is guarded by if [ ! -s ... ]; then ... ; fi
    assert "if [ ! -s " in cmd
    assert "; then " in cmd
    assert "; fi" in cmd
    # --config-name = job_name
    assert "--config-name complex1" in cmd


def test_rfaa_skip_finished_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    msa_root = _make_msa_dir(tmp_path)
    jobs = models.setUpRFAACalculation(
        str(tmp_path / "rfaa_run"), precomputed_msa_dir=str(msa_root)
    )
    assert jobs[0].startswith("[ -f ")
    # RFAA writes <mdir>/<model>.pdb — sentinel must reference that, not best_model.pdb.
    assert "complex1.pdb" in jobs[0]
    assert "best_model.pdb" not in jobs[0]


def test_rfaa_custom_paths_in_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    msa_root = _make_msa_dir(tmp_path)
    weights = "/some/path/weights.pt"
    hhdb = "/some/path/pdb100/pdb100"
    repo = "/some/path/RFAA"
    bin_ = "/some/path/hhsuite/bin"
    jobs = models.setUpRFAACalculation(
        str(tmp_path / "rfaa_run"),
        weights_path=weights,
        hhdb_path=hhdb,
        rfaa_repo=repo,
        hhsuite_bin=bin_,
        precomputed_msa_dir=str(msa_root),
    )
    cfg_text = (tmp_path / "rfaa_run" / "complex1" / "complex1.yaml").read_text()
    assert f"hhdb: \"{hhdb}\"" in cfg_text
    assert f"checkpoint_path: {weights}" in cfg_text
    assert repo in jobs[0]
    # hhsearch binary is invoked from the custom hhsuite_bin
    assert f"{bin_}/hhsearch" in jobs[0]
    # hhsearch -d points at the custom hhdb
    assert f"-d {hhdb}" in jobs[0]


def test_rfaa_precomputed_msa_symlinks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    msa_root = _make_msa_dir(tmp_path)

    job_folder = tmp_path / "rfaa_run"
    models.setUpRFAACalculation(
        str(job_folder), precomputed_msa_dir=str(msa_root)
    )

    for cid in ("A", "B"):
        cache_a = job_folder / "complex1" / "complex1" / cid / "t000_.msa0.a3m"
        assert cache_a.is_symlink() or cache_a.is_file()


def test_rfaa_defaults_point_at_shared_paths(tmp_path, monkeypatch):
    """Library defaults must point at the bsc72 shared locations."""
    monkeypatch.chdir(tmp_path)
    models = _make_dimer_models()
    msa_root = _make_msa_dir(tmp_path)
    jobs = models.setUpRFAACalculation(
        str(tmp_path / "rfaa_run"), precomputed_msa_dir=str(msa_root)
    )
    cfg_text = (tmp_path / "rfaa_run" / "complex1" / "complex1.yaml").read_text()
    assert "/gpfs/projects/bsc72/databases/pdb100_2021Mar03/pdb100_2021Mar03" in cfg_text
    assert "/gpfs/projects/bsc72/weights/rfaa/RFAA_paper_weights.pt" in cfg_text
    assert "/gpfs/projects/bsc72/Programs/RoseTTAFold-All-Atom" in jobs[0]
    assert "/gpfs/projects/bsc72/Programs/hhsuite-3.3.0/bin/hhsearch" in jobs[0]
