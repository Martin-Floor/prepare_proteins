"""Tests for sequenceModels.setUpESMFold2."""
import json
import os
import shlex

import prepare_proteins
import pytest


def test_setup_esmfold2_creates_per_model_dirs_and_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "esm_jobs"

    jobs = models.setUpESMFold2(str(job_folder))

    assert len(jobs) == 1
    assert (job_folder / "modelA").is_dir()
    assert (job_folder / "modelA" / "output").is_dir()
    assert (job_folder / "modelA" / "fold.py").is_file()


def test_setup_esmfold2_command_uses_relative_cd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "esm jobs"

    jobs = models.setUpESMFold2(str(job_folder), esmfold2_env="/path/to/env",
                                hf_cache="/path/to/cache")

    expected_model_dir = os.path.relpath(job_folder / "modelA", start=tmp_path)
    expected_lines = [
        'ESMFOLD2_START_DIR="$(pwd)"',
        f"cd {shlex.quote(expected_model_dir)}",
        f"export HF_HOME={shlex.quote('/path/to/cache')}",
        f"{shlex.quote('/path/to/env/bin/python')} fold.py",
        'cd "$ESMFOLD2_START_DIR"',
    ]
    assert jobs[0].splitlines() == expected_lines


def test_setup_esmfold2_ccd_ligands_in_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "esm"
    models.setUpESMFold2(str(job_folder), ligands=["HEM", "MG"])

    script = (job_folder / "modelA" / "fold.py").read_text()
    assert "'ccd': 'HEM'" in script or '"ccd": "HEM"' in script
    assert "'ccd': 'MG'" in script or '"ccd": "MG"' in script


def test_setup_esmfold2_ccd_ligands_dict_with_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "esm"
    models.setUpESMFold2(str(job_folder), ligands={"HEM": 2, "MG": 1})

    script = (job_folder / "modelA" / "fold.py").read_text()
    # 2 HEMs + 1 MG → 3 ligand entries
    assert script.count("'ccd': 'HEM'") + script.count('"ccd": "HEM"') == 2
    assert script.count("'ccd': 'MG'") + script.count('"ccd": "MG"') == 1


def test_setup_esmfold2_smiles_ligands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "esm"
    models.setUpESMFold2(str(job_folder), ligand_smiles=["CCO"])

    script = (job_folder / "modelA" / "fold.py").read_text()
    assert "'smiles': 'CCO'" in script or '"smiles": "CCO"' in script


def test_setup_esmfold2_msa_dir_autodetected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    (msa_dir / "modelA.a3m").write_text(">q\nACDEFGHIK\n")

    job_folder = tmp_path / "esm"
    models.setUpESMFold2(str(job_folder), msa_dir=str(msa_dir))

    script = (job_folder / "modelA" / "fold.py").read_text()
    # MSA wired as the 3rd element of the (chain_id, seq, msa) tuple in CHAINS
    assert "CHAINS = " in script
    assert "modelA.a3m" in script


def test_setup_esmfold2_msa_disabled_without_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "esm"
    models.setUpESMFold2(str(job_folder))

    script = (job_folder / "modelA" / "fold.py").read_text()
    # single chain, no MSA -> msa element is None
    assert "('A', 'ACDEFGHIK', None)" in script


def test_setup_esmfold2_receptor_peptide_two_chains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels(
        {"cx": {"A": "ACDEFGHIKLMN", "B": "GRGDS"}}
    )
    job_folder = tmp_path / "esm"
    models.setUpESMFold2(str(job_folder))

    script = (job_folder / "cx" / "fold.py").read_text()
    # two protein chains in CHAINS, folded as separate ProteinInputs
    assert "('A', 'ACDEFGHIKLMN'" in script
    assert "('B', 'GRGDS'" in script
    assert "ProteinInput(id=cid, sequence=seq, msa=_load_msa(mp))" in script


def test_setup_esmfold2_peptide_chain_single_seq_when_msa_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels(
        {"cx": {"A": "ACDEFGHIKLMN", "B": "GRGDS"}}
    )
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    (msa_dir / "cx.a3m").write_text(">q\nACDEFGHIKLMN\n")

    models.setUpESMFold2(str(tmp_path / "esm"), msa_dir=str(msa_dir))
    script = (tmp_path / "esm" / "cx" / "fold.py").read_text()
    # receptor (first chain) gets the MSA; peptide stays single-sequence (None)
    assert "cx.a3m" in script
    assert "('B', 'GRGDS', None)" in script


def test_setup_esmfold2_pose_ensemble(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"cx": {"A": "ACDEFGHIKLMN", "B": "GRGDS"}})
    models.setUpESMFold2(
        str(tmp_path / "esm"), num_diffusion_samples=100, samples_per_call=10,
    )
    script = (tmp_path / "esm" / "cx" / "fold.py").read_text()
    assert "NUM_DIFFUSION_SAMPLES = 100" in script
    assert "SAMPLES_PER_CALL = 10" in script
    assert "num_diffusion_samples=k" in script
    assert 'pose_{i:03d}.cif' in script


def test_setup_esmfold2_ensemble_skip_finished_checks_last_pose(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"cx": "ACDEFGHIK"})
    jf = tmp_path / "esm"
    (jf / "cx" / "output").mkdir(parents=True)
    (jf / "cx" / "output" / "pose_009.cif").write_text("# done\n")
    jobs = models.setUpESMFold2(str(jf), num_diffusion_samples=10, skip_finished=True)
    assert jobs == []


def test_setup_esmfold2_skip_finished(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"modelA": "ACDEFGHIK"})
    job_folder = tmp_path / "esm"
    # Simulate prior completion
    (job_folder / "modelA" / "output").mkdir(parents=True)
    (job_folder / "modelA" / "output" / "modelA.cif").write_text("# already done\n")

    jobs = models.setUpESMFold2(str(job_folder), skip_finished=True)
    assert jobs == []


def test_setup_esmfold2_only_models_and_exclude(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models = prepare_proteins.sequenceModels({"a": "ACDEFG", "b": "GHIKLM", "c": "NPQRST"})
    job_folder = tmp_path / "esm"

    jobs_only = models.setUpESMFold2(str(job_folder / "only"), only_models="b")
    assert len(jobs_only) == 1
    assert (job_folder / "only" / "b").is_dir()
    assert not (job_folder / "only" / "a").is_dir()

    jobs_excl = models.setUpESMFold2(str(job_folder / "excl"), exclude_models=["b"])
    assert len(jobs_excl) == 2
