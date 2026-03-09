import os
import shlex

import prepare_proteins


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
