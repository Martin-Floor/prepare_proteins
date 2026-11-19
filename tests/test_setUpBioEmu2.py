import json
import subprocess

import pytest

from prepare_proteins import sequenceModels


def test_bioemu2_equilibrium_launcher(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    executable = tmp_path / "bioemu2 env" / "bioemu2"
    executable.parent.mkdir()
    executable.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$BIOEMU2_MSA_CACHE" "$@" > received_args\n'
    )
    executable.chmod(0o755)
    checkpoint = tmp_path / "local weights" / "checkpoint.ckpt"
    cache = tmp_path / "msa cache"
    models = sequenceModels({"control model": "GYDPETGTWG", "other": "ACDE"})

    jobs = models.setUpBioEmu2(
        "job root",
        models="control model",
        num_samples=4,
        batch_size=2,
        seed=0,
        num_steps=5,
        bioemu2_executable=executable,
        checkpoint_path=checkpoint,
        msa_cache_dir=cache,
    )

    assert len(jobs) == 1
    subprocess.run(["bash", "-c", jobs[0]], check=True)
    model_folder = tmp_path / "job root" / "control model"
    assert json.loads((model_folder / "system.yaml").read_text()) == {
        "entities": [
            {"entity_type": "protein", "asym_ids": ["A"], "sequence": "GYDPETGTWG"}
        ],
        "bonds": [],
    }
    assert (model_folder / "received_args").read_text().splitlines() == [
        str(cache), "sample-equilibrium", "--system", "system.yaml",
        "--out", "output", "--num-samples", "4", "--batch-size", "2",
        "--seed", "0", "--num-steps", "5", "--checkpoint-path", str(checkpoint),
    ]
    assert not (tmp_path / "job root" / "other").exists()


def test_bioemu2_structure_reuses_only_matching_job_inputs(tmp_path):
    models = sequenceModels({"complex": {"A": "acde", "B2": "FGHI"}})
    jobs = models.setUpBioEmu2(tmp_path, mode="structure")
    model_folder = tmp_path / "complex"
    system = json.loads((model_folder / "system.yaml").read_text())
    assert system["entities"] == [
        {"entity_type": "protein", "asym_ids": ["A"], "sequence": "ACDE"},
        {"entity_type": "protein", "asym_ids": ["B2"], "sequence": "FGHI"},
    ]
    assert "bioemu2 predict-structure " in (model_folder / "run.sh").read_text()
    assert models.setUpBioEmu2(tmp_path, mode="structure") == jobs
    with pytest.raises(FileExistsError, match="run.sh"):
        models.setUpBioEmu2(tmp_path, num_samples=16, mode="structure")
    models.sequences["complex"]["A"] = "AAAA"
    with pytest.raises(FileExistsError, match="system.yaml"):
        models.setUpBioEmu2(tmp_path, mode="structure")
    assert json.loads((model_folder / "system.yaml").read_text()) == system


@pytest.mark.parametrize(
    "options, message",
    [
        ({"num_samples": 0}, "num_samples"),
        ({"batch_size": True}, "batch_size"),
        ({"num_steps": 0}, "num_steps"),
        ({"seed": 2.5}, "seed"),
        ({"mode": "propagate"}, "mode"),
    ],
)
def test_bioemu2_rejects_invalid_settings(tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        sequenceModels({"control": "ACDE"}).setUpBioEmu2(tmp_path, **options)
    assert not list(tmp_path.iterdir())


def test_bioemu2_cache_dir_sets_every_upstream_cache(tmp_path, monkeypatch):
    """All four upstream caches must be redirected, not just the MSA one.

    Each defaults beneath ~/.cache/bioemu2 on the execution host, which on a cluster writes into a
    quota-limited home directory.
    """
    monkeypatch.chdir(tmp_path)
    executable = tmp_path / "bioemu2"
    executable.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$BIOEMU2_MSA_CACHE" "$BIOEMU2_EMBEDS_CACHE" '
        '"$BIOEMU2_CCD_CACHE" "$BIOEMU2_TEMPLATE_CACHE" > received_env\n'
    )
    executable.chmod(0o755)
    models = sequenceModels({"m": "GYDPETGTWG"})

    jobs = models.setUpBioEmu2(
        "job root", num_samples=2, bioemu2_executable=executable, cache_dir=tmp_path / "caches"
    )
    subprocess.run(["bash", "-c", jobs[0]], check=True)

    received = (tmp_path / "job root" / "m" / "received_env").read_text().split("\n")
    root = str(tmp_path / "caches")
    assert received[0] == f"{root}/colabfold_msa"
    assert received[1] == f"{root}/embeds"
    assert received[2] == f"{root}/ccd"
    assert received[3] == f"{root}/templates"


def test_bioemu2_msa_cache_dir_overrides_cache_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    executable = tmp_path / "bioemu2"
    executable.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$BIOEMU2_MSA_CACHE" "$BIOEMU2_CCD_CACHE" > received_env\n'
    )
    executable.chmod(0o755)
    models = sequenceModels({"m": "GYDPETGTWG"})

    jobs = models.setUpBioEmu2(
        "job root",
        num_samples=2,
        bioemu2_executable=executable,
        cache_dir=tmp_path / "caches",
        msa_cache_dir=tmp_path / "explicit msa",
    )
    subprocess.run(["bash", "-c", jobs[0]], check=True)

    received = (tmp_path / "job root" / "m" / "received_env").read_text().split("\n")
    assert received[0] == str(tmp_path / "explicit msa")
    assert received[1] == str(tmp_path / "caches" / "ccd")


def test_bioemu2_sampler_and_physicality_filter_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    executable = tmp_path / "bioemu2"
    executable.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > received_args\n')
    executable.chmod(0o755)
    models = sequenceModels({"m": "GYDPETGTWG"})

    jobs = models.setUpBioEmu2(
        "job root",
        num_samples=2,
        bioemu2_executable=executable,
        sampler="edm",
        physicality_filter="high_quality",
    )
    subprocess.run(["bash", "-c", jobs[0]], check=True)

    received = (tmp_path / "job root" / "m" / "received_args").read_text().split("\n")
    assert "--sampler" in received and received[received.index("--sampler") + 1] == "edm"
    assert "--physicality-filter" in received
    assert received[received.index("--physicality-filter") + 1] == "high_quality"
    # the launcher is the record of effective settings
    launcher = (tmp_path / "job root" / "m" / "run.sh").read_text()
    assert "--sampler edm" in launcher and "--physicality-filter high_quality" in launcher


@pytest.mark.parametrize("bad", ["", 3, True])
def test_bioemu2_rejects_invalid_sampler(tmp_path, bad):
    models = sequenceModels({"m": "GYDPETGTWG"})
    with pytest.raises(ValueError):
        models.setUpBioEmu2(tmp_path / "job", num_samples=2, sampler=bad)
