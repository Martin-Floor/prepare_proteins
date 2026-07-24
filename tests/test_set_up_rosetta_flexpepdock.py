import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

import prepare_proteins


MODEL_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
ATOM      5  N   GLY A   2      14.220  11.690   3.050  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.830  10.370   3.330  1.00 20.00           C
ATOM      7  C   GLY A   2      15.412   9.850   2.010  1.00 20.00           C
ATOM      8  O   GLY A   2      15.962   8.743   1.970  1.00 20.00           O
TER
ATOM      9  N   GLY B   1      15.500  12.400   4.400  1.00 20.00           N
ATOM     10  CA  GLY B   1      16.040  11.110   4.880  1.00 20.00           C
ATOM     11  C   GLY B   1      15.220  10.050   4.140  1.00 20.00           C
ATOM     12  O   GLY B   1      15.620   8.890   4.030  1.00 20.00           O
TER
END
"""


def _write_model(models_dir: Path, model_name: str = "modelA") -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.pdb"
    model_path.write_text(MODEL_PDB)
    return model_path


def _write_scorefile(score_path: Path, model_name: str, nstruct: int) -> None:
    lines = ["SCORE: total_score description"]
    for pose in range(1, nstruct + 1):
        lines.append(f"SCORE: {-1.0 * pose:.1f} {model_name}_{pose:04d}")
    score_path.write_text("\n".join(lines) + "\n")


def test_set_up_rosetta_flexpepdock_writes_expected_xml_and_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir))

    job_folder = tmp_path / "flexpepdock jobs"
    jobs = models.setUpRosettaFlexPepDock(
        str(job_folder),
        peptide_chain="B",
        parallelisation=None,
        nstruct=3,
    )

    assert len(jobs) == 1
    expected_model_dir = os.path.relpath(
        job_folder / "output_models" / "modelA", start=tmp_path
    )
    assert f"cd {shlex.quote(expected_model_dir)}" in jobs[0]
    assert str(job_folder / "output_models" / "modelA") not in jobs[0]
    assert "PREPARE_PROTEINS_LAUNCH_DIR" not in jobs[0]
    assert "cd ../../.." in jobs[0]
    assert "rosetta_scripts.mpi.linuxgccrelease @ ../../flags/modelA_flexpepdock.flags" in jobs[0]

    xml_path = job_folder / "xml" / "modelA_flexpepdock.xml"
    xml_text = xml_path.read_text()
    assert "<FlexPepDock" in xml_text
    assert 'pep_refine="true"' in xml_text

    flags_path = job_folder / "flags" / "modelA_flexpepdock.flags"
    flags_text = flags_path.read_text()
    assert "-parser:protocol ../../xml/modelA_flexpepdock.xml" in flags_text
    assert "-s ../../input_models/modelA.pdb" in flags_text
    assert "-out:file:silent modelA_flexpepdock.out" in flags_text
    assert "-out:file:scorefile modelA_flexpepdock.sc" in flags_text
    assert "-flexPepDocking:peptide_chain B" in flags_text
    assert "-flexPepDocking:receptor_chain A" in flags_text
    assert "-score:weights ref2015" in flags_text
    assert "-use_input_sc" in flags_text
    assert "-ex1aro" in flags_text
    assert "-ex2aro" in flags_text


def test_set_up_rosetta_flexpepdock_skips_finished_models(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir))

    job_folder = tmp_path / "flexpepdock jobs"
    output_dir = job_folder / "output_models" / "modelA"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_scorefile(output_dir / "modelA_flexpepdock.sc", "modelA", nstruct=3)

    jobs = models.setUpRosettaFlexPepDock(
        str(job_folder),
        peptide_chain="B",
        parallelisation=None,
        nstruct=3,
        skip_finished=True,
    )

    assert jobs == []


def test_set_up_rosetta_flexpepdock_can_use_default_chain_assignment(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir))

    job_folder = tmp_path / "flexpepdock jobs"
    models.setUpRosettaFlexPepDock(
        str(job_folder),
        peptide_chain=None,
        receptor_chain=None,
        parallelisation=None,
        nstruct=1,
    )

    xml_path = job_folder / "xml" / "modelA_flexpepdock.xml"
    xml_text = xml_path.read_text()
    flags_path = job_folder / "flags" / "modelA_flexpepdock.flags"
    flags_text = flags_path.read_text()
    assert 'receptor_chain="' not in xml_text
    assert 'peptide_chain="' not in xml_text
    assert "-flexPepDocking:receptor_chain A" in flags_text
    assert "-flexPepDocking:peptide_chain B" in flags_text


def test_set_up_rosetta_flexpepdock_rejects_multiple_doc_modes(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir))

    with pytest.raises(ValueError):
        models.setUpRosettaFlexPepDock(
            str(tmp_path / "flexpepdock jobs"),
            peptide_chain="B",
            pep_refine=True,
            extra_scoring=True,
            parallelisation=None,
        )


def test_analyse_rosetta_calculation_detects_non_relax_silent_files_for_job_generation(
    tmp_path,
):
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir))

    rosetta_folder = tmp_path / "rosetta"
    output_dir = rosetta_folder / "output_models" / "modelA"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "modelA_flexpepdock.out").write_text("")

    jobs = models.analyseRosettaCalculation(
        str(rosetta_folder),
        return_jobs=True,
    )

    assert len(jobs) == 1
    assert "--models modelA" in jobs[0]


def test_set_up_rosetta_flexpepdock_adds_nma_params_for_capped_peptides(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir), conect_update=False)
    models.addCappingGroups(style="rosetta", backend="internal", chains="B")

    job_folder = tmp_path / "flexpepdock jobs"
    models.setUpRosettaFlexPepDock(
        str(job_folder),
        peptide_chain="B",
        parallelisation=None,
        nstruct=1,
    )

    params_path = job_folder / "params" / "NMA.params"
    assert params_path.exists()

    flags_text = (job_folder / "flags" / "modelA_flexpepdock.flags").read_text()
    assert "-in:file:extra_res_path ../../params" in flags_text


def test_set_up_rosetta_flexpepdock_worker_script_runs_from_launcher_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir))

    job_folder = tmp_path / "flexpepdock jobs"
    jobs = models.setUpRosettaFlexPepDock(
        str(job_folder),
        peptide_chain="B",
        parallelisation=None,
        nstruct=1,
        executable='python3 -c "import os, pathlib; pathlib.Path(\'marker.txt\').write_text(os.getcwd())"',
    )

    script_path = tmp_path / "run_job.sh"
    script_path.write_text("#!/bin/sh\n" + jobs[0])
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    subprocess.run(["bash", str(script_path)], cwd=tmp_path, check=True)

    marker_path = job_folder / "output_models" / "modelA" / "marker.txt"
    assert marker_path.exists()
    assert marker_path.read_text() == str(job_folder / "output_models" / "modelA")
