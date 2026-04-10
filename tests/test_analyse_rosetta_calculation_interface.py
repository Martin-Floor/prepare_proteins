from pathlib import Path

import pandas as pd
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


def _prepare_rosetta_folder(rosetta_folder: Path, model_name: str = "modelA") -> Path:
    output_dir = rosetta_folder / "output_models" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    silent_file = output_dir / f"{model_name}_relax.out"
    silent_file.write_text("")
    return silent_file


def _write_scores_csv(score_path: Path, model_name: str = "modelA") -> None:
    score_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "score": [-10.0, -9.5],
            "total_score": [-10.0, -9.5],
            "description": [f"{model_name}_0001", f"{model_name}_0002"],
            "Model": [model_name, model_name],
            "Pose": [1, 2],
        }
    ).to_csv(score_path, index=False)


def _write_complete_interface_csv(interface_path: Path, suffix: str = "B") -> None:
    interface_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Model": ["modelA", "modelA"],
            "Pose": [1, 2],
            f"interface_score_{suffix}": [-5.5, -4.5],
            f"interface_dG_{suffix}": [-5.5, -4.5],
            f"interface_delta_sasa_{suffix}": [420.0, 380.0],
            f"interface_packstat_{suffix}": [0.61, 0.58],
            f"interface_delta_hbond_unsat_{suffix}": [1, 2],
            f"interface_nres_{suffix}": [14, 12],
            f"interface_hbonds_{suffix}": [6, 5],
            f"interface_sc_{suffix}": [0.72, 0.68],
            f"interface_dG_dSASA_ratio_{suffix}": [-0.0131, -0.0118],
            f"interface_total_hbond_E_{suffix}": [-2.4, -2.0],
        }
    ).to_csv(interface_path, index=False)


def test_analyse_rosetta_calculation_return_jobs_accepts_interface_chains(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir))

    rosetta_folder = tmp_path / "rosetta"
    _prepare_rosetta_folder(rosetta_folder)

    jobs = models.analyseRosettaCalculation(
        str(rosetta_folder),
        return_jobs=True,
        interface_chains=["B"],
    )

    assert len(jobs) == 1
    assert "--interface_metrics B" in jobs[0]

    legacy_jobs = models.analyseRosettaCalculation(
        str(rosetta_folder),
        return_jobs=True,
        binding_energy="B",
    )

    assert len(legacy_jobs) == 1
    assert "--interface_metrics B" in legacy_jobs[0]


def test_analyse_rosetta_calculation_skip_finished_checks_interface_metrics(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir))

    rosetta_folder = tmp_path / "rosetta"
    _prepare_rosetta_folder(rosetta_folder)

    scores_csv = rosetta_folder / ".analysis" / "scores" / "modelA.csv"
    _write_scores_csv(scores_csv)

    jobs = models.analyseRosettaCalculation(
        str(rosetta_folder),
        return_jobs=True,
        interface_chains="B",
        skip_finished=True,
    )
    assert len(jobs) == 1

    incomplete_interface_csv = rosetta_folder / ".analysis" / "binding_energy" / "modelA.csv"
    incomplete_interface_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Model": ["modelA"],
            "Pose": [1],
            "interface_score_B": [-5.5],
        }
    ).to_csv(incomplete_interface_csv, index=False)

    jobs = models.analyseRosettaCalculation(
        str(rosetta_folder),
        return_jobs=True,
        interface_chains="B",
        skip_finished=True,
    )
    assert len(jobs) == 1

    _write_complete_interface_csv(incomplete_interface_csv)

    jobs = models.analyseRosettaCalculation(
        str(rosetta_folder),
        return_jobs=True,
        interface_chains="B",
        skip_finished=True,
    )
    assert jobs == []


def test_analyse_rosetta_calculation_merges_interface_metrics_into_rosetta_data(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    models = prepare_proteins.proteinModels(str(models_dir))

    rosetta_folder = tmp_path / "rosetta"
    _prepare_rosetta_folder(rosetta_folder)

    scores_csv = rosetta_folder / ".analysis" / "scores" / "modelA.csv"
    interface_csv = rosetta_folder / ".analysis" / "binding_energy" / "modelA.csv"
    _write_scores_csv(scores_csv)
    _write_complete_interface_csv(interface_csv)

    rosetta_data = models.analyseRosettaCalculation(
        str(rosetta_folder),
        interface_chains="B",
    )

    assert rosetta_data.index.names == ["Model", "Pose"]
    assert rosetta_data.loc[("modelA", 1), "interface_dG_B"] == pytest.approx(-5.5)
    assert rosetta_data.loc[("modelA", 1), "interface_score_B"] == pytest.approx(-5.5)
    assert rosetta_data.loc[("modelA", 2), "interface_sc_B"] == pytest.approx(0.68)
