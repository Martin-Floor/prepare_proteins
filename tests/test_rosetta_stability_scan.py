import os
from pathlib import Path

import pytest

import prepare_proteins


MODEL_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
ATOM      5  CB  ALA A   1      13.070  14.100   1.000  1.00 20.00           C
ATOM      6  N   GLY A   2      14.220  11.690   3.050  1.00 20.00           N
ATOM      7  CA  GLY A   2      14.830  10.370   3.330  1.00 20.00           C
ATOM      8  C   GLY A   2      15.412   9.850   2.010  1.00 20.00           C
ATOM      9  O   GLY A   2      15.962   8.743   1.970  1.00 20.00           O
TER
END
"""


def _write_model(models_dir: Path, name: str) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{name}.pdb").write_text(MODEL_PDB)


def _write_score_file(path: Path, scores: list) -> None:
    """Write a minimal Rosetta score file with given total_score values."""
    lines = ["SCORE:     total_score description"]
    for i, s in enumerate(scores, 1):
        lines.append(f"SCORE:     {s:.3f} model_{i:04d}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ── setUpRosettaStabilityScan ──────────────────────────────────────────────────

def test_stability_scan_setup_returns_one_job_per_model(tmp_path):
    models_dir = tmp_path / "models"
    for name in ("wt", "design1", "design2"):
        _write_model(models_dir, name)
    models = prepare_proteins.proteinModels(str(models_dir))

    jobs = models.setUpRosettaStabilityScan(
        str(tmp_path / "stability"),
        nstruct=5,
        parallelisation=None,
    )

    assert len(jobs) == 3


def test_stability_scan_setup_uses_cartesian_by_default(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir, "wt")
    models = prepare_proteins.proteinModels(str(models_dir))

    models.setUpRosettaStabilityScan(
        str(tmp_path / "stability"),
        nstruct=1,
        parallelisation=None,
    )

    xml_text = (tmp_path / "stability" / "xml" / "wt_relax.xml").read_text()
    assert "ref2015_cart" in xml_text


def test_stability_scan_setup_skips_finished_models(tmp_path):
    models_dir = tmp_path / "models"
    for name in ("wt", "design1"):
        _write_model(models_dir, name)
    models = prepare_proteins.proteinModels(str(models_dir))

    stability_dir = tmp_path / "stability"
    wt_score = stability_dir / "output_models" / "wt" / "wt_relax.out"
    _write_score_file(wt_score, scores=[-10.0] * 5)

    jobs = models.setUpRosettaStabilityScan(
        str(stability_dir),
        nstruct=5,
        parallelisation=None,
        skip_finished=True,
    )

    model_names_in_jobs = [j for j in jobs if "wt" in j]
    assert not model_names_in_jobs, "wt should have been skipped"
    assert len(jobs) == 1


# ── analyseRosettaStabilityScan ───────────────────────────────────────────────

def _make_models_with_score_files(tmp_path, score_map: dict):
    """Load models and write pre-baked score files. Returns the models object."""
    models_dir = tmp_path / "models"
    stability_dir = tmp_path / "stability"
    for name in score_map:
        _write_model(models_dir, name)
        sf = stability_dir / "output_models" / name / f"{name}_relax.out"
        _write_score_file(sf, score_map[name])
    return prepare_proteins.proteinModels(str(models_dir)), str(stability_dir)


def test_analyse_stability_scan_ddg_zero_for_wt(tmp_path):
    models, stability_dir = _make_models_with_score_files(
        tmp_path, {"wt": [-100.0] * 9, "design1": [-90.0] * 9}
    )
    result = models.analyseRosettaStabilityScan(stability_dir, wt_model="wt")
    wt_ddg = float(result.loc[result["model"] == "wt", "ddg_kcal_mol"])
    assert wt_ddg == pytest.approx(0.0)


def test_analyse_stability_scan_positive_ddg_for_less_stable_design(tmp_path):
    models, stability_dir = _make_models_with_score_files(
        tmp_path, {"wt": [-100.0] * 9, "design1": [-90.0] * 9}
    )
    result = models.analyseRosettaStabilityScan(stability_dir, wt_model="wt")
    d1_ddg = float(result.loc[result["model"] == "design1", "ddg_kcal_mol"])
    assert d1_ddg == pytest.approx(10.0, abs=0.1)


def test_analyse_stability_scan_negative_ddg_for_more_stable_design(tmp_path):
    models, stability_dir = _make_models_with_score_files(
        tmp_path, {"wt": [-100.0] * 9, "design1": [-110.0] * 9}
    )
    result = models.analyseRosettaStabilityScan(stability_dir, wt_model="wt")
    d1_ddg = float(result.loc[result["model"] == "design1", "ddg_kcal_mol"])
    assert d1_ddg == pytest.approx(-10.0, abs=0.1)


def test_analyse_stability_scan_uses_lowest_third_by_default(tmp_path):
    # WT: 9 scores; lowest third (3 models) average = mean([-30,-20,-10]) = -20
    # Design: 9 scores; lowest third average = mean([-50,-40,-30]) = -40
    models, stability_dir = _make_models_with_score_files(
        tmp_path,
        {
            "wt":      [-30.0, -20.0, -10.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "design1": [-50.0, -40.0, -30.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        },
    )
    result = models.analyseRosettaStabilityScan(stability_dir, wt_model="wt")
    d1_ddg = float(result.loc[result["model"] == "design1", "ddg_kcal_mol"])
    assert d1_ddg == pytest.approx(-20.0, abs=0.5)


def test_analyse_stability_scan_sorted_by_ddg(tmp_path):
    models, stability_dir = _make_models_with_score_files(
        tmp_path,
        {
            "wt":      [-100.0] * 9,
            "better":  [-110.0] * 9,
            "worse":   [ -90.0] * 9,
        },
    )
    result = models.analyseRosettaStabilityScan(stability_dir, wt_model="wt")
    assert list(result["model"]) == ["better", "wt", "worse"]


def test_analyse_stability_scan_stores_result_on_object(tmp_path):
    models, stability_dir = _make_models_with_score_files(
        tmp_path, {"wt": [-100.0] * 9, "design1": [-95.0] * 9}
    )
    result = models.analyseRosettaStabilityScan(stability_dir, wt_model="wt")
    assert hasattr(models, "rosetta_stability")
    assert models.rosetta_stability is result


def test_analyse_stability_scan_raises_if_no_score_files(tmp_path):
    models_dir = tmp_path / "models"
    _write_model(models_dir, "wt")
    models = prepare_proteins.proteinModels(str(models_dir))
    with pytest.raises(ValueError, match="No score files found"):
        models.analyseRosettaStabilityScan(str(tmp_path / "empty"), wt_model="wt")


def test_analyse_stability_scan_raises_if_wt_not_in_models(tmp_path):
    models, stability_dir = _make_models_with_score_files(
        tmp_path, {"design1": [-90.0] * 9}
    )
    with pytest.raises(ValueError, match="wt_model 'wt' not found"):
        models.analyseRosettaStabilityScan(stability_dir, wt_model="wt")


def test_analyse_stability_scan_raises_on_bad_top_fraction(tmp_path):
    models, stability_dir = _make_models_with_score_files(
        tmp_path, {"wt": [-100.0] * 9}
    )
    with pytest.raises(ValueError, match="top_fraction"):
        models.analyseRosettaStabilityScan(
            stability_dir, wt_model="wt", top_fraction=0.0
        )
