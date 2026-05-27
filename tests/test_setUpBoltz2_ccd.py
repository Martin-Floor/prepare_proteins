"""Tests for proteinModels.setUpBoltz2Calculation CCD-ligand extension."""
from pathlib import Path

import prepare_proteins


MODEL_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
ATOM      5  N   GLY A   2      14.220  11.690   3.050  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.830  10.370   3.330  1.00 20.00           C
ATOM      7  C   GLY A   2      15.412   9.850   2.010  1.00 20.00           C
ATOM      8  OXT GLY A   2      15.962   8.743   1.970  1.00 20.00           O
TER
END
"""


def _load_models(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "modelA.pdb").write_text(MODEL_PDB)
    return prepare_proteins.proteinModels(str(models_dir), conect_update=False)


def _read_yaml(p):
    return Path(p).read_text()


def test_setup_boltz2_ccd_ligand_only(tmp_path):
    models = _load_models(tmp_path)
    job_folder = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(str(job_folder), ccd_ligands=["HEM", "MG"])
    yaml = _read_yaml(job_folder / "modelA" / "boltz.yaml")
    assert "- protein:" in yaml
    # First ligand at chain B (after protein at A), second at C
    assert "id: [B]" in yaml
    assert "ccd: HEM" in yaml
    assert "id: [C]" in yaml
    assert "ccd: MG" in yaml
    # No SMILES section
    assert "smiles" not in yaml


def test_setup_boltz2_smiles_only_backward_compat(tmp_path):
    """Existing SMILES-only call must keep working unchanged."""
    models = _load_models(tmp_path)
    job_folder = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(str(job_folder), ligands=["CCO"])
    yaml = _read_yaml(job_folder / "modelA" / "boltz.yaml")
    assert "smiles: 'CCO'" in yaml
    assert "id: [B]" in yaml
    assert "ccd:" not in yaml


def test_setup_boltz2_ccd_then_smiles_chain_ordering(tmp_path):
    """CCD ligands occupy chains starting at B, then SMILES continue."""
    models = _load_models(tmp_path)
    job_folder = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(
        str(job_folder),
        ccd_ligands=["HEM", "MG"],
        ligands=["CCO"],
    )
    yaml = _read_yaml(job_folder / "modelA" / "boltz.yaml")
    # Expected chains: A (protein), B (HEM), C (MG), D (CCO)
    lines = yaml.splitlines()
    # Find each ligand block and verify chain id
    chain_to_kind = {}
    cur_chain = None
    for line in lines:
        s = line.strip()
        if s.startswith("id: [") and s.endswith("]"):
            cur_chain = s[len("id: ["):-1]
        elif s.startswith("ccd: "):
            chain_to_kind[cur_chain] = ("ccd", s.split("ccd:", 1)[1].strip())
        elif s.startswith("smiles:"):
            chain_to_kind[cur_chain] = ("smiles", s.split("smiles:", 1)[1].strip())
    assert chain_to_kind["B"] == ("ccd", "HEM")
    assert chain_to_kind["C"] == ("ccd", "MG")
    assert chain_to_kind["D"] == ("smiles", "'CCO'")


def test_setup_boltz2_no_ligands(tmp_path):
    """Calling without ligands or ccd_ligands still produces a protein-only YAML."""
    models = _load_models(tmp_path)
    job_folder = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(str(job_folder))
    yaml = _read_yaml(job_folder / "modelA" / "boltz.yaml")
    assert "- protein:" in yaml
    assert "- ligand:" not in yaml
