"""Tests for proteinModels.setUpBoltz2Calculation multi-chain and MSA handling.

Covers the two behaviours added for interfacial ligand sites: emitting more than
one protein chain, and writing a usable msa path. Single-chain behaviour must be
unchanged, so the backward-compatibility cases are asserted here too.
"""
from pathlib import Path

import prepare_proteins


# Two chains: A and B share a sequence (homodimer), C differs.
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
ATOM      9  N   ALA B   1      21.104  13.207   2.100  1.00 20.00           N
ATOM     10  CA  ALA B   1      22.560  13.207   2.100  1.00 20.00           C
ATOM     11  C   ALA B   1      23.050  11.780   2.400  1.00 20.00           C
ATOM     12  O   ALA B   1      22.410  10.801   2.000  1.00 20.00           O
ATOM     13  N   GLY B   2      24.220  11.690   3.050  1.00 20.00           N
ATOM     14  CA  GLY B   2      24.830  10.370   3.330  1.00 20.00           C
ATOM     15  C   GLY B   2      25.412   9.850   2.010  1.00 20.00           C
ATOM     16  OXT GLY B   2      25.962   8.743   1.970  1.00 20.00           O
TER
ATOM     17  N   SER C   1      31.104  13.207   2.100  1.00 20.00           N
ATOM     18  CA  SER C   1      32.560  13.207   2.100  1.00 20.00           C
ATOM     19  C   SER C   1      33.050  11.780   2.400  1.00 20.00           C
ATOM     20  O   SER C   1      32.410  10.801   2.000  1.00 20.00           O
ATOM     21  N   TRP C   2      34.220  11.690   3.050  1.00 20.00           N
ATOM     22  CA  TRP C   2      34.830  10.370   3.330  1.00 20.00           C
ATOM     23  C   TRP C   2      35.412   9.850   2.010  1.00 20.00           C
ATOM     24  OXT TRP C   2      35.962   8.743   1.970  1.00 20.00           O
TER
END
"""


def _load_models(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "modelA.pdb").write_text(MODEL_PDB)
    return prepare_proteins.proteinModels(str(models_dir), conect_update=False)


def _yaml(job_folder):
    return Path(job_folder, "modelA", "boltz.yaml").read_text()


def _protein_blocks(yaml_text):
    """Return the list of chain-id strings for each '- protein:' block."""
    ids, in_protein = [], False
    for line in yaml_text.splitlines():
        s = line.strip()
        if s == "- protein:":
            in_protein = True
        elif s == "- ligand:":
            in_protein = False
        elif in_protein and s.startswith("id: ["):
            ids.append(s[len("id: ["):-1])
    return ids


def test_identical_chains_grouped_into_one_entry(tmp_path):
    """A and B share a sequence, so they become one entry with two ids."""
    models = _load_models(tmp_path)
    job = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(str(job), chains=["A", "B"])
    blocks = _protein_blocks(_yaml(job))
    assert blocks == ["A, B"], blocks


def test_distinct_sequences_get_separate_entries(tmp_path):
    """C differs from A/B, so it gets its own protein entry."""
    models = _load_models(tmp_path)
    job = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(str(job), chains=["A", "B", "C"])
    assert _protein_blocks(_yaml(job)) == ["A, B", "C"]


def test_ligand_chains_follow_protein_chains(tmp_path):
    """With proteins on A/B/C the ligand must not collide at B."""
    models = _load_models(tmp_path)
    job = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(
        str(job), chains=["A", "B", "C"], ccd_ligands=["FMN"], ligands=["CCO"]
    )
    text = _yaml(job)
    assert "id: [D]" in text and "ccd: FMN" in text
    assert "id: [E]" in text and "smiles: 'CCO'" in text


def test_single_chain_ligand_still_starts_at_B(tmp_path):
    """Backward compatibility: one protein chain keeps the ligand at B."""
    models = _load_models(tmp_path)
    job = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(str(job), chains="A", ccd_ligands=["HEM"])
    text = _yaml(job)
    assert _protein_blocks(text) == ["A"]
    assert "id: [B]" in text and "ccd: HEM" in text


def test_msa_path_is_joined_not_concatenated(tmp_path):
    """Regression: the path used to be written as '<msa_path>+<model>.a3m'."""
    models = _load_models(tmp_path)
    job = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(
        str(job), chains="A", use_msa_server=False, msa_path="/msas"
    )
    text = _yaml(job)
    assert "msa: /msas/modelA.a3m" in text
    assert "+" not in text


def test_msa_path_per_entry_when_multiple_entries(tmp_path):
    """Distinct sequences need distinct MSA files; shared sequences share one."""
    models = _load_models(tmp_path)
    job = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(
        str(job), chains=["A", "B", "C"], use_msa_server=False, msa_path="/msas"
    )
    text = _yaml(job)
    assert "msa: /msas/modelA_A.a3m" in text  # entry for chains A+B
    assert "msa: /msas/modelA_C.a3m" in text  # entry for chain C
    assert text.count("msa:") == 2


def test_msa_empty_means_single_sequence_mode(tmp_path):
    models = _load_models(tmp_path)
    job = tmp_path / "boltz_jobs"
    models.setUpBoltz2Calculation(
        str(job), chains="A", use_msa_server=False, msa_path="empty"
    )
    assert "msa: empty" in _yaml(job)
