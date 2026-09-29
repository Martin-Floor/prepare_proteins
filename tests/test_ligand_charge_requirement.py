import warnings
from pathlib import Path

import pytest

import prepare_proteins.MD.openmm_setup as openmm_setup

LIG_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
TER
HETATM    5  C1  LIG B   1      18.000  13.000   2.000  1.00 20.00           C
HETATM    6  O1  LIG B   1      19.200  13.000   2.000  1.00 20.00           O
HETATM    7  O2  LIG B   1      17.300  14.100   2.000  1.00 20.00           O
TER
END
"""


def _build(tmp_path, monkeypatch, stage=False, **kwargs):
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(LIG_PDB)
    md = openmm_setup.openmm_md(str(pdb_path))
    md.setUpFF("amber14")
    folder = tmp_path / "parameters"
    commands = []
    if stage:
        (folder / "LIG_parameters").mkdir(parents=True)
        (folder / "LIG_parameters" / "LIG.prepi").write_text("staged")
        (folder / "LIG_parameters" / "LIG.frcmod").write_text("staged")

    def fake(command, command_log=None):
        commands.append(command.strip())
        words = command.split()
        if words and words[0] in ("pdb4amber", "antechamber", "parmchk2") and "-o" in words:
            out = words[words.index("-o") + 1]
            if out != "/dev/null":
                if words[0] == "pdb4amber" and "-i" in words and Path(words[words.index("-i") + 1]).exists():
                    Path(out).write_text(Path(words[words.index("-i") + 1]).read_text())
                else:
                    Path(out).write_text("fake")
        (folder / "model.prmtop").write_text("%FLAG POINTERS\n")
        (folder / "model.inpcrd").write_text("")
        return 0

    class FakePrmtop:
        def __init__(self, path):
            self.topology = md.modeller.topology

    class FakeInpcrd:
        def __init__(self, path):
            self.positions = md.modeller.positions

    monkeypatch.setattr(openmm_setup, "_run_command", fake)
    monkeypatch.setattr(openmm_setup, "AmberPrmtopFile", FakePrmtop)
    monkeypatch.setattr(openmm_setup, "AmberInpcrdFile", FakeInpcrd)
    md.parameterizePDBLigands(str(folder), run_acdoctor=False, solvate=False, **kwargs)
    return commands


def test_missing_ligand_charge_warns(tmp_path, monkeypatch):
    with pytest.warns(UserWarning, match="No net charge given for ligand LIG"):
        cmds = _build(tmp_path, monkeypatch)
    assert any(c.startswith("antechamber") and "-nc 0" in c for c in cmds)


def test_require_ligand_charges_raises_before_any_antechamber(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="require_ligand_charges"):
        _build(tmp_path, monkeypatch, require_ligand_charges=True)


@pytest.mark.parametrize("key", ["LIG", "lig", "Lig"])
def test_charge_keys_are_case_insensitive(tmp_path, monkeypatch, key):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        cmds = _build(tmp_path, monkeypatch, charges={key: -1})
    ante = [c for c in cmds if c.startswith("antechamber") and "-c bcc" in c]
    assert ante and all("-nc -1" in c for c in ante)


def test_no_warning_when_charges_are_not_computed(tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _build(tmp_path, monkeypatch, stage=True)  # staged prepi + frcmod: nothing is computed
