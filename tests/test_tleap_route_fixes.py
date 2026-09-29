from pathlib import Path
from types import SimpleNamespace

import pytest

import prepare_proteins.MD.openmm_setup as openmm_setup


def test_pdb_conect_bonds_reads_pairs_by_residue_and_name(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "HETATM    1  C   IMZ A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  N   PRO A   2       1.300   0.000   0.000  1.00  0.00           N\n"
        "ATOM      3  SG  CYX A   8       3.000   0.000   0.000  1.00  0.00           S\n"
        "HETATM    4  CS  IMZ A   1       4.800   0.000   0.000  1.00  0.00           C\n"
        "CONECT    1    2\n"
        "CONECT    4    3\n"
        "END\n")
    bonds = openmm_setup._pdb_conect_bonds(str(pdb))
    assert frozenset({(1, "C"), (2, "N")}) in bonds
    assert frozenset({(1, "CS"), (8, "SG")}) in bonds
    assert len(bonds) == 2


@pytest.mark.parametrize("returncode,content", [(1, "%FLAG POINTERS\n"), (0, ""), (0, "garbage\n"), (0, None)])
def test_check_tleap_output_raises_on_failure(tmp_path, returncode, content):
    prmtop = tmp_path / "m.prmtop"
    if content is not None:
        prmtop.write_text(content)
    with pytest.raises(RuntimeError, match="valid topology"):
        openmm_setup._check_tleap_output(returncode, str(prmtop), "tleap.in")


def test_check_tleap_output_accepts_valid_topology(tmp_path):
    prmtop = tmp_path / "m.prmtop"
    prmtop.write_text("%VERSION\n%FLAG TITLE\n\n%FLAG POINTERS\n")
    openmm_setup._check_tleap_output(0, str(prmtop), "tleap.in")


class _FakePDBFile:
    def __init__(self, _path):
        residue = SimpleNamespace(name="LIG")
        self.topology = SimpleNamespace(residues=lambda: [residue])


def test_staged_prepi_and_frcmod_skip_the_antechamber_mol2_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(openmm_setup, "PDBFile", _FakePDBFile, raising=False)
    commands = []

    def fake(command, command_log=None):
        commands.append(command.strip())
        if command.startswith("pdb4amber"):
            Path("LIG_renum.pdb").write_text("")
        return 0

    monkeypatch.setattr(openmm_setup, "_run_command", fake)
    (tmp_path / "ligand.pdb").write_text("HETATM\nEND\n")
    (tmp_path / "LIG.prepi").write_text("staged")
    (tmp_path / "LIG.frcmod").write_text("staged")
    openmm_setup.ligandParameters(str(tmp_path / "ligand.pdb")).getAmberParameters(run_acdoctor=False)
    assert not any(c.startswith(("antechamber", "parmchk2")) for c in commands)
    assert (tmp_path / "LIG.prepi").read_text() == "staged"


def test_without_staged_files_the_mol2_pass_still_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(openmm_setup, "PDBFile", _FakePDBFile, raising=False)
    commands = []
    monkeypatch.setattr(openmm_setup, "_run_command", lambda c, command_log=None: commands.append(c.strip()) or 0)
    (tmp_path / "ligand.pdb").write_text("HETATM\nEND\n")
    (tmp_path / "LIG.prepi").write_text("staged")  # prepi alone is not enough
    openmm_setup.ligandParameters(str(tmp_path / "ligand.pdb")).getAmberParameters(run_acdoctor=False)
    assert any(c.startswith("antechamber") and "-fo mol2" in c for c in commands)


NHE_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
ATOM      5  N   NHE A   2      14.220  11.690   3.050  1.00 20.00           N
TER
END
"""


def _build(tmp_path, monkeypatch, pdb_text, **kwargs):
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(pdb_text)
    md = openmm_setup.openmm_md(str(pdb_path))
    md.setUpFF("amber14")
    folder = tmp_path / "parameters"

    class FakePrmtop:
        def __init__(self, path):
            self.topology = md.modeller.topology

    class FakeInpcrd:
        def __init__(self, path):
            self.positions = md.modeller.positions

    def fake(command, command_log=None):
        (folder / "model.prmtop").write_text("%FLAG POINTERS\n")
        (folder / "model.inpcrd").write_text("")
        return 0

    monkeypatch.setattr(openmm_setup, "_run_command", fake)
    monkeypatch.setattr(openmm_setup, "AmberPrmtopFile", FakePrmtop)
    monkeypatch.setattr(openmm_setup, "AmberInpcrdFile", FakeInpcrd)
    md.parameterizePDBLigands(str(folder), **kwargs)
    return md, folder


def test_nhe_cap_is_kept_as_nhe_and_not_parameterized(tmp_path, monkeypatch):
    md, folder = _build(tmp_path, monkeypatch, NHE_PDB)
    names = [r.name for r in md.modeller.topology.residues()]
    assert "NHE" in names and "NH2" not in names
    saved = (folder / "model.pdb").read_text()  # the PDB handed to pdb4amber/tleap
    assert " NHE " in saved and " NH2 " not in saved
    assert not (folder / "NHE_parameters").exists() and not (folder / "NH2_parameters").exists()


def test_tleap_input_ends_with_quit(tmp_path, monkeypatch):
    _, folder = _build(tmp_path, monkeypatch, NHE_PDB)
    assert (folder / "tleap.in").read_text().splitlines()[-1] == "quit"


def test_failed_tleap_raises_instead_of_parsing_an_empty_topology(tmp_path, monkeypatch):
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(NHE_PDB)
    md = openmm_setup.openmm_md(str(pdb_path))
    md.setUpFF("amber14")
    folder = tmp_path / "parameters"

    def fake(command, command_log=None):
        (folder / "model.prmtop").write_text("")  # tleap died after opening the file
        return 0

    monkeypatch.setattr(openmm_setup, "_run_command", fake)
    with pytest.raises(RuntimeError, match="valid topology"):
        md.parameterizePDBLigands(str(folder))
