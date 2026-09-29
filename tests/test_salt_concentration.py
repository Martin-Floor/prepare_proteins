import json
from pathlib import Path

import pytest

import prepare_proteins.MD.openmm_setup as openmm_setup


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


def _water_pdb(n):
    lines = []
    for i in range(n):
        for name in ("O", "H1", "H2"):
            lines.append(f"ATOM  {i*3+1:5d} {name:<4s} WAT W{(i % 9999) + 1:4d}       0.000   0.000   0.000")
    return "\n".join(lines) + "\nEND\n"


def _build(tmp_path, monkeypatch, n_water=1000, **kwargs):
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(MODEL_PDB)
    md = openmm_setup.openmm_md(str(pdb_path))
    md.setUpFF("amber14")
    parameters_folder = tmp_path / "parameters"
    commands = []

    class FakePrmtop:
        def __init__(self, path):
            self.topology = md.modeller.topology

    class FakeInpcrd:
        def __init__(self, path):
            self.positions = md.modeller.positions

    def fake_run_command(command, command_log=None):
        commands.append(command)
        if "tleap_salt_count.in" in command:
            script = Path(command.split("-f ", 1)[1].strip()).read_text()
            out = [l for l in script.splitlines() if l.startswith("savepdb")][0].split()[-1]
            Path(out).write_text(_water_pdb(n_water))
        else:
            (parameters_folder / f"{md.pdb_name}.prmtop").write_text("%FLAG POINTERS\n")
            (parameters_folder / f"{md.pdb_name}.inpcrd").write_text("")
        return 0

    monkeypatch.setattr(openmm_setup, "_run_command", fake_run_command)
    monkeypatch.setattr(openmm_setup, "AmberPrmtopFile", FakePrmtop)
    monkeypatch.setattr(openmm_setup, "AmberInpcrdFile", FakeInpcrd)
    md.parameterizePDBLigands(str(parameters_folder), **kwargs)
    return (parameters_folder / "tleap.in").read_text().splitlines(), parameters_folder, commands


def test_default_adds_only_neutralizing_ions_and_runs_one_tleap(tmp_path, monkeypatch):
    lines, folder, commands = _build(tmp_path, monkeypatch)
    ions = [l for l in lines if l.startswith("addIons")]
    assert ions == ["addIons2 mol Na+ 0", "addIons2 mol Cl- 0"]
    assert not any("tleap_salt_count" in c for c in commands)
    assert not (folder / "model_salt.json").exists()


def test_salt_adds_pairs_from_water_count_after_neutralizing(tmp_path, monkeypatch):
    lines, folder, commands = _build(tmp_path, monkeypatch, n_water=3000, salt_concentration=0.15)
    ions = [l for l in lines if l.startswith("addIons")]
    n = round(0.15 * 3000 / 55.5)  # 8
    assert ions == ["addIons2 mol Na+ 0", "addIons2 mol Cl- 0", f"addIons2 mol Na+ {n}", f"addIons2 mol Cl- {n}"]
    assert lines.index(f"addIons2 mol Na+ {n}") > lines.index([l for l in lines if l.startswith("solvatebox")][0])
    record = json.loads((folder / "model_salt.json").read_text())
    assert record["n_water"] == 3000 and record["nacl_pairs_added"] == n
    assert sum("tleap_salt_count" in c for c in commands) == 1


def test_water_count_survives_residue_number_wraparound(tmp_path, monkeypatch):
    _, folder, _ = _build(tmp_path, monkeypatch, n_water=12000, salt_concentration=0.15)
    assert json.loads((folder / "model_salt.json").read_text())["n_water"] == 12000


@pytest.mark.parametrize("bad", [0, -0.1, "0.15", True])
def test_invalid_concentration_is_rejected(tmp_path, monkeypatch, bad):
    with pytest.raises(ValueError, match="salt_concentration"):
        _build(tmp_path, monkeypatch, salt_concentration=bad)


@pytest.mark.parametrize("kwargs", [{"solvate": False}, {"add_counterions": False}, {"add_counterionsRand": True}])
def test_salt_needs_solvent_and_addions2(tmp_path, monkeypatch, kwargs):
    with pytest.raises(ValueError, match="salt_concentration needs"):
        _build(tmp_path, monkeypatch, salt_concentration=0.15, **kwargs)


def test_ambertools_backend_forwards_salt_concentration(monkeypatch):
    from prepare_proteins.MD.parameterization.ambertools import AmberToolsBackend
    received = {}

    class FakeMD:
        def parameterizePDBLigands(self, parameters_folder, **kwargs):
            received.update(kwargs)

    backend = AmberToolsBackend(salt_concentration=0.15, gaff_version="gaff2")
    monkeypatch.setattr(backend, "describe_model", lambda md: None)
    backend.prepare_model(FakeMD(), "params")
    assert received == {"gaff_version": "gaff2", "salt_concentration": 0.15}


@pytest.mark.parametrize("first,second", [(None, 0.15), (0.15, None), (0.15, 0.10)])
def test_reusing_a_system_with_other_salt_is_refused(tmp_path, monkeypatch, first, second):
    kw1 = {} if first is None else {"salt_concentration": first}
    _build(tmp_path, monkeypatch, n_water=3000, **kw1)
    kw2 = {} if second is None else {"salt_concentration": second}
    with pytest.raises(ValueError, match="salt_concentration"):
        _build(tmp_path, monkeypatch, n_water=3000, **kw2)


@pytest.mark.parametrize("salt", [None, 0.15])
def test_reusing_a_system_with_the_same_salt_is_accepted(tmp_path, monkeypatch, salt):
    kw = {} if salt is None else {"salt_concentration": salt}
    _build(tmp_path, monkeypatch, n_water=3000, **kw)
    _build(tmp_path, monkeypatch, n_water=3000, **kw)
