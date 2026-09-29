from pathlib import Path
from types import SimpleNamespace

import pytest

import prepare_proteins.MD.openmm_setup as openmm_setup
from prepare_proteins.MD.parameterization.ambertools import AmberToolsBackend


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


class _FakePDBFile:
    def __init__(self, _path):
        residue = SimpleNamespace(name="LIG")
        self.topology = SimpleNamespace(residues=lambda: [residue])


def _run_get_amber_parameters(tmp_path, monkeypatch, fail=(), **kwargs):
    """Run getAmberParameters with fake AmberTools that write each '-o' output, except for the
    programs named in ``fail``, which return 1 and write nothing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(openmm_setup, "PDBFile", _FakePDBFile, raising=False)
    command_log = []

    def _fake_run_command(command, command_log_arg=None):
        command = command.rstrip()
        program = command.split()[0]
        if command_log_arg is not None:
            command_log_arg.append({"command": command, "returncode": 0})
        if program in fail:
            return 1
        words = command.split()
        if "-o" in words:
            output = words[words.index("-o") + 1]
            if output != "/dev/null":
                Path(output).write_text(program)
        return 0

    monkeypatch.setattr(openmm_setup, "_run_command", _fake_run_command)
    ligand_pdb = tmp_path / "ligand.pdb"
    ligand_pdb.write_text("HETATM\nEND\n")
    lig_par = openmm_setup.ligandParameters(str(ligand_pdb), command_log=command_log)
    lig_par.getAmberParameters(run_acdoctor=False, **kwargs)
    return [entry["command"] for entry in command_log]


def _antechamber_and_parmchk2(commands):
    antechamber = [c for c in commands if c.startswith("antechamber ") and "-dr y" not in c]
    parmchk2 = [c for c in commands if c.startswith("parmchk2 ")]
    assert len(antechamber) == 2  # mol2 and prepi
    assert len(parmchk2) == 1
    return antechamber, parmchk2[0]


def test_get_amber_parameters_default_keeps_gaff_commands(tmp_path, monkeypatch):
    antechamber, parmchk2 = _antechamber_and_parmchk2(_run_get_amber_parameters(tmp_path, monkeypatch))
    assert antechamber == [
        "antechamber -i LIG_renum.pdb -fi pdb -o LIG.mol2 -fo mol2 -pf y -c bcc -nc 0 -s 2",
        "antechamber -i LIG_renum.pdb -fi pdb -o LIG.prepi -fo prepi -pf y -c bcc -nc 0 -s 2",
    ]
    assert parmchk2 == "parmchk2 -i LIG.mol2 -o LIG.frcmod -f mol2"
    assert not (tmp_path / "LIG.gaff_version").exists()


def test_get_amber_parameters_gaff2_sets_atom_types_and_parmchk2_set(tmp_path, monkeypatch):
    commands = _run_get_amber_parameters(tmp_path, monkeypatch, gaff_version="gaff2")
    antechamber, parmchk2 = _antechamber_and_parmchk2(commands)
    assert all("-at gaff2 " in c for c in antechamber)
    assert parmchk2 == "parmchk2 -i LIG.mol2 -o LIG.frcmod -s 2 -f mol2"
    assert (tmp_path / "LIG.gaff_version").read_text() == "gaff2\n"


@pytest.mark.parametrize("bad", ["GAFF2", "gaff3", "", None])
def test_invalid_gaff_version_is_rejected_before_any_command(tmp_path, monkeypatch, bad):
    with pytest.raises(ValueError, match="gaff_version"):
        _run_get_amber_parameters(tmp_path, monkeypatch, gaff_version=bad)
    assert not (tmp_path / "LIG_renum.pdb").exists()


def _write_tleap(tmp_path, monkeypatch, _expect_tleap=True, **kwargs):
    expect_tleap = _expect_tleap
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(MODEL_PDB)
    md = openmm_setup.openmm_md(str(pdb_path))
    md.setUpFF("amber14")
    parameters_folder = tmp_path / "parameters"

    class FakeAmberPrmtopFile:
        def __init__(self, path):
            self.topology = md.modeller.topology

    class FakeAmberInpcrdFile:
        def __init__(self, path):
            self.positions = md.modeller.positions

    def fake_run_command(command, command_log=None):
        (parameters_folder / f"{md.pdb_name}.prmtop").write_text("%FLAG POINTERS\n")
        (parameters_folder / f"{md.pdb_name}.inpcrd").write_text("")
        return 0

    monkeypatch.setattr(openmm_setup, "_run_command", fake_run_command)
    monkeypatch.setattr(openmm_setup, "AmberPrmtopFile", FakeAmberPrmtopFile)
    monkeypatch.setattr(openmm_setup, "AmberInpcrdFile", FakeAmberInpcrdFile)
    md.parameterizePDBLigands(str(parameters_folder), **kwargs)
    if not expect_tleap:
        return None
    return (parameters_folder / "tleap.in").read_text().splitlines()


def test_tleap_sources_gaff_by_default(tmp_path, monkeypatch):
    lines = _write_tleap(tmp_path, monkeypatch)
    assert "source leaprc.gaff" in lines
    assert "source leaprc.gaff2" not in lines


def test_tleap_sources_only_gaff2_when_requested(tmp_path, monkeypatch):
    lines = _write_tleap(tmp_path, monkeypatch, gaff_version="gaff2")
    assert "source leaprc.gaff2" in lines
    assert "source leaprc.gaff" not in lines


def test_parameterize_rejects_invalid_gaff_version(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="gaff_version"):
        _write_tleap(tmp_path, monkeypatch, gaff_version="gaff1")


def test_ambertools_backend_forwards_gaff_version_option(monkeypatch):
    received = {}

    class FakeMD:
        def parameterizePDBLigands(self, parameters_folder, **kwargs):
            received.update(kwargs)

    backend = AmberToolsBackend(gaff_version="gaff2", ligand_smiles={"X": "C"})
    monkeypatch.setattr(backend, "describe_model", lambda md: None)
    backend.prepare_model(FakeMD(), "params", verbose=True)
    assert received == {"gaff_version": "gaff2"}


def test_ambertools_backend_without_option_passes_nothing_extra(monkeypatch):
    received = {}

    class FakeMD:
        def parameterizePDBLigands(self, parameters_folder, **kwargs):
            received.update(kwargs)

    backend = AmberToolsBackend()
    monkeypatch.setattr(backend, "describe_model", lambda md: None)
    backend.prepare_model(FakeMD(), "params")
    assert received == {}


def _marker(tmp_path):
    path = tmp_path / "LIG.gaff_version"
    return path.read_text() if path.exists() else None


def test_reusing_gaff2_parameters_with_default_version_raises(tmp_path, monkeypatch):
    _run_get_amber_parameters(tmp_path, monkeypatch, gaff_version="gaff2")
    with pytest.raises(ValueError, match="generated with gaff2"):
        _run_get_amber_parameters(tmp_path, monkeypatch)


def test_switching_versions_with_overwrite_updates_the_marker(tmp_path, monkeypatch):
    _run_get_amber_parameters(tmp_path, monkeypatch, gaff_version="gaff2")
    commands = _run_get_amber_parameters(tmp_path, monkeypatch, overwrite=True)
    antechamber, _ = _antechamber_and_parmchk2(commands)
    assert all("-at " not in c for c in antechamber)
    assert _marker(tmp_path) == "gaff\n"
    _run_get_amber_parameters(tmp_path, monkeypatch)  # reuse with the same version is fine
    with pytest.raises(ValueError, match="generated with gaff,"):
        _run_get_amber_parameters(tmp_path, monkeypatch, gaff_version="gaff2")


def test_reusing_unrecorded_parameters_with_gaff2_warns_and_does_not_certify(tmp_path, monkeypatch):
    (tmp_path / "LIG.mol2").write_text("unknown provenance")
    with pytest.warns(RuntimeWarning, match="unrecorded GAFF version"):
        commands = _run_get_amber_parameters(tmp_path, monkeypatch, gaff_version="gaff2",
                                             skip_ligand_charge_computation=True)
    assert any(c.startswith("parmchk2 ") for c in commands)  # only the frcmod was made here
    assert _marker(tmp_path) is None


@pytest.mark.parametrize("program", ["antechamber", "parmchk2"])
def test_failed_generation_records_no_version(tmp_path, monkeypatch, program):
    _run_get_amber_parameters(tmp_path, monkeypatch, fail=(program,), gaff_version="gaff2")
    assert _marker(tmp_path) is None


def test_failed_regeneration_removes_an_existing_marker(tmp_path, monkeypatch):
    _run_get_amber_parameters(tmp_path, monkeypatch, gaff_version="gaff2")
    _run_get_amber_parameters(tmp_path, monkeypatch, fail=("parmchk2",), gaff_version="gaff2", overwrite=True)
    assert _marker(tmp_path) is None


def test_cached_system_built_with_other_version_is_refused(tmp_path, monkeypatch):
    _write_tleap(tmp_path, monkeypatch)  # builds with gaff; fake tleap leaves prmtop/inpcrd behind
    with pytest.raises(ValueError, match="built with gaff"):
        _write_tleap(tmp_path, monkeypatch, gaff_version="gaff2")


def test_cached_system_without_tleap_record_warns_with_gaff2(tmp_path, monkeypatch):
    _write_tleap(tmp_path, monkeypatch)
    (tmp_path / "parameters" / "tleap.in").unlink()
    with pytest.warns(RuntimeWarning, match="not recorded"):
        _write_tleap(tmp_path, monkeypatch, gaff_version="gaff2", _expect_tleap=False)


def test_tleap_gaff_version_reader(tmp_path):
    path = tmp_path / "tleap.in"
    assert openmm_setup._tleap_gaff_version(str(path)) is None
    path.write_text("source leaprc.protein.ff14SB\nsource leaprc.gaff2\n")
    assert openmm_setup._tleap_gaff_version(str(path)) == "gaff2"
    path.write_text("source leaprc.protein.ff14SB\nsource leaprc.gaff\n")
    assert openmm_setup._tleap_gaff_version(str(path)) == "gaff"


VALID_MCPB_SITE = {
    "metal": {"chain": "A", "resid": 601, "resname": "HEM", "atom": "FE"},
    "coordinating_atoms": [
        {"chain": "A", "resid": 463, "resname": "CYS", "atom": "SG", "role": "proximal"},
    ],
}


@pytest.mark.parametrize("option", ["mcpb_config", "mcpb_site"])
def test_gaff2_is_refused_with_mcpb_sites(tmp_path, monkeypatch, option):
    with pytest.raises(ValueError, match="MCPB"):
        _write_tleap(tmp_path, monkeypatch, gaff_version="gaff2", **{option: VALID_MCPB_SITE})


def _supplied_pair(tmp_path, version=None):
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    (supplied / "LIG.mol2").write_text("@<TRIPOS>MOLECULE\nLIG\n")
    (supplied / "LIG.frcmod").write_text("Remark\n")
    if version:
        (supplied / "LIG.gaff_version").write_text(version + "\n")
    return {"LIG": str(supplied / "LIG.mol2")}, {"LIG": str(supplied / "LIG.frcmod")}


def test_supplied_pair_of_unknown_version_warns_with_gaff2(tmp_path, monkeypatch):
    mol2, frcmod = _supplied_pair(tmp_path)
    with pytest.warns(RuntimeWarning, match="Supplied mol2/frcmod for LIG"):
        _write_tleap(tmp_path, monkeypatch, gaff_version="gaff2", extra_mol2=mol2, extra_frcmod=frcmod)


def test_supplied_pair_recorded_as_other_version_is_refused(tmp_path, monkeypatch):
    mol2, frcmod = _supplied_pair(tmp_path, version="gaff")
    with pytest.raises(ValueError, match="recorded as gaff"):
        _write_tleap(tmp_path, monkeypatch, gaff_version="gaff2", extra_mol2=mol2, extra_frcmod=frcmod)


def test_supplied_pair_provenance_is_carried_to_the_exported_copies(tmp_path, monkeypatch):
    mol2, frcmod = _supplied_pair(tmp_path, version="gaff2")
    _write_tleap(tmp_path, monkeypatch, gaff_version="gaff2", extra_mol2=mol2, extra_frcmod=frcmod)
    root_marker = tmp_path / "parameters" / "LIG.gaff_version"
    assert root_marker.read_text() == "gaff2\n"
    # Reusing the exported copies under the default version is refused, not silently accepted.
    exported = {"LIG": str(tmp_path / "parameters" / "LIG.mol2")}, {"LIG": str(tmp_path / "parameters" / "LIG.frcmod")}
    reuse = tmp_path / "reuse"
    reuse.mkdir()
    with pytest.raises(ValueError, match="recorded as gaff2"):
        _write_tleap(reuse, monkeypatch, extra_mol2=exported[0], extra_frcmod=exported[1])
