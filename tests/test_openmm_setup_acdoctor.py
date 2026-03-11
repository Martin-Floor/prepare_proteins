from pathlib import Path
from types import SimpleNamespace

import prepare_proteins.MD.openmm_setup as openmm_setup


class _FakePDBFile:
    def __init__(self, _path):
        residue = SimpleNamespace(name="LIG")
        self.topology = SimpleNamespace(residues=lambda: [residue])


def test_get_amber_parameters_can_skip_acdoctor_for_reused_mol2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(openmm_setup, "PDBFile", _FakePDBFile, raising=False)

    command_log = []

    def _fake_run_command(command, command_log_arg=None):
        command = command.rstrip()
        if command_log_arg is not None:
            command_log_arg.append({"command": command, "returncode": 0})

        if command.startswith("pdb4amber "):
            Path("LIG_renum.pdb").write_text("")
        elif command.startswith("parmchk2 "):
            Path("LIG.frcmod").write_text("")
        return 0

    monkeypatch.setattr(openmm_setup, "_run_command", _fake_run_command)

    ligand_pdb = tmp_path / "ligand.pdb"
    ligand_pdb.write_text("HETATM\nEND\n")
    (tmp_path / "LIG.mol2").write_text("@<TRIPOS>MOLECULE\nLIG\n")

    lig_par = openmm_setup.ligandParameters(str(ligand_pdb), command_log=command_log)
    lig_par.getAmberParameters(skip_ligand_charge_computation=True, run_acdoctor=False)

    commands = [entry["command"] for entry in command_log]
    assert any(command.startswith("pdb4amber ") for command in commands)
    assert any(command.startswith("parmchk2 ") for command in commands)
    assert not any("-j 0" in command and "-dr y" in command for command in commands)
