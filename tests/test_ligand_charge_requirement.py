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


SDF = """LIG
  test

  3  2  0  0  0  0  0  0  0  0999 V2000
   18.0000   13.0000    2.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   19.2000   13.0000    2.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   17.3000   14.1000    2.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  1  3  1  0
M  END
$$$$
"""


def _build(tmp_path, monkeypatch, stage=(), **kwargs):
    """Run parameterizePDBLigands with fake AmberTools; `stage` names files pre-placed in LIG_parameters.
    Returns (commands, warnings)."""
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(LIG_PDB)
    md = openmm_setup.openmm_md(str(pdb_path))
    md.setUpFF("amber14")
    folder = tmp_path / "parameters"
    commands = []
    if stage:
        (folder / "LIG_parameters").mkdir(parents=True, exist_ok=True)
        for name in stage:
            (folder / "LIG_parameters" / name).write_text("staged")

    def fake(command, command_log=None):
        commands.append(command.strip())
        words = command.split()
        if words and words[0] in ("pdb4amber", "antechamber", "parmchk2") and "-o" in words:
            out = words[words.index("-o") + 1]
            if out != "/dev/null":
                src = words[words.index("-i") + 1] if "-i" in words else None
                if words[0] == "pdb4amber" and src and Path(src).exists():
                    Path(out).write_text(Path(src).read_text())
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        md.parameterizePDBLigands(str(folder), run_acdoctor=False, solvate=False, **kwargs)
    msgs = [str(w.message) for w in caught if "No net charge given" in str(w.message)]
    return commands, msgs


def _charge_runs(cmds):
    """antechamber commands that compute charges (any -c method other than rc)."""
    return [c for c in cmds if c.startswith("antechamber") and " -c " in f" {c} " and " -c rc" not in c]


def test_missing_charge_warns_and_computes_neutral(tmp_path, monkeypatch):
    cmds, msgs = _build(tmp_path, monkeypatch)
    assert msgs and "LIG" in msgs[0] and "bcc" in msgs[0]
    assert _charge_runs(cmds) and all("-nc 0" in c for c in _charge_runs(cmds))


def test_require_raises_with_no_charge_command_run(tmp_path, monkeypatch):
    seen = []
    orig = openmm_setup._missing_ligand_charge

    def spy(residue, method, require):
        seen.append("check")
        return orig(residue, method, require)

    monkeypatch.setattr(openmm_setup, "_missing_ligand_charge", spy)
    ran = []
    import prepare_proteins.MD.openmm_setup as m
    real_fake = None
    with pytest.raises(ValueError):
        cmds, _ = _build(tmp_path, monkeypatch, require_ligand_charges=True)
    # _build's runner is installed inside _build; inspect the parameters folder instead: no mol2 was produced
    assert seen == ["check"]
    assert not (tmp_path / "parameters" / "LIG_parameters" / "LIG.mol2").exists()


@pytest.mark.parametrize("key", ["LIG", "lig", "Lig"])
def test_charge_keys_are_case_insensitive(tmp_path, monkeypatch, key):
    cmds, msgs = _build(tmp_path, monkeypatch, charges={key: -1})
    assert not msgs
    assert _charge_runs(cmds) and all("-nc -1" in c for c in _charge_runs(cmds))


def test_conflicting_case_variants_are_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="differing only in case"):
        _build(tmp_path, monkeypatch, charges={"Lig": -1, "lIg": 1})


def test_agreeing_case_variants_are_accepted(tmp_path, monkeypatch):
    cmds, msgs = _build(tmp_path, monkeypatch, charges={"Lig": -1, "LIG": -1})
    assert not msgs and all("-nc -1" in c for c in _charge_runs(cmds))


def test_explicit_zero_is_not_a_missing_charge(tmp_path, monkeypatch):
    cmds, msgs = _build(tmp_path, monkeypatch, charges={"LIG": 0})
    assert not msgs and _charge_runs(cmds)


def test_staged_prepi_and_frcmod_compute_nothing_and_do_not_warn(tmp_path, monkeypatch):
    cmds, msgs = _build(tmp_path, monkeypatch, stage=("LIG.prepi", "LIG.frcmod"))
    assert not msgs and not _charge_runs(cmds)


def test_cached_ligand_only_files_compute_nothing_and_do_not_warn(tmp_path, monkeypatch):
    cmds, msgs = _build(tmp_path, monkeypatch, stage=("LIG.mol2", "LIG.prepi", "LIG.frcmod"), build_full_system=False)
    assert not msgs and not _charge_runs(cmds)


def test_skip_with_overwrite_still_computes_and_warns(tmp_path, monkeypatch):
    cmds, msgs = _build(tmp_path, monkeypatch, stage=("LIG.mol2",), skip_ligand_charge_computation=True,
                        overwrite=True)
    assert _charge_runs(cmds) and msgs


def test_supplied_pair_is_not_warned(tmp_path, monkeypatch):
    sup = tmp_path / "sup"; sup.mkdir()
    (sup / "LIG.mol2").write_text("@<TRIPOS>MOLECULE\nLIG\n"); (sup / "LIG.frcmod").write_text("Remark\n")
    cmds, msgs = _build(tmp_path, monkeypatch, extra_mol2={"LIG": str(sup / "LIG.mol2")},
                        extra_frcmod={"LIG": str(sup / "LIG.frcmod")})
    assert not msgs and not _charge_runs(cmds)


def test_extra_mol2_without_frcmod_is_converted_and_warned(tmp_path, monkeypatch):
    sup = tmp_path / "sup"; sup.mkdir()
    (sup / "LIG.mol2").write_text("@<TRIPOS>MOLECULE\nLIG\n")
    cmds, msgs = _build(tmp_path, monkeypatch, extra_mol2={"LIG": str(sup / "LIG.mol2")})
    assert _charge_runs(cmds) and msgs


@pytest.mark.parametrize("with_frcmod", [False, True])
def test_sdf_route_warns_before_its_charge_run(tmp_path, monkeypatch, with_frcmod):
    sdf = tmp_path / "lig.sdf"; sdf.write_text(SDF)
    kw = {"ligand_sdf_files": {"LIG": {"path": str(sdf)}}}
    if with_frcmod:
        sup = tmp_path / "sup"; sup.mkdir(); (sup / "LIG.frcmod").write_text("Remark\n")
        kw["extra_frcmod"] = {"LIG": str(sup / "LIG.frcmod")}
    cmds, msgs = _build(tmp_path, monkeypatch, **kw)
    assert msgs and any("-fi mdl" in c for c in _charge_runs(cmds))


def test_sdf_route_strict_mode_raises_before_the_sdf_charge_run(tmp_path, monkeypatch):
    sdf = tmp_path / "lig.sdf"; sdf.write_text(SDF)
    with pytest.raises(ValueError, match="require_ligand_charges"):
        _build(tmp_path, monkeypatch, require_ligand_charges=True, ligand_sdf_files={"LIG": {"path": str(sdf)}})
    assert not (tmp_path / "parameters" / "LIG_parameters" / "LIG.mol2").exists()
