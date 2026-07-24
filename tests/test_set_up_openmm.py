import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import prepare_proteins
import prepare_proteins._protein_models as protein_models_module
from prepare_proteins.MD import PolymerBuilder


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
ATOM      9  N   GLY B   1      15.500  12.400   4.400  1.00 20.00           N
ATOM     10  H1  GLY B   1      15.300  13.250   4.900  1.00 20.00           H
ATOM     11  H2  GLY B   1      16.350  12.450   4.250  1.00 20.00           H
ATOM     12  H3  GLY B   1      15.050  12.150   3.520  1.00 20.00           H
ATOM     13  CA  GLY B   1      16.040  11.110   4.880  1.00 20.00           C
ATOM     14  C   GLY B   1      15.220  10.050   4.140  1.00 20.00           C
ATOM     15  O   GLY B   1      15.620   8.890   4.030  1.00 20.00           O
ATOM     16  OXT GLY B   1      14.120  10.250   3.610  1.00 20.00           O
ATOM     17  HXT GLY B   1      13.700   9.510   3.120  1.00 20.00           H
TER
END
"""


def _load_models(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "modelA.pdb").write_text(MODEL_PDB)
    return prepare_proteins.proteinModels(str(models_dir), conect_update=False)


def _write_pet_template_pdb(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "HETATM    1   C1 PET A   1       0.000   0.000   0.000  1.00  0.00           C",
                "HETATM    2   O1 PET A   1       1.210   0.000   0.180  1.00  0.00           O",
                "HETATM    3   C2 PET A   1       2.050   0.620   0.920  1.00  0.00           C",
                "HETATM    4   C3 PET A   1       2.940  -0.080   1.920  1.00  0.00           C",
                "HETATM    5   O2 PET A   1       3.980   0.320   2.540  1.00  0.00           O",
                "HETATM    6   C4 PET A   1       4.920   0.040   3.280  1.00  0.00           C",
                "HETATM    7   C5 PET A   1       5.780   0.810   4.060  1.00  0.00           C",
                "HETATM    8   C6 PET A   1       6.970   0.470   4.640  1.00  0.00           C",
                "HETATM    9   C7 PET A   1       7.840  -0.260   5.420  1.00  0.00           C",
                "END",
            ]
        )
        + "\n"
    )
    return path


def _load_pet_crystal_models(tmp_path):
    project_dir = tmp_path / "PET_polymer"
    project_dir.mkdir(parents=True, exist_ok=True)
    template_pdb = _write_pet_template_pdb(project_dir / "pet_repeat_unit.pdb")
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    builder = PolymerBuilder()
    builder.write_polyethylene_terephthalate_crystal_pdb(
        models_dir / "pet_crystal_like.pdb",
        residue_template=template_pdb,
        n_units=3,
        n_rows=2,
        n_columns=2,
    )

    return prepare_proteins.proteinModels(str(models_dir), conect_update=False)


def test_set_up_openmm_simulations_uses_current_saved_models(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.addCappingGroups(style="openmm", backend="internal", chains="B")

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.prmtop"
    inpcrd_file = tmp_path / "modelA.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            captured["input_pdb"] = input_pdb
            captured["pdb_text"] = Path(input_pdb).read_text()
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name):
            captured["ff_name"] = ff_name

        def getProtonationStates(self):
            return []

        def removeHydrogens(self):
            captured["removed_hydrogens"] = True

        def addHydrogens(self, variants=None):
            captured["variants"] = variants

    class FakeBackend:
        name = "ambertools"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
    )

    input_pdb = Path(captured["input_pdb"])
    assert input_pdb.parent == tmp_path / "jobs" / "input_models"
    assert " ACE B   0" in captured["pdb_text"]
    assert " NME B   2" in captured["pdb_text"]
    assert captured["prepare_kwargs"]["run_acdoctor"] is True
    assert len(jobs) == 1


def test_set_up_openmm_simulations_can_disable_acdoctor(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.prmtop"
    inpcrd_file = tmp_path / "modelA.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name):
            return None

        def getProtonationStates(self):
            return []

        def removeHydrogens(self):
            return None

        def addHydrogens(self, variants=None):
            return None

    class FakeBackend:
        name = "ambertools"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        run_acdoctor=False,
    )

    assert captured["prepare_kwargs"]["run_acdoctor"] is False


def test_set_up_openmm_simulations_supports_pet_crystal_like_models(tmp_path, monkeypatch):
    models = _load_pet_crystal_models(tmp_path)
    models.conects.setdefault("pet_crystal_like", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "pet_crystal_like.prmtop"
    inpcrd_file = tmp_path / "pet_crystal_like.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            captured["input_pdb"] = input_pdb
            captured["pdb_text"] = Path(input_pdb).read_text()
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name):
            captured["ff_name"] = ff_name

    class FakeBackend:
        name = "ambertools"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        ff="charmm36",
        add_hydrogens=False,
    )

    input_pdb = Path(captured["input_pdb"])
    assert input_pdb.parent == tmp_path / "jobs" / "input_models"
    assert captured["ff_name"] == "charmm36"
    assert " PET " in captured["pdb_text"]
    assert "TER" in captured["pdb_text"]
    assert captured["prepare_kwargs"]["run_acdoctor"] is True
    assert len(jobs) == 1


def test_set_up_openmm_simulations_forwards_atom_level_mcpb_site(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.prmtop"
    inpcrd_file = tmp_path / "modelA.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name):
            return None

        def getProtonationStates(self):
            return []

        def removeHydrogens(self):
            return None

        def addHydrogens(self, variants=None):
            return None

    class FakeBackend:
        name = "ambertools"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    mcpb_site = {
        "metal": {"chain": "A", "resid": 601, "resname": "HEM", "atom": "FE"},
        "coordinating_atoms": [
            {"chain": "A", "resid": 463, "resname": "CYS", "atom": "SG", "role": "proximal"},
            {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
        ],
    }

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        mcpb_site=mcpb_site,
    )

    assert captured["prepare_kwargs"]["mcpb_site"] == mcpb_site


def test_set_up_openmm_simulations_accepts_mcpb_config_alias(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.prmtop"
    inpcrd_file = tmp_path / "modelA.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name):
            return None

        def getProtonationStates(self):
            return []

        def removeHydrogens(self):
            return None

        def addHydrogens(self, variants=None):
            return None

    class FakeBackend:
        name = "ambertools"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    mcpb_config = {
        "metal": {"chain": "A", "resid": 601, "resname": "HEM", "atom": "FE"},
        "coordinating_atoms": [
            {"chain": "A", "resid": 463, "resname": "CYS", "atom": "SG", "role": "proximal"},
            {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
        ],
    }

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        mcpb_config=mcpb_config,
    )

    assert captured["prepare_kwargs"]["mcpb_site"] == mcpb_config


def test_set_up_openmm_simulations_forwards_membrane_system(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.prmtop"
    inpcrd_file = tmp_path / "modelA.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}
    membrane_system = {
        "lipid_type": "POPC",
        "minimum_padding_nm": 1.2,
        "ionic_strength_molar": 0.15,
        "orientation_mode": "as_is",
    }

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name, membrane_system=None):
            captured["ff_name"] = ff_name
            captured["ff_membrane_system"] = membrane_system

        def getProtonationStates(self):
            return []

        def removeHydrogens(self):
            return None

        def addHydrogens(self, variants=None):
            return None

        def addMembrane(self, membrane_system):
            captured["add_membrane_system"] = membrane_system

    class FakeBackend:
        name = "ambertools"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        membrane_system=membrane_system,
    )

    assert captured["ff_name"] == "amber14"
    assert captured["ff_membrane_system"] == membrane_system
    assert captured["add_membrane_system"] == membrane_system
    assert captured["prepare_kwargs"]["membrane_system"] == membrane_system
    assert captured["prepare_kwargs"]["solvate"] is False
    assert captured["prepare_kwargs"]["add_counterions"] is False
    assert captured["prepare_kwargs"]["add_counterionsRand"] is False


def test_set_up_openmm_simulations_accepts_per_model_membrane_system_mapping(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.prmtop"
    inpcrd_file = tmp_path / "modelA.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}
    membrane_system = {"modelA": {"lipid_type": "POPC", "minimum_padding_nm": 1.0}}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name, membrane_system=None):
            captured["ff_membrane_system"] = membrane_system

        def getProtonationStates(self):
            return []

        def removeHydrogens(self):
            return None

        def addHydrogens(self, variants=None):
            return None

        def addMembrane(self, membrane_system):
            captured["add_membrane_system"] = membrane_system

    class FakeBackend:
        name = "ambertools"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        membrane_system=membrane_system,
    )

    expected = {"lipid_type": "POPC", "minimum_padding_nm": 1.0}
    assert captured["ff_membrane_system"] == expected
    assert captured["add_membrane_system"] == expected
    assert captured["prepare_kwargs"]["membrane_system"] == expected


def test_set_up_openmm_simulations_uses_oriented_pdb_when_requested(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.prmtop"
    inpcrd_file = tmp_path / "modelA.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    oriented_pdb = tmp_path / "oriented_modelA.pdb"
    oriented_pdb.write_text(MODEL_PDB.replace("ALA A   1", "SER A   1"))

    captured = {}
    membrane_system = {
        "lipid_type": "POPC",
        "orientation_mode": "use_oriented_pdb",
        "oriented_pdb": str(oriented_pdb),
    }

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            captured["input_pdb"] = input_pdb
            captured["input_pdb_text"] = Path(input_pdb).read_text()
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name, membrane_system=None):
            return None

        def getProtonationStates(self):
            return []

        def removeHydrogens(self):
            return None

        def addHydrogens(self, variants=None):
            return None

        def addMembrane(self, membrane_system):
            return None

    class FakeBackend:
        name = "ambertools"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        membrane_system=membrane_system,
    )

    assert "SER A   1" in captured["input_pdb_text"]
    assert "ALA A   1" not in captured["input_pdb_text"]


def test_set_up_openmm_simulations_can_use_membrane_positioning_folder(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.prmtop"
    inpcrd_file = tmp_path / "modelA.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    positioning_folder = tmp_path / "membrane_positioning"
    output_model_dir = positioning_folder / "output_models" / "modelA"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    (output_model_dir / "modelA.pdb").write_text(MODEL_PDB.replace("ALA A   1", "LYS A   1"))

    captured = {}
    membrane_system = {
        "lipid_type": "POPC",
        "orientation_mode": "use_oriented_pdb",
        "membrane_positioning_folder": str(positioning_folder),
    }

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            captured["input_pdb_text"] = Path(input_pdb).read_text()
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name, membrane_system=None):
            return None

        def getProtonationStates(self):
            return []

        def removeHydrogens(self):
            return None

        def addHydrogens(self, variants=None):
            return None

        def addMembrane(self, membrane_system):
            return None

    class FakeBackend:
        name = "ambertools"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        membrane_system=membrane_system,
    )

    assert "LYS A   1" in captured["input_pdb_text"]
    assert "ALA A   1" not in captured["input_pdb_text"]


def test_set_up_openmm_simulations_allows_membrane_system_with_openff(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.prmtop"
    inpcrd_file = tmp_path / "modelA.inpcrd"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name, membrane_system=None):
            captured["ff_name"] = ff_name
            captured["membrane_ff"] = membrane_system

        def getProtonationStates(self):
            return []

        def removeHydrogens(self):
            return None

        def addHydrogens(self, variants=None):
            return None

        def addMembrane(self, membrane_system):
            captured["membrane_added"] = membrane_system

    class FakeBackend:
        name = "openff"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    def fake_get_backend(method, **kwargs):
        captured["backend_method"] = method
        captured["backend_kwargs"] = dict(kwargs)
        return FakeBackend()

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(protein_models_module, "get_backend", fake_get_backend)

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        parameterization_method="openff",
        membrane_system={"lipid_type": "POPC"},
    )

    assert captured["backend_method"] == "openff"
    assert captured["backend_kwargs"]["forcefield_files"] == (
        "amber14/protein.ff14SB.xml",
        "amber14/tip3pfb.xml",
        "amber14/lipid17.xml",
    )
    assert captured["prepare_kwargs"]["membrane_system"] == {"lipid_type": "POPC"}


def test_set_up_openmm_simulations_skips_local_openmm_prep_for_charmm_gui_backend(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.parm7"
    inpcrd_file = tmp_path / "modelA.rst7"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

        def setUpFF(self, ff_name, membrane_system=None):
            raise AssertionError("CHARMM-GUI backend should not trigger local setUpFF.")

        def getProtonationStates(self):
            raise AssertionError("CHARMM-GUI backend should not inspect local protonation states.")

        def removeHydrogens(self):
            raise AssertionError("CHARMM-GUI backend should not remove hydrogens locally.")

        def addHydrogens(self, variants=None):
            raise AssertionError("CHARMM-GUI backend should not add hydrogens locally.")

        def addMembrane(self, membrane_system):
            raise AssertionError("CHARMM-GUI backend should not build a local membrane.")

    class FakeBackend:
        name = "charmm_gui"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    def fake_get_backend(method, **kwargs):
        captured["backend_method"] = method
        captured["backend_kwargs"] = dict(kwargs)
        return FakeBackend()

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(protein_models_module, "get_backend", fake_get_backend)

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        parameterization_method="charmm_gui",
        parameterization_options={
            "token": "token",
            "pdb_reader_jobid": {"modelA": "12345"},
            "quick_bilayer": {"modelA": {"membtype": "PMm", "margin": 20}},
        },
    )

    assert captured["backend_method"] == "charmm_gui"
    assert captured["prepare_kwargs"]["allow_remote"] is True
    assert jobs


def test_set_up_openmm_simulations_rejects_membrane_system_with_charmm_gui_backend(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")

    class FakeBackend:
        name = "charmm_gui"

    fake_openmm_setup = SimpleNamespace(
        openmm_md=lambda input_pdb: None,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    with pytest.raises(ValueError, match="does not support membrane_system"):
        models.setUpOpenMMSimulations(
            str(tmp_path / "jobs"),
            replicas=1,
            simulation_time=1,
            script_file=str(script_file),
            parameterization_method="charmm_gui",
            parameterization_options={
                "token": "token",
                "pdb_reader_jobid": "12345",
                "quick_bilayer": {"membtype": "PMm", "margin": 20},
            },
            membrane_system={"lipid_type": "POPC"},
        )


def test_set_up_openmm_simulations_charmm_gui_uses_cache_only_when_skip_preparation(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")
    prmtop_file = tmp_path / "modelA.parm7"
    inpcrd_file = tmp_path / "modelA.rst7"
    prmtop_file.write_text("")
    inpcrd_file.write_text("")

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

    class FakeBackend:
        name = "charmm_gui"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            md_obj.prmtop_file = str(prmtop_file)
            md_obj.inpcrd_file = str(inpcrd_file)
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

        def describe_model(self, md_obj):
            return SimpleNamespace(
                input_format="amber",
                prmtop_path=md_obj.prmtop_file,
                coordinates_path=md_obj.inpcrd_file,
            )

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        parameterization_method="charmm_gui",
        parameterization_options={
            "token": "token",
            "pdb_reader_jobid": "12345",
            "quick_bilayer": {"membtype": "PMm", "margin": 20},
        },
        skip_preparation=True,
    )

    assert captured["prepare_kwargs"]["allow_remote"] is False
    assert jobs


def test_set_up_openmm_simulations_submit_only_skips_replica_generation_for_pending_charmm_gui_jobs(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

    class FakeBackend:
        name = "charmm_gui"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            return SimpleNamespace(
                input_format="remote_pending",
                metadata={"state": "submitted", "quick_bilayer_jobid": "5804504324"},
            )

        def describe_model(self, md_obj):
            raise AssertionError("Pending CHARMM-GUI submissions should not need describe_model().")

    def fake_get_backend(method, **kwargs):
        captured["backend_method"] = method
        captured["backend_kwargs"] = dict(kwargs)
        return FakeBackend()

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(protein_models_module, "get_backend", fake_get_backend)

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=2,
        simulation_time=1,
        script_file=str(script_file),
        parameterization_method="charmm_gui",
        parameterization_options={
            "token": "token",
            "workflow_mode": "submit_only",
            "pdb_reader_jobid": "12345",
            "quick_bilayer": {"membtype": "PMm", "margin": 20},
        },
    )

    assert captured["backend_method"] == "charmm_gui"
    assert captured["backend_kwargs"]["workflow_mode"] == "submit_only"
    assert jobs == []


def test_set_up_openmm_simulations_stages_charmm_inputs_for_openmm_runner(tmp_path, monkeypatch):
    models = _load_models(tmp_path)
    models.conects.setdefault("modelA", [])

    script_file = tmp_path / "openmm_simulation.py"
    script_file.write_text("print('stub')\n")

    psf_file = tmp_path / "modelA.psf"
    crd_file = tmp_path / "modelA.crd"
    pdb_file = tmp_path / "modelA.pdb"
    toppar_file = tmp_path / "toppar.str"
    sysinfo_file = tmp_path / "sysinfo.dat"
    toppar_dir = tmp_path / "toppar"
    toppar_dir.mkdir()
    (toppar_dir / "dummy.prm").write_text("*\n")
    psf_file.write_text("")
    crd_file.write_text("")
    pdb_file.write_text("END\n")
    toppar_file.write_text("../toppar/dummy.prm\n")
    sysinfo_file.write_text('{"dimensions": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]}\n')

    captured = {}

    class FakeOpenMMMD:
        def __init__(self, input_pdb):
            self.input_pdb = input_pdb
            self.pdb_name = Path(input_pdb).stem
            self.command_log = []

    class FakeBackend:
        name = "charmm_gui"

        def prepare_model(self, md_obj, ligand_parameters_folder, **kwargs):
            captured["prepare_kwargs"] = dict(kwargs)
            return SimpleNamespace(
                input_format="charmm",
                files={
                    "psf": str(psf_file),
                    "crd": str(crd_file),
                    "coordinates": str(crd_file),
                    "pdb": str(pdb_file),
                    "toppar_str": str(toppar_file),
                    "sysinfo": str(sysinfo_file),
                    "toppar_dir": str(toppar_dir),
                },
                metadata={"backend": "charmm_gui"},
            )

        def describe_model(self, md_obj):
            raise AssertionError("Prepared CHARMM results should not need describe_model().")

    fake_openmm_setup = SimpleNamespace(
        openmm_md=FakeOpenMMMD,
        aa3=["ALA", "GLY", "ACE", "NME"],
    )
    monkeypatch.setattr(
        protein_models_module,
        "_require_openmm_support",
        lambda feature: fake_openmm_setup,
    )
    monkeypatch.setattr(
        protein_models_module,
        "get_backend",
        lambda method, **kwargs: FakeBackend(),
    )

    jobs = models.setUpOpenMMSimulations(
        str(tmp_path / "jobs"),
        replicas=1,
        simulation_time=1,
        script_file=str(script_file),
        parameterization_method="charmm_gui",
        parameterization_options={
            "token": "token",
            "pdb_reader_jobid": "12345",
            "quick_bilayer": {"membtype": "PMm", "margin": 20},
        },
    )

    replica_root = tmp_path / "jobs" / "modelA" / "replica_01"
    assert jobs
    assert "--input_format charmm" in jobs[0]
    assert "--charmm-toppar input_files/openmm/toppar.str" in jobs[0]
    assert "--charmm-sysinfo input_files/openmm/sysinfo.dat" in jobs[0]
    assert (replica_root / "input_files" / "openmm" / "modelA.psf").exists()
    assert (replica_root / "input_files" / "openmm" / "modelA.crd").exists()
    assert (replica_root / "input_files" / "openmm" / "toppar.str").exists()
    assert (replica_root / "input_files" / "openmm" / "sysinfo.dat").exists()
    assert (replica_root / "input_files" / "toppar" / "dummy.prm").exists()


def test_set_up_openmm_interaction_energy_calculations_builds_polymer_commands(tmp_path):
    models = _load_models(tmp_path)

    jobs_root = tmp_path / "jobs"
    for replica_name in ("replica_01", "replica_02"):
        input_folder = jobs_root / "modelA" / replica_name / "input_files"
        input_folder.mkdir(parents=True, exist_ok=True)
        (input_folder / "modelA.prmtop").write_text("")
        (input_folder / "modelA.inpcrd").write_text("")

    script_file = tmp_path / "computeInteractions.py"
    script_file.write_text("print('interaction stub')\n")
    output_folder = tmp_path / "polymer_interactions"

    jobs = models.setUpOpenMMInteractionEnergyCalculations(
        str(jobs_root),
        replicas=2,
        partner_residue_names=["ROH", "4GB", "0GB"],
        by_residue=True,
        residue_indexes=[0, 1],
        only_indexes={1: [1, 10], 2: [2, 20]},
        partner_interaction_groups={"ring": ["C1", "C2"]},
        script_file=str(script_file),
        output_folder=str(output_folder),
        root_dir_env_var="CBM_INTERACTION_ROOT",
    )

    copied_script = jobs_root / "scripts" / "computeInteractions.py"
    assert copied_script.exists()
    assert copied_script.read_text() == "print('interaction stub')\n"

    groups_json = jobs_root / "scripts" / "._partner_interaction_groups.json"
    assert groups_json.exists()
    assert json.loads(groups_json.read_text()) == {"ring": ["C1", "C2"]}

    assert len(jobs) == 2
    assert "ROH,4GB,0GB" in jobs[0]
    assert "--by_residue" in jobs[0]
    assert "--residue_indexes" in jobs[0]
    assert "--partner_interaction_groups" in jobs[0]
    assert "--only_indexes" in jobs[0]
    assert "1,10" in jobs[0]
    assert "Replica_1.csv" in jobs[0]
    assert "2,20" in jobs[1]
    assert "Replica_2.csv" in jobs[1]
    assert "CBM_INTERACTION_ROOT" in jobs[0]
