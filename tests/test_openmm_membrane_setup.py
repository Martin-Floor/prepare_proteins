from pathlib import Path

import pytest

from openmm.unit import molar, nanometer

from prepare_proteins.MD import openmm_setup as openmm_setup_module
from prepare_proteins.MD.parameterization import openff as openff_backend_module
from prepare_proteins.MD.parameterization.openff import OpenFFBackend


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

MODEL_WITH_NAG_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
ATOM      5  N   GLY A   2      14.220  11.690   3.050  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.830  10.370   3.330  1.00 20.00           C
ATOM      7  C   GLY A   2      15.412   9.850   2.010  1.00 20.00           C
ATOM      8  OXT GLY A   2      15.962   8.743   1.970  1.00 20.00           O
TER
HETATM    9  C1  NAG G   1      17.100  10.500   4.000  1.00 20.00           C
HETATM   10  O4  NAG G   1      17.900  11.200   4.700  1.00 20.00           O
HETATM   11  C1  NAG G   2      18.700  11.900   5.300  1.00 20.00           C
TER
END
"""

MODEL_WITH_WATER_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
ATOM      5  N   GLY A   2      14.220  11.690   3.050  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.830  10.370   3.330  1.00 20.00           C
ATOM      7  C   GLY A   2      15.412   9.850   2.010  1.00 20.00           C
ATOM      8  OXT GLY A   2      15.962   8.743   1.970  1.00 20.00           O
TER
HETATM    9  O   HOH W   1      16.500  12.000   3.800  1.00 20.00           O
HETATM   10  H1  HOH W   1      16.100  12.600   4.400  1.00 20.00           H
HETATM   11  H2  HOH W   1      17.200  12.400   3.300  1.00 20.00           H
TER
END
"""

WATER_ONLY_PDB = """\
HETATM    1  O   HOH W   1      16.500  12.000   3.800  1.00 20.00           O
HETATM    2  H1  HOH W   1      16.100  12.600   4.400  1.00 20.00           H
HETATM    3  H2  HOH W   1      17.200  12.400   3.300  1.00 20.00           H
TER
END
"""

MODEL_TWO_CHAIN_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
ATOM      5  N   GLY A   2      14.220  11.690   3.050  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.830  10.370   3.330  1.00 20.00           C
ATOM      7  C   GLY A   2      15.412   9.850   2.010  1.00 20.00           C
ATOM      8  OXT GLY A   2      15.962   8.743   1.970  1.00 20.00           O
TER
ATOM      9  N   ALA B   1      16.900  10.100   5.100  1.00 20.00           N
ATOM     10  CA  ALA B   1      17.700   9.100   5.700  1.00 20.00           C
ATOM     11  C   ALA B   1      18.800   9.700   6.600  1.00 20.00           C
ATOM     12  OXT ALA B   1      19.700   8.900   6.900  1.00 20.00           O
TER
END
"""

MODEL_ASN_WITH_NAG_PDB = """\
ATOM      1  N   ASN B   4      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  H   ASN B   4      10.500  13.900   2.400  1.00 20.00           H
ATOM      3  CA  ASN B   4      12.560  13.207   2.100  1.00 20.00           C
ATOM      4  HA  ASN B   4      12.900  13.700   1.300  1.00 20.00           H
ATOM      5  C   ASN B   4      13.050  11.780   2.400  1.00 20.00           C
ATOM      6  O   ASN B   4      12.410  10.801   2.000  1.00 20.00           O
ATOM      7  CB  ASN B   4      13.100  14.100   3.200  1.00 20.00           C
ATOM      8 HB2  ASN B   4      12.500  14.900   3.400  1.00 20.00           H
ATOM      9 HB3  ASN B   4      13.900  14.500   2.800  1.00 20.00           H
ATOM     10  CG  ASN B   4      13.700  13.300   4.300  1.00 20.00           C
ATOM     11 OD1  ASN B   4      14.300  12.200   4.100  1.00 20.00           O
ATOM     12 ND2  ASN B   4      13.500  13.900   5.500  1.00 20.00           N
ATOM     13 HD21 ASN B   4      13.900  13.400   6.200  1.00 20.00           H
ATOM     14 HD22 ASN B   4      12.900  14.700   5.600  1.00 20.00           H
TER
HETATM   15  C1  NAG G   1      17.100  10.500   4.000  1.00 20.00           C
HETATM   16  C2  NAG G   1      17.700   9.300   4.500  1.00 20.00           C
HETATM   17  C3  NAG G   1      18.900   9.600   5.300  1.00 20.00           C
HETATM   18  C4  NAG G   1      19.400  10.900   4.800  1.00 20.00           C
HETATM   19  C5  NAG G   1      18.700  12.000   5.600  1.00 20.00           C
HETATM   20  C6  NAG G   1      19.200  13.400   5.100  1.00 20.00           C
HETATM   21  C7  NAG G   1      16.600   8.100   3.300  1.00 20.00           C
HETATM   22  C8  NAG G   1      15.200   7.700   3.900  1.00 20.00           C
HETATM   23  N2  NAG G   1      17.100   8.200   3.700  1.00 20.00           N
HETATM   24  O3  NAG G   1      19.500   8.500   5.900  1.00 20.00           O
HETATM   25  O4  NAG G   1      20.700  11.100   5.300  1.00 20.00           O
HETATM   26  O5  NAG G   1      17.500  11.700   4.900  1.00 20.00           O
HETATM   27  O6  NAG G   1      18.700  14.400   5.900  1.00 20.00           O
HETATM   28  O7  NAG G   1      16.900   7.000   2.900  1.00 20.00           O
TER
END
"""

HIS_MODEL_PDB = """\
ATOM      1  N   HIS A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  HIS A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   HIS A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   HIS A   1      12.410  10.801   2.000  1.00 20.00           O
ATOM      5  CB  HIS A   1      13.123  14.612   2.400  1.00 20.00           C
ATOM      6  CG  HIS A   1      14.587  14.662   2.680  1.00 20.00           C
ATOM      7  ND1 HIS A   1      15.255  15.835   2.980  1.00 20.00           N
ATOM      8  CD2 HIS A   1      15.496  13.682   2.690  1.00 20.00           C
ATOM      9  CE1 HIS A   1      16.526  15.575   3.190  1.00 20.00           C
ATOM     10  NE2 HIS A   1      16.706  14.281   3.000  1.00 20.00           N
ATOM     11  OXT HIS A   1      13.900  11.710   3.250  1.00 20.00           O
TER
END
"""


def _build_openmm_md(tmp_path, pdb_text=MODEL_PDB):
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(pdb_text)
    return openmm_setup_module.openmm_md(str(pdb_path))


def test_openmm_md_set_up_ff_uses_lipid17_for_membrane_system(tmp_path):
    md = _build_openmm_md(tmp_path)

    md.setUpFF("amber14", membrane_system={"lipid_type": "POPC"})

    assert md.ff_files == [
        "amber14/protein.ff14SB.xml",
        "amber14/tip3pfb.xml",
        "amber14/lipid17.xml",
    ]
    assert "POP" in md.membrane_system["skip_residue_names"]


def test_openmm_md_add_membrane_normalizes_units_and_tracks_state(tmp_path, monkeypatch):
    md = _build_openmm_md(tmp_path)
    md.setUpFF("amber14", membrane_system={"lipid_type": "POPC"})

    captured = {}

    def fake_add_membrane(forcefield, **kwargs):
        captured["forcefield"] = forcefield
        captured["kwargs"] = kwargs

    monkeypatch.setattr(md.modeller, "addMembrane", fake_add_membrane)

    md.addMembrane(
        {
            "lipid_type": "POPC",
            "minimum_padding_nm": 1.2,
            "membrane_center_z_nm": 0.3,
            "ionic_strength_molar": 0.15,
            "orientation_mode": "as_is",
        }
    )

    assert captured["forcefield"] is md.forcefield
    assert captured["kwargs"]["lipidType"] == "POPC"
    assert captured["kwargs"]["minimumPadding"].value_in_unit(nanometer) == pytest.approx(1.2)
    assert captured["kwargs"]["membraneCenterZ"].value_in_unit(nanometer) == pytest.approx(0.3)
    assert captured["kwargs"]["ionicStrength"].value_in_unit(molar) == pytest.approx(0.15)
    assert md._membrane_built is True
    assert md.membrane_system["lipid_type"] == "POPC"


def test_parameterize_pdb_ligands_writes_lipid_tleap_setup_for_membranes(tmp_path, monkeypatch):
    md = _build_openmm_md(tmp_path)
    md.setUpFF("amber14", membrane_system={"lipid_type": "POPC"})

    normalized_membrane = md.membrane_system

    def fake_add_membrane(membrane_system=None, platform=None):
        md.membrane_system = openmm_setup_module._normalize_membrane_system(membrane_system)
        md._membrane_built = True

    class FakeAmberPrmtopFile:
        def __init__(self, path):
            self.topology = md.modeller.topology

    class FakeAmberInpcrdFile:
        def __init__(self, path):
            self.positions = md.modeller.positions

    def fake_run_command(command, command_log=None):
        if command_log is not None:
            command_log.append({"command": command.rstrip(), "returncode": 0})
        prmtop = parameters_folder / f"{md.pdb_name}.prmtop"
        inpcrd = parameters_folder / f"{md.pdb_name}.inpcrd"
        prmtop.write_text("")
        inpcrd.write_text("")
        return 0

    parameters_folder = tmp_path / "parameters"
    monkeypatch.setattr(md, "addMembrane", fake_add_membrane)
    monkeypatch.setattr(openmm_setup_module, "_run_command", fake_run_command)
    monkeypatch.setattr(openmm_setup_module, "AmberPrmtopFile", FakeAmberPrmtopFile)
    monkeypatch.setattr(openmm_setup_module, "AmberInpcrdFile", FakeAmberInpcrdFile)

    md.parameterizePDBLigands(str(parameters_folder), membrane_system=normalized_membrane)

    tleap_text = (parameters_folder / "tleap.in").read_text()
    assert "source oldff/leaprc.lipid17" in tleap_text
    assert "solvatebox" not in tleap_text
    assert "addIons2" not in tleap_text
    assert "addIonsRand" not in tleap_text


def test_openmm_md_add_membrane_temporarily_removes_nonprotein_residues(tmp_path, monkeypatch):
    md = _build_openmm_md(tmp_path, pdb_text=MODEL_WITH_NAG_PDB)
    md.setUpFF("amber14", membrane_system={"lipid_type": "POPC"})

    captured = {}

    def fake_add_membrane(forcefield, **kwargs):
        captured["residue_names_during_call"] = [res.name for res in md.modeller.topology.residues()]

    monkeypatch.setattr(md.modeller, "addMembrane", fake_add_membrane)

    md.addMembrane({"lipid_type": "POPC"})

    assert "NAG" not in captured["residue_names_during_call"]
    residue_names_after = [res.name for res in md.modeller.topology.residues()]
    assert residue_names_after.count("NAG") == 2


def test_openmm_md_add_membrane_can_exclude_protein_chains_during_build(tmp_path, monkeypatch):
    md = _build_openmm_md(tmp_path, pdb_text=MODEL_TWO_CHAIN_PDB)
    md.setUpFF("amber14", membrane_system={"lipid_type": "POPC", "exclude_chain_ids": ("B",)})

    captured = {}

    def fake_add_membrane(forcefield, **kwargs):
        captured["chain_ids_during_call"] = [chain.id for chain in md.modeller.topology.chains()]
        captured["residue_chain_ids_during_call"] = [res.chain.id for res in md.modeller.topology.residues()]

    monkeypatch.setattr(md.modeller, "addMembrane", fake_add_membrane)

    md.addMembrane({"lipid_type": "POPC", "exclude_chain_ids": ("B",)})

    assert "B" not in captured["chain_ids_during_call"]
    assert all(chain_id != "B" for chain_id in captured["residue_chain_ids_during_call"])
    restored_chain_ids = [chain.id for chain in md.modeller.topology.chains()]
    assert "B" in restored_chain_ids


def test_openmm_md_add_membrane_retries_after_nan_build_failure(tmp_path, monkeypatch):
    md = _build_openmm_md(tmp_path)
    md.setUpFF("amber14", membrane_system={"lipid_type": "POPC", "build_retries": 2})

    calls = {"count": 0}

    def fake_add_membrane(self, forcefield, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise Exception("Particle coordinate is NaN.")

    monkeypatch.setattr(openmm_setup_module.Modeller, "addMembrane", fake_add_membrane)

    md.addMembrane({"lipid_type": "POPC", "build_retries": 2})

    assert calls["count"] == 2
    assert md._membrane_built is True


def test_openff_backend_preserves_existing_water_for_membrane_like_systems(tmp_path):
    import parmed as pmd

    md = _build_openmm_md(tmp_path, pdb_text=WATER_ONLY_PDB)

    backend = OpenFFBackend(
        forcefield_files=(
            "amber14/tip3pfb.xml",
        ),
        solvate=False,
    )
    parameters_folder = tmp_path / "parameters"
    backend.prepare_model(
        md,
        str(parameters_folder),
        membrane_system={"lipid_type": "POPC"},
        verbose=False,
    )

    structure = pmd.load_file(str(parameters_folder / f"{md.pdb_name}.prmtop"))
    residue_names = [res.name for res in structure.residues]

    assert "HOH" in residue_names


def test_get_protonation_states_returns_none_for_unresolved_histidine(tmp_path):
    md = _build_openmm_md(tmp_path, pdb_text=HIS_MODEL_PDB)

    variants = md.getProtonationStates()

    assert variants == [None]


def test_openff_backend_can_rename_atoms_by_chain_residue_and_name(tmp_path):
    md = _build_openmm_md(tmp_path, pdb_text=MODEL_WITH_NAG_PDB)
    modeller = md.modeller

    OpenFFBackend._apply_atom_name_overrides(
        modeller,
        {
            ("G", 1, "C1"): "C2N",
            ("G", 1, "O4"): "O2N",
        },
    )

    renamed = []
    for atom in modeller.topology.atoms():
        if atom.residue.chain.id == "G" and int(atom.residue.id) == 1:
            renamed.append(atom.name)

    assert "C2N" in renamed
    assert "O2N" in renamed
    assert "C1" not in renamed
    assert "O4" not in renamed


def test_openff_backend_can_reconcile_template_backed_residues(tmp_path):
    md = _build_openmm_md(tmp_path, pdb_text=MODEL_ASN_WITH_NAG_PDB)

    for chain in md.modeller.topology.chains():
        for residue in chain.residues():
            if chain.id == "B" and str(residue.id) == "4":
                residue.name = "NLN"
            if chain.id == "G" and str(residue.id) == "1":
                residue.name = "4YB"

    OpenFFBackend._apply_atom_name_overrides(
        md.modeller,
        {
            ("G", 1, "C7"): "C2N",
            ("G", 1, "C8"): "CME",
            ("G", 1, "O7"): "O2N",
        },
    )

    class FakeTemplateAtom:
        def __init__(self, name):
            self.name = name

    class FakeTemplate:
        def __init__(self, atom_names, bonds):
            self.atoms = [FakeTemplateAtom(name) for name in atom_names]
            self.bonds = bonds

    class FakeForceField:
        def __init__(self):
                self._templates = {
                    "NLN": FakeTemplate(
                        ["N", "H", "CA", "HA", "CB", "HB2", "HB3", "CG", "OD1", "ND2", "HD21", "C", "O"],
                        [(0, 2), (2, 4), (4, 7), (7, 8), (7, 9), (9, 10), (2, 11), (11, 12)],
                    ),
                    "4YB": FakeTemplate(
                        [
                            "C1", "H1", "C2", "H2", "C3", "H3", "C4", "H4", "C5", "H5",
                            "C6", "H61", "H62", "N2", "H2N", "O3", "O4", "O5", "O6",
                            "C2N", "CME", "O2N",
                        ],
                        [
                            (0, 1), (0, 2), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9),
                            (8, 10), (10, 11), (10, 12), (2, 13), (13, 14), (13, 19), (19, 20), (19, 21),
                        ],
                    ),
                }

    forcefield = FakeForceField()
    OpenFFBackend._reconcile_residue_templates(md.modeller, forcefield, ["NLN", "4YB"])
    OpenFFBackend._add_missing_template_hydrogens(md.modeller, forcefield, ["NLN", "4YB"])

    nln_atoms = []
    glycan_atoms = []
    glycan_bond_names = set()
    for chain in md.modeller.topology.chains():
        for residue in chain.residues():
            if chain.id == "B" and str(residue.id) == "4":
                nln_atoms = [atom.name for atom in residue.atoms()]
            if chain.id == "G" and str(residue.id) == "1":
                glycan_atoms = [atom.name for atom in residue.atoms()]
                for atom1, atom2 in md.modeller.topology.bonds():
                    if atom1.residue == residue and atom2.residue == residue:
                        glycan_bond_names.add((atom1.name, atom2.name))
                    elif atom2.residue == residue and atom1.residue == residue:
                        glycan_bond_names.add((atom2.name, atom1.name))

    assert "HD22" not in nln_atoms
    assert {"C1", "C2", "C3", "C4", "C5", "C6", "N2", "C2N", "CME", "O2N"} <= set(glycan_atoms)
    assert {"H1", "H2", "H3", "H4", "H5", "H61", "H62", "H2N"} <= set(glycan_atoms)
    assert ("C1", "C2") in glycan_bond_names or ("C2", "C1") in glycan_bond_names
    assert ("N2", "C2N") in glycan_bond_names or ("C2N", "N2") in glycan_bond_names


def test_openff_backend_can_prepare_xml_only_system_without_openff_toolkit(tmp_path, monkeypatch):
    md = _build_openmm_md(tmp_path)
    backend = OpenFFBackend(
        forcefield_files=("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml"),
        solvate=False,
    )
    parameters_folder = tmp_path / "parameters"

    class FakeForceField:
        def __init__(self, *args, **kwargs):
            self._templates = {}
            self._templateGenerators = []

        def loadFile(self, *args, **kwargs):
            return None

        def createSystem(self, **kwargs):
            return object()

    class FakeStructure:
        def save(self, path, overwrite=False):
            Path(path).write_text("")

    monkeypatch.setattr(openff_backend_module, "Molecule", None)
    monkeypatch.setattr(openff_backend_module, "ToolkitRegistry", None)
    monkeypatch.setattr(openff_backend_module, "RDKitToolkitWrapper", None)
    monkeypatch.setattr(openff_backend_module, "AmberToolsToolkitWrapper", None)
    monkeypatch.setattr("openmm.app.ForceField", FakeForceField)

    import parmed as pmd

    monkeypatch.setattr(pmd.openmm, "load_topology", lambda topology, system, xyz=None: FakeStructure())

    result = backend.prepare_model(md, str(parameters_folder))

    assert Path(result.prmtop_path).exists()
    assert Path(result.coordinates_path).exists()
