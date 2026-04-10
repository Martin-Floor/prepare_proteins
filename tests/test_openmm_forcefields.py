from pathlib import Path

import pytest

import prepare_proteins.MD.openmm_setup as openmm_setup
from prepare_proteins.MD import PolymerBuilder


MODEL_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  OXT ALA A   1      12.410  10.801   2.000  1.00 20.00           O
TER
END
"""


pytestmark = pytest.mark.skipif(
    not getattr(openmm_setup, "OPENMM_AVAILABLE", False),
    reason="OpenMM is not available in the test environment.",
)


def _write_model(tmp_path):
    model_path = tmp_path / "model.pdb"
    model_path.write_text(MODEL_PDB)
    return model_path


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
                "CONECT    1    2",
                "CONECT    2    3",
                "CONECT    3    4",
                "CONECT    4    5",
                "CONECT    5    6",
                "CONECT    6    7",
                "CONECT    7    8",
                "CONECT    8    9",
                "END",
            ]
        )
        + "\n"
    )
    return path


def _write_minimal_pet_ffxml(path: Path) -> Path:
    path.write_text(
        """<ForceField>
 <AtomTypes>
  <Type name="PET-C" class="PETC" element="C" mass="12.011"/>
  <Type name="PET-O" class="PETO" element="O" mass="15.999"/>
 </AtomTypes>
 <Residues>
  <Residue name="PET">
   <Atom name="C1" type="PET-C" charge="0.0"/>
   <Atom name="O1" type="PET-O" charge="0.0"/>
   <Atom name="C2" type="PET-C" charge="0.0"/>
   <Atom name="C3" type="PET-C" charge="0.0"/>
   <Atom name="O2" type="PET-O" charge="0.0"/>
   <Atom name="C4" type="PET-C" charge="0.0"/>
   <Atom name="C5" type="PET-C" charge="0.0"/>
   <Atom name="C6" type="PET-C" charge="0.0"/>
   <Atom name="C7" type="PET-C" charge="0.0"/>
   <Bond atomName1="C1" atomName2="O1"/>
   <Bond atomName1="O1" atomName2="C2"/>
   <Bond atomName1="C2" atomName2="C3"/>
   <Bond atomName1="C3" atomName2="O2"/>
   <Bond atomName1="O2" atomName2="C4"/>
   <Bond atomName1="C4" atomName2="C5"/>
   <Bond atomName1="C5" atomName2="C6"/>
   <Bond atomName1="C6" atomName2="C7"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="PETC" class2="PETC" length="0.152" k="265265.6"/>
  <Bond class1="PETC" class2="PETO" length="0.134" k="376560.0"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="PETC" class2="PETC" class3="PETC" angle="2.09439510239" k="418.4"/>
  <Angle class1="PETC" class2="PETC" class3="PETO" angle="2.09439510239" k="418.4"/>
  <Angle class1="PETO" class2="PETC" class3="PETC" angle="2.09439510239" k="418.4"/>
  <Angle class1="PETC" class2="PETO" class3="PETC" angle="2.09439510239" k="502.08"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="PETC" class2="PETC" class3="PETC" class4="PETC" periodicity1="3" phase1="0.0" k1="1.0"/>
  <Proper class1="PETC" class2="PETC" class3="PETC" class4="PETO" periodicity1="3" phase1="0.0" k1="1.0"/>
  <Proper class1="PETO" class2="PETC" class3="PETC" class4="PETC" periodicity1="3" phase1="0.0" k1="1.0"/>
  <Proper class1="PETC" class2="PETO" class3="PETC" class4="PETC" periodicity1="3" phase1="0.0" k1="1.0"/>
  <Proper class1="PETC" class2="PETC" class3="PETO" class4="PETC" periodicity1="3" phase1="0.0" k1="1.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="PET-C" charge="0.0" sigma="0.34" epsilon="0.276144"/>
  <Atom type="PET-O" charge="0.0" sigma="0.30" epsilon="0.2100"/>
 </NonbondedForce>
</ForceField>
"""
    )
    return path


def _write_pet_validation_model(tmp_path):
    builder = PolymerBuilder()
    template_pdb = _write_pet_template_pdb(tmp_path / "pet_repeat_unit.pdb")
    model_path = tmp_path / "pet_validation_model.pdb"
    builder.write_polyethylene_terephthalate_crystal_pdb(
        model_path,
        residue_template=template_pdb,
        n_units=1,
        n_rows=2,
        n_columns=2,
        residue_mode="by_unit",
        include_ter_records=True,
        include_conect_records=True,
    )
    return model_path


def test_openmm_setup_supports_charmm36_preset(tmp_path):
    md_obj = openmm_setup.openmm_md(str(_write_model(tmp_path)))
    md_obj.setUpFF("charmm36")

    assert md_obj.ff_name == "charmm36"
    assert md_obj.ff_files == ["charmm36.xml"]
    assert md_obj.forcefield is not None


def test_openmm_setup_supports_explicit_xml_lists(tmp_path):
    md_obj = openmm_setup.openmm_md(str(_write_model(tmp_path)))
    md_obj.setUpFF(["amber14-all.xml", "amber14/tip3pfb.xml"])

    assert md_obj.ff_name == "custom"
    assert md_obj.ff_files == ["amber14-all.xml", "amber14/tip3pfb.xml"]
    assert md_obj.forcefield is not None


def test_openmm_setup_rejects_empty_custom_forcefield_lists(tmp_path):
    md_obj = openmm_setup.openmm_md(str(_write_model(tmp_path)))

    with pytest.raises(ValueError, match="must not be empty"):
        md_obj.setUpFF([])


def test_openmm_setup_supports_custom_pet_ffxml(tmp_path):
    model_path = _write_pet_validation_model(tmp_path)
    ffxml_path = _write_minimal_pet_ffxml(tmp_path / "PET.ffxml")

    md_obj = openmm_setup.openmm_md(str(model_path))
    md_obj.setUpFF([str(ffxml_path)])

    system = md_obj.forcefield.createSystem(md_obj.pdb.topology)

    assert md_obj.ff_name == "custom"
    assert md_obj.ff_files == [str(ffxml_path)]
    assert system.getNumParticles() == 36
