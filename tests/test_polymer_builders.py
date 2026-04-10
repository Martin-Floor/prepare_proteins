import json
import shutil
from pathlib import Path

import pytest

from prepare_proteins.MD import (
    CelluloseCrystalChain,
    CelluloseCrystalResidue,
    CelluloseCrystalSurface,
    CelluloseCrystalSurfaceResidue,
    PolymerAtomTemplate,
    PolymerBuilder,
    PolymerBuildSpec,
    PolymerRepeatUnit,
    PolymerResidueTemplate,
    PolymerTemplate,
    assign_pdb_chain_ids_by_ter,
)


def _cellulose_like_template():
    repeat_unit = PolymerRepeatUnit(
        name="glucose_beta14",
        residue_name="GLC",
        head_atom="O4",
        tail_atom="C1",
        atom_names=("C1", "O4", "C4", "O5"),
    )
    return PolymerTemplate(
        name="cellulose_beta14",
        repeat_unit=repeat_unit,
        description="Representative beta-1,4-linked glucose polymer.",
        bond_length_angstrom=1.43,
    )


def _pet_like_residue_template():
    return PolymerResidueTemplate(
        residue_name="PET",
        atoms=(
            PolymerAtomTemplate(name="C1", element="C", position=(0.000, 0.000, 0.000)),
            PolymerAtomTemplate(name="O1", element="O", position=(1.210, 0.000, 0.180)),
            PolymerAtomTemplate(name="C2", element="C", position=(2.050, 0.620, 0.920)),
            PolymerAtomTemplate(name="C3", element="C", position=(2.940, -0.080, 1.920)),
            PolymerAtomTemplate(name="O2", element="O", position=(3.980, 0.320, 2.540)),
            PolymerAtomTemplate(name="C4", element="C", position=(4.920, 0.040, 3.280)),
            PolymerAtomTemplate(name="C5", element="C", position=(5.780, 0.810, 4.060)),
            PolymerAtomTemplate(name="C6", element="C", position=(6.970, 0.470, 4.640)),
            PolymerAtomTemplate(name="C7", element="C", position=(7.840, -0.260, 5.420)),
        ),
        bonds=(
            ("C1", "O1"),
            ("O1", "C2"),
            ("C2", "C3"),
            ("C3", "O2"),
            ("O2", "C4"),
            ("C4", "C5"),
            ("C5", "C6"),
            ("C6", "C7"),
        ),
    )


def _write_pet_template_pdb(path: Path) -> Path:
    residue = _pet_like_residue_template()
    lines = []
    for serial, atom in enumerate(residue.atoms, start=1):
        lines.append(
            (
                f"HETATM{serial:5d} {atom.name.rjust(4)} {residue.residue_name:>3s} A   1    "
                f"{atom.position[0]:8.3f}{atom.position[1]:8.3f}{atom.position[2]:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}          {atom.element.upper():>2s}"
            )
        )
    atom_serials = {atom.name: serial for serial, atom in enumerate(residue.atoms, start=1)}
    for atom_a, atom_b in residue.bonds:
        lines.append(f"CONECT{atom_serials[atom_a]:5d}{atom_serials[atom_b]:5d}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_polymer_builder_registers_templates():
    template = _cellulose_like_template()
    builder = PolymerBuilder()
    builder.register_template(template)

    assert builder.available_templates() == ("cellulose_beta14",)
    assert builder.get_template("cellulose_beta14") is template


def test_polymer_builder_creates_linear_and_bundle_specs():
    builder = PolymerBuilder([_cellulose_like_template()])

    linear = builder.build_linear_spec("cellulose_beta14", n_units=8)
    assert isinstance(linear, PolymerBuildSpec)
    assert linear.arrangement == "linear"
    assert len(linear.chains) == 1
    assert linear.chains[0].chain_id == "A"
    assert linear.chains[0].n_units == 8

    bundle = builder.build_bundle_spec("cellulose_beta14", n_units=6, n_chains=3)
    assert bundle.arrangement == "bundle"
    assert [chain.chain_id for chain in bundle.chains] == ["A", "B", "C"]
    assert bundle.chains[1].translation == (0.0, 6.0, 0.0)
    assert bundle.metadata["linkage_atoms"] == ("C1", "O4")


def test_polymer_builder_creates_surface_fragment_spec():
    builder = PolymerBuilder([_cellulose_like_template()])

    surface = builder.build_surface_fragment_spec(
        "cellulose_beta14",
        n_units=5,
        n_rows=2,
        n_columns=3,
        row_spacing_angstrom=5.0,
        column_spacing_angstrom=7.5,
    )

    assert surface.arrangement == "surface_fragment"
    assert len(surface.chains) == 6
    assert surface.chains[0].translation == (0.0, 0.0, 0.0)
    assert surface.chains[1].translation == (7.5, 0.0, 0.0)
    assert surface.chains[3].translation == (0.0, 5.0, 0.0)
    assert surface.metadata["n_rows"] == 2
    assert surface.metadata["n_columns"] == 3


def test_polymer_builder_builds_pet_crystal_like_spec():
    builder = PolymerBuilder()

    spec = builder.build_polyethylene_terephthalate_crystal_spec(
        n_units=4,
        n_rows=2,
        n_columns=2,
    )

    assert spec.template_name == "polyethylene_terephthalate"
    assert spec.arrangement == "surface_fragment"
    assert len(spec.chains) == 4
    assert [chain.chain_id for chain in spec.chains] == ["A", "B", "C", "D"]
    assert spec.chains[0].translation == pytest.approx((0.0, 0.0, 0.0))
    assert spec.chains[1].translation == pytest.approx((4.56, 0.0, 0.0))
    assert spec.chains[2].translation == pytest.approx((0.0, 5.94, 5.375))
    assert spec.chains[3].translation == pytest.approx((4.56, 5.94, 5.375))
    assert spec.metadata["crystal_like"] is True
    assert spec.metadata["repeat_vector_angstrom"] == pytest.approx((0.0, 0.0, 10.75))


def test_polymer_builder_builds_pet_triclinic_crystal_spec():
    builder = PolymerBuilder()

    spec = builder.build_polyethylene_terephthalate_triclinic_crystal_spec(
        n_units=3,
        n_a=2,
        n_b=2,
    )

    assert spec.template_name == "polyethylene_terephthalate"
    assert spec.arrangement == "surface_fragment"
    assert len(spec.chains) == 4
    assert [chain.chain_id for chain in spec.chains] == ["A", "B", "C", "D"]
    assert spec.chains[0].translation == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert spec.chains[1].translation == pytest.approx((4.026241, 0.0, -2.140790), abs=1e-6)
    assert spec.chains[2].translation == pytest.approx((-2.986988, 5.058719, -0.877988), abs=1e-6)
    assert spec.metadata["repeat_vector_angstrom"] == pytest.approx((0.0, 0.0, 10.75))
    assert spec.metadata["crystal_model"] == "triclinic_lattice_parameter_approximation"
    assert spec.metadata["lattice_vectors_angstrom"]["c"] == pytest.approx((0.0, 0.0, 10.75))


def test_polymer_builder_builds_pet_wimzex01_crystal():
    builder = PolymerBuilder()

    chains = builder.build_polyethylene_terephthalate_wimzex01_crystal(
        n_units=3,
        n_a=2,
        n_b=2,
    )

    assert len(chains) == 4
    assert [chain.chain_id for chain in chains] == ["A", "B", "C", "D"]
    assert [len(chain.residues) for chain in chains] == [3, 3, 3, 3]
    assert [len(chain.bonds) for chain in chains] == [44, 44, 44, 44]
    assert chains[0].metadata["source_refcode"] == "WIMZEX01"
    assert chains[0].residues[0].cell_indices == ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0))
    assert len(chains[0].residues[0].atom_positions) == 14
    assert chains[0].residues[0].get_atom("C1")[0] == pytest.approx(0.806268, abs=1e-6)
    assert chains[1].residues[0].get_atom("C1")[0] - chains[0].residues[0].get_atom("C1")[0] == pytest.approx(
        4.026241,
        abs=1e-6,
    )
    assert chains[2].residues[0].get_atom("C1")[1] - chains[0].residues[0].get_atom("C1")[1] == pytest.approx(
        5.058719,
        abs=1e-6,
    )


def test_polymer_builder_validates_inputs():
    builder = PolymerBuilder([_cellulose_like_template()])

    with pytest.raises(ValueError, match="at least 1"):
        builder.build_linear_spec("cellulose_beta14", n_units=0)

    with pytest.raises(ValueError, match="at least 2 chains"):
        builder.build_bundle_spec("cellulose_beta14", n_units=4, n_chains=1)

    with pytest.raises(ValueError, match="Expected 2 chain IDs"):
        builder.build_bundle_spec("cellulose_beta14", n_units=4, n_chains=2, chain_ids=["A"])


def test_polymer_build_spec_writes_manifest(tmp_path):
    builder = PolymerBuilder([_cellulose_like_template()])
    spec = builder.build_surface_fragment_spec("cellulose_beta14", n_units=4, n_rows=1, n_columns=2)

    manifest = tmp_path / "surface_manifest.json"
    out_path = spec.write_manifest(manifest)

    data = json.loads(manifest.read_text())
    assert out_path == str(manifest.resolve())
    assert data["template_name"] == "cellulose_beta14"
    assert data["arrangement"] == "surface_fragment"
    assert data["chains"][1]["chain_id"] == "B"


def test_polymer_builder_generates_glycam_codes():
    builder = PolymerBuilder()

    assert builder.build_glycam_codes(1, sugar_code="G", anomer="B", linkage_position=4) == ("0GB",)
    assert builder.build_glycam_codes(3, sugar_code="G", anomer="B", linkage_position=4) == (
        "4GB",
        "4GB",
        "0GB",
    )


def test_polymer_builder_loads_residue_template_from_pdb(tmp_path):
    pdb_path = tmp_path / "glc_template.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "HETATM    1  C1  GLC A   1       0.000   0.000   0.000  1.00  0.00           C",
                "HETATM    2  O4  GLC A   1       1.400   0.000   0.000  1.00  0.00           O",
                "HETATM    3  C4  GLC A   1       2.100   1.200   0.000  1.00  0.00           C",
                "END",
            ]
        )
        + "\n"
    )

    builder = PolymerBuilder()
    residue = builder.residue_template_from_pdb(pdb_path)

    assert isinstance(residue, PolymerResidueTemplate)
    assert residue.residue_name == "GLC"
    assert [atom.name for atom in residue.atoms] == ["C1", "O4", "C4"]
    assert residue.get_atom("O4").position == pytest.approx((1.4, 0.0, 0.0))


def test_polymer_builder_writes_bundle_pdb_from_residue_template(tmp_path):
    builder = PolymerBuilder([_cellulose_like_template()])
    residue = PolymerResidueTemplate(
        residue_name="GLC",
        atoms=(
            PolymerAtomTemplate(name="C1", element="C", position=(0.0, 0.0, 0.0)),
            PolymerAtomTemplate(name="O4", element="O", position=(1.4, 0.0, 0.0)),
            PolymerAtomTemplate(name="C4", element="C", position=(2.1, 1.2, 0.0)),
        ),
    )
    spec = builder.build_bundle_spec("cellulose_beta14", n_units=3, n_chains=2)

    out_pdb = tmp_path / "bundle.pdb"
    out_path = builder.write_pdb(
        spec,
        residue_template=residue,
        output_pdb=out_pdb,
        repeat_vector_angstrom=(5.2, 0.0, 0.0),
        remark_lines=("test bundle",),
    )

    contents = out_pdb.read_text().splitlines()
    hetatm = [line for line in contents if line.startswith("HETATM")]

    assert out_path == str(out_pdb.resolve())
    assert contents[0] == "REMARK test bundle"
    assert len(hetatm) == 18
    assert " GLC A   1" in hetatm[0]
    assert " GLC A   3" in hetatm[6]
    assert " GLC B   1" in hetatm[9]
    assert hetatm[9][30:38].strip() == "0.000"
    assert hetatm[9][38:46].strip() == "6.000"
    assert contents[-1] == "END"


def test_polymer_builder_writes_pet_crystal_like_pdb(tmp_path):
    builder = PolymerBuilder()
    template_pdb = _write_pet_template_pdb(tmp_path / "pet_repeat_unit.pdb")
    output_pdb = tmp_path / "pet_crystal_like.pdb"

    summary = builder.write_polyethylene_terephthalate_crystal_pdb(
        output_pdb,
        residue_template=template_pdb,
        n_units=3,
        n_rows=2,
        n_columns=2,
        include_conect_records=True,
    )

    lines = output_pdb.read_text().splitlines()
    atom_lines = [line for line in lines if line.startswith("HETATM")]
    conect_lines = [line for line in lines if line.startswith("CONECT")]
    ter_count = sum(1 for line in lines if line == "TER")
    residue_keys = {
        (line[17:20].strip(), line[21].strip(), int(line[22:26]))
        for line in atom_lines
    }
    first_chain_a = next(line for line in atom_lines if line[21].strip() == "A")
    first_chain_c = next(line for line in atom_lines if line[21].strip() == "C")

    assert summary["pdb_path"] == str(output_pdb.resolve())
    assert summary["n_chains"] == 4
    assert summary["chain_lengths"] == (3, 3, 3, 3)
    assert summary["residue_mode"] == "by_chain"
    assert summary["include_ter_records"] is True
    assert summary["include_conect_records"] is True
    assert len(atom_lines) == 108
    assert len(conect_lines) == 96
    assert len(residue_keys) == 4
    assert ter_count == 4
    assert lines[-1] == "END"
    assert first_chain_a[46:54].strip() == "0.000"
    assert first_chain_c[46:54].strip() == "5.375"


def test_polymer_builder_writes_pet_triclinic_crystal_pdb(tmp_path):
    builder = PolymerBuilder()
    template_pdb = _write_pet_template_pdb(tmp_path / "pet_repeat_unit.pdb")
    output_pdb = tmp_path / "pet_triclinic_slab.pdb"

    summary = builder.write_polyethylene_terephthalate_triclinic_crystal_pdb(
        output_pdb,
        residue_template=template_pdb,
        n_units=2,
        n_a=2,
        n_b=2,
        include_conect_records=True,
    )

    lines = output_pdb.read_text().splitlines()
    atom_lines = [line for line in lines if line.startswith("HETATM")]
    conect_lines = [line for line in lines if line.startswith("CONECT")]
    first_chain_c = next(line for line in atom_lines if line[21].strip() == "C")

    assert summary["pdb_path"] == str(output_pdb.resolve())
    assert summary["n_chains"] == 4
    assert summary["chain_lengths"] == (2, 2, 2, 2)
    assert summary["n_a"] == 2
    assert summary["n_b"] == 2
    assert summary["a_angstrom"] == pytest.approx(4.56)
    assert summary["gamma_degrees"] == pytest.approx(112.0)
    assert lines[0].startswith("CRYST1")
    assert "  9.120" in lines[0]
    assert " 11.880" in lines[0]
    assert " 21.500" in lines[0]
    assert len(atom_lines) == 72
    assert len(conect_lines) == 64
    assert float(first_chain_c[30:38]) == pytest.approx(-2.987, abs=1e-3)
    assert float(first_chain_c[38:46]) == pytest.approx(5.059, abs=1e-3)
    assert float(first_chain_c[46:54]) == pytest.approx(-0.878, abs=1e-3)
    assert lines[-1] == "END"


def test_polymer_builder_writes_pet_wimzex01_pdb(tmp_path):
    builder = PolymerBuilder()
    output_pdb = tmp_path / "pet_wimzex01_slab.pdb"

    summary = builder.write_polyethylene_terephthalate_wimzex01_pdb(
        output_pdb,
        n_units=3,
        n_a=2,
        n_b=2,
        include_conect_records=True,
    )

    lines = output_pdb.read_text().splitlines()
    atom_lines = [line for line in lines if line.startswith("HETATM")]
    conect_lines = [line for line in lines if line.startswith("CONECT")]
    ter_count = sum(1 for line in lines if line == "TER")

    assert summary["pdb_path"] == str(output_pdb.resolve())
    assert summary["n_chains"] == 4
    assert summary["chain_lengths"] == (3, 3, 3, 3)
    assert summary["n_atoms"] == 168
    assert summary["unit_cells_a"] == 3
    assert summary["unit_cells_b"] == 3
    assert summary["crystal_model"] == "WIMZEX01_cif_basis"
    assert lines[0].startswith("CRYST1")
    assert " 13.680" in lines[0]
    assert " 17.820" in lines[0]
    assert " 32.250" in lines[0]
    assert any("WIMZEX01" in line for line in lines[:6])
    assert len(atom_lines) == 168
    assert len(conect_lines) == 176
    assert ter_count == 4
    assert lines[-1] == "END"


def test_assign_pdb_chain_ids_by_ter(tmp_path):
    input_pdb = tmp_path / "segments.pdb"
    input_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  C1  GLC     1       0.000   0.000   0.000  1.00  0.00           C",
                "ATOM      2  O1  GLC     1       1.200   0.000   0.000  1.00  0.00           O",
                "TER",
                "ATOM      3  C1  GLC     2       0.000   2.000   0.000  1.00  0.00           C",
                "ATOM      4  O1  GLC     2       1.200   2.000   0.000  1.00  0.00           O",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    output_pdb = tmp_path / "segments_chain_ids.pdb"
    out_path = assign_pdb_chain_ids_by_ter(input_pdb, output_pdb, chain_ids=("A", "B"))

    text = output_pdb.read_text().splitlines()
    assert out_path == str(output_pdb.resolve())
    assert text[0][21] == "A"
    assert text[3][21] == "B"
    assert text[-1] == "END"


def test_polymer_builder_builds_cellulose_ibeta_crystal():
    builder = PolymerBuilder()
    chains = builder.build_cellulose_ibeta_crystal(n_a=2, n_b=2, n_cellobiose=2)

    assert len(chains) == 5
    assert all(isinstance(chain, CelluloseCrystalChain) for chain in chains)
    assert [chain.chain_id for chain in chains] == ["A", "B", "C", "D", "E"]
    assert [len(chain.residues) for chain in chains] == [4, 4, 4, 4, 4]
    assert all(isinstance(chain.residues[0], CelluloseCrystalResidue) for chain in chains)
    assert chains[0].residues[0].basis_index == 3
    assert chains[0].residues[-1].basis_index == 1
    assert chains[2].residues[0].basis_index == 4
    assert chains[0].residues[0].get_atom("C1") == pytest.approx((-0.148, 0.342, 16.019), abs=2e-3)


def test_polymer_builder_describes_cellulose_ibeta_surface():
    builder = PolymerBuilder()
    chains = builder.build_cellulose_ibeta_crystal(n_a=3, n_b=4, n_cellobiose=2)

    surface = builder.describe_cellulose_ibeta_surface(chains, exposed_face="bc", side="max")

    assert isinstance(surface, CelluloseCrystalSurface)
    assert surface.face_name == "bc"
    assert surface.side == "max"
    assert surface.normal_axis == "a"
    assert surface.normal_vector == pytest.approx((1.0, 0.0, 0.0))
    assert surface.surface_axes == ("b", "c")
    assert surface.thickness_angstrom == pytest.approx(surface.bounding_box_size_angstrom[0])
    assert surface.surface_span_angstrom == pytest.approx(
        (surface.bounding_box_size_angstrom[1], surface.bounding_box_size_angstrom[2])
    )
    assert all(isinstance(ref, CelluloseCrystalSurfaceResidue) for ref in surface.residue_refs)
    assert all(ref.cell_index[0] == 3 for ref in surface.residue_refs)


def test_polymer_builder_selects_central_surface_patch():
    builder = PolymerBuilder()
    chains = builder.build_cellulose_ibeta_crystal(n_a=3, n_b=4, n_cellobiose=3)
    surface = builder.describe_cellulose_ibeta_surface(chains, exposed_face="ac", side="min")

    patch = builder.select_cellulose_ibeta_surface_patch(surface, max_residues=6)

    assert len(patch) == 6
    assert all(ref in surface.residue_refs for ref in patch)
    assert all(ref.cell_index[1] == 1 for ref in patch)

    axis_to_index = {"a": 0, "b": 1, "c": 2}
    surface_axes = tuple(axis_to_index[axis] for axis in surface.surface_axes)
    center = surface.surface_center

    def in_plane_distance_squared(ref):
        return sum((ref.center[index] - center[index]) ** 2 for index in surface_axes)

    surface_distances = sorted(in_plane_distance_squared(ref) for ref in surface.residue_refs)
    patch_distances = sorted(in_plane_distance_squared(ref) for ref in patch)
    assert patch_distances == pytest.approx(surface_distances[:6])


def test_polymer_builder_writes_glycam_tleap_bundle_script(tmp_path):
    builder = PolymerBuilder([_cellulose_like_template()])
    spec = builder.build_bundle_spec("cellulose_beta14", n_units=3, n_chains=2)

    script_path = builder.write_glycam_tleap_bundle_script(
        spec,
        tmp_path / "cellulose_bundle",
        sugar_code="G",
        anomer="B",
        linkage_position=4,
        reducing_end_cap="ROH",
    )

    text = Path(script_path).read_text()
    assert "source leaprc.GLYCAM_06j-1" in text
    assert "sequence { ROH 4GB 4GB 0GB }" in text
    assert "translate chain_02 { 0.000 6.000 0.000 }" in text
    assert "saveamberparm polymer_bundle" in text


def test_polymer_builder_writes_solvated_glycam_tleap_bundle_script(tmp_path):
    builder = PolymerBuilder([_cellulose_like_template()])
    spec = builder.build_bundle_spec("cellulose_beta14", n_units=3, n_chains=2)

    script_path = builder.write_glycam_tleap_bundle_script(
        spec,
        tmp_path / "cellulose_bundle_solvated",
        sugar_code="G",
        anomer="B",
        linkage_position=4,
        reducing_end_cap="ROH",
        solvate=True,
        solvent_box="TIP3PBOX",
        solvent_buffer_angstrom=12.0,
        neutralize=True,
    )

    text = Path(script_path).read_text()
    assert "solvateBox polymer_bundle TIP3PBOX 12.000" in text
    assert "addIonsRand polymer_bundle Na+ 0" in text
    assert "addIonsRand polymer_bundle Cl- 0" in text


@pytest.mark.skipif(shutil.which("tleap") is None, reason="tleap is not available.")
def test_polymer_builder_builds_glycam_bundle_with_tleap(tmp_path):
    builder = PolymerBuilder([_cellulose_like_template()])
    spec = builder.build_bundle_spec("cellulose_beta14", n_units=3, n_chains=2, chain_spacing_angstrom=8.0)

    result = builder.build_glycam_bundle(
        spec,
        tmp_path / "cellulose_bundle",
        sugar_code="G",
        anomer="B",
        linkage_position=4,
        reducing_end_cap="ROH",
    )

    assert result["returncode"] == 0
    assert Path(result["script_path"]).is_file()
    assert Path(result["pdb_path"]).is_file()
    assert Path(result["prmtop_path"]).is_file()
    assert Path(result["rst7_path"]).is_file()

    pdb_text = Path(result["pdb_path"]).read_text()
    assert " ROH " in pdb_text
    assert " 4GB " in pdb_text
    assert " 0GB " in pdb_text


@pytest.mark.skipif(shutil.which("tleap") is None, reason="tleap is not available.")
def test_polymer_builder_writes_cellulose_ibeta_glycam_pdb(tmp_path):
    builder = PolymerBuilder()
    output_pdb = tmp_path / "cellulose_ibeta_crystal.pdb"

    summary = builder.write_cellulose_ibeta_glycam_pdb(
        output_pdb,
        n_a=2,
        n_b=2,
        n_cellobiose=1,
    )

    pdb_text = output_pdb.read_text()
    assert summary["n_chains"] == 5
    assert summary["chain_lengths"] == (2, 2, 2, 2, 2)
    assert summary["generated_template"] is True
    residue_keys = {
        (line[17:20].strip(), line[21].strip(), int(line[22:26]))
        for line in pdb_text.splitlines()
        if line.startswith(("ATOM", "HETATM"))
    }
    residue_names = [key[0] for key in residue_keys]
    assert residue_names.count("ROH") == 5
    assert residue_names.count("4GB") == 5
    assert residue_names.count("0GB") == 5

    script_path = tmp_path / "check_cellulose_ibeta.tleap.in"
    script_path.write_text(
        "\n".join(
            [
                "source leaprc.GLYCAM_06j-1",
                f'mol = loadpdb "{output_pdb.resolve()}"',
                "check mol",
                "quit",
            ]
        )
        + "\n"
    )
    result = builder.run_tleap(script_path)
    assert result.returncode == 0
    assert "Errors = 0" in result.stdout


@pytest.mark.skipif(shutil.which("tleap") is None, reason="tleap is not available.")
def test_polymer_builder_builds_solvated_glycam_bundle_with_tleap(tmp_path):
    builder = PolymerBuilder([_cellulose_like_template()])
    spec = builder.build_bundle_spec("cellulose_beta14", n_units=3, n_chains=2, chain_spacing_angstrom=8.0)

    result = builder.build_glycam_bundle(
        spec,
        tmp_path / "cellulose_bundle_solvated",
        sugar_code="G",
        anomer="B",
        linkage_position=4,
        reducing_end_cap="ROH",
        solvate=True,
        solvent_box="TIP3PBOX",
        solvent_buffer_angstrom=12.0,
    )

    assert result["returncode"] == 0
    assert Path(result["pdb_path"]).is_file()

    pdb_text = Path(result["pdb_path"]).read_text()
    assert "CRYST1" in pdb_text
    assert " WAT " in pdb_text
