from types import SimpleNamespace

import numpy as np
from collections import defaultdict
import pytest

import prepare_proteins.MD.openmm_setup as openmm_setup


class _FakeChain:
    def __init__(self, chain_id):
        self.id = chain_id
        self._residues = []

    def residues(self):
        return iter(self._residues)


class _FakeResidue:
    def __init__(self, name, residue_id, chain):
        self.name = name
        self.id = str(residue_id)
        self.chain = chain
        self._atoms = []
        chain._residues.append(self)

    def atoms(self):
        return iter(self._atoms)


class _FakeAtom:
    def __init__(self, index, name, residue, element_name):
        self.index = index
        self.name = name
        self.residue = residue
        self.element = SimpleNamespace(name=element_name.lower())
        residue._atoms.append(self)


class _FakeTopology:
    def __init__(self, chains):
        self._chains = chains

    def chains(self):
        return iter(self._chains)


def _build_embedded_heme_topology():
    chain = _FakeChain("A")
    cys = _FakeResidue("CYS", 463, chain)
    hem = _FakeResidue("HEM", 601, chain)
    lfv = _FakeResidue("LFV", 602, chain)

    atoms = [
        _FakeAtom(0, "SG", cys, "S"),
        _FakeAtom(1, "FE", hem, "FE"),
        _FakeAtom(2, "NA", hem, "N"),
        _FakeAtom(3, "NAW", lfv, "N"),
    ]
    positions = [
        np.array([0.233, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.10, 0.0]),
        np.array([0.202, 0.0, 0.0]),
    ]
    return _FakeTopology([chain]), positions, atoms


def _build_two_site_embedded_heme_topology():
    chain_a = _FakeChain("A")
    cys_a = _FakeResidue("CYS", 463, chain_a)
    hem_a = _FakeResidue("HEM", 601, chain_a)
    lfv_a = _FakeResidue("LFV", 602, chain_a)

    chain_b = _FakeChain("B")
    cys_b = _FakeResidue("CYS", 463, chain_b)
    hem_b = _FakeResidue("HEM", 601, chain_b)
    lfv_b = _FakeResidue("LFV", 602, chain_b)

    atoms = [
        _FakeAtom(0, "SG", cys_a, "S"),
        _FakeAtom(1, "FE", hem_a, "FE"),
        _FakeAtom(2, "NA", hem_a, "N"),
        _FakeAtom(3, "NAW", lfv_a, "N"),
        _FakeAtom(4, "SG", cys_b, "S"),
        _FakeAtom(5, "FE", hem_b, "FE"),
        _FakeAtom(6, "NA", hem_b, "N"),
        _FakeAtom(7, "NAW", lfv_b, "N"),
    ]
    positions = [
        np.array([0.233, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.10, 0.0]),
        np.array([0.202, 0.0, 0.0]),
        np.array([1.233, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.10, 0.0]),
        np.array([1.202, 0.0, 0.0]),
    ]
    return _FakeTopology([chain_a, chain_b]), positions, atoms


def _build_single_atom_metal_topology():
    chain = _FakeChain("A")
    zn = _FakeResidue("ZN", 700, chain)
    lig = _FakeResidue("LIG", 701, chain)

    _FakeAtom(0, "ZN", zn, "ZN")
    _FakeAtom(1, "N1", lig, "N")
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.210, 0.0, 0.0]),
    ]
    return _FakeTopology([chain]), positions


def test_resolve_mcpb_config_handles_embedded_heme_like_metals():
    topology, positions, _atoms = _build_embedded_heme_topology()
    mcpb_config = {
        "metal": {"chain": "A", "resid": 601, "resname": "HEM", "atom": "FE"},
        "coordinating_atoms": [
            {"chain": "A", "resid": 463, "resname": "CYS", "atom": "SG", "role": "proximal"},
            {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
        ],
    }

    resolved = openmm_setup._resolve_mcpb_config(topology, positions, mcpb_config)

    assert resolved["contains_embedded_metal"] is True
    assert resolved["legacy_compatible"] is False
    assert len(resolved["sites"]) == 1

    site = resolved["sites"][0]
    assert site["metal"]["residue_name"] == "HEM"
    assert site["metal"]["atom_name"] == "FE"
    assert site["metal"]["is_embedded_metal"] is True
    assert [atom["atom_name"] for atom in site["coordinating_atoms"]] == ["SG", "NAW"]
    assert round(site["coordinating_atoms"][0]["distance_to_metal_angstrom"], 2) == 2.33
    assert round(site["coordinating_atoms"][1]["distance_to_metal_angstrom"], 2) == 2.02


def test_resolve_mcpb_config_accepts_cym_against_cys_in_structure():
    topology, positions, _atoms = _build_embedded_heme_topology()
    mcpb_config = {
        "metal": {"chain": "A", "resid": 601, "resname": "HEM", "atom": "FE"},
        "coordinating_atoms": [
            {"chain": "A", "resid": 463, "resname": "CYM", "atom": "SG", "role": "proximal"},
            {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
        ],
    }

    resolved = openmm_setup._resolve_mcpb_config(topology, positions, mcpb_config)
    site = resolved["sites"][0]
    proximal = site["coordinating_atoms"][0]
    selection = openmm_setup._collect_mcpb_nonprotein_residue_selection(resolved)

    assert proximal["residue_name"] == "CYM"
    assert proximal["structure_residue_name"] == "CYS"
    assert proximal["is_protein_residue"] is True
    assert selection["keys"] == [("A", 601, "HEM"), ("A", 602, "LFV")]


def test_legacy_metal_ligand_conversion_supports_single_atom_metals():
    topology, positions = _build_single_atom_metal_topology()
    mcpb_config = {
        "metal": {"chain": "A", "resid": 700, "resname": "ZN", "atom": "ZN"},
        "coordinating_atoms": [
            {"chain": "A", "resid": 701, "resname": "LIG", "atom": "N1", "role": "ligand"}
        ],
    }

    resolved = openmm_setup._resolve_mcpb_config(topology, positions, mcpb_config)
    legacy = openmm_setup._legacy_metal_ligand_from_mcpb_config(resolved)

    assert resolved["contains_embedded_metal"] is False
    assert resolved["legacy_compatible"] is True
    assert legacy == {"LIG": ["ZN"]}


def test_collect_mcpb_nonprotein_residue_selection_tracks_exact_site_residues():
    topology, positions, _atoms = _build_embedded_heme_topology()
    resolved = openmm_setup._resolve_mcpb_config(
        topology,
        positions,
        {
            "metal": {"chain": "A", "resid": 601, "resname": "HEM", "atom": "FE"},
            "coordinating_atoms": [
                {"chain": "A", "resid": 463, "resname": "CYS", "atom": "SG", "role": "proximal"},
                {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
            ],
        },
    )

    selection = openmm_setup._collect_mcpb_nonprotein_residue_selection(resolved)

    assert selection["keys"] == [("A", 601, "HEM"), ("A", 602, "LFV")]
    assert selection["names"] == ["HEM", "LFV"]
    assert selection["embedded_residue_names"] == ["HEM"]
    assert selection["duplicate_names"] == {}


def test_collect_mcpb_nonprotein_residue_selection_reports_duplicate_residue_names():
    topology, positions, _atoms = _build_two_site_embedded_heme_topology()
    resolved = openmm_setup._resolve_mcpb_config(
        topology,
        positions,
        {
            "sites": [
                {
                    "site_id": "site_a",
                    "metal": {"chain": "A", "resid": 601, "resname": "HEM", "atom": "FE"},
                    "coordinating_atoms": [
                        {"chain": "A", "resid": 463, "resname": "CYS", "atom": "SG", "role": "proximal"},
                        {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
                    ],
                },
                {
                    "site_id": "site_b",
                    "metal": {"chain": "B", "resid": 601, "resname": "HEM", "atom": "FE"},
                    "coordinating_atoms": [
                        {"chain": "B", "resid": 463, "resname": "CYS", "atom": "SG", "role": "proximal"},
                        {"chain": "B", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
                    ],
                },
            ]
        },
    )

    selection = openmm_setup._collect_mcpb_nonprotein_residue_selection(resolved)

    assert sorted(selection["duplicate_names"]) == ["HEM", "LFV"]
    assert [entry["chain_id"] for entry in selection["duplicate_names"]["HEM"]] == ["A", "B"]


def _write_parameter_file(path, contents):
    path.write_text(contents)
    return str(path)


def test_build_site_driven_mcpb_inputs_handles_embedded_heme_like_sites(tmp_path):
    topology, positions, _atoms = _build_embedded_heme_topology()
    resolved = openmm_setup._resolve_mcpb_config(
        topology,
        positions,
        {
            "metal": {"chain": "A", "resid": 601, "resname": "HEM", "atom": "FE"},
            "coordinating_atoms": [
                {"chain": "A", "resid": 463, "resname": "CYS", "atom": "SG", "role": "proximal"},
                {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
            ],
        },
    )

    parameters_folder = tmp_path / "parameters"
    parameters_folder.mkdir()
    hem_folder = tmp_path / "HEM_parameters"
    hem_folder.mkdir()
    lfv_folder = tmp_path / "LFV_parameters"
    lfv_folder.mkdir()

    hem_mol2 = _write_parameter_file(hem_folder / "HEM.mol2", "@<TRIPOS>ATOM\n")
    hem_frcmod = _write_parameter_file(hem_folder / "HEM.frcmod", "MASS\n")
    lfv_mol2 = _write_parameter_file(lfv_folder / "LFV.mol2", "@<TRIPOS>ATOM\n")
    lfv_frcmod = _write_parameter_file(lfv_folder / "LFV.frcmod", "MASS\n")

    spec = openmm_setup._build_site_driven_mcpb_inputs(
        resolved,
        {"HEM": str(hem_folder), "LFV": str(lfv_folder)},
        defaultdict(list, {"HEM": [hem_mol2], "LFV": [lfv_mol2]}),
        defaultdict(list, {"HEM": [hem_frcmod], "LFV": [lfv_frcmod]}),
        str(parameters_folder),
        "heme_site",
    )

    assert spec["group_name"] == "heme_site"
    assert spec["ion_ids"] == [2]
    assert spec["ion_mol2files"] == []
    assert spec["naa_mol2files"] == ["HEM.mol2", "LFV.mol2"]
    assert spec["frcmod_files"] == ["HEM.frcmod", "LFV.frcmod"]
    assert spec["add_bonded_pairs"] == [(2, 1), (2, 4)]
    assert spec["site_residue_names"] == ["HEM", "LFV"]
    assert spec["staged_residue_names"] == ["HEM", "LFV"]


def test_build_site_driven_mcpb_inputs_keeps_single_atom_metals_as_ion_mol2(tmp_path):
    topology, positions = _build_single_atom_metal_topology()
    resolved = openmm_setup._resolve_mcpb_config(
        topology,
        positions,
        {
            "metal": {"chain": "A", "resid": 700, "resname": "ZN", "atom": "ZN"},
            "coordinating_atoms": [
                {"chain": "A", "resid": 701, "resname": "LIG", "atom": "N1", "role": "ligand"}
            ],
        },
    )

    parameters_folder = tmp_path / "parameters"
    parameters_folder.mkdir()
    zn_folder = tmp_path / "ZN_parameters"
    zn_folder.mkdir()
    lig_folder = tmp_path / "LIG_parameters"
    lig_folder.mkdir()

    zn_mol2 = _write_parameter_file(zn_folder / "ZN.mol2", "@<TRIPOS>ATOM\n")
    lig_mol2 = _write_parameter_file(lig_folder / "LIG.mol2", "@<TRIPOS>ATOM\n")
    lig_frcmod = _write_parameter_file(lig_folder / "LIG.frcmod", "MASS\n")

    spec = openmm_setup._build_site_driven_mcpb_inputs(
        resolved,
        {"ZN": str(zn_folder), "LIG": str(lig_folder)},
        defaultdict(list, {"ZN": [zn_mol2], "LIG": [lig_mol2]}),
        defaultdict(list, {"LIG": [lig_frcmod]}),
        str(parameters_folder),
        "zn_site",
    )

    assert spec["group_name"] == "zn_site"
    assert spec["ion_ids"] == [1]
    assert spec["ion_mol2files"] == ["ZN.mol2"]
    assert spec["naa_mol2files"] == ["LIG.mol2"]
    assert spec["frcmod_files"] == ["LIG.frcmod"]
    assert spec["add_bonded_pairs"] == [(1, 2)]
    assert spec["site_residue_names"] == ["LIG"]
    assert spec["staged_residue_names"] == ["ZN", "LIG"]


def test_parameterize_pdb_ligands_requires_mcpb_site_when_metal_ligand_is_used(tmp_path):
    pdb_path = tmp_path / "zn_site.pdb"
    pdb_path.write_text(
        """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.050  11.780   2.400  1.00 20.00           C
ATOM      4  O   ALA A   1      12.410  10.801   2.000  1.00 20.00           O
HETATM    5 ZN   ZN  A 700      15.000  10.000   0.000  1.00 20.00          ZN
HETATM    6  N1  LIG A 701      16.900  10.000   0.000  1.00 20.00           N
END
"""
    )

    md = openmm_setup.openmm_md(str(pdb_path))

    with pytest.raises(ValueError, match="requires an explicit mcpb_site definition"):
        md.parameterizePDBLigands(
            str(tmp_path / "parameters"),
            metal_ligand={"LIG": ["ZN"]},
            build_full_system=False,
        )


def test_prepare_mcpb_site_writes_split_embedded_site_artifacts(tmp_path):
    pdb_path = tmp_path / "heme_site.pdb"
    pdb_path.write_text(
        """\
ATOM      1  N   CYS A 463       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  CYS A 463       1.200   0.000   0.000  1.00 20.00           C
ATOM      3  C   CYS A 463       1.800   1.200   0.000  1.00 20.00           C
ATOM      4  O   CYS A 463       1.200   2.200   0.000  1.00 20.00           O
ATOM      5  CB  CYS A 463       1.800  -0.900   1.200  1.00 20.00           C
ATOM      6  SG  CYS A 463       3.500  -0.500   1.200  1.00 20.00           S
HETATM    7  FE  HEM A 601       5.200  -0.500   1.200  1.00 20.00          FE
HETATM    8  NA  HEM A 601       5.700   0.800   1.200  1.00 20.00           N
HETATM    9  C1  HEM A 601       6.700   1.100   1.200  1.00 20.00           C
HETATM   10  NAW LFV A 602       5.200  -2.500   1.200  1.00 20.00           N
HETATM   11  C2  LFV A 602       6.000  -3.500   1.200  1.00 20.00           C
END
"""
    )

    md = openmm_setup.openmm_md(str(pdb_path))
    manifest = md.prepareMCPBSite(
        str(tmp_path / "mcpb"),
        {
            "site_id": "cyp51_a",
            "group_name": "cyp51_a",
            "metal": {
                "chain": "A",
                "resid": 601,
                "resname": "HEM",
                "atom": "FE",
                "formal_charge": 3,
            },
            "coordinating_atoms": [
                {"chain": "A", "resid": 463, "resname": "CYS", "atom": "SG", "role": "proximal"},
                {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
            ],
            "fragments": [
                {"chain": "A", "resid": 601, "resname": "HEM", "net_charge": -2, "exclude_atoms": ["FE"]},
                {"chain": "A", "resid": 602, "resname": "LFV", "net_charge": 0},
            ],
            "qm": {
                "small_model_charge": 0,
                "small_model_spin": 1,
                "large_model_charge": 0,
                "large_model_spin": 1,
            },
        },
    )

    site_dir = tmp_path / "mcpb" / "cyp51_a"
    assert manifest["site_directory"] == str(site_dir)
    assert (site_dir / "cyp51_a.in").exists()
    assert (site_dir / "prepare_fragments.sh").exists()
    assert (site_dir / "site_manifest.json").exists()
    assert (site_dir / "fragments" / "HEM.pdb").exists()
    assert (site_dir / "fragments" / "LFV.pdb").exists()
    assert (site_dir / "fragments" / "FE.pdb").exists()

    hem_pdb = (site_dir / "fragments" / "HEM.pdb").read_text()
    fe_pdb = (site_dir / "fragments" / "FE.pdb").read_text()
    input_file = (site_dir / "cyp51_a.in").read_text()
    split_model = (site_dir / "cyp51_a_mcpb_model.pdb").read_text()

    assert " FE " not in hem_pdb
    assert " FE " in fe_pdb
    assert "ion_mol2files FE.mol2" in input_file
    assert "naa_mol2files HEM.mol2 LFV.mol2" in input_file
    assert "frcmod_files HEM.frcmod LFV.frcmod" in input_file
    assert "smmodel_chg 0" in input_file
    assert "lgmodel_spin 1" in input_file
    assert "HEM A" in split_model
    assert "LFV A" in split_model
    assert any(
        line.startswith("HETATM")
        and line[12:16].strip() == "FE"
        and line[17:20].strip() == "FE"
        for line in split_model.splitlines()
    )
    atom_lines = [line for line in split_model.splitlines() if line.startswith(("ATOM", "HETATM"))]
    residue_serials = []
    previous_residue_key = None
    for line in atom_lines:
        residue_key = (line[21], line[22:26], line[17:20])
        if residue_key != previous_residue_key:
            residue_serials.append(int(line[22:26]))
            previous_residue_key = residue_key
    assert len(residue_serials) == len(set(residue_serials))
    fe_serial = next(int(line[6:11]) for line in atom_lines if line[12:16].strip() == "FE" and line[17:20].strip() == "FE")
    sg_serial = next(int(line[6:11]) for line in atom_lines if line[12:16].strip() == "SG" and line[17:20].strip() == "CYS")
    naw_serial = next(int(line[6:11]) for line in atom_lines if line[12:16].strip() == "NAW" and line[17:20].strip() == "LFV")
    assert manifest["ion_ids"] == [fe_serial]
    assert manifest["add_bonded_pairs"] == [(fe_serial, sg_serial), (fe_serial, naw_serial)]
    assert f"ion_ids {fe_serial}" in input_file
    assert f"add_bonded_pairs {fe_serial}-{sg_serial} {fe_serial}-{naw_serial}" in input_file
    assert manifest["metal"]["formal_charge"] == 3
    assert str(manifest["metal"]["split_residue_id"]) != "601"


def test_prepare_mcpb_site_accepts_cym_against_cys_in_structure(tmp_path):
    pdb_path = tmp_path / "heme_site_cym_alias.pdb"
    pdb_path.write_text(
        """\
ATOM      1  N   CYS A 463       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  CYS A 463       1.200   0.000   0.000  1.00 20.00           C
ATOM      3  C   CYS A 463       1.800   1.200   0.000  1.00 20.00           C
ATOM      4  O   CYS A 463       1.200   2.200   0.000  1.00 20.00           O
ATOM      5  CB  CYS A 463       1.800  -0.900   1.200  1.00 20.00           C
ATOM      6  SG  CYS A 463       3.500  -0.500   1.200  1.00 20.00           S
HETATM    7  FE  HEM A 601       5.200  -0.500   1.200  1.00 20.00          FE
HETATM    8  NA  HEM A 601       5.700   0.800   1.200  1.00 20.00           N
HETATM    9  C1  HEM A 601       6.700   1.100   1.200  1.00 20.00           C
HETATM   10  NAW LFV A 602       5.200  -2.500   1.200  1.00 20.00           N
HETATM   11  C2  LFV A 602       6.000  -3.500   1.200  1.00 20.00           C
END
"""
    )

    md = openmm_setup.openmm_md(str(pdb_path))
    manifest = md.prepareMCPBSite(
        str(tmp_path / "mcpb"),
        {
            "site_id": "cyp51_a",
            "group_name": "cyp51_a",
            "metal": {
                "chain": "A",
                "resid": 601,
                "resname": "HEM",
                "atom": "FE",
                "formal_charge": 3,
            },
            "coordinating_atoms": [
                {"chain": "A", "resid": 463, "resname": "CYM", "atom": "SG", "role": "proximal"},
                {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
            ],
            "fragments": [
                {"chain": "A", "resid": 601, "resname": "HEM", "net_charge": -4, "exclude_atoms": ["FE"]},
                {"chain": "A", "resid": 602, "resname": "LFV", "net_charge": 0},
            ],
            "qm": {
                "small_model_charge": -2,
                "small_model_spin": 2,
                "large_model_charge": -2,
                "large_model_spin": 2,
            },
        },
    )

    assert manifest["qm"]["small_model_charge"] == -2
    assert manifest["qm"]["large_model_spin"] == 2
    assert [fragment["residue_name"] for fragment in manifest["fragments"]] == ["HEM", "LFV"]
    assert (tmp_path / "mcpb" / "cyp51_a" / "cyp51_a.in").exists()


def test_prepare_mcpb_site_splits_embedded_metal_into_new_chain_when_later_chains_exist(tmp_path):
    pdb_path = tmp_path / "heme_site_two_chains.pdb"
    pdb_path.write_text(
        """\
ATOM      1  N   CYS A 463       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  SG  CYS A 463       2.300   0.000   0.000  1.00 20.00           S
HETATM    3  FE  HEM A 601       0.000   0.000   0.000  1.00 20.00          FE
HETATM    4  NA  HEM A 601       0.000   1.300   0.000  1.00 20.00           N
HETATM    5  NAW LFV A 602       2.000   0.000   0.000  1.00 20.00           N
ATOM      6  N   GLY B   1       8.000   0.000   0.000  1.00 20.00           N
ATOM      7  CA  GLY B   1       9.200   0.000   0.000  1.00 20.00           C
END
"""
    )

    md = openmm_setup.openmm_md(str(pdb_path))
    manifest = md.prepareMCPBSite(
        str(tmp_path / "mcpb"),
        {
            "site_id": "cyp51_a",
            "group_name": "cyp51_a",
            "metal": {
                "chain": "A",
                "resid": 601,
                "resname": "HEM",
                "atom": "FE",
                "formal_charge": 3,
            },
            "coordinating_atoms": [
                {"chain": "A", "resid": 463, "resname": "CYM", "atom": "SG", "role": "proximal"},
                {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW", "role": "azole"},
            ],
            "fragments": [
                {"chain": "A", "resid": 601, "resname": "HEM", "net_charge": -4, "exclude_atoms": ["FE"]},
                {"chain": "A", "resid": 602, "resname": "LFV", "net_charge": 0},
            ],
            "qm": {
                "small_model_charge": -2,
                "small_model_spin": 2,
                "large_model_charge": -2,
                "large_model_spin": 2,
            },
        },
    )

    assert manifest["metal"]["split_chain_id"] != "A"
    split_model = (tmp_path / "mcpb" / "cyp51_a" / "cyp51_a_mcpb_model.pdb").read_text().splitlines()
    assert any(
        line.startswith("HETATM")
        and line[12:16].strip() == "FE"
        and line[21].strip()
        and line[21] != "A"
        for line in split_model
    )


def test_prepare_mcpb_site_requires_mandatory_chemistry_fields(tmp_path):
    pdb_path = tmp_path / "heme_site_missing_fields.pdb"
    pdb_path.write_text(
        """\
ATOM      1  N   CYS A 463       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  SG  CYS A 463       2.300   0.000   0.000  1.00 20.00           S
HETATM    3  FE  HEM A 601       0.000   0.000   0.000  1.00 20.00          FE
HETATM    4  NAW LFV A 602       2.000   0.000   0.000  1.00 20.00           N
END
"""
    )
    md = openmm_setup.openmm_md(str(pdb_path))

    with pytest.raises(ValueError, match="formal_charge"):
        md.prepareMCPBSite(
            str(tmp_path / "mcpb"),
            {
                "site_id": "missing",
                "metal": {"chain": "A", "resid": 601, "resname": "HEM", "atom": "FE"},
                "coordinating_atoms": [
                    {"chain": "A", "resid": 463, "resname": "CYS", "atom": "SG"},
                    {"chain": "A", "resid": 602, "resname": "LFV", "atom": "NAW"},
                ],
                "fragments": [
                    {"chain": "A", "resid": 601, "resname": "HEM", "net_charge": -2, "exclude_atoms": ["FE"]},
                    {"chain": "A", "resid": 602, "resname": "LFV", "net_charge": 0},
                ],
                "qm": {
                    "small_model_charge": 0,
                    "small_model_spin": 1,
                    "large_model_charge": 0,
                    "large_model_spin": 1,
                },
            },
        )
