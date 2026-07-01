"""Tests for prepare_proteins.MD.mm_minimize (restrained cluster MM-min)."""
import numpy as np
import pytest

pytest.importorskip("openmm")
pytest.importorskip("parmed")

from prepare_proteins.MD import mm_minimize


@pytest.fixture
def mini_system(tmp_path):
    """Three Argon-like LJ particles at 0/3/6 Å along x → a tiny valid Amber prmtop."""
    import parmed as pmd

    at = pmd.AtomType("Ar", 1, 39.948, atomic_number=18)
    at.set_lj_params(0.2384, 1.908)  # epsilon kcal/mol, rmin/2 Å
    s = pmd.Structure()
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    for i in range(3):
        a = pmd.Atom(name="Ar", type="Ar", atomic_number=18, mass=39.948, charge=0.0)
        a.epsilon = at.epsilon
        a.rmin = at.rmin
        a.atom_type = at
        s.add_atom(a, "AR", i + 1)
    s.coordinates = coords
    s.box = None
    prm = tmp_path / "mini.prmtop"
    crd = tmp_path / "mini.inpcrd"
    s.save(str(prm), overwrite=True)
    s.save(str(crd), format="rst7", overwrite=True)
    return prm, crd, coords


def test_restrained_atom_stays_put(mini_system):
    prm, crd, coords = mini_system
    res = mm_minimize(
        prm, crd,
        restrain_indices=[0],
        restraint_constant=1000.0,   # stiff → atom 0 essentially frozen
        report_pairs={"d01": (0, 1)},
    )
    out = res["positions"]  # Å
    assert out.shape == (3, 3)
    disp0 = np.linalg.norm(out[0] - coords[0])
    disp1 = np.linalg.norm(out[1] - coords[1])
    # the restrained atom barely moves; the free atoms relax away from the LJ clash
    assert disp0 < 0.05
    assert disp1 > disp0
    assert isinstance(res["potential_energy_kj_mol"], float)
    # report_pairs recorded before/after distances (Å)
    before, after = res["distances"]["d01"]
    assert before == pytest.approx(3.0, abs=1e-3)
    assert after > before  # LJ repulsion pushes atom 1 out


def test_unrestrained_runs_and_lowers_clash(mini_system):
    prm, crd, _ = mini_system
    res = mm_minimize(prm, crd, restrain_indices=None, report_pairs={"d01": (0, 1)})
    before, after = res["distances"]["d01"]
    assert after > before  # atoms separate from the 3 Å (sub-rmin) start


def test_positions_array_overrides_inpcrd(mini_system):
    prm, _, coords = mini_system
    # pass coordinates directly (nanometres), no inpcrd
    res = mm_minimize(prm, positions=coords * 0.1, restrain_indices=[0], restraint_constant=1000.0)
    assert res["positions"].shape == (3, 3)


def test_writes_output_pdb(mini_system, tmp_path):
    prm, crd, _ = mini_system
    out_pdb = tmp_path / "min.pdb"
    res = mm_minimize(prm, crd, output_pdb=out_pdb, max_iterations=10)
    assert res["output_pdb"] == str(out_pdb)
    assert out_pdb.is_file() and out_pdb.stat().st_size > 0


def test_requires_coordinates(mini_system):
    prm, _, _ = mini_system
    with pytest.raises(ValueError, match="either 'inpcrd' or 'positions'"):
        mm_minimize(prm)


def test_rejects_out_of_range_restraint(mini_system):
    prm, crd, _ = mini_system
    with pytest.raises(ValueError, match="out of range"):
        mm_minimize(prm, crd, restrain_indices=[99])


def test_rejects_coord_topology_mismatch(mini_system):
    prm, _, _ = mini_system
    with pytest.raises(ValueError, match="mismatch"):
        mm_minimize(prm, positions=np.zeros((2, 3)))
