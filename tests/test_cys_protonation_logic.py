"""Unit tests for the Cys-protonation classifier and validator.

Targets the helpers added to fix the historical "CYX with no SS partner"
failure mode where PROPKA-stripped Cys's were silently renamed CYX,
producing prmtops with non-standard SG charges and no S-S bond.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prepare_proteins.MD.openmm_setup import (
    _classify_cys_form,
    _collect_ss_partner_sg_indices,
    validate_prmtop_cys_protonation,
)


class _Atom:
    """Tiny duck-type matching the bits of openmm.app.topology.Atom we use."""
    def __init__(self, name: str, index: int, residue=None):
        self.name = name
        self.index = index
        self.residue = residue


class _Residue:
    def __init__(self, name: str, atoms: list[_Atom]):
        self.name = name
        self._atoms = atoms
        for a in atoms:
            a.residue = self

    def atoms(self):
        return iter(self._atoms)


def _make_cys(name: str, sg_index: int, with_hg: bool, with_other_h: bool = True):
    """Build a Cys-family residue. ``with_other_h`` controls whether HB2/HB3
    are present — needed to distinguish heavy-atom-only inputs (e.g. raw
    AF3 output) from PROPKA-stripped inputs."""
    atoms = [
        _Atom("N",   sg_index - 4),
        _Atom("CA",  sg_index - 3),
        _Atom("CB",  sg_index - 2),
        _Atom("SG",  sg_index),
    ]
    if with_other_h:
        atoms.append(_Atom("HB2", sg_index - 6))
        atoms.append(_Atom("HB3", sg_index - 5))
    if with_hg:
        atoms.append(_Atom("HG", sg_index - 1))
    return _Residue(name, atoms)


# ----- _classify_cys_form --------------------------------------------------

def test_classify_cys_with_hg_is_cys():
    r = _make_cys("CYS", sg_index=100, with_hg=True)
    assert _classify_cys_form(r, ss_partner_sg_indices=set()) == "CYS"


def test_classify_cys_no_hg_with_ss_partner_is_cyx():
    r = _make_cys("CYS", sg_index=100, with_hg=False)
    assert _classify_cys_form(r, ss_partner_sg_indices={100, 200}) == "CYX"


def test_classify_cys_no_hg_no_ss_partner_with_other_hs_is_cym():
    """HG selectively stripped (other Hs present) → PROPKA-style thiolate."""
    r = _make_cys("CYS", sg_index=100, with_hg=False, with_other_h=True)
    assert _classify_cys_form(r, ss_partner_sg_indices=set()) == "CYM"


def test_classify_cys_no_hg_no_ss_partner_no_hs_at_all_is_cys():
    """Heavy-atom-only input (e.g. raw AF3) → tleap will add HG → CYS."""
    r = _make_cys("CYS", sg_index=100, with_hg=False, with_other_h=False)
    assert _classify_cys_form(r, ss_partner_sg_indices=set()) == "CYS"


def test_classify_cys_pathological_no_sg_no_hg_returns_cys():
    """No SG and no HG is non-physical; falls back to CYS so tleap surfaces
    the issue rather than silently producing thiolate parameters."""
    r = _Residue("CYS", [_Atom("N", 1), _Atom("CA", 2), _Atom("CB", 3)])
    assert _classify_cys_form(r, ss_partner_sg_indices=set()) == "CYS"


# ----- _collect_ss_partner_sg_indices --------------------------------------

class _Topology:
    def __init__(self, residues: list[_Residue]):
        self._residues = residues
        self._atoms = [a for r in residues for a in r.atoms()]

    def atoms(self):
        return iter(self._atoms)


class _Vec3:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z


def _make_modeller(residues_with_sg_positions: list[tuple[_Residue, _Vec3]]):
    residues = [r for r, _ in residues_with_sg_positions]
    topology = _Topology(residues)
    positions = [None] * len(topology._atoms)
    for r, sg_pos in residues_with_sg_positions:
        for a in r.atoms():
            if a.name == "SG":
                positions[a.index] = sg_pos
                break
    return SimpleNamespace(topology=topology, positions=positions)


def test_ss_partners_detects_close_sg_pair():
    a = _make_cys("CYS", sg_index=0, with_hg=False)
    b = _make_cys("CYS", sg_index=4, with_hg=False)
    modeller = _make_modeller([(a, _Vec3(0, 0, 0)), (b, _Vec3(0.205, 0, 0))])
    assert _collect_ss_partner_sg_indices(modeller) == {0, 4}


def test_ss_partners_ignores_distant_sg_pair():
    a = _make_cys("CYS", sg_index=0, with_hg=False)
    b = _make_cys("CYS", sg_index=4, with_hg=False)
    # 0.5 nm = 5 Å, well beyond the 0.30 nm cutoff
    modeller = _make_modeller([(a, _Vec3(0, 0, 0)), (b, _Vec3(0.5, 0, 0))])
    assert _collect_ss_partner_sg_indices(modeller) == set()


def test_ss_partners_handles_lone_cys():
    a = _make_cys("CYS", sg_index=0, with_hg=True)
    modeller = _make_modeller([(a, _Vec3(0, 0, 0))])
    assert _collect_ss_partner_sg_indices(modeller) == set()


def test_ss_partners_does_not_pair_residue_with_itself():
    a = _make_cys("CYS", sg_index=0, with_hg=False)
    modeller = _make_modeller([(a, _Vec3(0, 0, 0))])
    assert _collect_ss_partner_sg_indices(modeller) == set()


# ----- validate_prmtop_cys_protonation -------------------------------------

class _PMDAtom:
    def __init__(self, name: str, idx: int, charge: float, residue=None):
        self.name = name
        self.idx = idx
        self.charge = charge
        self.residue = residue


class _PMDResidue:
    def __init__(self, name: str, number: int, atoms: list[_PMDAtom]):
        self.name = name
        self.number = number
        self.atoms = atoms
        for a in atoms:
            a.residue = self


class _PMDBond:
    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2


class _PMDTop:
    def __init__(self, residues, bonds):
        self.residues = residues
        self.bonds = bonds


def _stub_parmed_load(top):
    """Monkeypatch helper: makes parmed.load_file(...) return our stub."""
    class _StubLoader:
        @staticmethod
        def load_file(_path):
            return top
    return _StubLoader


def _build_cys(name: str, resseq: int, sg_idx: int, *, with_hg: bool, sg_charge: float = -0.31, hg_charge: float = 0.20, backbone_charge_balance: float = 0.0):
    """Build a Cys-family residue with a known total charge."""
    atoms = [
        _PMDAtom("N",  sg_idx - 4, 0.0),
        _PMDAtom("CA", sg_idx - 3, 0.0),
        _PMDAtom("CB", sg_idx - 2, 0.0),
        _PMDAtom("SG", sg_idx, sg_charge),
    ]
    if with_hg:
        atoms.append(_PMDAtom("HG", sg_idx - 1, hg_charge))
    # nudge backbone to hit the desired residue total
    atoms[0].charge += backbone_charge_balance
    return _PMDResidue(name, resseq, atoms)


def test_validator_passes_consistent_prmtop(monkeypatch):
    import prepare_proteins.MD.openmm_setup as m

    cys = _build_cys("CYS", 10, sg_idx=10, with_hg=True, backbone_charge_balance=0.11)
    cyx_a = _build_cys("CYX", 20, sg_idx=20, with_hg=False, backbone_charge_balance=0.31)
    cyx_b = _build_cys("CYX", 30, sg_idx=30, with_hg=False, backbone_charge_balance=0.31)
    cym = _build_cys("CYM", 40, sg_idx=40, with_hg=False, sg_charge=-0.88, backbone_charge_balance=-0.12)

    top = _PMDTop(
        residues=[cys, cyx_a, cyx_b, cym],
        bonds=[_PMDBond(cyx_a.atoms[3], cyx_b.atoms[3])],
    )
    monkeypatch.setattr("parmed.load_file", lambda _p: top, raising=False)
    import parmed  # noqa - ensures module is importable
    monkeypatch.setattr(parmed, "load_file", lambda _p: top)

    rows = validate_prmtop_cys_protonation("fake.prmtop", print_summary=False)
    assert all(r["ok"] for r in rows), rows


def test_validator_flags_cyx_without_ss_partner(monkeypatch):
    cyx_orphan = _build_cys("CYX", 70, sg_idx=70, with_hg=False, backbone_charge_balance=0.31)
    top = _PMDTop(residues=[cyx_orphan], bonds=[])
    import parmed
    monkeypatch.setattr(parmed, "load_file", lambda _p: top)

    with pytest.raises(ValueError, match=r"CYX70\(SS\)"):
        validate_prmtop_cys_protonation("fake.prmtop", print_summary=False)


def test_validator_flags_cym_with_wrong_charge(monkeypatch):
    bogus_cym = _build_cys("CYM", 80, sg_idx=80, with_hg=False, backbone_charge_balance=0.88)
    top = _PMDTop(residues=[bogus_cym], bonds=[])
    import parmed
    monkeypatch.setattr(parmed, "load_file", lambda _p: top)

    # Residue sum is ~0 but CYM expects ~-1
    with pytest.raises(ValueError, match=r"CYM80.*q"):
        validate_prmtop_cys_protonation("fake.prmtop", print_summary=False)


def test_validator_flags_cys_missing_hg(monkeypatch):
    cys_no_hg = _build_cys("CYS", 100, sg_idx=100, with_hg=False, backbone_charge_balance=0.11)
    top = _PMDTop(residues=[cys_no_hg], bonds=[])
    import parmed
    monkeypatch.setattr(parmed, "load_file", lambda _p: top)

    with pytest.raises(ValueError, match=r"CYS100.*HG"):
        validate_prmtop_cys_protonation("fake.prmtop", print_summary=False)
