"""Tests for force-field residue-name alias handling.

Regression guard for the bug where a heme-coordinated catalytic cysteine named
``CYP`` was silently dropped during structure->sequence conversion because the
canonical-20 maps did not know the alias.
"""
import warnings

import pytest

from prepare_proteins._residue_names import (
    PROTEIN_RESNAME_ALIASES,
    STANDARD_AMINO_ACIDS,
    canonical_resname,
    is_protein_resname,
    is_non_protein_resname,
    resname_to_one,
)


# Aliases that must resolve, with their canonical target and one-letter code.
ALIAS_CASES = [
    ("CYP", "CYS", "C"),   # the bug: heme-coordinated cysteinate
    ("CYX", "CYS", "C"),   # disulfide
    ("CYM", "CYS", "C"),   # thiolate
    ("HID", "HIS", "H"), ("HIE", "HIS", "H"), ("HIP", "HIS", "H"),
    ("HSD", "HIS", "H"), ("HSE", "HIS", "H"), ("HSP", "HIS", "H"),
    ("ASH", "ASP", "D"), ("GLH", "GLU", "E"),
    ("LYN", "LYS", "K"), ("AR0", "ARG", "R"),
    ("MSE", "MET", "M"),
]


@pytest.mark.parametrize("alias, canon, one", ALIAS_CASES)
def test_alias_resolves(alias, canon, one):
    assert canonical_resname(alias) == canon
    assert canonical_resname(alias.lower()) == canon          # case-insensitive
    assert canonical_resname(f"  {alias} ") == canon          # whitespace
    assert is_protein_resname(alias)
    assert resname_to_one(alias) == one


def test_cyp_is_the_regression_case():
    # The exact failure mode that motivated this module.
    assert resname_to_one("CYP") == "C"
    assert is_protein_resname("CYP")
    assert not is_non_protein_resname("CYP")


def test_standard_amino_acids_roundtrip():
    for aa in STANDARD_AMINO_ACIDS:
        assert canonical_resname(aa) == aa
        assert is_protein_resname(aa)
        assert len(resname_to_one(aa)) == 1


def test_unknown_behaviour():
    assert resname_to_one("HEM", unknown="X") == "X"
    assert resname_to_one("HEM", unknown=None) is None
    with pytest.raises(KeyError):
        resname_to_one("HEM", unknown="raise")
    with pytest.raises(KeyError):
        resname_to_one("ZZZ")                                  # default raises


def test_non_protein_classification():
    for name in ("HOH", "WAT", "NA", "CL", "MG", "ACE", "NME"):
        assert is_non_protein_resname(name)
        assert not is_protein_resname(name)
    # ligands are neither protein nor a recognised "skippable" non-protein
    assert not is_protein_resname("HEM")
    assert not is_protein_resname("MYR")


def test_every_alias_targets_a_standard_amino_acid():
    for alias, target in PROTEIN_RESNAME_ALIASES.items():
        assert target in STANDARD_AMINO_ACIDS, (alias, target)


def test_get_topology_sequence_includes_cyp_and_warns_on_unknown():
    md = pytest.importorskip("mdtraj")
    from prepare_proteins.MD.alignment import getTopologySequence

    top = md.Topology()
    chain = top.add_chain()
    C = md.element.carbon
    N = md.element.nitrogen
    O = md.element.oxygen

    def add_residue(name, backbone=True):
        res = top.add_residue(name, chain)
        if backbone:
            for an, el in (("N", N), ("CA", C), ("C", C), ("O", O)):
                top.add_atom(an, el, res)
        return res

    # A short chain: ALA - CYP(catalytic cys) - GLY, plus an unknown
    # backbone-bearing residue 'XYZ' that must trigger a warning.
    add_residue("ALA")
    add_residue("CYP")       # must become 'C', not be dropped
    add_residue("GLY")
    add_residue("XYZ")       # unknown, has backbone -> warn + dropped

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seq = getTopologySequence(top, 0)

    assert seq == "ACG"                                        # CYP -> C kept
    assert any("XYZ" in str(w.message) for w in caught)        # loud, not silent
