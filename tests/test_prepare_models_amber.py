"""Unit tests for the Schrodinger-independent rDock receptor preparation
(`proteinModels.prepareModelsAmber`).

The full method is an integration step requiring AmberTools (`tleap`) and
`propka`; it is exercised functionally elsewhere. Here we unit-test the pure,
rDock-critical transform that makes a tleap MOL2 parseable by rDock 2013.1.
"""
from __future__ import annotations

from prepare_proteins._protein_models import _mol2_text_to_rdock


def _mol2(mol_type="SMALL", charge_type="USER_CHARGES"):
    return (
        "@<TRIPOS>MOLECULE\n"
        "default_name\n"
        " 4823  4885   297     0     1\n"
        f"{mol_type}\n"
        f"{charge_type}\n"
        "@<TRIPOS>ATOM\n"
        "  1 N      38.426   41.398   38.865 N3    1 NGLY  0.2943 ****\n"
    )


def test_sets_protein_no_charges():
    out = _mol2_text_to_rdock(_mol2()).splitlines()
    assert out[3] == "PROTEIN"
    assert out[4] == "NO_CHARGES"


def test_preserves_counts_and_atoms():
    out = _mol2_text_to_rdock(_mol2()).splitlines()
    assert out[2].split()[0] == "4823"          # atom count untouched
    assert out[6].startswith("  1 N")            # atom record untouched


def test_idempotent():
    once = _mol2_text_to_rdock(_mol2())
    assert _mol2_text_to_rdock(once) == once


def test_no_molecule_record_is_safe():
    txt = "@<TRIPOS>ATOM\n1 C 0.0 0.0 0.0\n"
    assert "PROTEIN" not in _mol2_text_to_rdock(txt)
