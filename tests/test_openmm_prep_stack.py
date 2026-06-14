"""Tests for the openmm_md structure-prep stack: capTermini, addHydrogens(pH/PROPKA
+ fixed_protonations overrides), detectDisulfides and validate.

A short real peptide fragment (residues 20-26 of BsCM chain A) with BROKEN termini is
the fixture: it is only force-field-valid after ACE/NME capping, so a successful
``validate()`` (System build) is the propagation guard the prep stack is meant to give.
"""
from pathlib import Path

import pytest

openmm = pytest.importorskip("openmm")
from prepare_proteins.MD.openmm_setup import openmm_md  # noqa: E402

# ILE-LEU-GLN-LYS-THR-LYS-GLN fragment (cap-friendly: no Pro/His/Cys; two Lys to flip)
FRAG_PDB = """\
ATOM      1  N   ILE A  20      21.972  24.519 -11.685  1.00  0.00           N
ATOM      2  CA  ILE A  20      21.630  25.167 -10.414  1.00  0.00           C
ATOM      3  C   ILE A  20      22.462  26.430 -10.133  1.00  0.00           C
ATOM      4  O   ILE A  20      21.919  27.472  -9.744  1.00  0.00           O
ATOM      5  CB  ILE A  20      21.756  24.178  -9.238  1.00  0.00           C
ATOM      6  CG1 ILE A  20      20.787  23.007  -9.451  1.00  0.00           C
ATOM      7  CG2 ILE A  20      21.531  24.902  -7.917  1.00  0.00           C
ATOM      8  CD1 ILE A  20      21.055  21.777  -8.607  1.00  0.00           C
ATOM      9  N   LEU A  21      23.778  26.348 -10.323  1.00  0.00           N
ATOM     10  CA  LEU A  21      24.635  27.509 -10.098  1.00  0.00           C
ATOM     11  C   LEU A  21      24.295  28.654 -11.043  1.00  0.00           C
ATOM     12  O   LEU A  21      24.180  29.808 -10.613  1.00  0.00           O
ATOM     13  CB  LEU A  21      26.115  27.142 -10.212  1.00  0.00           C
ATOM     14  CG  LEU A  21      26.626  26.248  -9.088  1.00  0.00           C
ATOM     15  CD1 LEU A  21      28.114  26.477  -8.882  1.00  0.00           C
ATOM     16  CD2 LEU A  21      25.858  26.524  -7.802  1.00  0.00           C
ATOM     17  N   GLN A  22      24.134  28.324 -12.324  1.00  0.00           N
ATOM     18  CA  GLN A  22      23.772  29.307 -13.337  1.00  0.00           C
ATOM     19  C   GLN A  22      22.453  30.005 -13.030  1.00  0.00           C
ATOM     20  O   GLN A  22      22.367  31.224 -13.114  1.00  0.00           O
ATOM     21  CB  GLN A  22      23.687  28.656 -14.714  1.00  0.00           C
ATOM     22  CG  GLN A  22      22.714  29.350 -15.652  1.00  0.00           C
ATOM     23  CD  GLN A  22      22.656  28.683 -17.004  1.00  0.00           C
ATOM     24  OE1 GLN A  22      21.709  27.957 -17.309  1.00  0.00           O
ATOM     25  NE2 GLN A  22      23.681  28.915 -17.824  1.00  0.00           N
ATOM     26  N   LYS A  23      21.424  29.232 -12.701  1.00  0.00           N
ATOM     27  CA  LYS A  23      20.108  29.809 -12.440  1.00  0.00           C
ATOM     28  C   LYS A  23      20.074  30.591 -11.131  1.00  0.00           C
ATOM     29  O   LYS A  23      19.425  31.636 -11.046  1.00  0.00           O
ATOM     30  CB  LYS A  23      19.006  28.746 -12.464  1.00  0.00           C
ATOM     31  CG  LYS A  23      18.776  28.132 -13.836  1.00  0.00           C
ATOM     32  CD  LYS A  23      18.366  29.182 -14.854  1.00  0.00           C
ATOM     33  CE  LYS A  23      18.934  28.843 -16.224  1.00  0.00           C
ATOM     34  NZ  LYS A  23      18.992  30.012 -17.149  1.00  0.00           N
ATOM     35  N   THR A  24      20.757  30.086 -10.108  1.00  0.00           N
ATOM     36  CA  THR A  24      20.859  30.825  -8.849  1.00  0.00           C
ATOM     37  C   THR A  24      21.577  32.155  -9.079  1.00  0.00           C
ATOM     38  O   THR A  24      21.144  33.196  -8.559  1.00  0.00           O
ATOM     39  CB  THR A  24      21.575  30.009  -7.753  1.00  0.00           C
ATOM     40  OG1 THR A  24      20.815  28.832  -7.450  1.00  0.00           O
ATOM     41  CG2 THR A  24      21.742  30.826  -6.475  1.00  0.00           C
ATOM     42  N   LYS A  25      22.671  32.112  -9.848  1.00  0.00           N
ATOM     43  CA  LYS A  25      23.425  33.318 -10.216  1.00  0.00           C
ATOM     44  C   LYS A  25      22.582  34.338 -10.967  1.00  0.00           C
ATOM     45  O   LYS A  25      22.649  35.519 -10.684  1.00  0.00           O
ATOM     46  CB  LYS A  25      24.668  32.991 -11.054  1.00  0.00           C
ATOM     47  CG  LYS A  25      25.527  34.227 -11.325  1.00  0.00           C
ATOM     48  CD  LYS A  25      26.787  33.956 -12.137  1.00  0.00           C
ATOM     49  CE  LYS A  25      27.802  35.068 -11.888  1.00  0.00           C
ATOM     50  NZ  LYS A  25      29.009  35.015 -12.763  1.00  0.00           N
ATOM     51  N   GLN A  26      21.798  33.876 -11.932  1.00  0.00           N
ATOM     52  CA  GLN A  26      20.989  34.786 -12.732  1.00  0.00           C
ATOM     53  C   GLN A  26      19.855  35.402 -11.924  1.00  0.00           C
ATOM     54  O   GLN A  26      19.517  36.575 -12.100  1.00  0.00           O
ATOM     55  CB  GLN A  26      20.451  34.072 -13.961  1.00  0.00           C
ATOM     56  CG  GLN A  26      21.532  33.836 -14.993  1.00  0.00           C
ATOM     57  CD  GLN A  26      21.115  32.856 -16.063  1.00  0.00           C
ATOM     58  OE1 GLN A  26      20.014  32.311 -16.021  1.00  0.00           O
ATOM     59  NE2 GLN A  26      22.004  32.614 -17.020  1.00  0.00           N
TER
END
"""


@pytest.fixture()
def frag_pdb(tmp_path):
    p = tmp_path / "frag.pdb"
    p.write_text(FRAG_PDB)
    return str(p)


def _md(frag_pdb):
    md = openmm_md(frag_pdb)
    md.setUpFF("amber14")
    return md


def test_capTermini_makes_broken_fragment_valid(frag_pdb):
    """A broken fragment is not FF-valid until ACE+NME capped; validate() proves it."""
    # uncapped broken fragment cannot be force-field-prepared at all
    with pytest.raises(Exception):
        _md(frag_pdb).addHydrogens()
    # cap both ends, protonate, validate -> succeeds
    md = _md(frag_pdb)
    md.capTermini(n_cap="ACE", c_cap="NME")
    assert sum(1 for r in md.modeller.topology.residues() if r.name == "ACE") == 1
    assert sum(1 for r in md.modeller.topology.residues() if r.name == "NME") == 1
    md.addHydrogens()
    info = md.validate()
    assert info["n_particles"] > 59
    assert info["net_charge"] is not None


def test_fixed_protonations_override(frag_pdb):
    """Manual fixed_protonations flips a Lys to neutral LYN (the +-charge drops by 1)."""
    base = _md(frag_pdb)
    base.capTermini(n_cap="ACE", c_cap="NME")
    base.addHydrogens()
    q0 = base.validate()["net_charge"]

    md = _md(frag_pdb)
    md.capTermini(n_cap="ACE", c_cap="NME")
    variants = md.addHydrogens(fixed_protonations={("A", 23): "LYN"})
    # the override is present in the returned variants list
    assert "LYN" in [v for v in variants if v]
    q1 = md.validate()["net_charge"]
    assert q1 == pytest.approx(q0 - 1.0, abs=1e-3)


def test_fixed_protonations_bad_key_raises(frag_pdb):
    md = _md(frag_pdb)
    md.capTermini(c_cap="NME")
    with pytest.raises(ValueError):
        md.addHydrogens(fixed_protonations={"A23": "LYN"})  # not a (chain, resid) tuple


def test_detectDisulfides_none_when_no_cys(frag_pdb):
    md = _md(frag_pdb)
    md.capTermini(c_cap="NME")
    assert md.detectDisulfides() == []


def test_addHydrogens_pH_propka(frag_pdb):
    """PROPKA-driven protonation builds a valid System (skipped if propka absent)."""
    pytest.importorskip("propka")
    md = _md(frag_pdb)
    md.capTermini(n_cap="ACE", c_cap="NME")
    variants = md.addHydrogens(pH=7.0)
    assert isinstance(variants, list)
    info = md.validate()
    assert info["n_particles"] > 59
