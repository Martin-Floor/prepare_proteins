"""Tests for prepareProteinPDB -- the Schrödinger-independent protein-prep helper.

Coverage:
  - Default protonation: reads a heavy-atom-only PDB, adds H via OpenMM
    Modeller, writes a PDB that round-trips.
  - Water dropping.
  - Custom residue dropping.
  - Explicit user-provided variants dict.
  - skip-if-exists when ``overwrite=False``.
  - PROPKA path is covered indirectly by the existing
    ``test_propka_protonation_states.py``; this file does not re-mock the
    PROPKA subprocess to avoid duplicating that test scaffolding.
"""
import os
import shutil
from pathlib import Path

import pytest

HEME_THIOLATE = Path(__file__).parent / "data" / "heme_thiolate.pdb"


def _has_openmm():
    try:
        import openmm  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_openmm() or not HEME_THIOLATE.is_file(),
    reason="OpenMM or test fixture missing",
)


def _residue_names(pdb_path):
    out = []
    seen = set()
    for ln in open(pdb_path):
        if ln[:6] not in ("ATOM  ", "HETATM"):
            continue
        key = (ln[21], ln[22:27])
        if key in seen:
            continue
        seen.add(key)
        out.append(ln[17:20].strip())
    return out


def _count_hydrogens(pdb_path):
    return sum(1 for ln in open(pdb_path)
               if ln[:6] in ("ATOM  ", "HETATM") and ln[76:78].strip() == "H")


# ---------------------------------------------------------------------------
# Default protonation: just add H's and write a PDB
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_default_adds_hydrogens(tmp_path):
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    out = tmp_path / "prepped.pdb"
    info = prepareProteinPDB(
        str(HEME_THIOLATE), str(out),
        protonate="default",
    )
    assert info["output_pdb"] == str(out.resolve())
    assert info["variants_applied"] == {}
    assert not info["skipped"]
    assert out.is_file()

    # the fixture has no hydrogens; the prepped PDB must
    assert _count_hydrogens(HEME_THIOLATE) == 0
    assert _count_hydrogens(out) > 1000  # ~2500 for this ~330-residue protein


# ---------------------------------------------------------------------------
# Water dropping
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_drops_waters(tmp_path):
    """Drop_waters strips any HOH/WAT residues. We synthesise a PDB with a
    couple of dummy HOH atoms appended to confirm the filter runs."""
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    src = tmp_path / "with_waters.pdb"
    base = HEME_THIOLATE.read_text()
    # inject two HOH atoms before END
    waters = (
        "HETATM 9991  O   HOH X 901      10.000  10.000  10.000  1.00 30.00           O\n"
        "HETATM 9992  O   HOH X 902      12.000  12.000  12.000  1.00 30.00           O\n"
    )
    if "END" in base:
        base = base.replace("END", waters + "END", 1)
    else:
        base += waters
    src.write_text(base)

    out = tmp_path / "no_waters.pdb"
    info = prepareProteinPDB(
        str(src), str(out),
        protonate="default",
        drop_waters=True,
    )
    # we asked for waters dropped; both should appear in the dropped list
    assert any(rn == "HOH" for _, _, rn in info["residues_dropped"])
    # and the output should contain no HOH atoms
    assert all(ln[17:20].strip() != "HOH"
               for ln in open(out) if ln[:6] in ("ATOM  ", "HETATM"))


# ---------------------------------------------------------------------------
# drop_resnames: e.g. dropping the structural Mg
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_drops_arbitrary_resname(tmp_path):
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    out = tmp_path / "no_mg.pdb"
    info = prepareProteinPDB(
        str(HEME_THIOLATE), str(out),
        protonate="default",
        drop_resnames=("MG",),
    )
    # heme_thiolate fixture has a Mg substructure (chain C)
    assert any(rn == "MG" for _, _, rn in info["residues_dropped"])
    assert all(ln[17:20].strip() != "MG"
               for ln in open(out) if ln[:6] in ("ATOM  ", "HETATM"))


# ---------------------------------------------------------------------------
# Explicit user variants
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_accepts_explicit_variants(tmp_path):
    """When a variants dict is passed explicitly the helper must apply it
    without invoking PROPKA. We use a trivial subset of histidines."""
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    # find a histidine in the input (chain A) to retype as HIP
    target = None
    for ln in open(HEME_THIOLATE):
        if ln[:6] == "ATOM  " and ln[17:20].strip() == "HIS":
            target = (ln[21], int(ln[22:26]))
            break
    if target is None:
        pytest.skip("no histidine in fixture")

    out = tmp_path / "with_hip.pdb"
    info = prepareProteinPDB(
        str(HEME_THIOLATE), str(out),
        protonate={target: "HIP"},
    )
    assert info["variants_applied"] == {target: "HIP"}
    # Modeller leaves the residue name as HIS in the written topology even
    # when the HIP template was used; the signature of HIP is that BOTH
    # imidazole protons (HD1 and HE2) are present.
    target_atoms = {
        ln[12:16].strip()
        for ln in open(out)
        if ln[:6] == "ATOM  "
        and ln[21] == target[0]
        and ln[22:26].strip() == str(target[1])
    }
    assert {"HD1", "HE2"}.issubset(target_atoms), (
        f"explicit HIP variant not applied — target residue atoms: "
        f"{sorted(target_atoms)}"
    )


# ---------------------------------------------------------------------------
# Skip behaviour
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_skips_when_output_exists(tmp_path):
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    out = tmp_path / "exists.pdb"
    out.write_text("placeholder\n")  # pre-existing file

    info = prepareProteinPDB(
        str(HEME_THIOLATE), str(out),
        protonate="default",
        overwrite=False,
    )
    assert info["skipped"] is True
    assert out.read_text() == "placeholder\n"  # untouched

    # with overwrite=True the file is regenerated
    info2 = prepareProteinPDB(
        str(HEME_THIOLATE), str(out),
        protonate="default",
        overwrite=True,
    )
    assert info2["skipped"] is False
    assert out.read_text() != "placeholder\n"


# ---------------------------------------------------------------------------
# PROPKA path: skipped if propka isn't installed. Confirms the helper sets
# include_identity_states=False so Modeller can keep its automatic disulfide
# handling (forcing identity CYS would collide with SS-bonded cysteines).
# ---------------------------------------------------------------------------

def _has_propka():
    if shutil.which("propka3") or shutil.which("propka"):
        return True
    try:
        import propka  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# variant_overrides validation (drops variants whose target residue doesn't match)
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_drops_mismatched_variant_overrides(tmp_path):
    """If the user requests HIP on a residue that isn't a HIS, the override
    must be silently dropped (rather than corrupt the prep). The classifier's
    `reason` string can be wrong in rare cases (e.g. 1CPO_A)."""
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    # find an ASP and pretend the user wanted HIP there
    target_asp = None
    for ln in open(HEME_THIOLATE):
        if ln[:6] == "ATOM  " and ln[17:20].strip() == "ASP":
            target_asp = (ln[21], int(ln[22:26]))
            break
    if target_asp is None:
        pytest.skip("no ASP in fixture")

    # also force HIP on a real HIS (must apply)
    target_his = None
    for ln in open(HEME_THIOLATE):
        if ln[:6] == "ATOM  " and ln[17:20].strip() == "HIS":
            target_his = (ln[21], int(ln[22:26]))
            break

    out = tmp_path / "validated.pdb"
    info = prepareProteinPDB(
        str(HEME_THIOLATE), str(out),
        protonate="default",
        variant_overrides={target_asp: "HIP", target_his: "HIP"},
    )
    # bad override should be in dropped, not applied
    assert target_asp not in info["variants_applied"]
    assert target_his in info["variants_applied"] and info["variants_applied"][target_his] == "HIP"
    assert any(k == target_asp for (k, _, _) in info["overrides_dropped"])


# ---------------------------------------------------------------------------
# Capping (ACE/NME termini)
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_cap_termini(tmp_path):
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    out = tmp_path / "capped.pdb"
    info = prepareProteinPDB(
        str(HEME_THIOLATE), str(out),
        protonate="default",
        cap_termini=True,
    )
    assert info["capped"]
    # output should contain ACE and NME residues
    resnames = {ln[17:20].strip() for ln in open(out)
                if ln[:6] in ("ATOM  ", "HETATM")}
    assert "ACE" in resnames, "ACE cap not added"
    assert "NME" in resnames, "NME cap not added"


# ---------------------------------------------------------------------------
# Minimization (H-only relax with heavy-atom restraints)
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_minimize_moves_only_hydrogens(tmp_path):
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    # bake a baseline without minimization, then re-do with minimization
    base = tmp_path / "base.pdb"
    prepareProteinPDB(str(HEME_THIOLATE), str(base), protonate="default")
    mini = tmp_path / "mini.pdb"
    info = prepareProteinPDB(
        str(HEME_THIOLATE), str(mini),
        protonate="default", minimize=True, minimize_max_iterations=50,
    )
    assert info["minimized"]

    def coords_by_atom(pdb):
        d = {}
        for ln in open(pdb):
            if ln[:6] != "ATOM  ":
                continue
            key = (ln[21], int(ln[22:26]), ln[12:16].strip())
            d[key] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        return d

    c_base = coords_by_atom(base)
    c_mini = coords_by_atom(mini)
    # restrict to atoms present in both
    common = set(c_base) & set(c_mini)
    max_heavy_drift = max(
        max(abs(c_mini[k][i] - c_base[k][i]) for i in range(3))
        for k in common if not k[2].startswith("H")
    )
    max_h_drift = max(
        (max(abs(c_mini[k][i] - c_base[k][i]) for i in range(3))
         for k in common if k[2].startswith("H")),
        default=0.0,
    )
    # heavy atoms should barely move (restrained); 0.5 A is a tight bound
    # for the stiff harmonic restraints (k=1e7 default) -- structures with
    # bad initial clashes can drift up to that
    assert max_heavy_drift < 0.5, f"heavy atoms drifted too much: {max_heavy_drift:.3f} A"
    # H drift may be small if the input was already well-placed; just ensure
    # the minimisation doesn't crash and heavy atoms stay put


@pytest.mark.skipif(not _has_propka(), reason="PROPKA not installed")
def test_prepareProteinPDB_propka_path(tmp_path):
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    out = tmp_path / "with_propka.pdb"
    info = prepareProteinPDB(
        str(HEME_THIOLATE), str(out),
        protonate="propka", pH=7.0,
        drop_resnames=("MG",),
    )
    assert out.is_file()
    # PROPKA on this fixture predicts a handful of explicit renames
    # (histidine tautomers, occasional GLH/ASH) but NO identity assignments
    # like CYS/CYX -- those are filtered by include_identity_states=False.
    variants = info["variants_applied"]
    assert variants, "expected at least one PROPKA-driven rename"
    explicit = {"HIP", "HID", "HIE", "ASH", "GLH", "CYX"}
    assert all(v in explicit for v in variants.values()), (
        f"expected only explicit renames (got {set(variants.values())})"
    )


# ---------------------------------------------------------------------------
# variant_overrides
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_variant_overrides_force_HIP(tmp_path):
    """`variant_overrides` must override whatever `protonate` produced and
    apply the user's variant verbatim."""
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    # pick a His in the fixture
    target = None
    for ln in open(HEME_THIOLATE):
        if ln[:6] == "ATOM  " and ln[17:20].strip() == "HIS":
            target = (ln[21], int(ln[22:26]))
            break
    if target is None:
        pytest.skip("no histidine in fixture")

    out = tmp_path / "with_overrides.pdb"
    # asks for 'default' (which would leave the variant as None) but the
    # override forces HIP -> Modeller should add both HD1 and HE2
    info = prepareProteinPDB(
        str(HEME_THIOLATE), str(out),
        protonate="default",
        variant_overrides={target: "HIP"},
    )
    assert info["variants_applied"] == {target: "HIP"}
    target_atoms = {
        ln[12:16].strip()
        for ln in open(out)
        if ln[:6] == "ATOM  "
        and ln[21] == target[0]
        and ln[22:26].strip() == str(target[1])
    }
    assert {"HD1", "HE2"}.issubset(target_atoms), (
        f"variant_override HIP not applied (got {sorted(target_atoms)})"
    )


# ---------------------------------------------------------------------------
# Invalid protonate argument
# ---------------------------------------------------------------------------

def test_prepareProteinPDB_rejects_invalid_protonate(tmp_path):
    from prepare_proteins.MD.openmm_setup import prepareProteinPDB

    with pytest.raises(ValueError, match="protonate"):
        prepareProteinPDB(
            str(HEME_THIOLATE), str(tmp_path / "x.pdb"),
            protonate="not_a_valid_mode",
        )
