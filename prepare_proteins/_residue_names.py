"""Canonical residue-name handling for force-field / protonation aliases.

Single source of truth for mapping the non-standard residue names that AMBER,
CHARMM and constant-pH protocols use for protonation, tautomer, coordination and
disulfide states back to the 20 canonical amino acids -- e.g. ``CYP`` (the
heme-iron-coordinated cysteinate), ``CYX`` (disulfide), ``CYM`` (thiolate),
``HID/HIE/HIP`` and ``HSD/HSE/HSP`` (histidine tautomers), ``ASH``/``GLH``
(protonated acids), ``LYN`` (neutral lysine).

Motivation: code that extracts a sequence from a structure, or classifies a
residue as protein vs. ligand, by testing membership in a *canonical-20* set
will silently drop these residues. That once removed a catalytic cysteine
(``CYP``) from a design input without any warning. Route such checks through the
helpers here so a new alias is added in exactly one place, and so an unmapped
backbone-bearing residue is reported loudly instead of vanishing.

This module deliberately has no third-party dependencies so it can be imported
from anywhere in the package (including the MD submodules) without import cost
or cycles.
"""

#: The 20 canonical amino acids, three-letter upper-case.
STANDARD_AMINO_ACIDS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})

#: Force-field / protonation / tautomer / coordination aliases -> canonical
#: amino acid. Covers AMBER (HID/HIE/HIP, ASH, GLH, LYN, AR0, CYX/CYM/CYP),
#: CHARMM (HSD/HSE/HSP, ASPP, GLUP, LSN), constant-pH (AS4/GL4) and MSE.
PROTEIN_RESNAME_ALIASES = {
    # histidine tautomers / protonation
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    # cysteine: disulfide, thiolate, metal/heme-coordinated
    "CYX": "CYS", "CYM": "CYS", "CYP": "CYS",
    # protonated acids
    "ASH": "ASP", "ASPP": "ASP", "AS4": "ASP",
    "GLH": "GLU", "GLUP": "GLU", "GL4": "GLU",
    # lysine / arginine neutral forms
    "LYN": "LYS", "LSN": "LYS",
    "AR0": "ARG",
    # deprotonated tyrosine; selenomethionine
    "TYM": "TYR",
    "MSE": "MET",
}

#: Backbone capping groups: part of a polymer chain but NOT amino acids, so they
#: have no one-letter code and must not be counted as a sequence position.
CAPPING_GROUPS = frozenset({"ACE", "NME", "NMA", "NHE", "NH2", "FOR"})

#: Common crystallographic waters.
WATER_RESNAMES = frozenset({"HOH", "WAT", "H2O", "TIP", "TIP3", "SPC", "SOL"})

#: Common monatomic ions seen in MD/crystal structures.
ION_RESNAMES = frozenset({
    "NA", "K", "CL", "MG", "CA", "ZN", "FE", "FE2", "MN", "CU", "CU1",
    "NI", "CO", "CD", "BR", "IOD", "I", "F", "LI", "RB", "CS", "SR", "BA",
    "NA+", "K+", "CL-", "MG2", "CAL", "POT", "SOD", "CLA",
})

#: Canonical three-letter -> one-letter code.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def canonical_resname(resname):
    """Normalise a residue name (strip/upper) and resolve force-field aliases.

    ``canonical_resname('cyp')`` -> ``'CYS'``; an unknown name is returned
    normalised but unchanged.
    """
    if resname is None:
        return ""
    name = str(resname).strip().upper()
    return PROTEIN_RESNAME_ALIASES.get(name, name)


def is_protein_resname(resname):
    """True if ``resname`` is (or aliases to) one of the 20 amino acids.

    Recognises force-field aliases (``CYP``, ``HID``, ``ASH`` ...). Capping
    groups, ions and waters are NOT protein.
    """
    return canonical_resname(resname) in STANDARD_AMINO_ACIDS


def is_non_protein_resname(resname):
    """True for residues that are legitimately not a protein sequence position
    (waters, monatomic ions, capping groups)."""
    name = str(resname or "").strip().upper()
    return name in WATER_RESNAMES or name in ION_RESNAMES or name in CAPPING_GROUPS


def resname_to_one(resname, unknown="raise"):
    """One-letter code for a residue name, resolving force-field aliases.

    Parameters
    ----------
    resname : str
        Three-letter residue name (any case); aliases such as ``CYP`` resolve.
    unknown : {"raise", "X", None}
        Behaviour when the (canonicalised) name is not an amino acid:
        ``"raise"`` raises ``KeyError`` (default, fail-loud); ``"X"`` returns
        ``"X"``; ``None`` returns ``None`` so the caller can decide.
    """
    code = THREE_TO_ONE.get(canonical_resname(resname))
    if code is not None:
        return code
    if unknown == "raise":
        raise KeyError(
            f"Unrecognised residue name {resname!r}: not a standard amino acid "
            f"or a known force-field alias. If it is a force-field residue, add "
            f"it to prepare_proteins._residue_names.PROTEIN_RESNAME_ALIASES."
        )
    if isinstance(unknown, str):
        return unknown
    return None
