from Bio import PDB
import os
import shutil
import numpy as np

def retrievePDBs(pdb_codes, names=None, pdb_directory='PDB'):
    """
    Download a set of pdb structures from the PDB database.

    Parameters
    ----------
    pdb_codes : list or str
        A list with the PDB codes to retrieve their files. Optionally a string
        with a unique code can be given.
    names : dict
        Dictionary mapping the name of each PDB structure
    pdb_directory : str
        The name of the directory to store the retrieved files.

    Returns
    -------
    pdb_paths : list
        A list containing the paths to the retrieved PDB files.
    """

    pdb_paths = []

    pdbl = PDB.PDBList()

    # Create directory to store files
    if not os.path.exists(pdb_directory):
        os.mkdir(pdb_directory)

    # Convert to list if only a string is given
    if isinstance(pdb_codes, str):
        pdb_codes = [pdb_codes]

    if isinstance(pdb_codes, list):
        # Iterate pdb codes
        for code in pdb_codes:
            # Download PDB file
            if names == None:
                output_file = pdb_directory+'/'+code.upper()+'.pdb'
            else:
                output_file = pdb_directory+'/'+names[code.upper()]+'.pdb'

            if not os.path.exists(output_file):
                pdbl.retrieve_pdb_file(code, file_format='pdb', pdir=pdb_directory)
            else: # If file already exists
                print('Structure exists: '+code.upper()+'.pdb')
                if names != None:
                    print('It was named as: '+names[code.upper()]+'.pdb')
                pdb_paths.append(output_file)

    for f in os.listdir(pdb_directory):
        if f.endswith('.ent'):
            # Rename file
            if names == None:
                reanamed_file = pdb_directory+'/'+f.replace('pdb','').upper().replace('.ENT','.pdb')
            else:
                reanamed_file = pdb_directory+'/'+names[f.replace('pdb','').upper().replace('.ENT','')]+'.pdb'
            os.rename(pdb_directory+'/'+f, reanamed_file)
            # Append path
            pdb_paths.append(reanamed_file)

    # Remove unnecesary folders created by Bio.PDB method
    shutil.rmtree('obsolete')

    return pdb_paths

def renumberResidues(structure, by_chain=False):
    """
    Renumber residues in a structure object starting from one. Two methods are possible:
    if by_chain is set to True the renumbering is restarted at the begining of every
    chain.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        Input structure object
    by_chain : bool
        Whether each chain should be renumerated from one.

    Returns
    -------

    structure_copy : Bio.PDB.Structure
        Renumbered copy of the input structure
    """

    count = 0
    structure_copy = PDB.Structure.Structure(0)
    model = PDB.Model.Model(0)
    auxiliar = PDB.Chain.Chain(0)

    for chain in structure.get_chains():
        new_chain = PDB.Chain.Chain(chain.id)
        if by_chain:
            count = 0
        for residue in chain.get_residues():
            count += 1
            residue.set_parent(auxiliar)
            residue.id = (residue.id[0], count, residue.id[2])
            new_chain.add(residue)
        model.add(new_chain)
    structure_copy.add(model)

    return structure_copy

def chainsAsStructure(chains):
    """
    This method creates a new Structure object containing only the given chains.

    Parameters
    ----------
    chains : list or Bio.PDB.Chain
        Chain or chains to be added to the new structure object.

    Returns
    -------
    structure : Bio.PDB.Structure
    """

    if not isinstance(chains, list):
        chains = [chains]

    structure = PDB.Structure.Structure(0)
    model = PDB.Model.Model(0)
    for chain in chains:
        model.add(chain)
    structure.add(model)

    return structure

def getStructureCoordinates(structure):
    """
    Get PDB coordinates as a anumpy array.
    """
    coordinates = []
    for atom in structure.get_atoms():
        coordinates.append(atom.coord)
    coordinates = np.array(coordinates)

    return coordinates


# Atoms used by BioPython's chain-extension propagation. Hedra/dihedra that
# include any of these MUST be left at their input values, otherwise the chain
# drifts when rebuilt from internal coordinates. CA is propagating-safe.
_PROPAGATING_BACKBONE_ATOMS = frozenset(
    {"N", "C", "O", "OXT", "H", "HA", "H2", "H3", "HN"}
)


def _build_chi_keys():
    """The set of dihedron keys that encode rotamer (chi) torsions — to PRESERVE."""
    from Bio.PDB import ic_data

    keys = set()
    for resletter, template in ic_data.ic_data_sidechains.items():
        for row in template:
            if len(row) == 5 and row[4].startswith("chi"):
                keys.add("".join(resletter + a for a in row[:4]))
    return keys


def _build_secondary_default_map():
    """target_key -> (source_key, offset_deg) from ic_data secondary tables."""
    from Bio.PDB import ic_data

    sec = {}
    for src_dict in (
        ic_data.dihedra_secondary_defaults,
        ic_data.dihedra_secondary_xoxt_defaults,
    ):
        for key, value in src_dict.items():
            src_atoms, offset = value
            src_letter = key[0]
            sec[key] = ("".join(src_letter + a for a in src_atoms), offset)
    return sec


def idealizeInternalCoordinates(input_pdb, output_pdb):
    """Idealise sidechain bond lengths, bond angles, and ring planarity in place.

    Many neural-network folders (ESMFold, ESMFold2, OmegaFold, ...) emit
    Cartesian coordinates without enforcing canonical covalent geometry.
    Aromatic rings end up non-planar, sidechain bond lengths drift to
    non-physical values (0.5-2.0 Å for what should be a 1.39 Å aromatic C-C),
    and long sidechains (Lys, Arg, Glu, Gln, Met) get compressed. The atomic
    positions are still close to where they ought to be — the bonds between
    them are simply wrong.

    This function fixes the internal coordinates while preserving:

      * Every backbone atom position (no drift from the input fold)
      * Every χ angle (rotamer information is retained exactly)
      * Cross-residue geometry (peptide bonds, ϕ/ψ/ω)

    Implementation: extract internal coordinates with BioPython, overwrite
    sidechain hedra (bond lengths + bond angles) and non-χ dihedra with the
    canonical defaults from ``Bio.PDB.ic_data``, then re-emit Cartesian.
    No force field, no minimisation, fully deterministic — typical wall time
    is well under a second per 300-residue chain.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_pdb : str
        Destination PDB path. Parent directory is created if needed.

    Returns
    -------
    dict
        Diagnostics: ``{'n_hedra_reset', 'n_dihedra_primary_reset',
        'n_dihedra_secondary_reset'}``.
    """
    from Bio.PDB import PDBParser, PDBIO, ic_data

    chi_keys = _build_chi_keys()
    sec_defaults = _build_secondary_default_map()

    out_dir = os.path.dirname(os.path.abspath(output_pdb))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    structure = PDBParser(QUIET=True).get_structure("x", input_pdb)

    n_h = n_dp = n_ds = 0
    for chain in structure[0]:
        chain.atom_to_internal_coordinates(verbose=False)
        if chain.internal_coord is None:
            continue

        # --- Hedra: reset bond lengths + bond angles ---
        for h_key, hedron in chain.internal_coord.hedra.items():
            ak1, ak2, ak3 = h_key
            if not (ak1.akl[0] == ak2.akl[0] == ak3.akl[0]):
                continue  # cross-residue — leave alone
            atom_names = (ak1.akl[3], ak2.akl[3], ak3.akl[3])
            if any(a in _PROPAGATING_BACKBONE_ATOMS for a in atom_names):
                continue  # touches a propagating-backbone atom
            res_letter = ak1.akl[2]
            key = res_letter + atom_names[0] + res_letter + atom_names[1] + res_letter + atom_names[2]
            canon = ic_data.hedra_defaults.get(key)
            if canon is None:
                continue
            hedron.len12, hedron.angle, hedron.len23 = canon[0]
            n_h += 1

        # --- Dihedra: gather candidates (intra-residue, sidechain-only) ---
        dihedra_by_key = {}
        for d_key, dihedron in chain.internal_coord.dihedra.items():
            ak1, ak2, ak3, ak4 = d_key
            if not (ak1.akl[0] == ak2.akl[0] == ak3.akl[0] == ak4.akl[0]):
                continue
            atom_names = (ak1.akl[3], ak2.akl[3], ak3.akl[3], ak4.akl[3])
            if any(a in _PROPAGATING_BACKBONE_ATOMS for a in atom_names):
                continue
            res_letter = ak1.akl[2]
            key = "".join(res_letter + a for a in atom_names)
            dihedra_by_key[(res_letter, ak1.akl[0], key)] = dihedron

        # Primary defaults: in-ring dihedra and aromatic-closure dihedra
        for (res_letter, resnum, key), dihedron in dihedra_by_key.items():
            if key in chi_keys:
                continue
            canon = ic_data.dihedra_primary_defaults.get(key)
            if canon is None:
                continue
            dihedron.angle = canon[0]
            n_dp += 1

        # Secondary defaults: target = source_dihedron_angle + offset (e.g. CD2
        # in Phe/Tyr is the chi2 mirror, +180°)
        for (res_letter, resnum, key), dihedron in dihedra_by_key.items():
            if key in chi_keys:
                continue
            sec = sec_defaults.get(key)
            if sec is None:
                continue
            src_key, offset_deg = sec
            src_dihedron = dihedra_by_key.get((res_letter, resnum, src_key))
            if src_dihedron is None:
                continue
            dihedron.angle = ((src_dihedron.angle + offset_deg + 180.0) % 360.0) - 180.0
            n_ds += 1

        chain.internal_to_atom_coordinates(verbose=False)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    return {
        "n_hedra_reset": n_h,
        "n_dihedra_primary_reset": n_dp,
        "n_dihedra_secondary_reset": n_ds,
    }
