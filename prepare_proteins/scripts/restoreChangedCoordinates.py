import argparse
import numpy as np
from Bio import PDB

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('original_pdb', default=None, help='Path to the PDB file with the original coordinates')
parser.add_argument('modified_pdb', default=None, help='Path to the PDB file with the modified coordinates')
args=parser.parse_args()

original_pdb = args.original_pdb
modified_pdb = args.modified_pdb

# Define functions
def getCoordinates(structure, return_residue_names=False):
    """
    Get atom coordinates as a dictionary with atom tuples as keys (i.e., (chain_id, residue_id, atom_name))

    Parameters
    ==========
    structure : Bio.PDB.Structure.Structure
        Input structure objet

    Returns
    =======
    coordinates : dict
        Coordinates as an atom_tuple dictionary
    """
    coordinates = {}
    residue_names = {}
    for chain in structure.get_chains():
        for residue in chain:
            for atom in residue:
                atom_tuple = (chain.id, residue.id[1], atom.name)
                coordinates[atom_tuple] = list(atom.coord)
                residue_names[atom_tuple] = residue.resname
    if return_residue_names:
        return coordinates, residue_names
    else:
        return coordinates

def getTerminalAtoms(structure):
    """
    Return atoms belongingr to the N- and C-terminus.
    """
    NT = []
    CT = []
    for chain in structure.get_chains():
        first_residue = False
        chain_residues = list(chain)
        for residue in chain_residues:
            if not first_residue and residue.id[0] == ' ':
                first_residue = residue.id[1]
            for atom in residue:
                atom_tuple = (chain.id, residue.id[1], atom.name)
                if residue.id[1] == first_residue:
                    NT.append(atom_tuple)
        last_residue = False
        for residue in reversed(chain_residues):
            if not last_residue and residue.id[0] == ' ':
                last_residue = residue.id[1]
            for atom in residue:
                atom_tuple = (chain.id, residue.id[1], atom.name)
                if residue.id[1] == last_residue:
                    CT.append(atom_tuple)
    return NT, CT

# Define aliases for modified atoms
aliases = {
    'NTER' : {
        'H' : '1H',
    },
    'CTER' : {
        '' : '',
    },
    'ALA' : {
        'HB1' : '1HB',
        'HB2' : '2HB',
        'HB3' : '3HB'
    },
    'GLU' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
        'HG2' : '1HG',
        'HG3' : '2HG'
    },
    'PRO' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
        'HG2' : '1HG',
        'HG3' : '2HG',
        'HD2' : '1HD',
        'HD3' : '2HD'
    },
    'GLY' : {
        'HA2' : '1HA',
        'HA3' : '2HA'
    },
    'LEU' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
        'HD11' : '1HD1',
        'HD12' : '2HD1',
        'HD13' : '3HD1',
        'HD21' : '1HD2',
        'HD22' : '2HD2',
        'HD23' : '3HD2'
    },
    'ASN' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
        'HD21' : '1HD2',
        'HD22' : '2HD2'
    },
    'SER' : {
        'HB2' : '1HB',
        'HB3' : '2HB'
    },
    'LYS' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
        'HG2' : '1HG',
        'HG3' : '2HG',
        'HD2' : '1HD',
        'HD3' : '2HD',
        'HE2' : '1HE',
        'HE3' : '2HE',
        'HZ1' : '1HZ',
        'HZ2' : '2HZ',
        'HZ3' : '3HZ'
    },
    'VAL' : {
        'HG11' : '1HG1',
        'HG12' : '2HG1',
        'HG13' : '3HG1',
        'HG21' : '1HG2',
        'HG22' : '2HG2',
        'HG23' : '3HG2',
    },
    'ASP' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
    },
    'HID' : {
        'HB2' : '1HB',
        'HB3' : '2HB'
    },
    'TRP' : {
        'HB2' : '1HB',
        'HB3' : '2HB'
    },
    'ARG' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
        'HG2' : '1HG',
        'HG3' : '2HG',
        'HD2' : '1HD',
        'HD3' : '2HD',
        'HH11' : '1HH1',
        'HH12' : '2HH1',
        'HH21' : '1HH2',
        'HH22' : '2HH2'
    },
    'ILE' : {
        'HG12' : '1HG1',
        'HG13' : '2HG1',
        'HG21' : '1HG2',
        'HG22' : '2HG2',
        'HG23' : '3HG2',
        'HD11' : '1HD1',
        'HD12' : '2HD1',
        'HD13' : '3HD1',
    },
    'CYT' : {
        'HB2' : '1HB',
        'HB3' : '2HB'
    },
    'THR' : {
        'HG21' : '1HG2',
        'HG22' : '2HG2',
        'HG23' : '3HG2',
    },
    'TYR' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
    },
    'GLN' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
        'HG2' : '1HG',
        'HG3' : '2HG',
        'HE21' : '1HE2',
        'HE22' : '2HE2',
    },
    'PHE' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
    },
    'HIE' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
    },
    'ASH' : {
        'HB2' : '1HB',
        'HB3' : '2HB'
    },
    'HIP' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
    },
    'MET' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
        'HG2' : '1HG',
        'HG3' : '2HG',
        'HE1' : '1HE',
        'HE2' : '2HE',
        'HE3' : '3HE',
    },
    'CYS' : {
        'HB2' : '1HB',
        'HB3' : '2HB',
    },
    'HOH' : {
        'OW' : 'O',
        '1HW' : 'H1',
        '2HW' : 'H2',
    }

}

# Define PDB parser
parser = PDB.PDBParser()
io = PDB.PDBIO()

# Read original PDB
o_structure = parser.get_structure(original_pdb, original_pdb)
m_structure = parser.get_structure(modified_pdb, modified_pdb)

o_coordinates = getCoordinates(o_structure)
m_coordinates, m_residue_names = getCoordinates(m_structure, return_residue_names=True)

NT, CT = getTerminalAtoms(m_structure)

# Check for differences in atom and coordinates and update m_coordinates
for atom in m_coordinates:

    if atom not in o_coordinates:

        # Define residue name
        resname = m_residue_names[atom]

        if atom in NT and atom[2] not in aliases['NTER'] and atom[2] not in aliases[resname]:
            message = f'N-terminus atom {atom[2]} from {resname} {atom} has not been defined in the aliases dictionary. '
            message += 'Please do it or contact the developer!'
            raise ValueError(message)

        if atom in CT and atom[2] not in aliases['CTER'] and atom[2] not in aliases[resname]:
            message = f'C-terminus atom {atom[2]} from {resname} {atom} has not been defined in the aliases dictionary. '
            message += 'Please do it or contact the developer!'
            raise ValueError(message)

        if atom[2] not in aliases[resname]:
            message = f'Atom {atom[2]} from {resname} {atom} has not been defined in the aliases dictionary. '
            message += 'Please do it or contact the developer!'
            raise ValueError(message)

        if atom in NT and atom[2] in aliases['NTER']:
            resname = 'NTER'
        if atom in NT and atom[2] in aliases['CTER']:
            resname = 'CTER'

        if isinstance(aliases[resname][atom[2]], tuple):
            i = 0
            while i < len(aliases[resname][atom[2]]):
                alias_atom = (*atom[:2], aliases[resname][atom[2]][i])
                if alias_atom in o_coordinates:
                    break
                i += 1
        else:
            alias_atom = (*atom[:2], aliases[resname][atom[2]])

        if alias_atom not in o_coordinates:
            message = f'Atom {atom} was not found in PDB {original_pdb}, please check your aliases definitions.'
            raise ValueError(message)
        m_coordinates[atom] = o_coordinates[alias_atom]
    else:
        if o_coordinates[atom] != m_coordinates[atom]:
            m_coordinates[atom] = o_coordinates[atom]

# Change back modified coordinates into the modified PDB
for chain in m_structure.get_chains():
    for residue in chain:
        for atom in residue:
            atom_tuple = (chain.id, residue.id[1], atom.name)
            atom.coord = m_coordinates[atom_tuple]

# Write modified PDB new coordinates
io.set_structure(m_structure)
io.save(modified_pdb)
