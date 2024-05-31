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
def getCoordinates(structure):
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
    for chain in structure.get_chains():
        for residue in chain:
            for atom in residue:
                atom_tuple = (chain.id, residue.id[1], atom.name)
                coordinates[atom_tuple] = list(atom.coord)
    return coordinates

# Define aliases for atoms typically modified
aliases = {'H'  : ('H1', '1H'),
           'OW' : 'O',
           '1HW' : 'H1',
           '2HW' : 'H2'}

# Define PDB parser
parser = PDB.PDBParser()
io = PDB.PDBIO()

# Read original PDB
o_structure = parser.get_structure(original_pdb, original_pdb)
m_structure = parser.get_structure(modified_pdb, modified_pdb)

o_coordinates = getCoordinates(o_structure)
m_coordinates = getCoordinates(m_structure)

# Check for differences in atom and coordinates and update m_coordinates
for atom in m_coordinates:
    if atom not in o_coordinates:
        if atom[2] not in aliases:
            message = f'Atom {atom[2]} has not been defined in the aliases dictionary. '
            message += 'Please do it or contact the developer!'
            raise ValueError(message)
        if isinstance(aliases[atom[2]], tuple):
            i = 0
            while i < len(aliases[atom[2]]):
                alias_atom = (*atom[:2], aliases[atom[2]][i])
                if alias_atom in o_coordinates:
                    break
                i += 1
        else:
            alias_atom = (*atom[:2], aliases[atom[2]])
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
