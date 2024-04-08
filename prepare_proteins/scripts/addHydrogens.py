from schrodinger import structure
from schrodinger.structutils import build
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_pdb', default=None, help='Path to the input PDB file.')
parser.add_argument('output_pdb', default=None, help='Path to the output PDB file.')
parser.add_argument('--indexes', default=None, help='Coma-separated list of indexes (no spaces) of atoms to which add hydrogens')
parser.add_argument('--add_bond', action='append', nargs='*', default=None, help='Add a bond order between two atoms: give them as at1,at2,order. Multiple argumetns can be defined.')
parser.add_argument('--covalent', action='store_true', help='Check if atom names match HN or HTX. Useful for covalent ligand preparation')

args=parser.parse_args()

input_pdb = args.input_pdb
output_pdb = args.output_pdb
covalent = args.covalent

# Convert indexes to list of integers
indexes = args.indexes
if indexes != None:
    indexes = [int(i) for i in indexes.split(',')]

# Convert add_bond to list if is None
add_bond = args.add_bond
if add_bond == None:
    add_bond = []

# Read PDB structure
st = structure.StructureReader.read(input_pdb)

# Get index of last atom before adding structures
max_atom_index = max([a.index for a in st.atom])

# Add specified bonds to structure
for bond in add_bond:
    at1 = int(bond[0].split(',')[0])
    at1 = [a for a in st.atom if a.index == at1][0]
    at2 = int(bond[0].split(',')[1])
    at2 = [a for a in st.atom if a.index == at2][0]
    order = int(bond[0].split(',')[2])
    at1.addBond(at2, order)

# Check if any atom are named as HN or HXT
if covalent:
    for a in st.atom:
        if a.pdbname.strip() in ['HN', 'HXT']:
            print('Warning atom name %s was found!' % a.pdbname.strip())
            print('Renaming it to %3s' % a.pdbname.strip()+'1')
            a.pdbname = '%3s' % a.pdbname.strip()+'1' # Check if HN1 is not taken

# Select atoms to add hydrogens
build.add_hydrogens(st, atom_list=indexes)

with structure.StructureWriter(output_pdb) as writer:
    writer.append(st)
