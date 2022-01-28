from schrodinger import structure
from schrodinger.structutils.analyze import AslLigandSearcher
import argparse

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('input_pdb', default=None, help='Input PDB with ligand')
parser.add_argument('output_folder', default=None, help='path to the output folder to store models')
args=parser.parse_args()
input_pdb = args.input_pdb
output_folder = args.output_folder

# Read PDB structure
st = [*structure.StructureReader(input_pdb)][0]
asl_searcher = AslLigandSearcher()
input_name = input_pdb.split('/')[-1]

# Search ligand in structure
ligands = asl_searcher.search(st)
for lig in ligands:
    lig.st.write(output_folder+'/'+input_name.replace('.pdb','_ligand.mae'))
    for residue in lig.st.residue:
        ligand_name = residue.pdbres
    break

# Get ligand atoms
to_remove = []
for residue in st.residue:
    if residue.pdbres == ligand_name:
        for atom in residue.atom:
            to_remove.append(atom.index)

# Remove ligand atoms
st.deleteAtoms(to_remove)

# Save protein structure
st.write(output_folder+'/'+input_name.replace('.pdb','_protein.mae'))
