from Bio import PDB
import argparse

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('embedded_pdb', default=None, help='PDB output from a mp_transform.linuxgccrelease\
membrane optimization.')
args=parser.parse_args()

embedded_pdb = args.embedded_pdb
name = embedded_pdb.replace('.pdb', '')

parser = PDB.PDBParser()
structure = parser.get_structure(name, embedded_pdb)

to_remove = {}
for residue in structure.get_residues():
    if residue.resname == 'MEM':
        mem_residue = residue
    elif residue.resname == 'EMB':
        emb_residue = residue
        chain = emb_residue.get_parent()
        if chain not in to_remove:
            to_remove[chain] = []
        to_remove[chain].append(emb_residue)

mem_coordinates = []
for atom in emb_residue:
    mem_coordinates.append(atom.coord)

for i,atom in enumerate(mem_residue):
    atom.coord = mem_coordinates[i]

for chain in to_remove:
    for residue in to_remove[chain]:
        chain.detach_child(residue.id)

io = PDB.PDBIO()
io.set_structure(structure)
io.save(embedded_pdb)
