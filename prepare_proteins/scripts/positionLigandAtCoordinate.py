import argparse
from schrodinger import structure
from schrodinger.structutils import transform
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('protein_file', default=None, help='Path to input protein structure file.')
parser.add_argument('ligand_file', default=None, help='Path to input protein ligand file.')
parser.add_argument('coordinate', help='comma-separated x,y,z coordinates')
parser.add_argument('--separator', default='-', help='separator for protein,ligand,pose name.')
parser.add_argument('--pele_poses', action='store_true', help='Write in the format for setUpPELECalculation()')

args=parser.parse_args()

protein_file = args.protein_file
ligand_file = args.ligand_file
x,y,z = [float(x) for x in args.coordinate.split(',')]
coordinate = np.array([x,y,z])
separator = args.separator
pele_poses = args.pele_poses

protein_structure = structure.StructureReader(protein_file)
ligand_structure = structure.StructureReader(ligand_file)

for lst in ligand_structure:

    # Change ligand chain to L
    for chain in lst.chain:
        chain.name = 'L'

    for residue in lst.residue:
        residue.pdbres = 'LIG'
        residue.resnum = 1

    centroid = np.average(lst.getXYZ(), axis=0)
    v = coordinate-centroid
    transform.translate_structure(lst, x=v[0], y=v[1], z=v[2])

for pst in protein_structure:
    st = pst.merge(lst)
    if pele_poses:
        os.remove(protein_file)
        protein_file = protein_file.replace('.pdb', separator+'0.pdb')
    st.write(protein_file)
