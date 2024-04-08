from schrodinger import structure
from schrodinger.structutils.analyze import AslLigandSearcher
import shutil
import json
import os

import argparse
# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('--residue_names', default=None, help='Dictionary json file containing the residue names of each ligand.')
args=parser.parse_args()
residue_names = args.residue_names

if residue_names != None:
    with open(residue_names) as rnf:
        residue_names = json.load(rnf)
else:
    residue_names = 'LIG'

for pdb in os.listdir():
    if pdb.endswith('.pdb'):
        # Read PDB structure
        for i,st in enumerate(structure.StructureReader(pdb)):
            if i == 0:
                for chain in st.chain:
                    chain.name = 'L'
                    for residue in chain.residue:
                        if isinstance(residue_names, str):
                            residue.pdbres = residue_names
                        elif isinstance(residue_names, dict):
                            pdb_name = pdb.replace('.pdb', '')
                            residue.pdbres = residue_names[pdb_name]
                st.write(pdb.replace('.pdb', '.mae'))
