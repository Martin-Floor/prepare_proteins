from schrodinger import structure
from schrodinger.structutils.analyze import AslLigandSearcher
import shutil
import json
import os

import argparse
# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('--residue_names', default=None, help='Dictionary json file containing the residue names of each ligand.')
parser.add_argument('--change_ligand_name', default=False, action='store_true', help='Change the name of the ligand residue?')
parser.add_argument('--modify_maes', action='store_true', help='Modify input MAE files')
parser.add_argument('--assign_pdb_names', action='store_true', help='Assign PDB names to PDB and MAE files.')

args=parser.parse_args()
residue_names = args.residue_names
change_ligand_name = args.change_ligand_name
modify_maes = args.modify_maes
assign_pdb_names = args.assign_pdb_names

if residue_names and os.path.exists(residue_names):
    with open(residue_names) as rnf:
        residue_names = json.load(rnf)
    change_ligand_name = True
elif isinstance(residue_names, str) and len(residue_names) == 3:
    residue_names = residue_names
    change_ligand_name = True
elif change_ligand_name:
    residue_names = 'LIG'

for mae in os.listdir():
    if mae.endswith('.mae'):
        # Read PDB structure
        for i,st in enumerate(structure.StructureReader(mae)):
            if i == 0:
                for chain in st.chain:
                    chain.name = 'L'
                    for residue in chain.residue:
                        if change_ligand_name:
                            if isinstance(residue_names, str):
                                residue.pdbres = residue_names
                            elif isinstance(residue_names, dict):
                                mae_name = pdb.replace('.mae', '')
                                residue.pdbres = residue_names[mae_name]

                        element_counts = {}
                        if assign_pdb_names:
                            for atom in residue.atom:
                                element_counts.setdefault(atom.element, 0)
                                element_counts[atom.element] += 1
                                base_name = atom.element+str(element_counts[atom.element])
                                if len(base_name) == 1:
                                    atom_name = ' '+base_name+'  '
                                elif len(base_name) == 2:
                                    atom_name = ' '+base_name+' '
                                elif len(base_name) == 3:
                                    atom_name = ' '+base_name
                                else:
                                    atom_name = base_name

                                atom.name = atom_name
                                atom.pdbname = atom_name

                st.write(mae.replace('.mae', '.pdb'))
                if modify_maes:
                    st.write(mae)
