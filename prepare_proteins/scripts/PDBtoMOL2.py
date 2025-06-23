from schrodinger import structure
from schrodinger.structutils.analyze import AslLigandSearcher
import shutil
import json
import os
import subprocess

import argparse
# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('--residue_names', default=None, help='Dictionary json file containing the residue names of each ligand.')
parser.add_argument('--change_ligand_name', default=False, action='store_true', help='Change the name of the ligand residue?')
parser.add_argument('--keep_pdbs', action='store_true', help='Do not remove the PDB files')

args=parser.parse_args()
residue_names = args.residue_names
change_ligand_name = args.change_ligand_name
keep_pdbs = args.keep_pdbs

if residue_names != None:
    with open(residue_names) as rnf:
        residue_names = json.load(rnf)
else:
    residue_names = 'LIG'

for pdb in os.listdir():
    if pdb.endswith('.pdb'):
        print(pdb)

        # Read PDB structure
        for i,st in enumerate(structure.StructureReader(pdb)):

            # Only process the first structure
            if i == 0:

                # Change the ligand name
                if change_ligand_name:
                    for chain in st.chain:
                        chain.name = 'L'
                        for residue in chain.residue:
                            if isinstance(residue_names, str):
                                residue.pdbres = residue_names
                            elif isinstance(residue_names, dict):
                                pdb_name = pdb.replace('.pdb', '')
                                residue.pdbres = residue_names[pdb_name]

                # Write the MAE file
                #st.write(pdb.replace('.pdb', '.mae'))
                #subprocess.run(["obabel", pdb, "-O", pdb.replace('.pdb', '.sdf')], check=True)
                output = pdb.replace('.pdb', '.mol2')
                command = "obabel -i pdb {pdb} -o mol2 > {output}".format(pdb=pdb, output=output)
                print(command)
                subprocess.run(command, shell=True, check=True)
                #subprocess.run(["obabel", pdb, "-O", pdb.replace('.pdb', '.mol2'), "--partialcharge", "gasteiger"], check=True)
                #st.write(pdb.replace('.pdb', '.mol2'))
                #mae_file = pdb.replace('.pdb', '.mae')
                #print(mae_file)
                #mol2_file = pdb.replace('.pdb', '.mol2')
                #st.write(mae_file)

		        # Convert to MOL2
                #subprocess.run(['$SCHRODINGER/utilities/structconvert', mae_file, mol2_file], check=True)
		
                if not keep_pdbs:
                    os.remove(pdb)