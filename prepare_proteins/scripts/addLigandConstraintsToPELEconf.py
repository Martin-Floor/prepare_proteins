import os
import json
import argparse
from Bio import PDB
import shutil

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', default=None, help='Path to the PELE output folder.')
parser.add_argument('--constraint_value', default=5.0, help='Constraint energy constant.')

args=parser.parse_args()
pele_output = args.pele_output
constraint_value = float(args.constraint_value)

# Read input PDB file
input_pdb = [pele_output+'/input/'+pdb for pdb in os.listdir(pele_output+'/input') if pdb.endswith('_processed.pdb')][0]
parser = PDB.PDBParser()
structure = parser.get_structure(input_pdb, input_pdb)
ligand_atoms = []
for c in structure.get_chains():
    if c.id == 'L':
        for r in c:
            for a in r:
                ligand_atoms.append((c.id, r.id[1], a.fullname.replace(' ', '_')))

# Modify pele.conf file to remove unquoted words
pele_words = ['COMPLEXES', 'PELE_STEPS', 'SEED']
with open(pele_output+'/pele.conf.tmp', 'w') as tmp:
    with open(pele_output+'/pele.conf') as pc:
        for l in pc:
            for w in pele_words:
                if w in l:
                    l = l.replace('$'+w, '"$'+w+'"')
                    break
            tmp.write(l)

# Load modified pele.conf as json
with open(pele_output+'/pele.conf.tmp') as tmp:
    json_conf = json.load(tmp)

# Add ligand contraints
constraints_list = json_conf['commands'][0]['constraints']
for a in ligand_atoms:
    atom_cst = {}
    atom_cst['type'] = 'constrainAtomToPosition'
    atom_cst['springConstant'] = constraint_value
    atom_cst['equilibriumDistance'] = 0.0
    atom_cst['constrainThisAtom'] = ':'.join([str(x) for x in a])
    constraints_list.append(atom_cst)

# Write pele.conf
with open(pele_output+'/pele.conf.tmp', 'w') as tmp:
    json.dump(json_conf, tmp, indent=1)

# Add quotes to pele reserved words
with open(pele_output+'/pele.conf', 'w') as pc:
    with open(pele_output+'/pele.conf.tmp') as tmp:
        for l in tmp:
            for w in pele_words:
                if w in l:
                    l = l.replace('"$'+w+'"','$'+w)
            pc.write(l)

os.remove(pele_output+'/pele.conf.tmp')
