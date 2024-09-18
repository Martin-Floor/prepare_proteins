import os
import json
import argparse
from Bio import PDB

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', help='Path to the PELE output folder.')
parser.add_argument('com_groups', help='Json file containing atom definitions for group 1')
args=parser.parse_args()

pele_output = args.pele_output
with open(args.com_groups) as jf:
    com_groups = json.load(jf)

# Get atom names dictionary
parser = PDB.PDBParser()
atom_names = {}
for pdb in os.listdir(pele_output+'/input'):
    if pdb.endswith('_processed.pdb'):
        structure = parser.get_structure(pdb, pele_output+'/input/'+pdb)
        residue_atoms = {}
        for residue in structure.get_residues():
            chain = residue.get_parent()
            residue_atoms[(chain.id, residue.id[1])] = []
            for atom in residue:
                atom_names[(chain.id, residue.id[1], atom.name)] = atom.fullname.replace(' ','_')
                residue_atoms[(chain.id, residue.id[1])].append(atom.name)
        break

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

# Get PELE csts
pele_tasks = json_conf['commands'][0]['PeleTasks'][0]['metrics']

# Add com distance for each com distance group
for group in com_groups:
    task = {}
    task['type'] = 'com_distance'
    task['selection_group_1'] = {}
    task['selection_group_2'] = {}
    task['selection_group_1']['atoms'] = {}
    task['selection_group_2']['atoms'] = {}
    task['selection_group_1']['atoms']['ids'] = []
    task['selection_group_2']['atoms']['ids'] = []
    for atom in residue_atoms[group[0]]:
        atom_tuple = (group[0][0], group[0][1], atom)
        task['selection_group_1']['atoms']['ids'].append(atom_tuple[0]+':'+str(atom_tuple[1])+':'+atom_names[atom_tuple])
    for atom in residue_atoms[group[1]]:
        atom_tuple = (group[1][0], group[1][1], atom)
        task['selection_group_2']['atoms']['ids'].append(atom_tuple[0]+':'+str(atom_tuple[1])+':'+atom_names[atom_tuple])
    pele_tasks.append(task)

# Write pele.conf
with open(pele_output+'/pele.conf.tmp', 'w') as tmp:
    json.dump(json_conf, tmp, indent=4)

# Add quotes to pele reserved words
with open(pele_output+'/pele.conf', 'w') as pc:
    with open(pele_output+'/pele.conf.tmp') as tmp:
        for l in tmp:
            for w in pele_words:
                if w in l:
                    l = l.replace('"$'+w+'"','$'+w)
            pc.write(l)

os.remove(pele_output+'/pele.conf.tmp')

# Modify adaptive.conf
# Load modified pele.conf as json
with open(pele_output+'/adaptive.conf') as ac:
    adaptive_conf = json.load(ac)

spawning = {}
spawning['type'] = 'epsilon'
spawning['params'] = {}
spawning['params']['reportFilename'] = 'report'
spawning['params']['metricColumnInReport'] = task_index
spawning['params']['epsilon'] = epsilon
spawning['params']['condition'] = 'min'
spawning['density'] = {}
spawning['density']['type']= 'continuous'
adaptive_conf['spawning'] = spawning

# Write adaptive.conf
with open(pele_output+'/adaptive.conf', 'w') as ac:
    json.dump(adaptive_conf, ac, indent=4)
