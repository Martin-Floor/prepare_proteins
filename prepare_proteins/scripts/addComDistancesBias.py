import os
import json
import argparse
from Bio import PDB

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', help='Path to the PELE output folder.')
parser.add_argument('com_group_1', help='Json file containing atom definitions for group 1')
parser.add_argument('com_group_2', help='Json file containing atom definitions for group 2')
parser.add_argument('--epsilon', default=0.50, help='Bias epsilon parameter')
args=parser.parse_args()

pele_output = args.pele_output
with open(args.com_group_1) as jf:
    com_group_1 = json.load(jf)
with open(args.com_group_2) as jf:
    com_group_2 = json.load(jf)
epsilon = float(args.epsilon)

# Get atom names dictionary
parser = PDB.PDBParser()
atom_names = {}
for pdb in os.listdir(pele_output+'/input'):
    if pdb.endswith('_processed.pdb'):
        structure = parser.get_structure(pdb, pele_output+'/input/'+pdb)
        for a in structure.get_atoms():
            r = a.get_parent()
            c = r.get_parent()
            atom_names[(c.id, r.id[1], a.name)] = a.fullname.replace(' ','_')
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
random_tasks = [1 for t in pele_tasks if t['type'] == 'random']

# Add to pint bias
task = {}
task['type'] = 'com_distance'
task['selection_group_1'] = {}
task['selection_group_2'] = {}
task['selection_group_1']['atoms'] = {}
task['selection_group_2']['atoms'] = {}
task['selection_group_1']['atoms']['ids'] = []
task['selection_group_2']['atoms']['ids'] = []
for atom in com_group_1:
    task['selection_group_1']['atoms']['ids'].append(atom[0]+':'+str(atom[1])+':'+atom_names[tuple(atom)])
for atom in com_group_2:
    task['selection_group_2']['atoms']['ids'].append(atom[0]+':'+str(atom[1])+':'+atom_names[tuple(atom)])
pele_tasks.append(task)
task_index = len(pele_tasks) - len(random_tasks) + 4

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
