import os
import json
import argparse
from Bio import PDB
import shutil

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', default=None, help='Path to the PELE output folder.')
parser.add_argument('--energy_type', default=None, help='Path to the PELE output folder.')
args=parser.parse_args()
pele_output = args.pele_output
energy_type = args.energy_type

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

# Modify json parameters
metrics_list = json_conf['commands'][0]['PeleTasks'][0]['metrics']

# Read receptor structure
parser = PDB.PDBParser()
structure = parser.get_structure('receptor', pele_output+'/input/receptor.pdb')

for residue in structure.get_residues():
    chain = residue.get_parent().id
    resid = residue.id[1]
    resname = residue.resname

    # Add energy by residue metrics
    if energy_type == 'all':
        ebrt = ['lennard_jones', 'electrostatic', 'sgb']
    elif energy_type == 'lennard_jones':
        ebrt = ['lennard_jones']
    elif energy_type == 'electrostatic':
        ebrt = ['electrostatic']
    elif energy_type == 'sgb':
        ebrt = ['sgb']

    for et in ebrt
        metric = {
        'type': 'energyBySelection',
        'tag' : 'L:1_'+chain+':'+str(resid)+'_'+resname+'_'+et,
        'typeOfContribution' : et,
        'selection_group_1' : {
            'links': {'ids': ['L:1']},
            },
        'selection_group_2' : {
            'links': {'ids': [chain+':'+str(resid)]},
            }
        }

    metrics_list.append(metric)

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
