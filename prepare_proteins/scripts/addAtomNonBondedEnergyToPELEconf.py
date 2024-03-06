import os
import json
import argparse
from Bio import PDB
import shutil

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', default=None, help='Path to the PELE output folder.')
parser.add_argument('--energy_type', default=None, help='Type of energy by residue calculation.')
parser.add_argument('--target_atoms', default=None, help='Json file with atom tuples')
parser.add_argument('--protein_chain', default=None, help='Protein chain index.')
parser.add_argument('--new_version', action='store_true', default=False, help='Add the new PELE flag for atom nonbonded energy.')

args=parser.parse_args()
pele_output = args.pele_output
energy_type = args.energy_type
target_atoms = args.target_atoms
protein_chain = args.protein_chain
new_version = args.new_version

# Read ligand_energy_groups file
with open(args.target_atoms) as jf:
    target_atoms = json.load(jf)

### Ligand energy groups ###

def mapAtomNameToTemplateNames(template_file):

    # define mapping from index to template atom name
    atom_map = {}
    with open(template_file) as tf:
        cond = False
        for l in tf:
            if l.startswith('NBON'):
                cond = False
            if cond:
                atom_map[l.split()[4].replace('_','')] = l.split()[4]
            if l.startswith('LIG'):
                cond = True
    return atom_map

# Create a dictionary mapping atom names to four-position atom names with dashes
# Get path to ligand template file
lig_template_file = None
for ff in os.listdir(pele_output+'/DataLocal/Templates'):
    for f in os.listdir(pele_output+'/DataLocal/Templates/'+ff+'/HeteroAtoms'):
        if f == 'ligz':
            lig_template_file = pele_output+'/DataLocal/Templates/'+ff+'/HeteroAtoms/'+f
            break
    if lig_template_file != None:
        break

# Get dicionary mapping atom names to template atom names
atom_map = mapAtomNameToTemplateNames(lig_template_file)

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

if new_version:
    metricFlag = 'nonBondingEnergyBySelection'
else:
    metricFlag = 'energyBySelection'

for atom in target_atoms:
    chain = atom[0]
    resid = atom[1]
    atom_name = atom[2]

    # Add energy by residue metrics
    if energy_type == 'all':
        ebrt = ['all', 'lennard_jones', 'electrostatic', 'sgb']
    elif energy_type == 'lennard_jones':
        ebrt = ['lennard_jones']
    elif energy_type == 'electrostatic':
        ebrt = ['electrostatic']
    elif energy_type == 'sgb':
        ebrt = ['sgb']

    for et in ebrt:

        metric = {
        'type': metricFlag,
        'tag' : chain+':'+str(resid)+':'+atom_name+':'+et,
        'typeOfContribution' : et,
        'selection_group_1' : {
            "atoms": {'ids': [chain+':'+str(resid)+':'+atom_map[atom_name]]},
            },
        'selection_group_2' : {
            'chains': {'names': [protein_chain]},
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
