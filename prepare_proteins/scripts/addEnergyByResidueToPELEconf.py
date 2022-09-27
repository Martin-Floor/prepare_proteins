import os
import json
import argparse
from Bio import PDB
import shutil

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', default=None, help='Path to the PELE output folder.')
parser.add_argument('--energy_type', default=None, help='Type eneergy by residue calculation.')
parser.add_argument('--peptide', action='store_true', default=False, help='Is this a peptide run?')
parser.add_argument('--ligand_index', default=1, help='Ligand index')
parser.add_argument('--ligand_energy_groups', help='json file containing the ligand groups to include.')
parser.add_argument('--new_version', action='store_true', default=False, help='Add the new PELE flag for energy by residue.')

args=parser.parse_args()
pele_output = args.pele_output
energy_type = args.energy_type
peptide = args.peptide
ligand_index = args.ligand_index
ligand_energy_groups = args.ligand_energy_groups

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

if not isinstance(ligand_energy_groups, type(None)):

    # Read ligand_energy_groups file
    with open(ligand_energy_groups) as jf:
        ligand_energy_groups = json.load(jf)

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

# Read receptor structure
parser = PDB.PDBParser()
structure = parser.get_structure('receptor', pele_output+'/input/receptor.pdb')

if new_version:
    byResidueFlag = 'nonBondingEnergyBySelection'
else:
    byResidueFlag = 'energyBySelection'

for residue in structure.get_residues():
    chain = residue.get_parent().id
    resid = residue.id[1]
    resname = residue.resname

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
        if peptide:
            if chain != 'L':
                metric = {
                'type': byResidueFlag,
                'tag' : 'Peptide_'+chain+':'+str(resid)+'_'+resname+'_'+et,
                'typeOfContribution' : et,
                'selection_group_1' : {
                    'chains': {'names': ['L']},
                    },
                'selection_group_2' : {
                    'links': {'ids': [chain+':'+str(resid)]},
                    }
                }
        else:
            metric = {
            'type': byResidueFlag,
            'tag' : 'L:'+str(ligand_index)+'_'+chain+':'+str(resid)+'_'+resname+'_'+et,
            'typeOfContribution' : et,
            'selection_group_1' : {
                'links': {'ids': ['L:'+str(ligand_index)]},
                },
            'selection_group_2' : {
                'links': {'ids': [chain+':'+str(resid)]},
                }
            }

        metrics_list.append(metric)

        if isinstance(ligand_energy_groups, dict):
            for group in ligand_energy_groups:
                # print(['L:'+a for a in ligand_energy_groups[group]])
                metric = {
                'type': byResidueFlag,
                'tag' : 'L:'+group+'_'+chain+':'+str(resid)+'_'+resname+'_'+et,
                'typeOfContribution' : et,
                'selection_group_1' : {
                    "atoms": {'ids': ['L:'+str(ligand_index)+':'+atom_map[a] for a in ligand_energy_groups[group]]},
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
