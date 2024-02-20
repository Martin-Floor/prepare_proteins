import os
import json
import argparse
from Bio import PDB

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', help='Path to the PELE output folder.')
parser.add_argument('angles_dictionary', help='Json file containing angle definitions')
parser.add_argument('topology', help='Path to PDB file to extract full atom names')
args=parser.parse_args()

pele_output = args.pele_output
angles_dictionary = args.angles_dictionary
topology = args.topology

# Read angles as json file
with open(angles_dictionary) as jf:
    angles_dictionary = json.load(jf)

# Read topology structure
if not os.path.exists(topology):
    raise ValueError(f'Topology file {topology} was not found!')

parser = PDB.PDBParser()
topology_structure = parser.get_structure('topology', topology)

# Create full atom name mapping
fullname = {}
for atom in topology_structure.get_atoms():
    residue = atom.get_parent()
    chain = residue.get_parent()
    fullname[(chain.id, residue.id[1], atom.name)] = atom.fullname.replace(' ', '_')

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

metrics = json_conf['commands'][0]['PeleTasks'][0]['metrics']

# Add angle definitions
for a in angles_dictionary:

    angle_tag  = 'angle'
    angle_tag += '_'+a[0][0]+str(a[0][1])+a[0][2]
    angle_tag += '_'+a[1][0]+str(a[1][1])+a[1][2]
    angle_tag += '_'+a[2][0]+str(a[2][1])+a[2][2]

    pele_angle = {
        "type" : "atomsAngle",
        "tag"  : angle_tag,
    "selection_group_1" : {
        "atoms": { "ids" : [a[0][0]+":"+str(a[0][1])+":"+fullname[tuple(a[0])]]}},
    "selection_group_2" :{
        "atoms": { "ids" : [a[1][0]+":"+str(a[1][1])+":"+fullname[tuple(a[1])]]}},
    "selection_group_3" :{
        "atoms": { "ids" : [a[2][0]+":"+str(a[2][1])+":"+fullname[tuple(a[2])]]}}}

    metrics.append(pele_angle)

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
