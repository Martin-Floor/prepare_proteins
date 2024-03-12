import os
import json
from Bio import PDB
import argparse

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', help='Path to the PELE output folder.')
parser.add_argument('topology', help='Path to PDB file to extract full atom names')
args=parser.parse_args()

pele_output = args.pele_output
topology = args.topology

# Read topology structure
if not os.path.exists(topology):
    raise ValueError(f'Topology file {topology} was not found!')

parser = PDB.PDBParser()
topology_structure = parser.get_structure('topology', topology)

# Get coordinates
coordinates = {}
for atom in topology_structure.get_atoms():
    residue = atom.get_parent()
    chain = residue.get_parent()
    atom_tuple = (chain.id, residue.id[1], atom.name)
    coordinates[atom_tuple] = atom.coord

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

constraints = json_conf['commands'][0]['constraints']

for c in constraints:
    if c['type'] == 'constrainAtomToPosition':
        chain, residue, atom = c['constrainThisAtom'].split(':')
        residue = int(residue)
        atom = atom.replace('_', '')
        c['toThisPoint'] = list(coordinates[(chain, residue, atom)])

json_conf['commands'][0]['constraints'] = constraints

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
