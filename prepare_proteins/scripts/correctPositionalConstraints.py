import os
import json
from Bio import PDB
import argparse
import pandas as pd
import ast
import numpy as np

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', help='Path to the PELE output folder.')
parser.add_argument('topology', help='Path to PDB file to extract full atom names')
parser.add_argument('csv', help='Path to the csv file with the residues to be constrained')
parser.add_argument('separator', default="@", help='The separator for files')
args=parser.parse_args()

pele_output = args.pele_output
topology = args.topology
csv = pd.read_csv(args.csv, index_col=0)
separator = args.separator

# Read topology structure
if not os.path.exists(topology):
    raise ValueError(f'Topology file {topology} was not found!')

index = index = os.path.basename(topology).split(separator)[0].split("-")

if len(index) == 3:
    index = index[1]
elif len(index) == 2:
    index = csv[csv["uniprot accession"] == index[1]].index[0]
else:
    index = index[0]

resids = ast.literal_eval(csv.loc[index]["catalytic resids"])
motif = csv.loc[index]["catalytic motif"]

parser = PDB.PDBParser()
topology_structure = parser.get_structure('topology', topology)

res = list(topology_structure.get_residues())

b_cd1 = False
b_cd2 = False
b_ne2_5 = False
b_ne2_12 = False
for r in res:
    if r.get_parent().id == "M" and r.id[1] == 1:
        cu_coord = r["CU"].get_coord()
    if r.id[1] == resids[11]:
        if motif[11] == "I":
            cd1 = r["CD1"].get_coord()
        elif motif[11] == "V":
            cd1 = r["CG2"].get_coord()
        b_cd1 = True
    if r.id[1] == resids[-1]:
        if motif[-1] == "F":
            at1 = "CD1"
            cd2 = r["CD1"].get_coord()
        elif motif[-1] == "L":
            at1 = "CD2"
            cd2 = r["CD2"].get_coord()
        b_cd2 = True
    if r.id[1] == resids[5]:
        ne2_5 = r["NE2"].get_coord()
        b_ne2_5 = True
    if r.id[1] == resids[12]:
        b_ne2_12 = True
        ne2_12 = r["NE2"].get_coord()

if b_cd1:            
    dist_cd1_cu = float(np.linalg.norm(cu_coord - cd1))
if b_cd2:
    dist_cd2_cu = float(np.linalg.norm(cu_coord - cd2))
if b_ne2_5:
    distance_ne2_5_cu = float(np.linalg.norm(ne2_5 - cu_coord))
if b_ne2_12:
    distance_ne2_12_cu = float(np.linalg.norm(ne2_12 - cu_coord))

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

constraints[0:0] = [{'type': 'constrainAtomsDistance', 'springConstant': 50.0, 'equilibriumDistance': dist_cd1_cu, 'constrainThisAtom':  f'A:{resids[11]}:_CD1', 'toThisOtherAtom': 'M:1:CU__'},
                    {'type': 'constrainAtomsDistance', 'springConstant': 50.0, 'equilibriumDistance': dist_cd2_cu, 'constrainThisAtom':  f'A:{resids[-1]}:_{at1}', 'toThisOtherAtom': 'M:1:CU__'},
                    {'type': 'constrainAtomsDistance', 'springConstant': 50.0, 'equilibriumDistance': distance_ne2_5_cu, 'constrainThisAtom':  f'A:{resids[5]}:_NE2', 'toThisOtherAtom': 'M:1:CU__'},
                    {'type': 'constrainAtomsDistance', 'springConstant': 50.0, 'equilibriumDistance': distance_ne2_12_cu, 'constrainThisAtom':  f'A:{resids[12]}:_NE2', 'toThisOtherAtom': 'M:1:CU__'},
                    {'type': 'constrainAtomToPosition', 'springConstant': 50.0, 'equilibriumDistance': 0.0, "constrainThisAtom": "M:1:CU__", "toThisPoint": [float(x) for x in cu_coord.tolist()]}]


for c in constraints:
    if c['type'] == 'constrainAtomToPosition':
        chain, residue, atom = c['constrainThisAtom'].split(':')
        residue = int(residue)
        atom = atom.replace('_', '')
        c['toThisPoint'] = [float(x) for x in coordinates[(chain, residue, atom)]]

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
