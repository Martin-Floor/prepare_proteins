import os
import json
import argparse
from Bio import PDB
from Bio.PDB.Polypeptide import three_to_one

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', help='Path to the PELE output folder.')
parser.add_argument('--membrane_residues', help='Comma-separated-list of membrane residue indexes.')
parser.add_argument('--mem_spring_constant', default=0.5, help='Spring constant for membrane residues')
parser.add_argument('--mem_edge_spring_constant', default=0.5, help='Spring constant for the membrane edge residues')
parser.add_argument('--nomem_spring_constant', default=0.5, help='Spring constant for no-membrane residues')

parser.add_argument('--cst_frequency', default=4, help='Frequency to constraint residues.')
args=parser.parse_args()

pele_output = args.pele_output
membrane_residues = [int(r) for r in args.membrane_residues.split(',')]
mem_spring_constant = float(args.mem_spring_constant)
mem_edge_spring_constant = float(args.mem_edge_spring_constant)
nomem_spring_constant = float(args.nomem_spring_constant)
cst_frequency = int(args.cst_frequency)

protein_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

parser = PDB.PDBParser()
# Get input PDB
for f in os.listdir(pele_output+'/input'):
    if 'processed' in f and f.endswith('.pdb'):
        structure = parser.get_structure('protein', pele_output+'/input/'+f)
        break

if len(list(structure.get_chains())) > 2:
    raise ValueError('No more than one protein chains can be handle at the moment!')

csts = []
count = 0
for r in structure.get_residues():
    if r.resname in ['HID', 'HIE', 'HIP']:
        resname = 'HIS'
    else:
        resname = r.resname
    try:
        three_to_one(resname)
    except:
        continue
    if three_to_one(resname) in protein_aa:
        count += 1
    if count % cst_frequency == 0:
        cst = {}
        cst['type'] = 'constrainAtomToPosition'
        if r.id[1] in membrane_residues:
            cst['springConstant'] = mem_spring_constant
        elif r.id[1]-1 in membrane_residues or r.id[1]+1 in membrane_residues:
            cst['springConstant'] = mem_edge_spring_constant
        else:
            cst['springConstant'] = nomem_spring_constant
        cst['equilibriumDistance'] = 0.0
        cst['constrainThisAtom'] = r.get_parent().id+':'+str(r.id[1])+':'+'_CA_'
        csts.append(cst)

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
pele_csts = json_conf['commands'][0]['constraints']

# Remove CA constraints from list
remove = []
for cst in pele_csts:
    if cst['constrainThisAtom'].split(':')[-1] == '_CA_':
        remove.append(cst)
for index in remove:
    pele_csts.remove(index)

# Add new membrane csts
pele_csts += csts

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
