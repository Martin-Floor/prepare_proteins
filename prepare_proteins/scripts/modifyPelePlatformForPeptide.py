import os
import json
import argparse
from Bio import PDB
import shutil

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', default=None, help='Path to the PELE output folder.')
parser.add_argument('input_pdb', default=None, help='Name of input PDB')
args=parser.parse_args()
pele_output = args.pele_output
input_pdb = args.input_pdb

# Modify adaptive.conf file to remove unquoted words
pele_words = ['COMPLEXES', 'PELE_STEPS', 'SEED']
with open(pele_output+'/adaptive.conf.tmp', 'w') as tmp:
    with open(pele_output+'/adaptive.conf') as pc:
        for l in pc:
            for w in pele_words:
                if w in l:
                    l = l.replace('$'+w, '"$'+w+'"')
                    break
            tmp.write(l)

# Load modified pele.conf as json
with open(pele_output+'/adaptive.conf.tmp') as tmp:
    json_conf = json.load(tmp)

# Modify json parameters
del json_conf['clustering']['params']['ligandResname']
json_conf['clustering']['params']['ligandChain'] = 'L'

# Write pele.conf
with open(pele_output+'/adaptive.conf.tmp', 'w') as tmp:
    json.dump(json_conf, tmp, indent=1)

# Add quotes to pele reserved words
with open(pele_output+'/adaptive.conf', 'w') as pc:
    with open(pele_output+'/adaptive.conf.tmp') as tmp:
        for l in tmp:
            for w in pele_words:
                if w in l:
                    l = l.replace('"$'+w+'"','$'+w)
            pc.write(l)

os.remove(pele_output+'/adaptive.conf.tmp')

# Remove phantom ligand from processed PDB
processed = pele_output+'/input/'+input_pdb.replace('.pdb', '_processed.pdb')
peptide_residues = []
with open(processed+'.tmp', 'w') as tpf:
    with open(processed) as pf:
        for l in pf:
            if 'XXX' not in l:
                tpf.write(l)
            # Get peptide residues ids
            if len(l) > 5:
                chain = l.split()[4]
                if chain == 'L':
                    resid = l.split()[5]
                    peptide_residues.append(int(resid))
peptide_residues = list(set(peptide_residues))

shutil.move(processed+'.tmp', processed)

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

# Modify JSON parameters
# Omit peptide links from ANM
linksToOmit = {}
linksToOmit['links'] = {}
linksToOmit['links']['ranges'] = ['L:'+str(min(peptide_residues))+' '+'L:'+str(max(peptide_residues))]
ANM_dict = json_conf['commands'][0]['ANM']
ANM_dict['linksToOmit'] = linksToOmit

# Remove constraints added for peptide
keep_cst = []
for cst in json_conf['commands'][0]['constraints']:
    if not cst['constrainThisAtom'].startswith('L:'):
        keep_cst.append(cst)
json_conf['commands'][0]['constraints'] = keep_cst

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
