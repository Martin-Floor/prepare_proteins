import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('covalent_indexes', help='The residue indexes of the covalent residues separated by commas.')
args=parser.parse_args()

covalent_indexes = [int(x) for x in args.covalent_indexes.split(',')]

# Read dictionary for template atom names (with correct spaces)
atoms = False
template_atoms = {}
template_folder = 'DataLocal/Templates/OPLS2005/Protein/templates_generated'
for f in os.listdir('DataLocal/Templates/OPLS2005/Protein/templates_generated'):
    with open(template_folder+'/'+f) as t:
        for l in t:
            if len(l.split()) == 6:
                atoms = True
                continue
            elif l.startswith('NBON'):
                atoms = False
                continue
            if atoms and l != '':
                _atom_name = l.split()[4]
                atom_name = _atom_name.replace('_', '').strip()
                template_atoms[atom_name] = _atom_name

# Iterate processed PDBs to modify covalent residue
skip = []

# Modify processed PDB files
for f in os.listdir('input'):
    if 'processed' in f and f.endswith('.pdb'):
        # Check covalent ligands in PDB lines
        with open('input/'+f) as pdb:
            lines = pdb.readlines()
            for i,l in enumerate(lines):
                if l.startswith('ATOM') or l.startswith('HETATM'):
                    index, name, chain, resid = (int(l[7:12]), l[12:17].strip(), l[21], int(l[22:27]))
                    if resid in covalent_indexes:
                        if lines[i-1].startswith('TER'):
                            skip.append(i-1)
                        elif lines[i+1].startswith('TER'):
                            skip.append(i+1)
                        lines[i] = l.replace('HETATM', 'ATOM  ')

                        # Replace covalent residue atom names for template atom names
                        tan = template_atoms[name].replace('_', ' ')
                        lines[i] = lines[i][:12]+tan+lines[i][16:]

        with open('input/'+f, 'w') as pdb:
            for i,l in enumerate(lines):
                if i not in skip:
                    pdb.write(l)
