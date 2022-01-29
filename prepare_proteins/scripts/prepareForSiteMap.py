from schrodinger import structure
from schrodinger.structutils.analyze import AslLigandSearcher
import argparse
import shutil
import os

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('input_pdb', default=None, help='Input PDB with ligand')
parser.add_argument('output_folder', default=None, help='path to the output folder to store models')
args=parser.parse_args()
input_pdb = args.input_pdb
output_folder = args.output_folder

# Set output file names
input_name = input_pdb.split('/')[-1]
ligand_output = output_folder+'/'+input_name.replace('.pdb','_ligand.mae')
protein_output = output_folder+'/'+input_name.replace('.pdb','_protein.mae')

# Read PDB structure
st = [*structure.StructureReader(input_pdb)][0]
asl_searcher = AslLigandSearcher()

# Search ligand in structure
ligands = asl_searcher.search(st)

for lig in ligands:
    lig.st.write(ligand_output)
    for residue in lig.st.residue:
        ligand_name = residue.pdbres
    break

# Get ligand atoms
to_remove = []
for residue in st.residue:
    if residue.pdbres == ligand_name:
        for atom in residue.atom:
            to_remove.append(atom.index)

# Remove ligand atoms
st.deleteAtoms(to_remove)

# Save protein structure
st.write(protein_output)

# Modify ligand file to set it as the SiteMap ligand
f_m_ct = False
site_map = False
skip = False
with open(ligand_output+'.tmp', 'w') as lof:
    with open(protein_output) as lif:
        for l in lif:
            if not skip:
                lof.write(l)
            if l.startswith('f_m_ct'):
                f_m_ct = True
            elif l.startswith(' :::') and f_m_ct:
                site_map = True
                skip = False
                f_m_ct = False
                lof.write(l)
            elif l.startswith(' m_atom') and site_map:
                skip = False
                site_map = False
                lof.write(l)
            if f_m_ct and not skip:
                lof.write(' s_m_title\n')
                lof.write(' i_m_ct_format\n')
                skip = True
            if site_map and not skip:
                lof.write(' sitemap_1_ligand \n')
                lof.write('  2\n')
                skip = True

shutil.copyfile(ligand_output+'.tmp', ligand_output)
os.remove(ligand_output+'.tmp')

# Modify protein file to set it as SiteMap ligand
f_m_ct = False
site_map = False
skip = False
with open(protein_output+'.tmp', 'w') as pof:
    with open(protein_output) as pif:
        for l in pif:
            if not skip:
                pof.write(l)
            if l.startswith('f_m_ct'):
                f_m_ct = True
            elif l.startswith(' :::') and f_m_ct:
                site_map = True
                skip = False
                f_m_ct = False
                pof.write(l)
            elif l.startswith(' m_atom') and site_map:
                skip = False
                site_map = False
                pof.write(l)
            if f_m_ct and not skip:
                pof.write(' s_m_title\n')
                pof.write(' s_m_entry_id\n')
                pof.write(' s_m_entry_name\n')
                pof.write(' s_pdb_PDB_format_version\n')
                pof.write(' s_m_Source_Path\n')
                pof.write(' s_m_Source_File\n')
                pof.write(' i_m_Source_File_Index\n')
                pof.write(' i_m_ct_format\n')
                skip = True
            if site_map and not skip:
                pof.write(' sitemap_1_protein \n')
                pof.write('  1 \n')
                pof.write('  %s\n' % input_name.replace('.pdb', ''))
                pof.write('  3.0\n')
                pof.write('  %s\n' % input_name.replace('.pdb', '')) #Check
                pof.write('  %s\n' % input_name.replace('.pdb', '')) #Check
                pof.write('  1\n')
                pof.write('  2\n')
                skip = True

shutil.copyfile(protein_output+'.tmp', protein_output)
os.remove(protein_output+'.tmp')
