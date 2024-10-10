import argparse
import schrodinger
from schrodinger.protein import captermini
from schrodinger.structure import StructureReader, StructureWriter
import os

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('models_folder', help='Path to the folder where the PDB inputs are stored.')
parser.add_argument('output_folder', help='Path to the folder where the PDB outputs will be stored.')
parser.add_argument('--rosetta_style_caps', action='store_true', help='Change cap atom names to match those of Rosetta.')

args = parser.parse_args()

models_folder = args.models_folder
output_folder = args.output_folder
rosetta_style_caps = args.rosetta_style_caps

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop over files in the input folder
for f in os.listdir(models_folder):

    # Process only PDB files
    if not f.endswith('.pdb'):
        continue

    # Load structure
    input_path = os.path.join(models_folder, f)
    st = next(StructureReader(input_path))

    # Cap termini
    capped_st = captermini.cap_termini(st)

    # Modify residue names in the capped structure
    if rosetta_style_caps:

        # Get protein residues and added terminal caps
        residues = [r for r in capped_st.residue if r.getAlphaCarbon() or r.pdbres.strip() in ['ACE', 'NMA']]

        for i,residue in enumerate(residues):

            if residue.pdbres == 'ACE ':

                for atom in residue.atom:
                    if atom.pdbname.strip() == 'CH3':
                        atom.pdbname = ' CP2'
                    elif atom.pdbname.strip() == 'C':
                        atom.pdbname = ' CO '
                    elif atom.pdbname.strip() == 'O':
                        atom.pdbname = ' OP1'
                    elif atom.pdbname.strip() == '1H':
                        atom.pdbname = '1HP2'
                    elif atom.pdbname.strip() == '2H':
                        atom.pdbname = '2HP2'
                    elif atom.pdbname.strip() == '3H':
                        atom.pdbname = '3HP2'

                # Assign the same residue name and numer as the N-terminal
                residues[i].pdbres = residues[i+1].pdbres
                residues[i].resnum = residues[i+1].resnum

            elif residue.pdbres == 'NMA ':
                for atom in residue.atom:
                    print(atom.pdbname)
                    if atom.pdbname.strip() == 'H':
                        atom.pdbname = 'HN2 '
                    elif atom.pdbname.strip() == 'CA':
                        atom.pdbname = ' C  '
                    elif atom.pdbname.strip() == '1HA':
                        atom.pdbname = ' H1 '
                    elif atom.pdbname.strip() == '2HA':
                        atom.pdbname = ' H2 '
                    elif atom.pdbname.strip() == '3HA':
                        atom.pdbname = ' H3 '

    # Save modified structure to the output folder
    output_path = os.path.join(output_folder, f)
    with StructureWriter(output_path) as writer:
        writer.append(capped_st)

print("Processing completed.")
