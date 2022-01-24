from schrodinger import structure
import argparse

### Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('smile', default=None, help='SMILE string')
parser.add_argument('output_mae', default=None, help='Name of the output mae file.')
args=parser.parse_args()
smile = args.smile
output_mae = args.output_mae

# Read smile and convert it to 3D
smile_st = structure.SmilesStructure(smile)
ligand_st = smile_st.get3dStructure(smile_st)

# Write mae file
with structure.StructureWriter(output_mae) as writer:
    writer.append(ligand_st)
