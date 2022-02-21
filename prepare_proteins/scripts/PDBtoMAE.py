from schrodinger import structure
from schrodinger.structutils.analyze import AslLigandSearcher
import shutil
import os

for pdb in os.listdir():
    if pdb.endswith('.pdb'):
        # Read PDB structure
        st = [*structure.StructureReader(pdb)][0]
        st.write(pdb.replace('.pdb', '.mae'))
