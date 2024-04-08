from peleffy.topology import Molecule
from peleffy.forcefield import OPLS2005ForceField
from peleffy.topology import Topology
from peleffy.template import Impact
from peleffy.topology.rotamer import RotamerLibrary

from simtk.openmm.app import PDBFile

import os
import argparse

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('ligand_file')
parser.add_argument('--rotamer_resolution', default=10)
parser.add_argument('--include_terminal_rotamers', action='store_false', default=True)

args=parser.parse_args()

ligand_file = args.ligand_file
rotamer_resolution = args.rotamer_resolution
include_terminal_rotamers = args.include_terminal_rotamers

if include_terminal_rotamers:
    exclude_terminal_rotamers = False
else:
    exclude_terminal_rotamers = True

extension = ligand_file.split('.')[-1]
ligand_name = ligand_file.replace('.'+extension,'')

os.environ['SCHRODINGER'] = '/gpfs/projects/bsc72/SCHRODINGER_ACADEMIC' # Fill in path to SCHRODINGER

molecule = Molecule(ligand_file,
                    allow_undefined_stereo=True,
                    rotamer_resolution=rotamer_resolution,
                    exclude_terminal_rotamers=exclude_terminal_rotamers)

oplsff = OPLS2005ForceField()

parameters = oplsff.parameterize(molecule, charge_method='OPLS2005')
topology = Topology(molecule, parameters)

resname = molecule.tag
output_rotamers = resname+'.rot.assign'
output_impact = resname.lower()+'z'

rotamer_library = RotamerLibrary(molecule)
rotamer_library.to_file(output_rotamers)

impact = Impact(topology)
impact.to_file(output_impact)
