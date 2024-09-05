try:
    import openmm
except ImportError as e:
    raise ValueError('openmm python module not avaiable. Please install it to use this function.')

from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm.vec3 import Vec3

from Bio.PDB.Polypeptide import aa3

import numpy as np
from sys import stdout
import shutil
import os
from multiprocessing import cpu_count

aa3 = aa3+['HID', 'HIE', 'HIP', 'ASH', 'GLH', 'CYX']
ions = ['MG', 'NA', 'CL']
aa3 += ions


class openmm_md:

    def __init__(self, input_pdb):

        # Set variables
        self.input_pdb = input_pdb
        self.pdb_name = input_pdb.split('/')[-1].replace('.pdb', '')
        self.pdb = pdb = PDBFile(self.input_pdb)
        self.modeller = Modeller(pdb.topology, pdb.positions)
        self.positions = np.array([c.value_in_unit(nanometer) for c in self.modeller.positions])

    def setUpFF(self, ff_name):

        # Check forcefield
        available_ffs = ['amber14']
        if ff_name not in available_ffs:
            raise ValueError(f'{ff_name} not found in available forcefields: {available_ffs}')
        self.ff_name = ff_name

        # Define ff definition files
        if self.ff_name == 'amber14':
            self.ff_files = ['amber14-all.xml', 'amber14/tip3pfb.xml']
        self.forcefield = ForceField(*self.ff_files)

    def setPeriodicBoundaryConditions(self, radius=1.5):
        min_xyz = np.min(self.positions, axis=0)
        max_xyz = np.max(self.positions, axis=0)
        vectors = [Vec3((max_xyz[0]-min_xyz[0])+radius, 0, 0),
                   Vec3(0, (max_xyz[1]-min_xyz[1])+radius, 0),
                   Vec3(0, 0, (max_xyz[2]-min_xyz[2])+radius)]
        self.modeller.topology.setPeriodicBoxVectors(vectors)

    def removeHydrogens(self, keep_ligands=True):
        hydrogens = [a for a in self.modeller.topology.atoms() if a.element.name == 'hydrogen']
        if keep_ligands:
            hydrogens = [a for a in hydrogens if a.residue.name in aa3]
        self.modeller.delete(hydrogens)

    def addHydrogens(self, variants=None):

        # Create protein only state
        positions = {}
        for residue in self.modeller.topology.residues():
            if residue.name not in aa3:
                positions[residue] = []
                for atom in residue.atoms():
                    positions[residue].append(self.modeller.positions[atom.index])
        self.modeller.delete(list(positions.keys()))

        # Add hydrogens
        self.modeller.addHydrogens(self.forcefield, variants=variants)

        # Add remaning residues
        for residue in positions:
            chain_id = residue.chain.id
            c = self.modeller.topology.addChain(chain_id)
            r = self.modeller.topology.addResidue(residue.name, c)
            for atom in residue.atoms():
                self.modeller.topology.addAtom(atom.name, atom.element, r)
            for p in positions[residue]:
                self.modeller.positions.append(p)

    def getProtonationStates(self, keep_ligands=False):
        """
        Get residue names according to the protonation state.
        """

        residue_names = []
        for residue in self.modeller.topology.residues():
            if residue.name == 'HIS':
                atoms = []
                for atom in residue.atoms():
                    atoms.append(atom.name)
                if 'HD1' in atoms and 'HE2' in atoms:
                    his_name = 'HIP'
                elif 'HD1' in atoms:
                    his_name = 'HID'
                elif 'HE2' in atoms:
                    his_name = 'HIE'
                else:
                    his_name = 'HIS'
                residue_names.append(his_name)

            elif residue.name == 'ASP':
                atoms = []
                for atom in residue.atoms():
                    atoms.append(atom.name)
                if 'HD2' in atoms:
                    residue_names.append('ASH')
                else:
                    residue_names.append('ASP')

            elif residue.name == 'GLU':
                atoms = []
                for atom in residue.atoms():
                    atoms.append(atom.name)
                if 'HE2' in atoms:
                    residue_names.append('GLH')
                else:
                    residue_names.append('GLU')

            elif residue.name == 'CYS':
                atoms = []
                for atom in residue.atoms():
                    atoms.append(atom.name)
                if 'HG' in atoms:
                    residue_names.append('CYS')
                else:
                    residue_names.append('CYX')
            else:
                if residue.name not in aa3 and not keep_ligands:
                    continue
                residue_names.append(None)

        return residue_names

    def addSolvent(self):
        self.modeller.addSolvent(self.forcefield)

    def parameterizePDBLigands(self, parameters_folder, charges=None, skip_ligands=None, overwrite=False,
                               metal_ligand=None, add_bonds=None, cpus=None, return_qm_jobs=False,
                               force_field='ff14SB', residue_names=None, metal_parameters=None, extra_frcmod=None,
                               extra_mol2=None, add_counterions=None, save_amber_pdb=False, solvate=True,
                               regenerate_amber_files=False):

        def topologyFromResidue(residue, topol, positions):
            top = topology.Topology()
            c = top.addChain()
            c.id = residue.chain.id
            r = top.addResidue(residue.name, c)
            atom_indexes = []
            pos = openmm.unit.quantity.Quantity()
            pos.unit = nanometer

            # Get atoms
            for a in residue.atoms():
                atom_indexes.append(a.index)
                top.addAtom(a.name, a.element, r)
                pos.append(positions[a.index])

            # Get atoms by name
            atoms = {}
            for a in r.atoms():
                atoms[a.name] = a

            # Get bonds
            for a1, a2 in topol.bonds():
                if a1.residue == residue and a2.residue == residue:
                    top.addBond(atoms[a1.name], atoms[a2.name])

            assert top.getNumAtoms() == len(pos)

            return top, pos

        def _changeGaussianCPUS(gaussian_com_file, cpus):
            tmp = open('file.tmp', 'w')
            with open(gaussian_com_file) as gf:
                for l in gf:
                    if l.startswith('%NProcShared='):
                         l = l.strip()[:-1]+str(cpus)+'\n'
                    tmp.write(l)
            tmp.close()
            shutil.move('file.tmp', gaussian_com_file)

        def _checkIfGaussianFinished(gaussian_log_file):
            finished = False
            if os.path.exists(gaussian_log_file):
                with open(gaussian_log_file) as lf:
                    for l in lf:
                        if 'Normal termination of Gaussian 09' in l:
                            finished = True
            return finished

        def getNonProteinResidues(topology, skip_residues=None):
            residues = []
            for chain in topology.chains():
                for r in chain.residues():
                    residue = r.name.upper()
                    if residue not in aa3:
                        #Skip water molecules
                        if residue == 'HOH':
                            continue
                        # Skip given ligands
                        if residue in skip_ligands:
                            continue
                        residues.append(r)
            return residues

        def getMissingAtomTypes(mol2):
            cond = False
            missing = []
            with open(mol2) as mf:
                for l in mf:
                    if l.startswith('@<TRIPOS>ATOM'):
                        cond = True
                        continue
                    elif l.startswith('@<TRIPOS>BOND'):
                        cond = False
                        continue
                    if cond:
                        if l.split()[5].isupper():
                            missing.append(l.split()[5])
            return missing

        def getAtomTypes(frcmod, atoms):
            cond = False
            atom_types = {}
            with open(frcmod) as f:
                for l in f:
                    if l.startswith('MASS'):
                        cond = True
                        continue
                    elif l.startswith('BOND'):
                        cond = False
                        continue
                    if cond:
                        if l.strip() == '':
                            continue
                        if l.split()[3] == 'ion':
                            atom_types[l.split()[0]] = l.split()[2]
                        elif 'disulfide' in l:
                            atom_types[l.split()[0]] = l.split()[3]
                        else:
                            atom_types[l.split()[0]] = l.split()[4]
            return atom_types

        if regenerate_amber_files:
            if os.path.exists(parameters_folder+'/'+self.pdb_name+'.prmtop'):
                os.remove(parameters_folder+'/'+self.pdb_name+'.prmtop')
            if os.path.exists(parameters_folder+'/'+self.pdb_name+'.inpcrd'):
                os.remove(parameters_folder+'/'+self.pdb_name+'.inpcrd')

        if os.path.exists(parameters_folder+'/'+self.pdb_name+'.prmtop') and os.path.exists(parameters_folder+'/'+self.pdb_name+'.inpcrd'):
            if not overwrite:

                # Set topology and positions to amber's
                self.prmtop_file = parameters_folder+'/'+self.pdb_name+'.prmtop'
                self.inpcrd_file = parameters_folder+'/'+self.pdb_name+'.inpcrd'

                self.prmtop = AmberPrmtopFile(self.prmtop_file)
                self.inpcrd = AmberInpcrdFile(self.inpcrd_file)

                self.modeller.topology = self.prmtop.topology
                self.modeller.positions = self.inpcrd.positions

                print('Parameters were already created. Give overwrite=True to recompute them.')
                return

        # Create working folder
        if not os.path.exists(parameters_folder):
            os.mkdir(parameters_folder)

        if not skip_ligands:
            skip_ligands = []

        if not metal_ligand:
            metal_ligand = {}

        # Generate set of metal ligand values
        metal_ligand_values = []
        for r in metal_ligand:
            if isinstance(metal_ligand[r], str): # Convert into a list
                metal_ligand[r] = [metal_ligand[r]]
            for m in metal_ligand[r]:
                metal_ligand_values.append(m)

        # Get parameter folders from the given metal_parameters folder
        parameters_folders = {}
        parameters_mol2 = {}
        parameters_frcmod = {}
        if metal_parameters:
            parameters_folders = {}
            for d in os.listdir(metal_parameters):
                if d.endswith('_parameters'):
                    parameters_folders[d.split('_')[0]] = metal_parameters+'/'+d
                elif d.endswith('_mcpbpy.frcmod'):
                    parameters_folders['mcpbpy.frcmod'] = metal_parameters+'/'+d
                elif d.endswith('.mol2'):
                    parameters_mol2[d.split('.')[0]] = metal_parameters+'/'+d
                elif d.endswith('.frcmod'):
                    parameters_frcmod[d.split('.')[0]] = metal_parameters+'/'+d

        if not cpus:
            cpus = cpu_count()

        # Modify protonation state names and save state
        for chain in self.modeller.topology.chains():
            for i,residue in enumerate(chain.residues()):

                res_tuple = (chain.id, int(residue.id))
                if residue_names and res_tuple in residue_names:
                    residue.name = residue_names[res_tuple]

                if residue.name not in aa3:
                    continue

                if i == 0:
                    for atom in residue.atoms():
                        if atom.name == 'H':
                            atom.name = 'H1'

                if residue.name == 'HIS':
                    atoms = []
                    for atom in residue.atoms():
                        atoms.append(atom.name)
                    if 'HD1' in atoms and 'HE2' in atoms:
                        his_name = 'HIP'
                    elif 'HD1' in atoms:
                        his_name = 'HID'
                    elif 'HE2' in atoms:
                        his_name = 'HIE'
                    else:
                        his_name = 'HIS'
                    residue.name = his_name

                elif residue.name == 'ASP':
                    atoms = []
                    for atom in residue.atoms():
                        atoms.append(atom.name)
                    if 'HD2' in atoms:
                        residue.name = 'ASH'
                    else:
                        residue.name = 'ASP'

                elif residue.name == 'CYS':
                    atoms = []
                    for atom in residue.atoms():
                        atoms.append(atom.name)
                    if 'HG' in atoms:
                        residue.name = 'CYS'
                    else:
                        residue.name = 'CYX'

        pdb_file = parameters_folder+'/'+self.pdb_name+'.pdb'
        self.savePDB(pdb_file)

        # Get molecules that need parameterization
        par_folder = {}
        for r in getNonProteinResidues(self.modeller.topology, skip_residues=skip_ligands):

            residue = r.name.upper()

            # Create PDB for each ligand molecule
            lig_top, lig_pos = topologyFromResidue(r, self.modeller.topology,
                                                      self.modeller.positions)

            par_folder[residue] = parameters_folder+'/'+residue+'_parameters'

            if not os.path.exists(par_folder[residue]):
                os.mkdir(par_folder[residue])

            with open(par_folder[residue]+'/'+residue+'.pdb', 'w') as rf:
                PDBFile.writeFile(lig_top, lig_pos, rf)

            if residue in parameters_folders:
                for d in os.listdir(parameters_folders[residue]):
                    shutil.copyfile(parameters_folders[residue]+'/'+d,
                                    par_folder[residue]+'/'+d)

            if residue in parameters_mol2:
                shutil.copyfile(parameters_mol2[residue],
                                parameters_folder+'/'+residue+'.mol2')

        if extra_frcmod:
            for residue in extra_frcmod:
                if os.path.exists(residue):
                    shutil.copyfile(residue,
                                    parameters_folder+'/'+residue.split('/')[-1])
                elif residue in parameters_frcmod:
                    shutil.copyfile(parameters_frcmod[residue],
                                    parameters_folder+'/'+residue+'.frcmod')
                elif not os.path.exists(residue) and residue not in parameters_frcmod:
                    raise ValueError(f'Frcmod file {residue} was not found.')
                else:
                    raise ValueError(f'Frcmod file for residue {residue} was not found in {metal_parameters}')

        if extra_mol2:
            for residue in extra_mol2:
                if isinstance(extra_mol2, dict):
                    if os.path.exists(extra_mol2[residue]):
                        shutil.copyfile(extra_mol2[residue],
                                        parameters_folder+'/'+extra_mol2[residue].split('/')[-1])
                    else:
                        raise ValueError(f'Mol2 file for {residue} {extra_mol2[residue]} was not found.')
                else:
                    if os.path.exists(residue):
                        shutil.copyfile(residue,
                                        parameters_folder+'/'+residue.split('/')[-1])
                    elif residue in parameters_mol2:
                        shutil.copyfile(parameters_mol2[residue],
                                        parameters_folder+'/'+residue+'.mol2')
                    elif not os.path.exists(residue) and residue not in parameters_mol2:
                        raise ValueError(f'Mol2 file {residue} was not found.')
                    else:
                        raise ValueError(f'Mol2 file for residue {residue} was not found in {metal_parameters}')

        # Create parameters for each molecule
        for residue in par_folder:

            # Skip metal ions to be processed together with other ligands
            if residue in metal_ligand_values:
                continue

            metal_pdb = None
            metal_charge = None
            if residue in metal_ligand:
                metal_pdb = {}
                metal_charge = {}
                for metal_residue in metal_ligand[residue]:
                    metal_pdb[metal_residue] = '../'+metal_residue+'_parameters/'+metal_residue+'.pdb'
                    metal_charge[metal_residue] = charges[metal_residue]

            if charges and residue in charges:
                charge = charges[residue]
            else:
                charge = 0

            if metal_parameters and residue in parameters_folders:
                continue

            print(f'Computing parameters for residue {residue}')
            if residue in metal_ligand:
                print('\tConsidering the followng ions:')
                print(f'\t\t{metal_ligand[residue]}')
            os.chdir(par_folder[residue])

            lig_par = ligandParameters(residue+'.pdb', metal_pdb=metal_pdb)
            lig_par.getAmberParameters(ligand_charge=charge, overwrite=overwrite,
                                       metal_charge=metal_charge)
            os.chdir('../../')

        # Renumber PDB
        renum_pdb = pdb_file.replace('.pdb', '_renum.pdb')
        if not os.path.exists(renum_pdb) or regenerate_amber_files:
            command =  'pdb4amber -i '
            command += pdb_file+' '
            command += '-o '+renum_pdb+'\n'
            os.system(command)

        # Parameterize metal complex with MCPB.py
        if metal_ligand:

            # Copy frcmmod file from previous optimization
            if metal_parameters:
                shutil.copyfile(parameters_folders['mcpbpy.frcmod'],
                                parameters_folder+'/'+self.pdb_name+'_mcpbpy.frcmod')

            # Get ion ID
            metal_pdb = PDBFile(renum_pdb)
            ion_ids = {}
            for residue in metal_pdb.topology.residues():
                for r in metal_ligand:
                    if residue.name in metal_ligand[r]:
                        atoms = list(residue.atoms())
                        if len(atoms) > 1:
                            raise ValueError('Ion residue should contain only one atom!')
                        ion_ids[residue.name] = atoms[0].index+1

            # Copy input files
            for residue in par_folder:

                if residue in metal_ligand_values:
                    continue

                if metal_parameters and residue in parameters_folders:
                    continue

                # Copy metal ions files
                if residue in metal_ligand:
                    for m in metal_ligand[residue]:
                        shutil.copyfile(par_folder[residue]+'/'+m+'.mol2',
                                        parameters_folder+'/'+m+'.mol2')

                shutil.copyfile(par_folder[residue]+'/'+residue+'.mol2',
                                parameters_folder+'/'+residue+'.mol2')
                shutil.copyfile(par_folder[residue]+'/'+residue+'.frcmod',
                                parameters_folder+'/'+residue+'.frcmod')

            if not metal_parameters:

                input_file = pdb_file.replace('.pdb', '.in')
                with open(input_file, 'w') as f:
                    f.write('original_pdb '+self.pdb_name+'_renum.pdb'+'\n')
                    f.write('group_name '+self.pdb_name+'\n')
                    f.write('software_version g09\n')
                    f.write('force_field '+force_field+'\n')
                    f.write('cut_off 2.8\n')
                    ion_ids_line = 'ion_ids'
                    for residue in par_folder:
                        if residue in metal_ligand:
                            for m in metal_ligand[residue]:
                                ion_ids_line += ' '+str(ion_ids[m])
                    f.write(ion_ids_line+'\n')
                    ion_mol2 = 'ion_mol2files'
                    for residue in par_folder:
                        if residue in metal_ligand:
                            for m in metal_ligand[residue]:
                                 ion_mol2 += ' '+m+'.mol2'
                    f.write(ion_mol2+'\n')
                    naa_mol2 = 'naa_mol2files'
                    for residue in par_folder:
                        if residue in metal_ligand_values:
                            continue
                        naa_mol2 += ' '+residue+'.mol2'
                    f.write(naa_mol2+'\n')
                    frcmod = 'frcmod_files'
                    for residue in par_folder:
                        if residue in metal_ligand_values:
                            continue
                        frcmod += ' '+residue+'.frcmod'
                    f.write(frcmod+'\n')

                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 1\n'
                os.system(command)

                # Set gaussian calculations for small
                commands = []

                command = ''
                finished = _checkIfGaussianFinished(self.pdb_name+'_small_opt.log')
                if not finished:
                    command = 'cd '+parameters_folder+'\n'
                    _changeGaussianCPUS(self.pdb_name+'_small_opt.com', cpus)
                    command +=  'g09 '+self.pdb_name+'_small_opt.com > '+self.pdb_name+'_small_opt.log\n'

                finished = _checkIfGaussianFinished(self.pdb_name+'_small_fc.log')
                if not finished:
                    _changeGaussianCPUS(self.pdb_name+'_small_fc.com', cpus)
                    command += 'g09 '+self.pdb_name+'_small_fc.com > '+self.pdb_name+'_small_fc.log\n'
                    command += 'formchk '+self.pdb_name+'_small_opt.chk > '+self.pdb_name+'_small_opt.fchk\n'
                if command != '':
                    command += 'cd ../\n'
                    commands.append(command)

                # Set gaussian calculations for large
                command = ''
                finished = _checkIfGaussianFinished(self.pdb_name+'_large_mk.log')
                if not finished:
                    command = 'cd '+parameters_folder+'\n'
                    _changeGaussianCPUS(self.pdb_name+'_large_mk.com', cpus)
                    command += 'g09 '+self.pdb_name+'_large_mk.com > '+self.pdb_name+'_large_mk.log\n'
                if command != '':
                    command += 'cd ../\n'
                    commands.append(command)

                os.chdir('../')
                if commands:# and not metal_parameters:
                    if return_qm_jobs:
                        print('Returning QM jobs')
                        return commands
                    else:
                        print('Computing QM parameters')
                        for command in commands:
                            os.system(command)

                print('QM calculations finished.')

                # Run step 2 of the MCPB protocol
                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 2\n'
                os.system(command)
                os.chdir('../')

                # Run step 3 of the MCPB protocol
                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 3\n'
                os.system(command)
                os.chdir('../')

                # Run step 4 of the MCPB protocol
                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 4\n'
                os.system(command)
                os.chdir('../')

        # Generate set of metal ligand values
        metal_ligand_values = []
        for r in metal_ligand:
            if isinstance(metal_ligand[r], str): # Convert into a list
                metal_ligand[r] = [metal_ligand[r]]
            for m in metal_ligand[r]:
                metal_ligand_values.append(m)

        # Create tleap input file
        with open(parameters_folder+'/tleap.in', 'w') as tlf:
            tlf.write('source leaprc.protein.ff14SB\n')
            tlf.write('source leaprc.gaff\n')
            tlf.write('source leaprc.water.tip3p\n')

            if metal_ligand:

                # Get residues not parameterized as metal
                not_metal = []
                for residue in par_folder:
                    if residue in metal_ligand_values:
                        continue
                    not_metal.append(residue)

                # Generate frcmod files
                if metal_parameters:
                    mcpb_pdb = renum_pdb
                else:
                    mcpb_pdb = parameters_folder+'/'+self.pdb_name+'_mcpbpy.pdb'
                pdb = PDBFile(mcpb_pdb)
                missing_atoms = []
                for residue in getNonProteinResidues(pdb.topology):
                    if residue in not_metal:
                        continue
                    missing_atoms += getMissingAtomTypes(parameters_folder+'/'+residue.name+'.mol2')

                mcpb_frcmod = parameters_folder+'/'+self.pdb_name+'_mcpbpy.frcmod'
                atom_types = getAtomTypes(mcpb_frcmod, missing_atoms)
                tlf.write('addAtomTypes {\n')
                for atom in atom_types:
                    tlf.write('\t{ "'+atom+'"  "'+atom_types[atom]+'" "sp3" }\n')
                tlf.write('}\n')

                for residue in getNonProteinResidues(pdb.topology):
                    tlf.write(residue.name+' = loadmol2 '+parameters_folder+'/'+residue.name+'.mol2\n')

            if extra_mol2:
                for residue in extra_mol2:
                    if isinstance(extra_mol2, dict):
                        if os.path.exists(extra_mol2[residue]):
                            tlf.write(residue+' = loadmol2 '+parameters_folder+'/'+extra_mol2[residue].split('/')[-1]+'\n')
                    else:
                        if os.path.exists(residue):
                            tlf.write(residue.split('/')[-1].replace('.mol2', '')+' = loadmol2 '+parameters_folder+'/'+residue.split('/')[-1]+'\n')
                        else:
                            tlf.write(residue+' = loadmol2 '+parameters_folder+'/'+residue+'.mol2\n')

            for residue in par_folder:
                if residue in metal_ligand_values:
                    continue
                if not metal_ligand:
                    tlf.write('loadamberprep '+par_folder[residue]+'/'+residue+'.prepi\n')
                tlf.write('loadamberparams '+par_folder[residue]+'/'+residue+'.frcmod\n')

            if extra_frcmod:
                for residue in extra_frcmod:
                    if os.path.exists(residue):
                        tlf.write('loadamberparams '+parameters_folder+'/'+residue.split('/')[-1]+'\n')
                    else:
                        tlf.write('loadamberparams '+parameters_folder+'/'+residue+'.frcmod\n')

            if metal_ligand:
                tlf.write('loadamberparams '+mcpb_frcmod+'\n')
                tlf.write('mol = loadpdb '+mcpb_pdb+'\n')
            else:
                tlf.write('mol = loadpdb '+renum_pdb+'\n')

            # Add bonds
            if add_bonds:
                for bond in add_bonds:
                    tlf.write('bond mol.'+str(bond[0][0])+'.'+bond[0][1]+' '
                                   'mol.'+str(bond[1][0])+'.'+bond[1][1]+'\n')

            if solvate:
                tlf.write('solvatebox mol TIP3PBOX 12\n')

            if add_counterions:
                tlf.write('addIons2 mol Na+ 0\n')
                tlf.write('addIons2 mol Cl- 0\n')

            if save_amber_pdb:
                tlf.write('savepdb mol '+parameters_folder+'/'+self.pdb_name+'_amber.pdb\n')

            tlf.write('saveamberparm mol '+parameters_folder+'/'+self.pdb_name+'.prmtop '+parameters_folder+'/'+self.pdb_name+'.inpcrd\n')


        os.system('tleap -s -f '+parameters_folder+'/tleap.in')

        # Define prmtop and inpcrd file paths
        self.prmtop_file = parameters_folder+'/'+self.pdb_name+'.prmtop'
        self.inpcrd_file = parameters_folder+'/'+self.pdb_name+'.inpcrd'

        # Set topology and positions to amber's
        self.prmtop = AmberPrmtopFile(self.prmtop_file)
        self.inpcrd = AmberInpcrdFile(self.inpcrd_file)

        self.modeller.topology = self.prmtop.topology
        self.modeller.positions = self.inpcrd.positions

    def savePDB(self, output_file):
        with open(output_file, 'w') as of:
            PDBFile.writeFile(self.modeller.topology, self.modeller.positions, of)

    def centerPositionsAtCOM(self, output_unit=nanometer):
        masses = np.array([a.element.mass.value_in_unit(amu) for a in self.modeller.topology.atoms()])
        masses = np.repeat(masses, 3, axis=0).reshape((masses.shape[0], 3))
        com = np.sum(self.positions*masses, axis=0)/np.sum(masses[:,0])
        self.positions = self.positions-com
        self.modeller.positions = _getPositionsArrayAsVector(self.positions)

class ligandParameters:

    def __init__(self, ligand_pdb, metal_pdb=None):

        self.ligand_pdb = ligand_pdb
        self.pdb = PDBFile(self.ligand_pdb)
        self.pdb_name = ligand_pdb.split('/')[-1].replace('.pdb', '')
        self.metal_pdb = metal_pdb
        self.metal = bool(metal_pdb)

        res_count = 0
        for residue in self.pdb.topology.residues():
            res_count += 1
            self.resname = residue.name

        if res_count != 1:
            raise ValueError('A PDB with a single residue must be given for parameterization!')

    def getAmberParameters(self, charge_model='bcc', ligand_charge=0, metal_charge=None, overwrite=False):

        # Execute pdb4amber to generate a renumbered PDB
        if not os.path.exists(self.resname+'_renum.pdb') or overwrite:
            command  = 'pdb4amber '
            command += '-i '+self.ligand_pdb+' '
            command += '-o '+self.resname+'_renum.pdb '
            os.system(command)

        # Run antechamber to create a mol2 file with the atomic charges
        if not os.path.exists(self.resname+'.mol2') or overwrite:
            command = 'antechamber '
            command += '-i '+self.resname+'_renum.pdb '
            command += '-fi pdb '
            command += '-o '+self.resname+'.mol2 '
            command += '-fo mol2 '
            command += '-pf y '
            command += '-c '+charge_model+' '
            command += '-nc '+str(ligand_charge)+' '
            command += '-s 2\n'
            os.system(command)

        # Run antechamber to create a prepi file with the atomic charges
        if not os.path.exists(self.resname+'.prepi') or overwrite:
            command = 'antechamber '
            command += '-i '+self.resname+'_renum.pdb '
            command += '-fi pdb '
            command += '-o '+self.resname+'.prepi '
            command += '-fo prepi '
            command += '-pf y '
            command += '-c '+charge_model+' '
            command += '-nc '+str(ligand_charge)+' '
            command += '-s 2\n'
            os.system(command)

        # Run parmchk to check with forcefield parameters will be used
        if not os.path.exists(self.resname+'.frcmod') or overwrite:
            command  = 'parmchk2 '
            command += '-i '+self.resname+'.mol2 '
            command += '-o '+self.resname+'.frcmod '
            command += '-f mol2\n'
            os.system(command)

        # Parameterize metal pdb if given
        if self.metal_pdb:
            for m in self.metal_pdb:
                metal_mol2 = self.metal_pdb[m].split('/')[-1].replace('.pdb','.mol2')

                if not os.path.exists(metal_mol2) or overwrite:

                    if not metal_charge:
                        raise ValueError('You must give the metal_charge if using a ligand with metal!')
                    self.metal_charge = metal_charge
                    command  = 'metalpdb2mol2.py '
                    command += '-i '+self.metal_pdb[m]+' '
                    command += '-o '+metal_mol2+' '
                    command += '-c '+str(self.metal_charge[m])+'\n'
                    os.system(command)

def _getPositionsArrayAsVector(positions):
    v3_positions = []
    for p in positions:
        v3_positions.append(quantity.Quantity(Vec3(*p), unit=nanometer))
    return v3_positions
