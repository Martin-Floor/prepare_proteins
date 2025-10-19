try:
    import openmm
except ImportError as e:
    raise ValueError('openmm python module not avaiable. Please install it to use this function.')

from pkg_resources import resource_filename, Requirement

from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm.vec3 import Vec3

from Bio.PDB.Polypeptide import aa3

import numpy as np
from sys import stdout
from collections.abc import Mapping
import shutil
import os
import fileinput
from multiprocessing import cpu_count
import shlex
import warnings

aa3 = list(aa3)+['HID', 'HIE', 'HIP', 'ASH', 'GLH', 'CYX', 'ACE', 'NME']
ions = ['MG', 'NA', 'CL', 'CU']
aa3 += ions


def _copyfile_if_needed(src, dst):
    """Copy `src` to `dst` unless they resolve to the same file."""
    try:
        if os.path.samefile(src, dst):
            return
    except FileNotFoundError:
        # Destination does not exist yet, fall back to regular copy.
        pass
    shutil.copyfile(src, dst)


def _run_command(command, command_log=None):
    """Execute `command` via `os.system` while optionally recording the call."""
    entry = None
    if command_log is not None:
        entry = {"command": command.rstrip(), "returncode": None}
        command_log.append(entry)
    ret = os.system(command)
    if entry is not None:
        entry["returncode"] = ret
    return ret


class openmm_md:

    def __init__(self, input_pdb):

        # Set variables
        self.input_pdb = input_pdb
        self.pdb_name = input_pdb.split('/')[-1].replace('.pdb', '')
        self.pdb = pdb = PDBFile(self.input_pdb)
        self.modeller = Modeller(pdb.topology, pdb.positions)
        self.positions = np.array([c.value_in_unit(nanometer) for c in self.modeller.positions])
        self.command_log = []

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
            if residue.name not in set(aa3)-set(ions):
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
                if residue.name not in set(aa3)-set(ions) and not keep_ligands:
                    continue
                residue_names.append(None)

        return residue_names

    def addSolvent(self):
        self.modeller.addSolvent(self.forcefield)

    def parameterizePDBLigands(self, parameters_folder, charges=None, skip_ligands=None, overwrite=False,
                               metal_ligand=None, add_bonds=None, cpus=None, return_qm_jobs=False,
                               extra_force_field=None,
                               force_field='ff14SB', residue_names=None, metal_parameters=None, extra_frcmod=None,
                               extra_mol2=None, add_counterions=True, add_counterionsRand=False, save_amber_pdb=False, solvate=True,
                               regenerate_amber_files=False, non_standard_residues=None, strict_atom_name_check=True,
                               only_residues=None, build_full_system=True):

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
                        if residue in skip_residues:
                            continue
                        residues.append(r)
            return residues

        def getResiduesByName(topology, residue_name):
            residues = []
            for chain in topology.chains():
                for r in chain.residues():
                    if r.name.upper() == residue_name:
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

        def _extract_pdb_atom_names(pdb_path):
            atom_names = set()
            if not os.path.exists(pdb_path):
                return atom_names
            with open(pdb_path) as handle:
                for line in handle:
                    if line.startswith(('ATOM', 'HETATM')):
                        name = line[12:16].strip()
                        if name:
                            atom_names.add(name.upper())
            return atom_names

        def _validate_ligand_pdb_atoms(generated_path, reference_path, residue_name, source_label):
            if not os.path.exists(generated_path) or not os.path.exists(reference_path):
                return
            generated_atoms = _extract_pdb_atom_names(generated_path)
            reference_atoms = _extract_pdb_atom_names(reference_path)
            if not generated_atoms and not reference_atoms:
                return
            missing = reference_atoms - generated_atoms
            extra = generated_atoms - reference_atoms
            if missing or extra:
                mismatch = []
                if missing:
                    mismatch.append(f"missing atoms {sorted(missing)}")
                if extra:
                    mismatch.append(f"extra atoms {sorted(extra)}")
                message = (
                    f"Atom name mismatch for ligand {residue_name}: generated PDB "
                    f"({generated_path}) differs from {source_label} ({reference_path}); "
                    + ", ".join(mismatch)
                )
                if strict_atom_name_check:
                    raise ValueError(message)
                warnings.warn(message, UserWarning)

        if only_residues:
            if isinstance(only_residues, str):
                only_residue_set = {only_residues.strip().upper()} if only_residues.strip() else set()
            else:
                only_residue_set = {str(res).strip().upper() for res in only_residues if str(res).strip()}
            if not only_residue_set:
                only_residue_set = None
        else:
            only_residue_set = None

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

        extra_ffs = ['parmBSC1', 'parmBSC2']
        if extra_force_field:
            if extra_force_field not in extra_ffs:
                raise ValueError(f'The only implemented extra ff is: {extra_ffs[0]}')

            # Copy ff files
            extra_ff_folder = parameters_folder+'/'+extra_force_field
            if not os.path.exists(extra_ff_folder):
                os.mkdir(extra_ff_folder)
            _copyFFFiles(extra_ff_folder, extra_force_field)

            extra_force_field_source = extra_ff_folder+'/leaprc.bsc'+extra_force_field[-1]

            # Define ff residues
            ff_residues = [
                            'A', 'A3', 'A5', 'AN',
                            'C', 'C3', 'C5', 'CN',
                            'DA', 'DA3', 'DA5', 'DAN',
                            'DC', 'DC3', 'DC5', 'DCN',
                            'DG', 'DG3', 'DG5', 'DGN',
                            'DT', 'DT3', 'DT5', 'DTN',
                            'G', 'G3', 'G5', 'GN',
                            'OHE',
                            'U', 'U3', 'U5', 'UN'
                        ]

            if skip_ligands:
                skip_ligands += ff_residues
            else:
                skip_ligands = ff_residues

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

                elif residue.name == 'GLU':
                    atoms = []
                    for atom in residue.atoms():
                        atoms.append(atom.name)
                    if 'HE2' in atoms:
                        residue.name = 'GLH'
                    else:
                        residue.name = 'GLU'

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
        generated_pdb_paths = {}

        for r in getNonProteinResidues(self.modeller.topology, skip_residues=skip_ligands):

            residue = r.name.upper()
            if only_residue_set and residue not in only_residue_set:
                continue

            # Create PDB for each ligand molecule
            lig_top, lig_pos = topologyFromResidue(r, self.modeller.topology,
                                                      self.modeller.positions)

            par_folder[residue] = parameters_folder+'/'+residue+'_parameters'

            if not os.path.exists(par_folder[residue]):
                os.mkdir(par_folder[residue])

            generated_ligand_pdb = par_folder[residue]+'/'+residue+'.pdb'
            with open(generated_ligand_pdb, 'w') as rf:
                PDBFile.writeFile(lig_top, lig_pos, rf)
            generated_pdb_paths[residue] = generated_ligand_pdb
            model_specific_pdb = os.path.join(parameters_folder, f"{self.pdb_name}_{residue}.pdb")
            shutil.copyfile(generated_ligand_pdb, model_specific_pdb)

            if residue in parameters_folders:
                provided_pack = parameters_folders[residue]
                pack_pdb = os.path.join(provided_pack, residue+'.pdb')
                _validate_ligand_pdb_atoms(
                    generated_ligand_pdb,
                    pack_pdb,
                    residue,
                    f"parameter pack in {provided_pack}"
                )
                for d in os.listdir(parameters_folders[residue]):
                    _copyfile_if_needed(parameters_folders[residue]+'/'+d,
                                        par_folder[residue]+'/'+d)

            if residue in parameters_mol2:
                _copyfile_if_needed(parameters_mol2[residue],
                                    parameters_folder+'/'+residue+'.mol2')

        mol2_conversion_targets = set()
        provided_frcmod_residues = set()
        provided_mol2_residues = set()

        def _copy_into_par_folder(residue_name, source_path, extension):
            """Duplicate a provided parameter file into the residue-specific workspace."""
            if not source_path or not os.path.exists(source_path):
                return
            res_upper = residue_name.upper()
            if res_upper not in par_folder:
                return
            destination = os.path.join(par_folder[res_upper], f'{res_upper}.{extension}')
            _copyfile_if_needed(source_path, destination)

        def _get_case_insensitive(mapping, key):
            """Return a value from `mapping` regardless of key casing."""
            key_str = str(key)
            if key_str in mapping:
                return mapping[key_str]
            key_upper = key_str.upper()
            if key_upper in mapping:
                return mapping[key_upper]
            key_lower = key_str.lower()
            if key_lower in mapping:
                return mapping[key_lower]
            return None

        def _stage_external_mol2(residue_name, source_path, mark_for_conversion=False):
            """Place a provided MOL2 inside the working folder so antechamber can reuse it."""
            res_upper = residue_name.upper()
            if res_upper not in par_folder:
                return
            if not os.path.exists(source_path):
                raise ValueError(f'Mol2 file for {residue_name} at {source_path} was not found.')
            workdir = par_folder[res_upper]
            dest_path = os.path.join(workdir, f'{res_upper}.mol2')
            try:
                if os.path.exists(dest_path) and os.path.samefile(source_path, dest_path):
                    return
            except FileNotFoundError:
                pass
            shutil.copyfile(source_path, dest_path)
            if mark_for_conversion:
                mol2_conversion_targets.add(res_upper)

        for residue_name in par_folder:
            if residue_name in parameters_mol2:
                _stage_external_mol2(residue_name, parameters_mol2[residue_name])

        if extra_frcmod:
            def _iter_extra_frcmod_sources(values):
                """Yield (residue, source) pairs from user-provided extras."""
                if isinstance(values, Mapping):
                    for res_name, src in values.items():
                        yield res_name, src
                else:
                    if isinstance(values, (str, os.PathLike)):
                        values_iterable = [values]
                    else:
                        values_iterable = values
                    for src in values_iterable:
                        yield None, src

            for residue_key, source in _iter_extra_frcmod_sources(extra_frcmod):
                if residue_key is not None:
                    res_upper = str(residue_key).upper()
                    destination = os.path.join(parameters_folder, f"{res_upper}.frcmod")
                    source_path = None
                    if source and os.path.exists(source):
                        source_path = os.fspath(source)
                    else:
                        source_path = _get_case_insensitive(parameters_frcmod, residue_key)
                        if not source_path:
                            fallback_key = os.path.splitext(str(residue_key))[0]
                            source_path = _get_case_insensitive(parameters_frcmod, fallback_key)
                    if not source_path or not os.path.exists(source_path):
                        if source and not os.path.exists(source):
                            raise ValueError(f'Frcmod file for residue {residue_key} at {source} was not found.')
                        if metal_parameters:
                            raise ValueError(f'Frcmod file for residue {residue_key} was not found in {metal_parameters}.')
                        raise ValueError(f'Frcmod file for residue {residue_key} was not found.')
                    _copyfile_if_needed(source_path, destination)
                    _copy_into_par_folder(res_upper, destination, 'frcmod')
                    provided_frcmod_residues.add(res_upper)
                    continue

                candidate = source
                if candidate is None:
                    raise ValueError('Frcmod file None was not found.')
                candidate_str = os.fspath(candidate) if isinstance(candidate, os.PathLike) else str(candidate)
                if os.path.exists(candidate_str):
                    res_upper = os.path.splitext(os.path.basename(candidate_str))[0].upper()
                    destination = os.path.join(parameters_folder, f"{res_upper}.frcmod")
                    _copyfile_if_needed(candidate_str, destination)
                    _copy_into_par_folder(res_upper, destination, 'frcmod')
                    provided_frcmod_residues.add(res_upper)
                else:
                    pack_source = _get_case_insensitive(parameters_frcmod, candidate_str)
                    if not pack_source:
                        fallback_key = os.path.splitext(os.path.basename(candidate_str))[0]
                        pack_source = _get_case_insensitive(parameters_frcmod, fallback_key)
                    if not pack_source or not os.path.exists(pack_source):
                        if metal_parameters:
                            raise ValueError(f'Frcmod file {candidate_str} was not found in {metal_parameters}.')
                        raise ValueError(f'Frcmod file {candidate_str} was not found.')
                    res_upper = os.path.splitext(os.path.basename(candidate_str))[0].upper()
                    destination = os.path.join(parameters_folder, f"{res_upper}.frcmod")
                    _copyfile_if_needed(pack_source, destination)
                    _copy_into_par_folder(res_upper, destination, 'frcmod')
                    provided_frcmod_residues.add(res_upper)

        if extra_mol2:
            def _iter_extra_mol2_sources(values):
                if isinstance(values, Mapping):
                    for res_name, src in values.items():
                        yield res_name, src
                else:
                    if isinstance(values, (str, os.PathLike)):
                        values_iterable = [values]
                    else:
                        values_iterable = values
                    for src in values_iterable:
                        yield None, src

            for residue_key, source in _iter_extra_mol2_sources(extra_mol2):
                if residue_key is not None:
                    res_upper = str(residue_key).upper()
                    dest_root_mol2 = os.path.join(parameters_folder, f'{res_upper}.mol2')
                    source_is_user = False
                    if source and os.path.exists(source):
                        source_path = os.fspath(source)
                        source_is_user = True
                    else:
                        source_path = _get_case_insensitive(parameters_mol2, residue_key)
                        if not source_path:
                            fallback_key = os.path.splitext(str(residue_key))[0]
                            source_path = _get_case_insensitive(parameters_mol2, fallback_key)
                    if not source_path or not os.path.exists(source_path):
                        if source and not os.path.exists(source):
                            raise ValueError(f'Mol2 file for residue {residue_key} at {source} was not found.')
                        if metal_parameters:
                            raise ValueError(f'Mol2 file for residue {residue_key} was not found in {metal_parameters}.')
                        raise ValueError(f'Mol2 file for residue {residue_key} was not found.')
                    _copyfile_if_needed(source_path, dest_root_mol2)
                    mark_conversion = res_upper not in provided_frcmod_residues
                    _stage_external_mol2(res_upper, dest_root_mol2, mark_for_conversion=mark_conversion)
                    provided_mol2_residues.add(res_upper)
                    generated_path = generated_pdb_paths.get(res_upper)
                    if generated_path:
                        if source_is_user:
                            candidate_pdb = os.path.splitext(source_path)[0] + '.pdb'
                            if os.path.exists(candidate_pdb):
                                _validate_ligand_pdb_atoms(
                                    generated_path,
                                    candidate_pdb,
                                    res_upper,
                                    f"extra_mol2 source {source_path}"
                                )
                        else:
                            pack_pdb = os.path.join(os.path.dirname(source_path), f'{res_upper}.pdb')
                            if os.path.exists(pack_pdb):
                                _validate_ligand_pdb_atoms(
                                    generated_path,
                                    pack_pdb,
                                    res_upper,
                                    f"extra_mol2 metal_parameters source {source_path}"
                                )
                    continue

                if source is None:
                    raise ValueError('Mol2 file None was not found.')
                source_str = os.fspath(source) if isinstance(source, os.PathLike) else str(source)
                res_upper = os.path.splitext(os.path.basename(source_str))[0].upper()
                dest_root_mol2 = os.path.join(parameters_folder, f'{res_upper}.mol2')
                if os.path.exists(source_str):
                    _copyfile_if_needed(source_str, dest_root_mol2)
                    mark_conversion = res_upper not in provided_frcmod_residues
                    _stage_external_mol2(res_upper, dest_root_mol2, mark_for_conversion=mark_conversion)
                    provided_mol2_residues.add(res_upper)
                    generated_path = generated_pdb_paths.get(res_upper)
                    candidate_pdb = os.path.splitext(source_str)[0] + '.pdb'
                    if generated_path and os.path.exists(candidate_pdb):
                        _validate_ligand_pdb_atoms(
                            generated_path,
                            candidate_pdb,
                            res_upper,
                            f"extra_mol2 source {source_str}"
                        )
                else:
                    pack_source = _get_case_insensitive(parameters_mol2, source_str)
                    if not pack_source:
                        fallback_key = os.path.splitext(os.path.basename(source_str))[0]
                        pack_source = _get_case_insensitive(parameters_mol2, fallback_key)
                    if not pack_source or not os.path.exists(pack_source):
                        if metal_parameters:
                            raise ValueError(f'Mol2 file {source_str} was not found in {metal_parameters}.')
                        raise ValueError(f'Mol2 file {source_str} was not found.')
                    res_upper = os.path.splitext(os.path.basename(pack_source))[0].upper()
                    dest_root_mol2 = os.path.join(parameters_folder, f'{res_upper}.mol2')
                    _copyfile_if_needed(pack_source, dest_root_mol2)
                    mark_conversion = res_upper not in provided_frcmod_residues
                    _stage_external_mol2(res_upper, dest_root_mol2, mark_for_conversion=mark_conversion)
                    provided_mol2_residues.add(res_upper)
                    generated_path = generated_pdb_paths.get(res_upper)
                    pack_pdb = os.path.join(os.path.dirname(pack_source), f'{res_upper}.pdb')
                    if generated_path and os.path.exists(pack_pdb):
                        _validate_ligand_pdb_atoms(
                            generated_path,
                            pack_pdb,
                            res_upper,
                            f"extra_mol2 metal_parameters source {pack_source}"
                        )

        skip_parameterization_residues = provided_mol2_residues & provided_frcmod_residues

        # Create parameters for each molecule
        def _convert_staged_mol2(residue_name, ligand_charge):
            res_upper = residue_name.upper()
            if res_upper not in mol2_conversion_targets:
                return
            mol2_path = os.path.join(par_folder[res_upper], f"{res_upper}.mol2")
            if not os.path.exists(mol2_path):
                return
            temp_output = mol2_path + '.tmp'
            if os.path.exists(temp_output):
                os.remove(temp_output)
            charge_value = ligand_charge if ligand_charge is not None else 0
            command = (
                'antechamber '
                f'-i {shlex.quote(mol2_path)} '
                '-fi mol2 '
                f'-o {shlex.quote(temp_output)} '
                '-fo mol2 '
                '-c bcc '
                f'-nc {charge_value} '
                f'-rn {res_upper} '
                '-s 2 '
                '-pf y\n'
            )
            ret = _run_command(command, self.command_log)
            if ret != 0:
                raise RuntimeError(
                    f'antechamber conversion to GAFF failed for {mol2_path} with exit code {ret}.'
                )
            shutil.move(temp_output, mol2_path)
            mol2_conversion_targets.discard(res_upper)

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

            if residue in skip_parameterization_residues:
                continue

            _convert_staged_mol2(residue, charge)

            if metal_parameters and residue in parameters_folders:
                continue

            print(f'Computing parameters for residue {residue}')
            if residue in metal_ligand:
                print('\tConsidering the followng ions:')
                print(f'\t\t{metal_ligand[residue]}')
            os.chdir(par_folder[residue])

            lig_par = ligandParameters(residue+'.pdb', metal_pdb=metal_pdb, command_log=self.command_log)
            lig_par.getAmberParameters(ligand_charge=charge, overwrite=overwrite,
                                       metal_charge=metal_charge)
            os.chdir('../'*len(par_folder[residue].split('/')))

        # Copy newly generated parameters to the root folder so they can be reused directly.
        for residue, folder in par_folder.items():
            gen_mol2 = os.path.join(folder, f"{residue}.mol2")
            gen_frcmod = os.path.join(folder, f"{residue}.frcmod")
            if os.path.exists(gen_mol2):
                shutil.copyfile(gen_mol2, os.path.join(parameters_folder, f"{residue}.mol2"))
            if os.path.exists(gen_frcmod):
                shutil.copyfile(gen_frcmod, os.path.join(parameters_folder, f"{residue}.frcmod"))

        # Renumber PDB
        renum_pdb = pdb_file.replace('.pdb', '_renum.pdb')
        if not os.path.exists(renum_pdb) or regenerate_amber_files:
            command =  'pdb4amber -i '
            command += pdb_file+' '
            command += '-o '+renum_pdb+'\n'
            _run_command(command, self.command_log)

        # Parameterize metal complex with MCPB.py
        if metal_ligand:

            # Copy frcmmod file from previous optimization
            if metal_parameters:
                _copyfile_if_needed(parameters_folders['mcpbpy.frcmod'],
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
                        _copyfile_if_needed(par_folder[residue]+'/'+m+'.mol2',
                                            parameters_folder+'/'+m+'.mol2')

                _copyfile_if_needed(par_folder[residue]+'/'+residue+'.mol2',
                                    parameters_folder+'/'+residue+'.mol2')
                _copyfile_if_needed(par_folder[residue]+'/'+residue+'.frcmod',
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
                _run_command(command, self.command_log)

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

                os.chdir('../'*len(parameters_folder.split('/')))
                if commands:# and not metal_parameters:
                    if return_qm_jobs:
                        print('Returning QM jobs')
                        return commands
                    else:
                        print('Computing QM parameters')
                        for command in commands:
                            _run_command(command, self.command_log)

                print('QM calculations finished.')

                # Run step 2 of the MCPB protocol
                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 2\n'
                _run_command(command, self.command_log)
                os.chdir('../'*len(parameters_folder.split('/')))

                # Run step 3 of the MCPB protocol
                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 3\n'
                _run_command(command, self.command_log)
                os.chdir('../'*len(parameters_folder.split('/')))

                # Run step 4 of the MCPB protocol
                os.chdir(parameters_folder)
                command  = 'MCPB.py '
                command += '-i '+self.pdb_name+'.in '
                command += '-s 4\n'
                _run_command(command, self.command_log)
                os.chdir('../'*len(parameters_folder.split('/')))

        if not build_full_system:
            return

        # Generate set of metal ligand values
        metal_ligand_values = []
        for r in metal_ligand:
            if isinstance(metal_ligand[r], str): # Convert into a list
                metal_ligand[r] = [metal_ligand[r]]
            for m in metal_ligand[r]:
                metal_ligand_values.append(m)

        # Create tleap input file
        with open(parameters_folder+'/tleap.in', 'w') as tlf:
            mol2_loaded_residues = set()
            tlf.write('source leaprc.protein.ff14SB\n')
            tlf.write('source leaprc.gaff\n')
            tlf.write('source leaprc.water.tip3p\n')
            if extra_force_field:
                tlf.write('source '+extra_force_field_source+'\n')

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

                # Get mapping as tuples for residues
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
                for residue_key, source in _iter_extra_mol2_sources(extra_mol2):
                    if residue_key is not None:
                        res_upper = str(residue_key).upper()
                    else:
                        if source is None:
                            continue
                        source_str = os.fspath(source) if isinstance(source, os.PathLike) else str(source)
                        res_upper = os.path.splitext(os.path.basename(source_str))[0].upper()
                    mol2_path = os.path.join(parameters_folder, f'{res_upper}.mol2')
                    if os.path.exists(mol2_path) and res_upper not in mol2_loaded_residues:
                        tlf.write(f'{res_upper} = loadmol2 {mol2_path}\n')
                        mol2_loaded_residues.add(res_upper)

            for residue in par_folder:
                if residue in metal_ligand_values:
                    continue
                prepi_path = os.path.join(par_folder[residue], f'{residue}.prepi')
                frcmod_path = os.path.join(par_folder[residue], f'{residue}.frcmod')
                if not metal_ligand:
                    if os.path.exists(prepi_path):
                        tlf.write(f'loadamberprep {prepi_path}\n')
                    else:
                        mol2_path = os.path.join(parameters_folder, f'{residue}.mol2')
                        if os.path.exists(mol2_path) and residue not in mol2_loaded_residues:
                            tlf.write(f'{residue} = loadmol2 {mol2_path}\n')
                            mol2_loaded_residues.add(residue)
                if os.path.exists(frcmod_path):
                    tlf.write(f'loadamberparams {frcmod_path}\n')
                else:
                    frcmod_root = os.path.join(parameters_folder, f'{residue}.frcmod')
                    if os.path.exists(frcmod_root):
                        tlf.write(f'loadamberparams {frcmod_root}\n')

            if extra_frcmod:
                for residue_key, source in _iter_extra_frcmod_sources(extra_frcmod):
                    if residue_key is not None:
                        res_upper = str(residue_key).upper()
                    else:
                        if source is None:
                            continue
                        source_str = os.fspath(source) if isinstance(source, os.PathLike) else str(source)
                        res_upper = os.path.splitext(os.path.basename(source_str))[0].upper()
                    frcmod_path = os.path.join(parameters_folder, f'{res_upper}.frcmod')
                    if os.path.exists(frcmod_path):
                        tlf.write(f'loadamberparams {frcmod_path}\n')

            if metal_ligand:
                tlf.write('loadamberparams '+mcpb_frcmod+'\n')
                tlf.write('mol = loadpdb '+mcpb_pdb+'\n')
            else:
                tlf.write('mol = loadpdb '+renum_pdb+'\n')

            if non_standard_residues:
                if not add_bonds:
                    add_bonds = []

                for residue in non_standard_residues:
                    residues = getResiduesByName(self.modeller.topology, residue)
                    for r in residues:
                        tlf.write('set mol.'+r.id+' connect0 mol.'+r.id+'.N\n')
                        tlf.write('set mol.'+r.id+' connect1 mol.'+r.id+'.C\n')
                        # cn_bond = ((r.chain.id, int(r.id)-1, 'C'),(r.chain.id, int(r.id), 'N'))
                        # nc_bond = ((r.chain.id, int(r.id), 'C'),(r.chain.id, int(r.id)+1, 'N'))
                        # add_bonds.append(cn_bond)
                        # add_bonds.append(nc_bond)

            # Add bonds
            if add_bonds:

                # Map original residue indexes to the renumbered ones
                input_pdb_object = PDBFile(self.input_pdb)
                o_residues = []
                for chain in input_pdb_object.topology.chains():
                    for residue in chain.residues():
                        o_residues.append((chain.id, int(residue.id)))

                renum_pdb_object = PDBFile(renum_pdb)
                r_residues = []
                for chain in renum_pdb_object.topology.chains():
                    for residue in chain.residues():
                        r_residues.append((chain.id, int(residue.id)))

                res_mapping = {}
                for r1, r2 in zip(o_residues, r_residues):
                    res_mapping[r1] = r2

                for bond in add_bonds:
                    atom1 = (*res_mapping[bond[0][:2]], bond[0][2])
                    atom2 = (*res_mapping[bond[1][:2]], bond[1][2])
                    tlf.write('bond mol.'+str(atom1[1])+'.'+atom1[2]+' '+
                                   'mol.'+str(atom2[1])+'.'+atom2[2]+'\n')

            if solvate:
                tlf.write('solvatebox mol TIP3PBOX 12\n')

            #Add ions with addIons2 (fast) or addIonsRand (slow)
            if add_counterionsRand:
                add_counterions = False
            if add_counterions:
                tlf.write('addIons2 mol Na+ 0\n')
                tlf.write('addIons2 mol Cl- 0\n')
            if add_counterionsRand:
                tlf.write('addIonsRand mol Na+ 0\n')
                tlf.write('addIonsRand mol Cl- 0\n')


            if save_amber_pdb:
                tlf.write('savepdb mol '+parameters_folder+'/'+self.pdb_name+'_amber.pdb\n')

            tlf.write('saveamberparm mol '+parameters_folder+'/'+self.pdb_name+'.prmtop '+parameters_folder+'/'+self.pdb_name+'.inpcrd\n')

        _run_command('tleap -s -f '+parameters_folder+'/tleap.in', self.command_log)

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

    def get_command_log(self):
        """Return a copy of the recorded external command log."""
        return list(self.command_log)

class ligandParameters:

    def __init__(self, ligand_pdb, metal_pdb=None, command_log=None):

        self.ligand_pdb = ligand_pdb
        self.pdb = PDBFile(self.ligand_pdb)
        self.pdb_name = ligand_pdb.split('/')[-1].replace('.pdb', '')
        self.metal_pdb = metal_pdb
        self.metal = bool(metal_pdb)
        self.command_log = command_log

        res_count = 0
        for residue in self.pdb.topology.residues():
            res_count += 1
            self.resname = residue.name

        if res_count != 1:
            raise ValueError('A PDB with a single residue must be given for parameterization!')

    def getAmberParameters(self, charge_model='bcc', ligand_charge=0, metal_charge=None, overwrite=False):
        def _run_acdoctor_on_mol2(mol2_file):
            if not os.path.exists(mol2_file):
                raise FileNotFoundError(f'acdoctor input {mol2_file} not found.')
            command = (
                'antechamber '
                f'-i {shlex.quote(mol2_file)} '
                '-fi mol2 '
                '-o /dev/null '
                '-fo mol2 '
                '-j 0 '
                '-dr y '
                '-s 2 '
                '-pf y\n'
            )
            ret_code = _run_command(command, self.command_log)
            if ret_code != 0:
                raise RuntimeError(f'acdoctor (antechamber) failed for {mol2_file} with exit code {ret_code}.')

        # Execute pdb4amber to generate a renumbered PDB
        if not os.path.exists(self.resname+'_renum.pdb') or overwrite:
            command  = 'pdb4amber '
            command += '-i '+self.ligand_pdb+' '
            command += '-o '+self.resname+'_renum.pdb '
            _run_command(command, self.command_log)

        # Run antechamber to create a mol2 file with the atomic charges
        mol2_path = self.resname+'.mol2'
        mol2_generated = False
        if not os.path.exists(mol2_path) or overwrite:
            command = 'antechamber '
            command += '-i '+self.resname+'_renum.pdb '
            command += '-fi pdb '
            command += '-o '+self.resname+'.mol2 '
            command += '-fo mol2 '
            command += '-pf y '
            command += '-c '+charge_model+' '
            command += '-nc '+str(ligand_charge)+' '
            command += '-s 2\n'
            _run_command(command, self.command_log)
            mol2_generated = True

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
            _run_command(command, self.command_log)

        # Run parmchk to check which forcefield parameters will be used
        frcmod_path = self.resname+'.frcmod'
        need_parmchk = overwrite or not os.path.exists(frcmod_path)
        if need_parmchk and not mol2_generated:
            _run_acdoctor_on_mol2(mol2_path)
        if need_parmchk:
            command  = 'parmchk2 '
            command += '-i '+mol2_path+' '
            command += '-o '+frcmod_path+' '
            command += '-f mol2\n'
            _run_command(command, self.command_log)

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
                    _run_command(command, self.command_log)

def _getPositionsArrayAsVector(positions):
    v3_positions = []
    for p in positions:
        v3_positions.append(quantity.Quantity(Vec3(*p), unit=nanometer))
    return v3_positions

def _copyFFFiles(output_folder, ff):
    """
    Copy all forcefield files from the specified folder in the prepare_proteins package.

    Parameters
    ==========
    output_folder : str
        Path to the folder where the forcefield files will be copied.
    ff : str
        Name of the forcefield folder to copy files from.
    """
    ffs = ['parmBSC1', 'parmBSC2']
    if ff not in ffs:
        raise ValueError(f"{ff} not implemented!")

    # Path to the folder containing the forcefield files
    ff_folder_path = resource_filename(
        Requirement.parse("prepare_proteins"), f"prepare_proteins/MD/ff_files/{ff}"
    )

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the forcefield folder and copy them to the output folder
    for filename in os.listdir(ff_folder_path):
        source_file = os.path.join(ff_folder_path, filename)

        # Skip directories like __pycache__
        if os.path.isdir(source_file):
            continue

        if source_file == '__init__.py':
            continue

        destination_file = os.path.join(output_folder, filename)

        # Give ff path to leaprc file
        if 'leaprc.bsc' in destination_file:
            destination_file = open(destination_file, 'w')
            with open(source_file) as sf:
                for line in sf:
                    line = line.replace('FF_PATH', output_folder)
                    destination_file.write(line)
            destination_file.close()
        else:
            # Copy the file to the output directory
            shutil.copy(source_file, destination_file)
