from . import alignment
from . import _atom_selectors
from . import rosettaScripts
from . import MD

import os
import sys
import shutil
import uuid
import itertools
import io
import subprocess
import json
from pkg_resources import resource_stream, Requirement, resource_listdir

import numpy as np
from Bio import PDB
from Bio.PDB.DSSP import DSSP
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md
import fileinput


import prepare_proteins

class proteinModels:
    """
    Attributes
    ==========
    models_folder : str
        Path to folder were PDB models are located.
    models_paths : dict
        Contains the paths to each model folder.
    msa :
        Multiple sequence alignment object
    multi_chain :
        Whether any model contains multiple chains
    sequences : dict
        Contains the sequence of each model.
    structures : dict
        Contains the Bio.PDB.Structure object to each model.

    Methods
    =======
    readModelFromPDB(self, models_folder)
        Read a model from a PDB file.
    getModelsSequences(self)
        Updates the sequences of each model in the PDB.
    calculateMSA(self)
        Calculates a multiple sequence alignment from the current sequences.
    calculateSecondaryStructure(self)
        Calculate secondary structure strings using DSSP.
    removeTerminalUnstructuredRegions(self)
        Remove terminal unstructured regions from all models.
    saveModels(self, output_folder)
        Save current models into an output folder.

    Hidden Methods
    ==============
    _getChainSequence(self, chain):
        Get the sequnce from a Bio.PDB.Chain.Chain object.
    _getModelsPaths(self)
        Get the paths for all PDBs in the input_folder path.
    """

    def __init__(self, models_folder, get_sequences=True, get_ss=False, msa=False):
        """
        Read PDB models as Bio.PDB structure objects.

        Parameters
        ==========
        models_folder : str
            Path to the folder containing the PDB models.
        get_sequences : bool
            Get the sequences from the structure. They will be separated by chain and
            can be accessed through the .sequences attribute.
        get_ss : bool
            Get the strign representing the secondary structure of the models. they
            can be accessed through the .ss attribute.
        msa : bool
            single-chain structures at startup, othewise look for the calculateMSA()
            method.
        """

        self.models_folder = models_folder
        self.models_paths = self._getModelsPaths()
        self.models_names = [] # Store model names
        self.structures = {} # structures are stored here
        self.sequences = {} # sequences are stored here
        self.target_sequences = {} # Final sequences are stored here
        self.msa = None # multiple sequence alignment
        self.multi_chain = False
        self.ss = {} # secondary structure strings are stored here
        self.docking_data = None # secondary structure strings are stored here
        self.docking_ligands = {}
        self.rosetta_data = None # Rosetta data is stored here
        self.sequence_differences = {} # Store missing/changed sequence information
        self.conects = {} # Store the conection inforamtion for each model
        self.covalent = {} # Store covalent residues

        # Read PDB structures into Biopython
        for model in sorted(self.models_paths):
            self.models_names.append(model)
            self.readModelFromPDB(model, self.models_paths[model])

        if get_sequences:
            # Get sequence information based on stored structure objects
            self.getModelsSequences()

        if get_ss:
            # Calculate secondary structure inforamtion as strings
            self.calculateSecondaryStructure()

        # # Perform a multiple sequence aligment of models
        if msa:
            if self.multichain:
                print('MSA cannot be calculated at startup when multichain models \
are given. See the calculateMSA() method for selecting which chains will be algined.')
            else:
                self.calculateMSA()

    def addResidueToModel(self, model, chain_id, resname, atom_names, coordinates,
                          new_resid=None, elements=None, hetatom=True, water=False):
        """
        Add a residue to a specific model.

        Parameters
        ==========
        model : str
            Model name to edit
        chain_id : str
            Chain ID to which the residue will be added.
        resname : str
            Name of the residue to be added.
        atom_names : list ot tuple
            Atom names of each atom in the residue to add.
        coordinates : numpy.ndarray
            Atom coordinates array, it should match the order in the given
            atom_names.
        elements : list
            List of atomic elements. One per each atom.
        hetatom : bool
            Is the residue an hetatm?
        water : bool
            Is the residue a water residue?
        """

        # Check model name
        if model not in self.structures:
            raise ValueError('The input model was not found.')

        # Check chain ID
        chain = [chain for chain in self.structures[model].get_chains() if chain_id == chain.id]
        if len(chain) != 1:
            raise ValueError('Chain ID given was not found in the selected model.')

        # Check coordinates correctness
        if coordinates.shape == ():
            if np.isnan(coordinates):
                raise ValueError('Given Coordinate in nan!')
        elif np.isnan(coordinates.any()):
            raise ValueError('Some given Coordinates are nan!')
        if coordinates.shape[1] != 3:
            raise ValueError('Coordinates must have shape (x,3). X=number of atoms in residue.')
        if len(coordinates.shape) > 1:
            if coordinates.shape[0] != len(atom_names):
                raise ValueError('Mismatch between the number of atom_names and coordinates.')
        if len(coordinates.shape) == 1:
                if len(atom_names) != 1:
                    raise ValueError('Mismatch between the number of atom_names and coordinates.')

        # Create new residue
        if new_resid == None:
            new_resid = max([r.id[1] for r in chain[0].get_residues()])+1

        rt_flag = ' ' # Define the residue type flag for complete the residue ID.
        if hetatom:
            rt_flag = 'H'
        if water:
            rt_flag = 'W'
        residue = PDB.Residue.Residue((rt_flag, new_resid, ' '), resname, ' ')

        # Add new atoms to residue
        serial_number = max([a.serial_number for a in chain[0].get_atoms()])+1
        for i, atnm in enumerate(atom_names):
            if elements:
                atom = PDB.Atom.Atom(atom_names[i], coordinates[i], 0, 1.0, ' ',
                                     '%-4s' % atom_names[i], serial_number+i, elements[i])
            else:
                atom = PDB.Atom.Atom(atom_names[i], coordinates[i], 0, 1.0, ' ',
                                     '%-4s' % atom_names[i], serial_number+i)
            residue.add(atom)
        chain[0].add(residue)

        return new_resid

    def removeModelAtoms(self, model, atoms_list):
        """
        Remove specific atoms of a model. Atoms to delete are given as a list of tuples.
        Each tuple contains three positions specifying (chain_id, residue_id, atom_name).

        Paramters
        =========
        atom_lists : list
            Specifies the list of atoms to delete for the particular model.
        """
        for remove_atom in atoms_list:
            for chain in self.structures[model].get_chains():
                if chain.id == remove_atom[0]:
                    for residue in chain:
                        if residue.id[1] == remove_atom[1]:
                            for atom in residue:
                                if atom.name == remove_atom[2]:
                                    print('Removing atom: '+str(remove_atom)+' from model '+model)
                                    residue.detach_child(atom.id)

    def readModelFromPDB(self, model, pdb_file, wat_to_hoh=False, covalent_check=True,
                         atom_mapping=None):
        """
        Adds a model from a PDB file.

        Parameters
        ----------
        model : str
            Model name.
        pdb_file : str
            Path to the pdb file.
        wat_to_hoh : bool
            Change the water name from WAT to HOH. Specially useful when resing from
            Rosetta optimization output files containing water.

        Returns
        -------
        structure : Bio.PDB.Structure
            Structure object.
        """
        self.structures[model] = _readPDB(model, pdb_file)

        if wat_to_hoh:
            for residue in self.structures[model].get_residues():
                if residue.resname == 'WAT':
                    residue.resname = 'HOH'

        if model not in self.conects or self.conects[model] == []:
            # Read conect lines
            self.conects[model] = self._readPDBConectLines(pdb_file, model)

        # Check covalent ligands
        if covalent_check:
            self._checkCovalentLigands(model, pdb_file, atom_mapping=atom_mapping)

        # Update conect lines
        self.conects[model] = self._readPDBConectLines(pdb_file, model)

        self.models_paths[model] = pdb_file

        return self.structures[model]

    def getModelsSequences(self):
        """
        Get sequence information for all stored models. It modifies the self.multi_chain
        option to True if more than one chain is found in the models.

        Returns
        =======
        sequences : dict
            Contains the sequences of all models.
        """
        self.multi_chain = False
        # Add sequence information
        for model in self.models_names:
            chains = [c for c in self.structures[model].get_chains()]
            if len(chains) == 1:
                for c in chains:
                    self.sequences[model] = self._getChainSequence(c)
            else:
                self.sequences[model] = {}
                for c in chains:
                    self.sequences[model][c.id] = self._getChainSequence(c)
                # If any model has more than one chain set multi_chain to True.
                self.multi_chain = True

        return self.sequences

    def calculateMSA(self, chains=None):
        """
        Calculate a Multiple Sequence Alignment from the current models' sequences.

        Returns
        =======
        alignment : Bio.AlignIO
            Multiple sequence alignment in Biopython format.
        chains : dict
            Dictionary specifying which chain to use for each model
        """

        for model in self.models_names:
            if isinstance(self.sequences[model], dict) and chains == None:
                raise ValueError('There are multiple chains in model %s. Specify which \
chain to use for each model with the chains option.' % model)

        if chains != None:
            sequences = {}
            for model in self.models_names:
                if isinstance(self.sequences[model], dict):
                    sequences[model] = self.sequences[model][chains[model]]
                else:
                    sequences[model] = self.sequences[model]

            self.msa = alignment.mafft.multipleSequenceAlignment(sequences)
        else:
            self.msa = alignment.mafft.multipleSequenceAlignment(self.sequences)

        return self.msa

    def getConservedMSAPositions(self, msa):
        """
        Get all conserved MSA positions.

        Returns
        =======
        conserved : list
            All conserved MSA positions indexes and the conserved amino acid identity.
        """

        positions = {}
        conserved = []
        n_models = len(self.msa)
        for i in range(self.msa.get_alignment_length()):
            positions[i] = []
            for model in self.msa:
                positions[i].append(model.seq[i])
            positions[i] = set(positions[i])
            if len(positions[i]) == 1:
                conserved.append((i,list(positions[i])[0]))

        return conserved

    def getStructurePositionFromMSAindex(self, msa_index):
        """
        Get the individual model residue structure positions of a specific MSA index

        Paramters
        =========
        msa_index : int
            Zero-based MSA index

        Returns
        =======
        residue_indexes : dict
            Residue indexes for each protein at the MSA position
        """

        residue_positions = {}
        residue_ids = {}

        # Gather dictionary between sequence position and residue PDB index
        for model in self.models_names:
            residue_positions[model] = 0
            residue_ids[model] = {}
            for i,r in enumerate(self.structures[model].get_residues()):
                residue_ids[model][i+1] = r.id[1]

        # Gather sequence indexes for the given MSA index
        for i in range(self.msa.get_alignment_length()):

            # Count structure positions
            for entry in self.msa:
                if entry.seq[i] != '-':
                    residue_positions[entry.id] += 1

            # Get residue positions matching the MSA indexes
            if i == msa_index:
                for entry in self.msa:
                    if entry.seq[i] == '-':
                        residue_positions[entry.id] = None
                    else:
                        residue_positions[entry.id] = residue_ids[entry.id][residue_positions[entry.id]]
                break

        return residue_positions

    def calculateSecondaryStructure(self, _save_structure=False):
        """
        Calculate secondary structure information for each model.
        DSSP Code:
        H : Alpha helix (4-12)
        B : Isolated beta-bridge residue
        E : Strand
        G : 3-10 helix
        I : Pi helix
        T : Turn
        S : Bend
        - : None
        Parameters
        ==========
        _save_structure : bool
            Save structure model before computing secondary structure. This option
            is used if models have been modified.
        Returns
        ss : dict
            Contains the secondary structure strings for each model.
        """

        for model in self.models_names:
            structure_path = self.models_paths[model]
            if _save_structure:
                structure_path = '.'+str(uuid.uuid4())+'.pdb'
                _saveStructureToPDB(self.structures[model], structure_path)

            dssp = DSSP(self.structures[model][0], structure_path)
            if _save_structure:
                os.remove(structure_path)
            ss = []
            for k in dssp.keys():
                ss.append(dssp[k][2])
            ss = ''.join(ss)
            self.ss[model] = ss

        return self.ss

    def keepModelChains(self, model, chains):
        """
        Only keep the specified chains for the selected model.

        Parameters
        ==========
        model : str
            Model name
        chains : list or tuple or str
            Chain IDs to keep.
        """
        if isinstance(chains, str):
            chains = list(chains)

        remove = []
        for chain in self.structures[model].get_chains():
            if chain.id not in chains:
                print('From model %s Removing chain %s' % (model, chain.id))
                remove.append(chain)

        model = [*self.structures[model].get_models()][0]
        for chain in remove:
            model.detach_child(chain.id)

        self.getModelsSequences()

    def removeTerminalUnstructuredRegions(self, n_hanging=3):
        """
        Remove unstructured terminal regions from models.

        Parameters
        ==========
        n_hangin : int
            Maximum unstructured number of residues to keep at the unstructured terminal regions.
        """

        if self.multi_chain:
            raise ValueError('removeTerminalUnstructuredRegions() function only supports single chain models')

        # Calculate residues to be removed
        for model in self.models_names:

            # Get N-terminal residues to remove based on secondary structure.
            remove_indexes = []
            for i,r in enumerate(self.ss[model]):
                if r == '-':
                    remove_indexes.append(i)
                else:
                    break

            if len(remove_indexes) > n_hanging:
                remove_indexes = remove_indexes[:-n_hanging]
            else:
                remove_indexes = []

            # Get C-terminal residues to remove based on secondary structure.
            remove_C = []
            for i,r in enumerate(self.ss[model][::-1]):
                if r == '-':
                    remove_C.append(i)
                else:
                    break
            if len(remove_C) > n_hanging:
                remove_C = remove_C[:-n_hanging]
            else:
                remove_C = []

            for x in remove_C:
                remove_indexes.append(len(self.ss[model])-1-x)

            # Sort indexes
            remove_indexes = sorted(remove_indexes)

            # Get residues to remove from models structures
            remove_this = []
            for c in self.structures[model].get_chains():
                for i,r in enumerate(c.get_residues()):
                    if i in remove_indexes:
                        remove_this.append(r)
                chain = c

            # Remove residues
            for r in remove_this:
                chain.detach_child(r.id)

        self.getModelsSequences()
        self.calculateSecondaryStructure(_save_structure=True)

    def removeTerminiByConfidenceScore(self, confidence_threshold=70):
        """
        Remove terminal regions with low confidence scores from models.
        """

        ## Warning only single chain implemented
        for model in self.models_names:

            atoms = [a for a in self.structures[model].get_atoms()]
            n_terminus = set()
            for a in atoms:
                if a.bfactor < confidence_threshold:
                    n_terminus.add(a.get_parent().id[1])
                else:
                    break
            c_terminus = set()

            for a in reversed(atoms):
                if a.bfactor < confidence_threshold:
                    c_terminus.add(a.get_parent().id[1])
                else:
                    break
            n_terminus = sorted(list(n_terminus))
            c_terminus = sorted(list(c_terminus))

            remove_this = []
            for c in self.structures[model].get_chains():
                for r in c.get_residues():
                    if r.id[1] in n_terminus or r.id[1] in c_terminus:
                        remove_this.append(r)
                chain = c

            # Remove residues
            for r in remove_this:
                chain.detach_child(r.id)

        self.getModelsSequences()
        # self.calculateSecondaryStructure(_save_structure=True)

        # Missing save models and reload them to take effect.

    def alignModelsToReferencePDB(self, reference, output_folder, chain_indexes=None,
                                  trajectory_chain_indexes=None, reference_chain_indexes=None,
                                  aligment_mode='aligned'):
        """
        Align all models to a reference PDB based on a sequence alignemnt.

        The chains are specified using their indexes. When the trajectories have
        corresponding chains use the option chain_indexes to specify the list of
        chains to align. Otherwise, specify the chains with trajectory_chain_indexes
        and reference_chain_indexes options. Note that the list of chain indexes
        must be corresponding.

        Parameters
        ==========
        reference : str
            Path to the reference PDB
        output_folder : str
            Path to the output folder to store models
        mode : str
            The mode defines how sequences are aligned. 'exact' for structurally
            aligning positions with exactly the same aminoacids after the sequence
            alignemnt or 'aligned' for structurally aligining sequences using all
            positions aligned in the sequence alignment.
        """

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        reference = md.load(reference)
        for model in self.models_names:
            traj = md.load(self.models_paths[model])
            MD.alignTrajectoryBySequenceAlignment(traj, reference, chain_indexes=chain_indexes,
                                                  trajectory_chain_indexes=trajectory_chain_indexes,
                                                  aligment_mode=aligment_mode)

            traj.save(output_folder+'/'+model+'.pdb')

    def createMutants(self, job_folder, mutants, nstruct=100, relax_cycles=0, cst_optimization=True,
                      param_files=None, mpi_command='slurm'):
        """
        Create mutations from protein models. Mutations (mutants) must be given as a nested dictionary
        with each protein as the first key and the name of the particular mutant as the second key.
        The value of each inner dictionary is a list containing the mutations, with each mutation
        described by a 2-element tuple (residue_id, aa_to_mutate). E.g., (73, 'A').

        Parameters
        ==========
        job_folder : str
            Folder path where to place the mutation job.
        mutants : dict
            Dictionary specify the mutants to generate.
        relax_cycles : int
            Apply this number of relax cycles (default:0, i.e., no relax).
        nstruct : int
            Number of structures to generate when relaxing mutant
        param_files : list
            Params file to use when reading model with Rosetta.
        """

        mpi_commands = ['slurm', 'openmpi', None]
        if mpi_command not in ['slurm', 'openmpi', None]:
            raise ValueError('Wrong mpi_command it should either: '+' '.join(mpi_commands))

        # Create mutation job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder+'/input_models'):
            os.mkdir(job_folder+'/input_models')
        if not os.path.exists(job_folder+'/flags'):
            os.mkdir(job_folder+'/flags')
        if not os.path.exists(job_folder+'/xml'):
            os.mkdir(job_folder+'/xml')
        if not os.path.exists(job_folder+'/output_models'):
            os.mkdir(job_folder+'/output_models')

        # Save considered models
        considered_models = list(mutants.keys())
        self.saveModels(job_folder+'/input_models', models=considered_models)

        jobs = []

        # Create all-atom score function
        score_fxn_name = 'ref2015'
        sfxn = rosettaScripts.scorefunctions.new_scorefunction(score_fxn_name,
                                                               weights_file=score_fxn_name)

        for model in self.models_names:

            # Skip models not in given mutants
            if model not in considered_models:
                continue

            if not os.path.exists(job_folder+'/output_models/'+model):
                os.mkdir(job_folder+'/output_models/'+model)

            # Iterate each mutant
            for mutant in mutants[model]:

                # Create xml mutation (and minimization) protocol
                xml = rosettaScripts.xmlScript()
                protocol = []

                # Add score function
                xml.addScorefunction(sfxn)

                for m in mutants[model][mutant]:
                    mutate = rosettaScripts.movers.mutate(name='mutate_'+str(m[0]),
                                                          target_residue=m[0],
                                                          new_residue=PDB.Polypeptide.one_to_three(m[1]))
                    xml.addMover(mutate)
                    protocol.append(mutate)

                if relax_cycles:
                    # Create fastrelax mover
                    relax = rosettaScripts.movers.fastRelax(repeats=relax_cycles, scorefxn=sfxn)
                    xml.addMover(relax)
                    protocol.append(relax)
                else:
                    # Turn off more than one structure when relax is not performed
                    nstruct = 1

                # Set protocol
                xml.setProtocol(protocol)

                # Add scorefunction output
                xml.addOutputScorefunction(sfxn)

                # Write XMl protocol file
                xml.write_xml(job_folder+'/xml/'+model+'_'+mutant+'.xml')

                # Create options for minimization protocol
                flags = rosettaScripts.flags('../../xml/'+model+'_'+mutant+'.xml',
                                             nstruct=nstruct, s='../../input_models/'+model+'.pdb',
                                             output_silent_file=model+'_'+mutant+'.out')

                # Add relaxation with constraints options and write flags file
                if cst_optimization and relax_cycles:
                    flags.add_relax_cst_options()
                else:
                    flags.add_relax_options()

                # Add path to params files
                if param_files != None:
                    if not os.path.exists(job_folder+'/params'):
                        os.mkdir(job_folder+'/params')
                    if isinstance(param_files, str):
                        param_files = [param_files]
                    for param in param_files:
                        param_name = param.split('/')[-1]
                        shutil.copyfile(param, job_folder+'/params/'+param_name)
                    flags.addOption('in:file:extra_res_path', '../../params')

                flags.write_flags(job_folder+'/flags/'+model+'_'+mutant+'.flags')

                # Create and append execution command
                if mpi_command == None:
                    mpi_command = ''

                command = 'cd '+job_folder+'/output_models/'+model+'\n'
                command += mpi_command+' rosetta_scripts.mpi.linuxgccrelease @ '+'../../flags/'+model+'_'+mutant+'.flags\n'
                command += 'cd ../../..\n'
                jobs.append(command)

        return jobs

    def setUpRosettaOptimization(self, relax_folder, nstruct=1000, relax_cycles=5,
                                 cst_files=None, mutations=False, models=None, cst_optimization=True,
                                 membrane=False, membrane_thickness=15, param_files=None):
        """
        Set up minimizations using Rosetta FastRelax protocol.

        Parameters
        ==========
        relax_folder : str
            Folder path where to place the relax job.
        """

        # Create minimization job folders
        if not os.path.exists(relax_folder):
            os.mkdir(relax_folder)
        if not os.path.exists(relax_folder+'/input_models'):
            os.mkdir(relax_folder+'/input_models')
        if not os.path.exists(relax_folder+'/flags'):
            os.mkdir(relax_folder+'/flags')
        if not os.path.exists(relax_folder+'/xml'):
            os.mkdir(relax_folder+'/xml')
        if not os.path.exists(relax_folder+'/output_models'):
            os.mkdir(relax_folder+'/output_models')

        # Save all models
        self.saveModels(relax_folder+'/input_models', models=models)

        # Check that sequence comparison has been done before adding mutational steps
        if mutations:
            if self.sequence_differences == {}:
                raise ValueError('Mutations have been enabled but no sequence comparison\
has been carried out. Please run compareSequences() function before setting mutation=True.')

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Skip models not in the given list
            if models != None:
                    continue

            if not os.path.exists(relax_folder+'/output_models/'+model):
                os.mkdir(relax_folder+'/output_models/'+model)

            # Create xml minimization protocol
            xml = rosettaScripts.xmlScript()
            protocol = []

            # Create membrane scorefucntion
            if membrane:
                # Create all-atom score function
                sfxn = rosettaScripts.scorefunctions.new_scorefunction('mpframework_smooth_fa_2012',
                                                                       weights_file='mpframework_smooth_fa_2012')
                # Add constraint weights to membrane score function
                if cst_files != None:
                    reweights = (('chainbreak', 1.0),
                                ('coordinate_constraint', 1.0),
                                ('atom_pair_constraint', 1.0),
                                ('angle_constraint', 1.0),
                                ('dihedral_constraint', 1.0),
                                ('res_type_constraint', 1.0),
                                ('metalbinding_constraint', 1.0))

                    for rw in reweights:
                        sfxn.addReweight(rw[0],rw[1])

            # Create all-atom scorefucntion
            else:
                score_fxn_name = 'ref2015'
                # Check if constraints are given
                if cst_files != None:
                    score_fxn_name = score_fxn_name+'_cst'

                # Create all-atom score function
                sfxn = rosettaScripts.scorefunctions.new_scorefunction(score_fxn_name,
                                                                       weights_file=score_fxn_name)
            xml.addScorefunction(sfxn)

            # Create mutation movers if needed
            if mutations:
                if self.sequence_differences[model]['mutations'] != {}:
                    for m in self.sequence_differences[model]['mutations']:
                        mutate = rosettaScripts.movers.mutate(name='mutate_'+str(m[0]),
                                                              target_residue=m[0],
                                                              new_residue=PDB.Polypeptide.one_to_three(m[1]))
                        xml.addMover(mutate)
                        protocol.append(mutate)

            # Add constraint mover if constraint file is given.
            if cst_files != None:
                if model not in cst_files:
                    raise ValueError('Model %s is not in the cst_files dictionary!' % model)
                set_cst = rosettaScripts.movers.constraintSetMover(add_constraints=True,
                                                                   cst_file='../../../'+cst_files[model])
                xml.addMover(set_cst)
                protocol.append(set_cst)

            if membrane:
                add_membrane = rosettaScripts.rosetta_MP.movers.addMembraneMover()
                xml.addMover(add_membrane)
                protocol.append(add_membrane)

                init_membrane = rosettaScripts.rosetta_MP.movers.membranePositionFromTopologyMover()
                xml.addMover(init_membrane)
                protocol.append(init_membrane)

            # Create fastrelax mover
            relax = rosettaScripts.movers.fastRelax(repeats=relax_cycles, scorefxn=sfxn)
            xml.addMover(relax)
            protocol.append(relax)

            # Set protocol
            xml.setProtocol(protocol)

            # Add scorefunction output
            xml.addOutputScorefunction(sfxn)

            # Write XMl protocol file
            xml.write_xml(relax_folder+'/xml/'+model+'_relax.xml')

            # Create options for minimization protocol
            flags = rosettaScripts.flags('../../xml/'+model+'_relax.xml',
                                         nstruct=nstruct, s='../../input_models/'+model+'.pdb',
                                         output_silent_file=model+'_relax.out')

            # Add relaxation with constraints options and write flags file
            if cst_optimization:
                flags.add_relax_cst_options()
            else:
                flags.add_relax_options()

            # Add path to params files
            if param_files != None:

                if not os.path.exists(relax_folder+'/params'):
                    os.mkdir(relax_folder+'/params')

                if isinstance(param_files, str):
                    param_files = [param_files]
                for param in param_files:
                    param_name = param.split('/')[-1]
                    shutil.copyfile(param, relax_folder+'/params/'+param_name)
                flags.addOption('in:file:extra_res_path', '../../params')

            if membrane:
                flags.addOption('mp::setup::spans_from_structure', 'true')
                flags.addOption('relax:constrain_relax_to_start_coords', '')
            flags.write_flags(relax_folder+'/flags/'+model+'_relax.flags')

            # Create and append execution command
            command = 'cd '+relax_folder+'/output_models/'+model+'\n'
            command += 'srun rosetta_scripts.mpi.linuxgccrelease @ '+'../../flags/'+model+'_relax.flags\n'
            command += 'cd ../../..\n'
            jobs.append(command)

        return jobs

    def setUpMembranePositioning(self, job_folder, membrane_thickness=15, models=None):
        """
        """
        # Create minimization job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder+'/input_models'):
            os.mkdir(job_folder+'/input_models')
        if not os.path.exists(job_folder+'/flags'):
            os.mkdir(job_folder+'/flags')
        if not os.path.exists(job_folder+'/output_models'):
            os.mkdir(job_folder+'/output_models')

        # Save all models
        self.saveModels(job_folder+'/input_models', models=models)

        # Copy embeddingToMembrane.py script
        _copyScriptFile(job_folder, 'embeddingToMembrane.py')

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            if not os.path.exists(job_folder+'/output_models/'+model):
                os.mkdir(job_folder+'/output_models/'+model)

            flag_file = job_folder+'/flags/mp_span_from_pdb_'+model+'.flags'
            with open(flag_file, 'w') as flags:
                flags.write('-mp::thickness '+str(membrane_thickness)+'\n')
                flags.write('-s model.pdb\n')
                flags.write('-out:path:pdb .\n')

            flag_file = job_folder+'/flags/mp_transform_'+model+'.flags'
            with open(flag_file, 'w') as flags:
                flags.write('-s ../../input_models/'+model+'.pdb\n')
                flags.write('-mp:transform:optimize_embedding true\n')
                flags.write('-mp:setup:spanfiles '+model+'.span\n')
                flags.write('-out:no_nstruct_label\n')

            command = 'cd '+job_folder+'/output_models/'+model+'\n'
            command += 'cp ../../input_models/'+model+'.pdb model.pdb\n'
            command += 'mp_span_from_pdb.linuxgccrelease @ ../../flags/mp_span_from_pdb_'+model+'.flags\n'
            command += 'rm model.pdb \n'
            command += 'mv model.span '+model+'.span\n'
            command += 'mp_transform.linuxgccrelease @ ../../flags/mp_transform_'+model+'.flags\n'
            command += 'python ../../._embeddingToMembrane.py'+' '+model+'.pdb\n'
            command += 'cd ../../..\n'
            jobs.append(command)

        return jobs

    def addMissingLoops(self, job_folder, nstruct=1, sfxn='ref2015', param_files=None, idealize=True):
        """
        Create a Rosetta loop optimization protocol for missing loops in the structure.

        Parameters
        ==========
        job_folder : str
            Loop modeling calculation folder.
        """

        # Create minimization job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder+'/input_models'):
            os.mkdir(job_folder+'/input_models')
        if not os.path.exists(job_folder+'/flags'):
            os.mkdir(job_folder+'/flags')
        if not os.path.exists(job_folder+'/xml'):
            os.mkdir(job_folder+'/xml')
        if not os.path.exists(job_folder+'/output_models'):
            os.mkdir(job_folder+'/output_models')

        # Save all models
        self.saveModels(job_folder+'/input_models')

        # Check that sequence comparison has been done before checking missing loops
        if self.sequence_differences == {}:
            raise ValueError('No sequence comparison has been carried out. Please run \
compareSequences() function before adding missing loops.')

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Check that model has missing loops
            if self.sequence_differences[model]['missing_loops'] != []:

                missing_loops = self.sequence_differences[model]['missing_loops']

                for loop in missing_loops:

                    loop_name = str(loop[0])+'_'+str(loop[1])

                    if not os.path.exists(job_folder+'/output_models/'+model):
                        os.mkdir(job_folder+'/output_models/'+model)

                    if not os.path.exists(job_folder+'/output_models/'+model+'/'+loop_name):
                        os.mkdir(job_folder+'/output_models/'+model+'/'+loop_name)

                    # Create xml minimization protocol
                    xml = rosettaScripts.xmlScript()
                    protocol = []

                    # Create score function

                    scorefxn = rosettaScripts.scorefunctions.new_scorefunction(sfxn,
                                                                               weights_file=sfxn)

                    # Add loop remodel protocol
                    if len(loop[1]) == 1:
                        hanging_residues = 3
                    elif len(loop[1]) == 2:
                        hanging_residues = 2
                    else:
                        hanging_residues = 1
                    loop_movers = rosettaScripts.loop_modeling.loopRebuild(xml, loop[0], loop[1], scorefxn=sfxn,
                                                                           hanging_residues=hanging_residues)
                    for m in loop_movers:
                        protocol.append(m)

                    # Add idealize step
                    if idealize:
                        idealize = rosettaScripts.movers.idealize()
                        xml.addMover(idealize)
                        protocol.append(idealize)

                    # Set protocol
                    xml.setProtocol(protocol)

                    # Add scorefunction output
                    xml.addOutputScorefunction(scorefxn)
                    # Write XMl protocol file
                    xml.write_xml(job_folder+'/xml/'+model+'_'+loop_name+'.xml')

                    # Create options for minimization protocol
                    output_silent = 'output_models/'+model+'/'+loop_name+'/'+model+'_'+loop_name+'.out'
                    flags = rosettaScripts.flags('xml/'+model+'_'+loop_name+'.xml',
                                                 nstruct=nstruct, s='input_models/'+model+'.pdb',
                                                 output_silent_file=output_silent)

                    # Add path to params files
                    if param_files != None:
                        if not os.path.exists(job_folder+'/params'):
                            os.mkdir(job_folder+'/params')

                        if isinstance(param_files, str):
                            param_files = [param_files]
                        for param in param_files:
                            param_name = param.split('/')[-1]
                            shutil.copyfile(param, job_folder+'/params/'+param_name)
                        flags.addOption('in:file:extra_res_path', 'params')

                    # Write flags file
                    flags.write_flags(job_folder+'/flags/'+model+'_'+loop_name+'.flags')

                    # Create and append execution command
                    command = 'cd '+job_folder+'\n'
                    command += 'srun rosetta_scripts.mpi.linuxgccrelease @ '+'flags/'+model+'_'+loop_name+'.flags\n'
                    command += 'cd ..\n'

                    jobs.append(command)

        return jobs

    def setUpPrepwizardOptimization(self, prepare_folder, pH=7.0, epik_pH=False, samplewater=False,
                                    epik_pHt=False, remove_hydrogens=False, delwater_hbond_cutoff=False,
                                    fill_loops=False, protonation_states=None, no_epik=False, mae_input=False,
                                    use_new_version=False, **kwargs):
        """
        Set up an structure optimization with the Schrodinger Suite prepwizard.

        Parameters
        ==========
        prepare_folder : str
            Folder name for the prepwizard optimization.
        """

        # Create prepare job folders
        if not os.path.exists(prepare_folder):
            os.mkdir(prepare_folder)
        if not os.path.exists(prepare_folder+'/input_models'):
            os.mkdir(prepare_folder+'/input_models')
        if not os.path.exists(prepare_folder+'/output_models'):
            os.mkdir(prepare_folder+'/output_models')

        # Save all input models
        self.saveModels(prepare_folder+'/input_models', convert_to_mae=mae_input,
                        remove_hydrogens=remove_hydrogens, **kwargs)

        # Generate jobs
        jobs = []
        for model in self.models_names:
            if fill_loops:
                if model not in self.target_sequences:
                    raise ValueError('Target sequence for model %s was not given. First\
make sure of reading the target sequences with the function readTargetSequences()' % model)
                sequence = {}
                sequence[model] = self.target_sequences[model]
                fasta_file = prepare_folder+'/input_models/'+model+'.fasta'
                alignment.writeFastaFile(sequence, fasta_file)

            # Create model output folder
            output_folder = prepare_folder+'/output_models/'+model
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            if fill_loops:
                command = 'cd '+prepare_folder+'/input_models/\n'
                command += 'pwd=$(pwd)\n'
                command += 'cd ../output_models/'+model+'\n'
            else:
                command = 'cd '+output_folder+'\n'

            command += '"${SCHRODINGER}/utilities/prepwizard" '
            if mae_input:
                command += '../../input_models/'+model+'.mae '
            else:
                command += '../../input_models/'+model+'.pdb '
            command += model+'.pdb '
            command += '-fillsidechains '
            command += '-disulfides '
            if fill_loops:
                command += '-fillloops '
                command += '-fasta_file "$pwd"/'+model+'.fasta '
            if remove_hydrogens:
                command += '-rehtreat '
            if no_epik:
                command += '-noepik '
            else:
                if epik_pH:
                    command += '-epik_pH '+str(pH)+' '
                if epik_pHt:
                    command += '-epik_pHt '+str(epik_pHt)+' '
            command += '-propka_pH '+str(pH)+' '
            command += '-f 2005 '
            command += '-rmsd 0.3 '
            if samplewater:
                command += '-samplewater '
            if delwater_hbond_cutoff:
                command += '-delwater_hbond_cutoff '+str(delwater_hbond_cutoff)+' '

            if not isinstance(protonation_states, type(None)):
                for ps in protonation_states[model]:
                    if use_new_version:
                        command += '-force '+str(ps[0])+" "+str(ps[1])+' '
                    else:
                        command += '-force '+str(ps[0])+" "+str(ps[1])+' '

            command += '-JOBNAME '+model+' '
            command += '-HOST localhost:1 '
            command += '-WAIT\n'
            command += 'cd ../../..\n'
            jobs.append(command)

        return jobs

    def setUpDockingGrid(self, grid_folder, center_atoms, innerbox=(10,10,10),
                         outerbox=(30,30,30), useflexmae=True, peptide=False, mae_input=True):
        """
        Setup grid calculation for each model.

        Parameters
        ==========
        grid_folder : str
            Path to grid calculation folder
        center_atoms : tuple
        """

        # Create grid job folders
        if not os.path.exists(grid_folder):
            os.mkdir(grid_folder)

        if not os.path.exists(grid_folder+'/input_models'):
            os.mkdir(grid_folder+'/input_models')

        if not os.path.exists(grid_folder+'/grid_inputs'):
            os.mkdir(grid_folder+'/grid_inputs')

        if not os.path.exists(grid_folder+'/output_models'):
            os.mkdir(grid_folder+'/output_models')

        # Save all input models
        self.saveModels(grid_folder+'/input_models', convert_to_mae=mae_input)

        # Check that inner and outerbox values are given as integers
        for v in innerbox:
            if type(v) != int:
                raise ValueError('Innerbox values must be given as integers')
        for v in outerbox:
            if type(v) != int:
                raise ValueError('Outerbox values must be given as integers')

        # Create grid input files
        jobs = []
        for model in self.models_names:

            # Get coordinates of center residue
            chainid = center_atoms[model][0]
            resid = center_atoms[model][1]
            atom_name = center_atoms[model][2]

            x = None
            for c in self.structures[model].get_chains():
                if c.id == chainid:
                    for r in c.get_residues():
                        if r.id[1] == resid:
                            for a in r.get_atoms():
                                if a.name == atom_name:
                                    x = a.coord[0]
                                    y = a.coord[1]
                                    z = a.coord[2]

            # Check if any atom center was found.
            if x == None:
                raise ValueError('Given atom center not found for model %s' % model)

            # Write grid input file
            with open(grid_folder+'/grid_inputs/'+model+'.in', 'w') as gif:
                gif.write('GRID_CENTER %.14f, %.14f, %.14f\n' % (x,y,z))
                gif.write('GRIDFILE '+model+'.zip\n')
                gif.write('INNERBOX %s, %s, %s\n' % innerbox)
                gif.write('OUTERBOX %s, %s, %s\n' % outerbox)
                if mae_input:
                    gif.write('RECEP_FILE %s\n' % ('../input_models/'+model+'.mae'))
                else:
                    gif.write('RECEP_FILE %s\n' % ('../input_models/'+model+'.pdb'))
                if peptide:
                    gif.write('PEPTIDE True\n')
                if useflexmae:
                    gif.write('USEFLEXMAE YES\n')

            command = 'cd '+grid_folder+'/output_models\n'

            # Add convert PDB into mae format command
            # command += '"$SCHRODINGER/utilities/structconvert" '
            # if mae_input:
            #     command += '-ipdb ../input_models/'+model+'.pdb'+' '
            #     command += '-omae '+model+'.mae\n'

            # Add grid generation command
            command += '"${SCHRODINGER}/glide" '
            command += '../grid_inputs/'+model+'.in'+' '
            command += '-OVERWRITE '
            command += '-HOST localhost '
            command += '-TMPLAUNCHDIR '
            command += '-WAIT\n'

            command += 'cd ../..\n'

            jobs.append(command)

        return jobs

    def setUpGlideDocking(self, docking_folder, grids_folder, ligands_folder,
                          poses_per_lig=100, precision='SP', use_ligand_charges=False,
                          energy_by_residue=False, use_new_version=False,):
        """
        Set docking calculations for all the proteins and set of ligands located
        grid_folders and ligands_folder folders, respectively. The ligands must be provided
        in MAE format.

        Parameters
        ==========
        docking_folder : str

        ligands_folder : str
            Path to the folder containing the ligands to dock.
        residues : dict
            Dictionary with the residues for each model near which to position the
            ligand as the starting pose.
        """

        # Create docking job folders
        if not os.path.exists(docking_folder):
            os.mkdir(docking_folder)

        if not os.path.exists(docking_folder+'/input_models'):
            os.mkdir(docking_folder+'/input_models')

        if not os.path.exists(docking_folder+'/output_models'):
            os.mkdir(docking_folder+'/output_models')

        # Save all input models
        self.saveModels(docking_folder+'/input_models')

        # Read paths to grid files
        grids_paths = {}
        for f in os.listdir(grids_folder+'/output_models'):
            if f.endswith('.zip'):
                name = f.replace('.zip','')
                grids_paths[name] = grids_folder+'/output_models/'+f

        # Read paths to substrates
        substrates_paths = {}
        for f in os.listdir(ligands_folder):
            if f.endswith('.mae'):
                name = f.replace('.mae','')
                substrates_paths[name] = ligands_folder+'/'+f

        # Set up docking jobs
        jobs = []
        for grid in grids_paths:
            # Create ouput folder
            if not os.path.exists(docking_folder+'/output_models/'+grid):
                os.mkdir(docking_folder+'/output_models/'+grid)

            for substrate in substrates_paths:

                # Create glide dock input
                with open(docking_folder+'/output_models/'+grid+'/'+grid+'_'+substrate+'.in', 'w') as dif:
                    dif.write('GRIDFILE GRID_PATH/'+grid+'.zip\n')
                    dif.write('LIGANDFILE ../../../%s\n' % substrates_paths[substrate])
                    dif.write('POSES_PER_LIG %s\n' % poses_per_lig)
                    if use_ligand_charges:
                        dif.write('LIG_MAECHARGES true\n')
                    dif.write('PRECISION %s\n' % precision)
                    if energy_by_residue:
                        dif.write('WRITE_RES_INTERACTION true\n')

                # Create commands
                command = 'cd '+docking_folder+'/output_models/'+grid+'\n'

                # Schrodinger has problem with relative paths to the grid files
                # This is a quick fix for that (not elegant, but works).
                command += 'cwd=$(pwd)\n'
                grid_folder = '/'.join(grids_paths[grid].split('/')[:-1])
                command += 'cd ../../../%s\n' % grid_folder
                command += 'gd=$(pwd)\n'
                command += 'cd $cwd\n'
                command += 'sed -i "s@GRID_PATH@$gd@" %s \n' % (grid+'_'+substrate+'.in')

                # Add docking command
                command += '"${SCHRODINGER}/glide" '
                command += grid+'_'+substrate+'.in'+' '
                command += '-OVERWRITE '
                command += '-adjust '
                command += '-HOST localhost:1 '
                command += '-TMPLAUNCHDIR '
                command += '-WAIT\n'
                command += 'cd ../../..\n'
                jobs.append(command)

        return jobs

    def setUpSiteMapForModels(self, job_folder, target_residue, site_box=10,
                              resolution='fine', reportsize=100, overwrite=False):
        """
        Generates a SiteMap calculation for model poses (no ligand) near specified residues.

        Parameters
        ==========
        job_folder : str
            Path to the calculation folder
        """

        # Create site map job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if not os.path.exists(job_folder+'/input_models'):
            os.mkdir(job_folder+'/input_models')

        if not os.path.exists(job_folder+'/output_models'):
            os.mkdir(job_folder+'/output_models')

        # Copy script to generate protein and ligand mae inputs, separately.
        _copyScriptFile(job_folder, 'prepareForSiteMap.py')
        script_path = job_folder+'/._prepareForSiteMap.py'

        # Save all input models
        self.saveModels(job_folder+'/input_models')

        # Create input files
        jobs = []
        for model in self.models_names:

            # Createa output folder for each model
            if not os.path.exists(job_folder+'/output_models/'+model):
                os.mkdir(job_folder+'/output_models/'+model)

            # Generate input protein and ligand files
            input_protein = job_folder+'/input_models/'+model+'.pdb'
            if not os.path.exists(input_protein) or overwrite:
                command = 'run '+script_path+' '
                command += input_protein+' '
                command += job_folder+'/output_models/'+model+' '
                command += '--protein_only '
                os.system(command)

            # Add site map command
            command = 'cd '+job_folder+'/output_models/'+model+'\n'
            command += '"${SCHRODINGER}/sitemap" '
            command += '-prot ../../input_models/'+model+'/'+model+'_protein.mae'+' '
            command += '-sitebox '+str(site_box)+' '
            command += '-resolution '+str(resolution)+' '
            command += '-reportsize '+str(reportsize)+' '
            command += '-keepvolpts yes '
            command += '-keeplogs yes '
            command += '-siteasl \"res.num {'+target_residue+'}\" '
            command += '-HOST localhost:1 '
            command += '-TMPLAUNCHDIR\n'
            command += '-WAIT\n'
            command += 'cd ../../..\n'
            jobs.append(command)

        return jobs

    def setUpSiteMapForLigands(self, job_folder, poses_folder, site_box=10, resolution='fine', overwrite=False):
        """
        Generates a SiteMap calculation for Docking poses outputed by the extractDockingPoses()
        function.

        Parameters
        ==========
        job_folder : str
            Path to the calculation folder
        poses_folder : str
            Path to docking poses folder.
        """

        # Create site map job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if not os.path.exists(job_folder+'/input_models'):
            os.mkdir(job_folder+'/input_models')

        if not os.path.exists(job_folder+'/output_models'):
            os.mkdir(job_folder+'/output_models')

        # Copy script to generate protein and ligand mae inputs, separately.
        _copyScriptFile(job_folder, 'prepareForSiteMap.py')
        script_path = job_folder+'/._prepareForSiteMap.py'

        # Create input files
        jobs = []
        for model in os.listdir(poses_folder):
            if not os.path.isdir(poses_folder+'/'+model):
                continue
            if not os.path.exists(job_folder+'/input_models/'+model):
                os.mkdir(job_folder+'/input_models/'+model)
            if not os.path.exists(job_folder+'/output_models/'+model):
                os.mkdir(job_folder+'/output_models/'+model)

            for pose in os.listdir(poses_folder+'/'+model):
                if pose.endswith('.pdb'):
                    pose_name = pose.replace('.pdb','')

                    # Generate input protein and ligand files
                    input_ligand = job_folder+'/input_models/'+model+'/'+pose_name+'_ligand.mae'
                    input_protein = job_folder+'/input_models/'+model+'/'+pose_name+'_protein.mae'
                    if not os.path.exists(input_ligand) or not os.path.exists(input_protein) or overwrite:
                        command = 'run '+script_path+' '
                        command += poses_folder+'/'+model+'/'+pose+' '
                        command += job_folder+'/input_models/'+model
                        os.system(command)

                    # Write Site Map input file
                    with open(job_folder+'/output_models/'+model+'/'+pose_name+'.in', 'w') as smi:
                        smi.write('PROTEIN ../../input_models/'+model+'/'+pose_name+'_protein.mae\n')
                        smi.write('LIGMAE ../../input_models/'+model+'/'+pose_name+'_ligand.mae\n')
                        smi.write('SITEBOX '+str(site_box)+'\n')
                        smi.write('RESOLUTION '+resolution+'\n')
                        smi.write('REPORTSIZE 100\n')
                        smi.write('KEEPVOLPTS yes\n')
                        smi.write('KEEPLOGS yes\n')

                    # Add site map command
                    command = 'cd '+job_folder+'/output_models/'+model+'\n'
                    command += '"${SCHRODINGER}/sitemap" '
                    command += pose_name+'.in'+' '
                    command += '-HOST localhost:1 '
                    command += '-TMPLAUNCHDIR '
                    command += '-WAIT\n'
                    command += 'cd ../../..\n'
                    jobs.append(command)
        return jobs

    def setUpLigandParameterization(self, job_folder, ligands_folder, charge_method=None,
                                    only_ligands=None):
        """
        Run PELE platform for ligand parameterization

        Parameters
        ==========
        job_folder : str
            Path to the job input folder
        ligands_folder : str
            Path to the folder containing the ligand molecules in PDB format.
        """

        charge_methods = ['gasteiger', 'am1bcc', 'OPLS']
        if charge_method == None:
            charge_method = 'OPLS'

        if charge_method not in charge_methods:
            raise ValueError('The charge method should be one of: '+str(charge_methods))

        # Create PELE job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        # Copy script to generate protein and ligand mae inputs, separately.
        _copyScriptFile(job_folder, 'peleffy_ligand.py')

        jobs = []
        for ligand in os.listdir(ligands_folder):

            extension = ligand.split('.')[-1]

            if extension == 'pdb':
                ligand_name = ligand.replace('.'+extension, '')

                # Only process ligands given in only_ligands list
                if only_ligands != None:
                    if ligand_name not in only_ligands:
                        continue

                # structure = _readPDB(ligand_name, ligands_folder+'/'+ligand)
                if not os.path.exists(job_folder+'/'+ligand_name):
                    os.mkdir(job_folder+'/'+ligand_name)

                # _saveStructureToPDB(structure, job_folder+'/'+pdb_name+'/'+pdb_name+extension)
                shutil.copyfile(ligands_folder+'/'+ligand, job_folder+'/'+ligand_name+'/'+ligand_name+'.'+extension)

                # Create command
                command = 'cd '+job_folder+'/'+ligand_name+'\n'
                command += 'python  ../._peleffy_ligand.py '+ligand_name+'.'+extension+'\n'
                command += 'cd ../..\n'
                jobs.append(command)

        return jobs

    def setUpPELECalculation(self, pele_folder, models_folder, input_yaml, box_centers=None, distances=None, ligand_index=1,
                             box_radius=10, steps=100, debug=False, iterations=3, cpus=96, equilibration_steps=100, ligand_energy_groups=None,
                             separator='-', use_peleffy=True, usesrun=True, energy_by_residue=False, ebr_new_flag=False, ninety_degrees_version=False,
                             analysis=False, energy_by_residue_type='all', peptide=False, equilibration_mode='equilibrationLastSnapshot',
                             spawning='independent', continuation=False, equilibration=True,  skip_models=None, skip_ligands=None,
                             extend_iterations=False, only_models=None, only_ligands=None, ligand_templates=None, seed=12345, log_file=False):
        """
        Generates a PELE calculation for extracted poses. The function reads all the
        protein ligand poses and creates input for a PELE platform set up run.

        Parameters
        ==========
        pele_folder : str
            Path to the folder where PELE calcualtions will be located
        models_folder : str
            Path to input docking poses folder.
        input_yaml : str
            Path to the input YAML file to be used as template for all the runs.
        ligand_energy_groups : dict
            Additional groups to consider when doing energy by residue reports.
        Missing!
        """

        energy_by_residue_types = ['all', 'lennard_jones', 'sgb', 'electrostatic']
        if energy_by_residue_type not in energy_by_residue_types:
            raise ValueError('%s not found. Try: %s' % (energy_by_residue_type, energy_by_residue_types))

        spawnings = ['independent', 'inverselyProportional', 'epsilon', 'variableEpsilon',
                     'independentMetric', 'UCB', 'FAST', 'ProbabilityMSM', 'MetastabilityMSM',
                     'IndependentMSM']

        if spawning not in spawnings:
            message = 'Spawning method %s not found.' % spawning
            message = 'Allowed options are: '+str(spawnings)
            raise ValueError(message)

        # Create PELE job folder
        if not os.path.exists(pele_folder):
            os.mkdir(pele_folder)

        # Read docking poses information from models_folder and create pele input
        # folders.
        jobs = []
        for d in os.listdir(models_folder):
            if os.path.isdir(models_folder+'/'+d):
                models = {}
                ligand_pdb_name = {}
                for f in os.listdir(models_folder+'/'+d):
                    fs = f.split(separator)
                    protein = fs[0]
                    ligand = fs[1]
                    pose = fs[2].replace('.pdb','')

                    # Skip given protein models
                    if skip_models != None:
                        if protein in skip_models:
                            continue

                    # Skip given ligand models
                    if skip_ligands != None:
                        if ligand in skip_ligands:
                            continue

                    # Skip proteins not in only_proteins list
                    if only_models != None:
                        if protein not in only_models:
                            continue

                    # Skip proteins not in only_ligands list
                    if only_ligands != None:
                        if ligand not in only_ligands:
                            continue

                    # Create PELE job folder for each docking
                    if not os.path.exists(pele_folder+'/'+protein+'_'+ligand):
                        os.mkdir(pele_folder+'/'+protein+'_'+ligand)

                    structure = _readPDB(protein+'_'+ligand, models_folder+'/'+d+'/'+f)

                    # Change water names if any
                    for residue in structure.get_residues():
                        if residue.id[0] == 'W':
                            residue.resname = 'HOH'

                        if residue.get_parent().id == 'L':
                            ligand_pdb_name[ligand] = residue.resname

                    ## Add dummy atom if peptide docking ### Strange fix =)
                    if peptide:
                        for chain in structure.get_chains():
                            if chain.id == 'L':
                                # Create new residue
                                new_resid = max([r.id[1] for r in chain.get_residues()])+1
                                residue = PDB.Residue.Residue(('H', new_resid, ' '), 'XXX', ' ')
                                serial_number = max([a.serial_number for a in chain.get_atoms()])+1
                                atom = PDB.Atom.Atom('X', [0,0,0], 0, 1.0, ' ',
                                                     '%-4s' % 'X', serial_number+1, 'H')
                                residue.add(atom)
                                chain.add(residue)

                    _saveStructureToPDB(structure, pele_folder+'/'+protein+'_'+ligand+'/'+f)

                    if (protein, ligand) not in models:
                        models[(protein,ligand)] = []
                    models[(protein,ligand)].append(f)

                # If templates are given for ligands
                templates = {}
                if ligand_templates != None:

                    # Create templates folder
                    if not os.path.exists(pele_folder+'/templates'):
                        os.mkdir(pele_folder+'/templates')

                    for ligand in os.listdir(ligand_templates):

                        if not os.path.isdir(ligand_templates+'/'+ligand):
                            continue

                        # Create ligand template folder
                        if not os.path.exists(pele_folder+'/templates/'+ligand):
                            os.mkdir(pele_folder+'/templates/'+ligand)

                        templates[ligand] = []
                        for f in os.listdir(ligand_templates+'/'+ligand):
                            if f.endswith('.rot.assign') or f.endswith('z'):

                                # Copy template files
                                shutil.copyfile(ligand_templates+'/'+ligand+'/'+f,
                                                pele_folder+'/templates/'+ligand+'/'+f)

                                templates[ligand].append(f)

                # Create YAML file
                for model in models:
                    protein, ligand = model
                    keywords = ['system', 'chain', 'resname', 'steps', 'iterations', 'atom_dist', 'analyse',
                                'cpus', 'equilibration', 'equilibration_steps', 'traj', 'working_folder',
                                'usesrun', 'use_peleffy', 'debug', 'box_radius', 'box_center', 'equilibration_mode',
                                'seed' ,'spawning']

                    # Write input yaml
                    with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input.yaml', 'w') as iyf:
                        if energy_by_residue:
                            # Use new PELE version with implemented energy_by_residue
                            iyf.write('pele_exec: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/bin/PELE-1.7.2_mpi"\n')
                            iyf.write('pele_data: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/Data"\n')
                            iyf.write('pele_documents: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/Documents/"\n')
                        elif ninety_degrees_version:
                            # Use new PELE version with implemented 90 degrees fix
                            iyf.write('pele_exec: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/bin/PELE-1.8_mpi"\n')
                            iyf.write('pele_data: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Data"\n')
                            iyf.write('pele_documents: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Documents/"\n')
                        if len(models[model]) > 1:
                            equilibration_mode = 'equilibrationCluster'
                            iyf.write("system: '*.pdb'\n")
                        else:
                            iyf.write("system: '"+" ".join(models[model])+"'\n")
                        iyf.write("chain: 'L'\n")
                        if peptide:
                            iyf.write("resname: 'XXX'\n")
                            iyf.write("skip_ligand_prep:\n")
                            iyf.write(" - 'XXX'\n")
                        else:
                            iyf.write("resname: '"+ligand_pdb_name[ligand]+"'\n")
                        iyf.write("steps: "+str(steps)+"\n")
                        iyf.write("iterations: "+str(iterations)+"\n")
                        iyf.write("cpus: "+str(cpus)+"\n")
                        if equilibration:
                            iyf.write("equilibration: true\n")
                            iyf.write("equilibration_mode: '"+equilibration_mode+"'\n")
                            iyf.write("equilibration_steps: "+str(equilibration_steps)+"\n")
                        else:
                            iyf.write("equilibration: false\n")
                        if spawning != None:
                            iyf.write("spawning: '"+str(spawning)+"'\n")
                        iyf.write("traj: trajectory.xtc\n")
                        iyf.write("working_folder: 'output'\n")
                        if usesrun:
                            iyf.write("usesrun: true\n")
                        else:
                            iyf.write("usesrun: false\n")
                        if use_peleffy:
                            iyf.write("use_peleffy: true\n")
                        else:
                            iyf.write("use_peleffy: false\n")
                        if analysis:
                            iyf.write("analyse: true\n")
                        else:
                            iyf.write("analyse: false\n")

                        if ligand in templates:
                            iyf.write("templates:\n")
                            iyf.write(' - "LIGAND_TEMPLATE_PATH_ROT"\n')
                            iyf.write(' - "LIGAND_TEMPLATE_PATH_Z"\n')
                            iyf.write("skip_ligand_prep:\n")
                            iyf.write(' - "'+ligand_pdb_name[ligand]+'"\n')

                        iyf.write("box_radius: "+str(box_radius)+"\n")
                        if isinstance(box_centers, type(None)) and peptide:
                            raise ValueError('You must give per-protein box_centers when docking peptides!')
                        if not isinstance(box_centers, type(None)):
                            if not all(isinstance(x, float) for x in box_centers[model]):
                                # get coordinates from tuple
                                for chain in self.structures[model[0]].get_chains():
                                    if chain.id == box_centers[model][0]:
                                        for r in chain:
                                            if r.id[1] == box_centers[model][1]:
                                                for atom in r:
                                                    if atom.name == box_centers[model][2]:
                                                        coordinates = atom.coord
                            else:
                                coordinates = box_centers[model]

                            box_center = ''
                            for coord in coordinates:
                                #if not isinstance(coord, float):
                                #    raise ValueError('Box centers must be given as a (x,y,z) tuple or list of floats.')
                                box_center += '  - '+str(coord)+'\n'
                            iyf.write("box_center: \n"+box_center)

                        # energy by residue is not implemented in PELE platform, therefore
                        # a scond script will modify the PELE.conf file to set up the energy
                        # by residue calculation.
                        if debug or energy_by_residue or peptide:
                            iyf.write("debug: true\n")

                        if distances != None:
                            iyf.write("atom_dist:\n")
                            for d in distances[protein][ligand]:
                                if isinstance(d[0], str):
                                    d1 = "- 'L:"+str(ligand_index)+":"+d[0]+"'\n"
                                else:
                                    d1 = "- '"+d[0][0]+":"+str(d[0][1])+":"+d[0][2]+"'\n"
                                if isinstance(d[1], str):
                                    d2 = "- 'L:"+str(ligand_index)+":"+d[1]+"'\n"
                                else:
                                    d2 = "- '"+d[1][0]+":"+str(d[1][1])+":"+d[1][2]+"'\n"
                                iyf.write(d1)
                                iyf.write(d2)

                        if seed:
                            iyf.write('seed: '+str(seed)+'\n')

                        if log_file:
                            iyf.write('log: true\n')

                        iyf.write('\n')
                        iyf.write("#Options gathered from "+input_yaml+'\n')

                        with open(input_yaml) as tyf:
                            for l in tyf:
                                if l.startswith('#'):
                                    continue
                                elif l.startswith('-'):
                                    continue
                                elif l.strip() == '':
                                    continue
                                if l.split()[0].replace(':', '') not in keywords:
                                    iyf.write(l)

                    if energy_by_residue:
                        _copyScriptFile(pele_folder, 'addEnergyByResidueToPELEconf.py')
                        ebr_script_name = '._addEnergyByResidueToPELEconf.py'
                        if not isinstance(ligand_energy_groups, type(None)):
                            if not isinstance(ligand_energy_groups, dict):
                                raise ValueError('ligand_energy_groups, must be given as a dictionary')
                            with open(pele_folder+'/'+protein+'_'+ligand+'/ligand_energy_groups.json', 'w') as jf:
                                json.dump(ligand_energy_groups[ligand], jf)

                    if peptide:
                        _copyScriptFile(pele_folder, 'modifyPelePlatformForPeptide.py')
                        peptide_script_name = '._modifyPelePlatformForPeptide.py'

                    # Create command
                    command = 'cd '+pele_folder+'/'+protein+'_'+ligand+'\n'

                    # Add commands to write template folder absolute paths
                    if ligand in templates:
                        command += "export CWD=$(pwd)\n"
                        command += 'cd ../templates\n'
                        command += 'export TMPLT_DIR=$(pwd)\n'
                        command += 'cd $CWD\n'
                        for tf in templates[ligand]:
                            if continuation:
                                yaml_file = 'input_restart.yaml'
                            else:
                                yaml_file = 'input.yaml'
                            if tf.endswith('.assign'):
                                command += "sed -i s,LIGAND_TEMPLATE_PATH_ROT,$TMPLT_DIR/"+tf+",g "+yaml_file+"\n"
                            elif tf.endswith('z'):
                                command += "sed -i s,LIGAND_TEMPLATE_PATH_Z,$TMPLT_DIR/"+tf+",g "+yaml_file+"\n"
                    if not continuation:
                        command += 'python -m pele_platform.main input.yaml\n'
                    if continuation:
                        debug_line = False
                        restart_line = False
                        with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                            with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        debug_line = True
                                        oyml.write('restart: true\n')
                                        oyml.write('adaptive_restart: true\n')
                                    elif 'restart: true' in l:
                                        continue
                                    oyml.write(l)
                                if not debug_line:
                                    oyml.write('restart: true\n')
                                    oyml.write('adaptive_restart: true\n')

                        command += 'python -m pele_platform.main input_restart.yaml\n'

                    elif energy_by_residue:
                        command += 'python ../'+ebr_script_name+' output --energy_type '+energy_by_residue_type
                        if isinstance(ligand_energy_groups, dict):
                            command += ' --ligand_energy_groups ligand_energy_groups.json'
                            command += ' --ligand_index '+str(ligand_index)
                        if ebr_new_flag:
                            command += ' --new_version '
                        if peptide:
                            command += ' --peptide \n'
                            command += 'python ../'+peptide_script_name+' output '+" ".join(models[model])+'\n'
                        else:
                            command += '\n'
                        with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                            with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        l = 'restart: true\n'
                                    oyml.write(l)
                        command += 'python -m pele_platform.main input_restart.yaml\n'
                    elif peptide:
                        command += 'python ../'+peptide_script_name+' output '+" ".join(models[model])+'\n'
                        with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                            with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        l = 'restart: true\n'
                                    oyml.write(l)
                        command += 'python -m pele_platform.main input_restart.yaml\n'
                    elif extend_iterations and not continuation:
                        raise ValueEror('extend_iterations must be used together with the continuation keyword')
                    command += 'cd ../..'
                    jobs.append(command)

        return jobs

    def setUpMDSimulations(self,md_folder,sim_time,frags=5,program='gromacs',command_name='gmx_mpi',ff='amber99sb-star-ildn',benchmark=False,benchmark_steps=10,water_traj=False,ion_chain=False):
        """
        Sets up MD simulations for each model. The current state only allows to set
        up simulations for apo proteins and using the Gromacs software. WARNING: peptide chain
        must be named L and ions must be grouped in chain I.

        ######################################
        ###  TODO:                         ###
        ### - generalize selection numbers ###
        ### - fix genrestr itp files       ###
        ######################################
        Parameters
        ==========
        md_folder : str
            Path to the job folder where the MD input files are located.
        sim_time : int
            Number of simulation steps
        frags : int
            Number of fragments to divide the simulation.
        program : str
            Program to execute simulation.
        command : str
            Command to call program.
        ff : str
            Force field to use for simulation.

        """

        available_programs = ['gromacs']

        if program not in available_programs:
            raise ValueError('The program %s is not available for setting MD simulations.' % program)

        # Create MD job folders
        if benchmark == True:
            md_folder = md_folder + '_benchmark'

        if not os.path.exists(md_folder):
            os.mkdir(md_folder)
        if not os.path.exists(md_folder+'/scripts'):
            os.mkdir(md_folder+'/scripts')
        if not os.path.exists(md_folder+'/FF'):
            os.mkdir(md_folder+'/FF')
        if not os.path.exists(md_folder+'/FF/'+ff+".ff"):
            os.mkdir(md_folder+'/FF/'+ff+".ff")
        if not os.path.exists(md_folder+'/input_models'):
            os.mkdir(md_folder+'/input_models')
        if not os.path.exists(md_folder+'/output_models'):
            os.mkdir(md_folder+'/output_models')

        # Save all input models
        self.saveModels(md_folder+'/input_models')

        # Copy script files
        if program == 'gromacs':
            for file in resource_listdir(Requirement.parse("prepare_proteins"), 'prepare_proteins/scripts/md/gromacs/mdp'):
                if not file.startswith("__"):
                    _copyScriptFile(md_folder+'/scripts/', file, subfolder='md/gromacs/mdp',no_py=False,hidden=False)

            for file in resource_listdir(Requirement.parse("prepare_proteins"), 'prepare_proteins/scripts/md/gromacs/ff/'+ff):
                if not file.startswith("__"):
                    _copyScriptFile(md_folder+'/FF/'+ff+'.ff', file, subfolder='md/gromacs/ff/'+ff,no_py=False,hidden=False)


            for line in fileinput.input(md_folder+'/scripts/md.mdp', inplace=True):
                if line.strip().startswith('nsteps'):
                    line = 'nsteps = '+ str(int(sim_time/frags)) + '\n'
                #if water_traj == True:
                #    if line.strip().startswith('compressed-x-grps'):
                #        line = 'compressed_x_grps = '+'System'+ '\n'

                sys.stdout.write(line)


            jobs = []

            for model in self.models_names:
                # Create additional folders
                if not os.path.exists(md_folder+'/output_models/'+model):
                    os.mkdir(md_folder+'/output_models/'+model)

                parser = PDB.PDBParser()
                structure = parser.get_structure('protein', md_folder+'/input_models/'+model+'.pdb')

                gmx_codes = []

                for mdl in structure:
                    for chain in mdl:
                        for residue in chain:
                            HD1 = False
                            HE2 = False
                            if residue.resname == 'HIS':
                                for atom in residue:
                                    if atom.name == 'HD1':
                                        HD1 = True
                                    if atom.name == 'HE2':
                                        HE2 = True
                            if HD1 != False or HE2 != False:
                                if HD1 == True and HE2 == False:
                                    number = 0
                                if HD1 == False and HE2 == True:
                                    number = 1
                                if HD1 == True and HE2 == True:
                                    number = 2
                                gmx_codes.append(number)

                his_pro = (str(gmx_codes)[1:-1].replace(',',''))

                command = 'cd '+md_folder+'\n'
                command += "export GMXLIB=$(pwd)/FF" +'\n'

                # Set up commands
                if not os.path.exists(md_folder+'/output_models/'+model+"/topol/prot_ions.pdb"):
                    command += 'mkdir output_models/'+model+'/topol'+'\n'
                    command += 'cp input_models/'+model+'.pdb output_models/'+model+'/topol/protein.pdb'+'\n'
                    command += 'cd output_models/'+model+'/topol'+'\n'
                    if ion_chain:
                        command += 'echo '+his_pro+' | '+command_name+' pdb2gmx -f protein.pdb -o prot.gro -p topol.top -his -ignh -ff '+ff+' -water tip3p -vsite hydrogens -merge all'+'\n'
                    else:
                        command += 'echo '+his_pro+' | '+command_name+' pdb2gmx -f protein.pdb -o prot.gro -p topol.top -his -ignh -ff '+ff+' -water tip3p -vsite hydrogens'+'\n'

                    command += command_name+ ' editconf -f prot.gro -o prot_box.gro -c -d 1.0 -bt octahedron'+'\n'
                    command += command_name+' solvate -cp prot_box.gro -cs spc216.gro -o prot_solv.gro -p topol.top'+'\n'
                    command += command_name+' grompp -f ../../../scripts/ions.mdp -c prot_solv.gro -p topol.top -o prot_ions.tpr -maxwarn 1'+'\n'

                    if ion_chain:
                        selector = '15'
                    else:
                        selector = '13'

                    command += 'echo '+selector+' | '+command_name+' genion -s prot_ions.tpr -o prot_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.1'+'\n'

                    if ion_chain:
                        command += 'echo  -e "1|13\\nq"| gmx make_ndx -f  prot_ions.gro'+'\n'

                    command += 'cd ..'+'\n'
                else:
                    command += 'cd output_models/'+model+'\n'



                # Energy minimization
                if not os.path.exists(md_folder+'/output_models/'+model+"/em/prot_em.tpr"):
                    command += 'mkdir em'+'\n'
                    command += 'cd em'+'\n'
                    command += command_name+' grompp -f ../../../scripts/em.mdp -c ../topol/prot_ions.gro -p ../topol/topol.top -o prot_em.tpr'+'\n'
                    command += command_name+' mdrun -v -deffnm prot_em'+'\n'
                    command += 'cd ..'+'\n'


                # NVT equilibration
                if not os.path.exists(md_folder+'/output_models/'+model+"/nvt/prot_nvt.tpr"):
                    command += 'mkdir nvt'+'\n'
                    command += 'cd nvt'+'\n'
                    if ion_chain:
                        #command += 'echo 18 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre_Protein_chain_A.itp -fc 1000 1000 1000'+'\n'
                        #command += 'echo 18 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre_Protein_chain_L.itp -fc 1000 1000 1000'+'\n'
                        #command += 'echo 18 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre_Ion_chain_I.itp -fc 1000 1000 1000'+'\n'
                        command += 'echo 20 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre.itp -fc 1000 1000 1000 -n ../topol/index.ndx'+'\n'
                    else:
                        command += 'echo 1 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre.itp -fc 1000 1000 1000'+'\n'

                    command += command_name+' grompp -f ../../../scripts/nvt.mdp -c ../em/prot_em.gro -p ../topol/topol.top -o prot_nvt.tpr -r ../em/prot_em.gro'+'\n'
                    command += command_name+' mdrun -v -deffnm prot_nvt'+'\n'
                    command += 'cd ..'+'\n'

                # NPT equilibration
                FClist= ('550','300','170','90','50','30','15','10','5')
                if not os.path.exists(md_folder+'/output_models/'+model+'/npt'):
                    command += 'mkdir npt'+'\n'
                command += 'cd npt'+'\n'


                for i in range(len(FClist)+1):
                    if not os.path.exists(md_folder+'/output_models/'+model+'/npt/prot_npt_'+str(i+1)+'.tpr'):
                        if i == 0:
                            command += command_name+' grompp -f ../../../scripts/npt.mdp -c ../nvt/prot_nvt.gro -t ../nvt/prot_nvt.cpt -p ../topol/topol.top -o prot_npt_1.tpr -r ../nvt/prot_nvt.gro'+'\n'
                            command += command_name+' mdrun -v -deffnm prot_npt_'+str(i+1)+'\n'
                        else:
                            if ion_chain:
                                #command += 'echo 18 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre_Protein_chain_A.itp -fc '+FClist[i-1]+' '+FClist[i-1]+' '+FClist[i-1]+'\n'
                                #command += 'echo 18 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre_Protein_chain_L.itp -fc '+FClist[i-1]+' '+FClist[i-1]+' '+FClist[i-1]+'\n'
                                #command += 'echo 18 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre_Ion_chain_I.itp -fc '+FClist[i-1]+' '+FClist[i-1]+' '+FClist[i-1]+'\n'
                                command += 'echo 20 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre.itp -fc '+FClist[i-1]+' '+FClist[i-1]+' '+FClist[i-1]+' -n ../topol/index.ndx\n'
                            else:
                                command += 'echo 1 | '+command_name+' genrestr -f ../topol/prot_ions.gro -o ../topol/posre.itp -fc '+FClist[i-1]+' '+FClist[i-1]+' '+FClist[i-1]+'\n'

                            command += command_name+' grompp -f ../../../scripts/npt.mdp -c prot_npt_'+str(i)+'.gro -t prot_npt_'+str(i)+'.cpt -p ../topol/topol.top -o prot_npt_'+str(i+1)+'.tpr -r prot_npt_'+str(i)+'.gro'+'\n'
                            command += command_name+' mdrun -v -deffnm prot_npt_'+str(i+1)+'\n'
                command += 'cd ..'+'\n'


                #Production run
                if not os.path.exists(md_folder+'/output_models/'+model+'/md'):
                    command += 'mkdir md'+'\n'
                command += 'cd md'+'\n'
                for i in range(1,frags+1):
                    if not os.path.exists(md_folder+'/output_models/'+model+'/md/prot_md_'+str(i)+'.xtc'):
                        if i == 1:
                            command += command_name+' grompp -f ../../../scripts/md.mdp -c ../npt/prot_npt_' + str(len(FClist)+1) + '.gro  -t ../npt/prot_npt_' + str(len(FClist)+1) + '.cpt -p ../topol/topol.top -o prot_md_'+str(i)+'.tpr'+'\n'
                            command += command_name+' mdrun -v -deffnm prot_md_' + str(i) + '\n'
                        else:
                            command += command_name+' grompp -f ../../../scripts/md.mdp -c prot_md_'+str(i-1)+'.gro -t prot_md_'+str(i-1)+'.cpt -p ../topol/topol.top -o prot_md_'+str(i)+'.tpr'+'\n'
                            command += command_name+' mdrun -v -deffnm prot_md_'+str(i)+'\n'
                    else:
                        if os.path.exists(md_folder+'/output_models/'+model+'/md/prot_md_'+str(i)+'_prev.cpt'):
                            command += command_name+' mdrun -v -deffnm prot_md_'+str(i)+' -cpi prot_md_'+str(i)+'_prev.cpt'+'\n'

                jobs.append(command)

            return jobs


    def getTrajectoryPaths(self,path,step='md',traj_name='prot_md_cat_noPBC.xtc'):
        """
        """
        output_paths = []
        for folder in os.listdir(path+'/output_models/'):
            if folder in self.models_names:
                traj_path = path+'/output_models/'+folder+'/'+step
                output_paths.append(traj_path+'/'+traj_name)

        return(output_paths)



    def removeBoundaryConditions(self,path,command,step='md',remove_water=False):
        """
        Remove boundary conditions from gromacs simulation trajectory file

        Parameters
        ==========
        path : str
            Path to the job folder where the MD outputs files are located.
        command : str
            Command to call program.
        """
        for folder in os.listdir(path+'/output_models/'):
            if folder in self.models_names:
                traj_path = path+'/output_models/'+folder+'/'+step
                for file in os.listdir(traj_path):
                    if file.endswith('.xtc') and not file.endswith('_noPBC.xtc') and not os.path.exists(traj_path+'/'+file.split(".")[0]+'_noPBC.xtc'):
                        if remove_water == True:
                            option = '14'
                        else:
                            option = '0'
                        os.system('echo '+option+' | '+command+' trjconv -s '+ traj_path+'/'+file.split(".")[0] +'.tpr -f '+traj_path+'/'+file+' -o '+traj_path+'/'+file.split(".")[0]+'_noPBC.xtc -pbc mol -ur compact')

                if not os.path.exists(traj_path+'/prot_md_cat_noPBC.xtc'):
                    os.system(command+' trjcat -f '+traj_path+'/*_noPBC.xtc -o '+traj_path+'/prot_md_cat_noPBC.xtc -cat')

                ### md_1 or npt_10

                if not os.path.exists('/'.join(traj_path.split('/')[:-1])+'/npt/prot_npt_10_no_water.gro') and remove_water == True:
                    os.system('echo 1 | gmx editconf -ndef -f '+'/'.join(traj_path.split('/')[:-1])+'/npt/prot_npt_10.gro -o '+'/'.join(traj_path.split('/')[:-1])+'/npt/prot_npt_10_no_water.gro')



    def analyseDocking(self, docking_folder, protein_atoms=None, atom_pairs=None,
                       skip_chains=False, return_failed=False, ignore_hydrogens=False):
        """
        Analyse a Glide Docking simulation. The function allows to calculate ligand
        distances with the options protein_atoms or protein_pairs. With the first option
        the analysis will calculate the closest distance between the protein atoms given
        and any ligand atom (or heavy atom if ignore_hydrogens=True). The analysis will
        also return which ligand atom is the closest for each pose. On the other hand, with
        the atom_pairs option only distances for the specific atom pairs between the
        protein and the ligand will be calculated.

        The protein_atoms dictionary must contain as keys the model names (see iterable of this class),
        and as values a list of tuples, with each tuple representing a protein atom:
            {model1_name: [(chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name), ...], model2_name:...}

        The atom pairs must be given in a dicionary with each key representing the name
        of a model and each value  a sub dicionary with the ligands as keys and a list of the atom pairs
        to calculate in the format:
            {model1_name: { ligand_name : [((chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name)), ...],...} model2_name:...}

        Paramaeters
        ===========
        docking_folder : str
            Path to the folder where the docking resuts are (the format comes from the setUpGlideDocking() function.
        protein_atoms : dict
            Protein atoms to use for the closest distance calculation.
        atom_pairs : dict
            Protein and ligand atoms to use for distances calculation.
        skip_chains : bool
            Consider chains when atom tuples are given?
        return_failed : bool
            Return failed dockings as a list?
        ignore_hydrogens : bool
            With this option ligand hydrogens will be ignored for the closest distance (i.e., protein_atoms) calculation.
        """
        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(docking_folder, 'analyse_docking.py')
        script_path = docking_folder+'/._analyse_docking.py'

        # Write protein_atoms dictionary to json file
        if protein_atoms != None:
            with open(docking_folder+'/._protein_atoms.json', 'w') as jf:
                json.dump(protein_atoms, jf)

        # Write atom_pairs dictionary to json file
        if atom_pairs != None:
            with open(docking_folder+'/._atom_pairs.json', 'w') as jf:
                json.dump(atom_pairs, jf)

        # Execute docking analysis
        os.chdir(docking_folder)

        command = 'run ._analyse_docking.py'
        if atom_pairs != None:
            command += ' --atom_pairs ._atom_pairs.json'
        elif protein_atoms != None:
            command += ' --protein_atoms ._protein_atoms.json'
        if skip_chains:
            command += ' --skip_chains'
        if return_failed:
            command += ' --return_failed'
        if ignore_hydrogens:
            command += ' --ignore_hydrogens'
        os.system(command)

        # Read the CSV file into pandas
        if not os.path.exists('._docking_data.csv'):
            os.chdir('..')
            raise ValueError('Docking analysis failed. Check the ouput of the analyse_docking.py script.')

        self.docking_data = pd.read_csv('._docking_data.csv')
        # Create multiindex dataframe
        self.docking_data.set_index(['Protein', 'Ligand', 'Pose'], inplace=True)

        # Create dictionary with proteins and ligands
        for protein in self.docking_data.index.levels[0]:
            protein_series = self.docking_data[self.docking_data.index.get_level_values('Protein') == protein]
            self.docking_ligands[protein] = []
            ligands = [*set(protein_series.index.get_level_values('Ligand'))]
            for ligand in ligands:
                self.docking_ligands[protein].append(ligand)

        # Remove tmp files
        os.remove('._analyse_docking.py')
        os.remove('._docking_data.csv')
        if os.path.exists('._protein_atoms.json'):
            os.remove('._protein_atoms.json')

        if return_failed:
            with open('._failed_dockings.json') as jifd:
                failed_dockings = json.load(jifd)
            os.remove('._failed_dockings.json')
            os.chdir('..')
            return failed_dockings

        os.chdir('..')

    def convertLigandPDBtoMae(self, ligands_folder, change_ligand_name=True):
        """
        Convert ligand PDBs into MAE files.

        Parameters
        ==========
        ligands_folder : str
            Path to the folder where ligands are in PDB format
        """

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(ligands_folder, 'PDBtoMAE.py')
        script_name = '._PDBtoMAE.py'

        cwd = os.getcwd()
        os.chdir(ligands_folder)
        command = 'run ._PDBtoMAE.py'
        if change_ligand_name:
            command += ' --change_ligand_name'
        os.system(command)
        os.chdir(cwd)

    def convertLigandMAEtoPDB(self, ligands_folder):
        """
        Convert ligand MAEs into PDB files.

        Parameters
        ==========
        ligands_folder : str
            Path to the folder where ligands are in MAE format
        """

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(ligands_folder, 'MAEtoPDB.py')
        script_name = '._MAEtoPDB.py'

        cwd = os.getcwd()
        os.chdir(ligands_folder)
        os.system('run ._MAEtoPDB.py')
        os.chdir(cwd)

    def getDockingDistances(self, protein, ligand):
        """
        Get the distances related to a protein and ligand docking.
        """
        protein_series = self.docking_data[self.docking_data.index.get_level_values('Protein') == protein]
        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
        if not ligand_series.empty:
            distances = []
            for d in ligand_series:
                if d not in ['Score', 'RMSD', 'Catalytic distance']:
                    if not ligand_series[d].dropna().empty:
                        distances.append(d)
            return distances
        else:
            return None

    def calculateModelsDistances(self, atom_pairs):
        """
        Calculate models distances for a set of atom pairs.

        The atom pairs must be given in a dicionary with each key representing the name
        of a model and each value a list of the atom pairs to calculate in the format:
            {model_name: [((chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name)), ...], ...}

        Paramters
        =========
        atom_pairs : dict
            Atom pairs to calculate for each model
        """

        self.distance_data = {}
        self.distance_data['model'] = []

        ### Add all label entries to dictionary
        for model in self.structures:
            for d in atom_pairs[model]:
                # Generate label for distance
                label = 'distance_'
                label += '_'.join([str(x) for x in d[0]])+'-'
                label += '_'.join([str(x) for x in d[1]])
                if label not in self.distance_data:
                    self.distance_data[label] = []

        for model in self.structures:


            self.distance_data['model'].append(model)

            # Get atoms in atom_pairs as dictionary
            atoms = {}
            for d in atom_pairs[model]:
                for t in d:
                    if t[0] not in atoms:
                        atoms[t[0]] = {}
                    if t[1] not in atoms[t[0]]:
                        atoms[t[0]][t[1]] = []
                    if t[2] not in atoms[t[0]][t[1]]:
                        atoms[t[0]][t[1]].append(t[2])

            # Get atom coordinates for each atom in atom_pairs
            coordinates = {}
            for chain in self.structures[model].get_chains():
                if chain.id in atoms:
                    coordinates[chain.id] = {}
                    for r in chain:
                        if r.id[1] in atoms[chain.id]:
                            coordinates[chain.id][r.id[1]] = {}
                            for atom in r:
                                if atom.name in atoms[chain.id][r.id[1]]:
                                    coordinates[chain.id][r.id[1]][atom.name] = atom.coord

            # Calculate atom distances
            for d in atom_pairs[model]:

                # Generate label for distance
                label = 'distance_'
                label += '_'.join([str(x) for x in d[0]])+'-'
                label += '_'.join([str(x) for x in d[1]])

                # Calculate distance
                atom1 = d[0]
                atom2 = d[1]
                coord1 = coordinates[atom1[0]][atom1[1]][atom1[2]]
                coord2 = coordinates[atom2[0]][atom2[1]][atom2[2]]
                value = np.linalg.norm(coord1-coord2)

                # Add data to dataframe
                self.distance_data[label].append(value)

            # Check length of each label
            for label in self.distance_data:
                if label not in ['model']:
                    delta = len(self.distance_data['model'])-len(self.distance_data[label])
                    for x in range(delta):
                        self.distance_data[label].append(None)

        self.distance_data = pd.DataFrame(self.distance_data)
        self.distance_data.set_index('model', inplace=True)

        return self.distance_data



    def getModelDistances(self, model):
        """
        Get the distances associated with a specific model included in the
        self.distance_data atrribute. This attribute must be calculated in advance
        by running the calculateModelsDistances() function.

        Parameters
        ==========
        model : str
            model name
        """

        model_data = self.distance_data[self.distance_data.index == model]
        distances = []
        for d in model_data:
            if 'distance_' in d:
                if not model_data[d].dropna().empty:
                    distances.append(d)
        return distances

    def combineModelDistancesIntoMetric(self, metric_distances, overwrite=False):
        """
        Combine different equivalent distances contained in the self.distance_data
        attribute into specific named metrics. The function takes as input a
        dictionary (catalytic_labels) composed of inner dictionaries as follows:

            catalytic_labels = {
                metric_name = {
                    protein = distances_list}}}

        The innermost distances_list object contains all equivalent distance names for
        a specific protein to be combined under the same metric_name column.

        The combination is done by taking the minimum value of all equivalent distances.

        Parameters
        ==========
        catalytic_labels : dict
            Dictionary defining which distances will be combined under a common name.
            (for details see above).
        """
        for name in metric_distances:
            if 'metric_'+name in self.distance_data.keys() and not overwrite:
                print('Combined metric %s already added. Give overwrite=True to recombine' % name)
            else:
                values = []
                models = []

                for model in self.models_names:
                    mask = []
                    for index in self.distance_data.index:
                        if model == index:
                            mask.append(True)
                        else:
                            mask.append(False)

                    model_data = self.distance_data[mask]
                    model_distances = metric_distances[name][model]
                    values += model_data[model_distances].min(axis=1).tolist()

                self.distance_data['metric_'+name] = values

        return self.distance_data

    def getModelsProtonationStates(self, residues=None):
        """
        Get the protonation state of all or specific residues in all protein models.

        For getting the protonation states of only a subset of residues a dictionary must
        be given with the 'residues' option. The keys of the dictionary are the models'
        names, and, the values, lists of tuples defining each residue to be query. The
        residue's tuples are defined as: (chain_id, residue_id).

        Parameters
        ==========
        residues : dict
            Dictionary with lists of tuples of residues (e.g., (chain_id, residue_id)) to query for each model.

        Returns
        =======
        protonation_states : pandas.DataFrame
            Data frame with protonation information.
        """

        # Set input dictionary to store protonation states
        self.protonation_states = {}
        self.protonation_states['model'] = []
        self.protonation_states['chain'] = []
        self.protonation_states['residue'] = []
        self.protonation_states['name'] = []
        self.protonation_states['state'] = []

        # Iterate all models' structures
        for model in self.models_names:
            structure = self.structures[model]
            for r in structure.get_residues():

                # Skip if a list of residues is given per model
                if residues != None:
                    if (r.get_parent().id, r.id[1]) not in residues[model]:
                        continue

                # Get Histidine protonation states
                if r.resname == 'HIS':
                    atoms = [a.name for a in r]
                    self.protonation_states['model'].append(model)
                    self.protonation_states['chain'].append(r.get_parent().id)
                    self.protonation_states['residue'].append(r.id[1])
                    self.protonation_states['name'].append(r.resname)
                    if 'HE2' in atoms and 'HD1' in atoms:
                        self.protonation_states['state'].append('HIP')
                    elif 'HD1' in atoms:
                        self.protonation_states['state'].append('HID')
                    elif 'HE2' in atoms:
                        self.protonation_states['state'].append('HIE')

        # Convert dictionary to Pandas
        self.protonation_states = pd.DataFrame(self.protonation_states)
        self.protonation_states.set_index(['model', 'chain', 'residue'], inplace=True)

        return self.protonation_states

    def combineDockingDistancesIntoMetrics(self, catalytic_labels, exclude=None,overwrite=False):
        """
        Combine different equivalent distances into specific named metrics. The function
        takes as input a dictionary (catalytic_labels) composed of inner dictionaries as follows:

            catalytic_labels = {
                metric_name = {
                    protein = {
                        ligand = distances_list}}}

        The innermost distances_list object contains all equivalent distance names for
        a specific protein and ligand pair to be combined under the same metric_name column.

        The combination is done by taking the minimum value of all equivalent distances.

        Parameters
        ==========
        catalytic_labels : dict
            Dictionary defining which distances will be combined under a common name.
            (for details see above).
        """

        for name in catalytic_labels:
            if 'metric_'+name in self.docking_data.keys() and not overwrite:
                print('Combined metric %s already added. Give overwrite=True to recombine' % name)
            else:
                changed = True
                values = []
                for protein in self.docking_data.index.levels[0]:
                    protein_series = self.docking_data[self.docking_data.index.get_level_values('Protein') == protein]
                    for ligand in self.docking_data.index.levels[1]:
                        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                        if not ligand_series.empty:
                            distances = catalytic_labels[name][protein][ligand]
                            values += ligand_series[distances].min(axis=1).tolist()
                self.docking_data['metric_'+name] = values

    def getBestDockingPoses(self, filter_values, n_models=1, return_failed=False):
        """
        Get best models based on the best SCORE and a set of metrics with specified thresholds.
        The filter thresholds must be provided with a dictionary using the metric names as keys
        and the thresholds as the values.

        Parameters
        ==========
        n_models : int
            The number of models to select for each protein + ligand docking.
        filter_values : dict
            Thresholds for the filter.
        return_failed : bool
            Whether to return a list of the dockings without any models fulfilling
            the selection criteria. It is returned as a tuple (index 0) alongside
            the filtered data frame (index 1).
        """
        best_poses = pd.DataFrame()
        bp = []
        failed = []
        for model in self.docking_ligands:
            protein_series = self.docking_data[self.docking_data.index.get_level_values('Protein') == model]
            for ligand in self.docking_ligands[model]:
                ligand_data = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                for metric in filter_values:
                    ligand_data = ligand_data[ligand_data['metric_'+metric] < filter_values[metric]]
                if ligand_data.empty:
                    failed.append((model, ligand))
                    continue
                if ligand_data.shape[0] < n_models:
                    print('WARNING: less than %s models available for docking %s + %s' % (n_models, model, ligand))
                for i in ligand_data['Score'].nsmallest(n_models).index:
                    bp.append(i)

        if return_failed:
            return failed, self.docking_data[self.docking_data.index.isin(bp)]
        return self.docking_data[self.docking_data.index.isin(bp)]

    def getBestDockingPosesIteratively(self, metrics, ligands=None, min_threshold=3.5, max_threshold=5.0, step_size=0.1):
        extracted = []
        selected_indexes = []

        for t in np.arange(min_threshold, max_threshold+(step_size/10), step_size):
            filter_values = {m:t for m in metrics}
            best_poses = self.getBestDockingPoses(filter_values, n_models=1)
            mask = []
            if not isinstance(ligands, type(None)):
                for level in best_poses.index.get_level_values('Ligand'):
                    if level in ligands:
                        mask.append(True)
                    else:
                        mask.append(False)
                pele_data = best_poses[mask]
            else:
                pele_data = best_poses

            for row in pele_data.index:
                if row[:2] not in extracted:
                    selected_indexes.append(row)
                if row[:2] not in extracted:
                    extracted.append(row[:2])

        final_mask = []
        for row in self.docking_data.index:
            if row in selected_indexes:
                final_mask.append(True)
            else:
                final_mask.append(False)
        pele_data = self.docking_data[final_mask]

        return pele_data

    def extractDockingPoses(self, docking_data, docking_folder, output_folder, separator='-'):
        """
        Extract docking poses present in a docking_data dataframe. The docking DataFrame
        contains the same structure as the self.docking_data dataframe, parameter of
        this class. This dataframe makes reference to the docking_folder where the
        docking results are contained.

        Parameters
        ==========
        dockign_data : pandas.DataFrame
            Datframe containing the poses to be extracted
        docking_folder : str
            Path the folder containing the docking results
        output_folder : str
            Path to the folder where the docking structures will be saved.
        separator : str
            Symbol used to separate protein, ligand, and docking pose index.
        """

        # Check the separator is not in model or ligand names
        for model in self.docking_ligands:
            if separator in model:
                raise ValueError('The separator %s was found in model name %s. Please use a different separator symbol.' % (separator, model))
            for ligand in self.docking_ligands[model]:
                if separator in ligand:
                    raise ValueError('The separator %s was found in ligand name %s. Please use a different separator symbol.' % (separator, ligand))

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Copy analyse docking script (it depends on schrodinger so we leave it out.)
        _copyScriptFile(output_folder, 'extract_docking.py')
        script_path = output_folder+'/._extract_docking.py'

        # Move to output folder
        os.chdir(output_folder)

        # Save given docking data to csv
        dd = docking_data.reset_index()
        dd.to_csv('._docking_data.csv', index=False)

        # Execute docking analysis
        command = 'run ._extract_docking.py ._docking_data.csv ../'+docking_folder+' --separator '+separator
        os.system(command)

        # Remove docking data
        os.remove('._docking_data.csv')

        # move back to folder
        os.chdir('..')

        # Check models for covalent residues
        for protein in os.listdir(output_folder):
            if not os.path.isdir(output_folder+'/'+protein):
                continue
            for f in os.listdir(output_folder+'/'+protein):
                self._checkCovalentLigands(protein, output_folder+'/'+protein+'/'+f,
                                           check_file=True)

    def getSingleDockingData(self, protein, ligand, data_frame=None):
        """
        Get the docking data for a particular combination of protein and ligand

        Parameters
        ==========
        protein : str
            Protein model name
        ligad : str
            Ligand name
        data_frame : pandas.DataFrame
            Optional dataframe to get docking data from.
        """

        if ligand not in self.docking_ligands[protein]:
            raise ValueError('has no docking data')

        if isinstance(data_frame, type(None)):
            data_frame = self.docking_data

        protein_series = data_frame[data_frame.index.get_level_values('Protein') == protein]
        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]

        return ligand_series

    def plotDocking(self, protein, ligand, x='RMSD', y='Score', z=None, colormap='Blues_r', output_folder=None, extension='.png',
                    dpi=200):

        if output_folder != None:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

        protein_series = self.docking_data[self.docking_data.index.get_level_values('Protein') == protein]
        if protein_series.empty:
            print('Model %s not found in Docking data' % protein)
            return None
        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
        if ligand_series.empty:
            print('Ligand %s not found in Docking data for protein %s' % (ligand, protein))
            return None

        fig, ax = plt.subplots()
        if z != None:
            ligand_series.plot(kind='scatter', x=x, y=y, c=z, colormap=colormap, ax=ax)
        else:
            ligand_series.plot(kind='scatter', x=x, y=y, ax=ax)

        plt.title(protein+' + '+ligand)
        if output_folder != None:
            plt.savefig(output_folder+'/'+protein+'_'+ligand+extension, dpi=dpi)
            plt.close()

    def loadModelsFromPrepwizardFolder(self, prepwizard_folder, return_missing=False,
                                       return_failed=False, covalent_check=True,
                                       atom_mapping=None):
        """
        Read structures from a Schrodinger calculation.

        Parameters
        ==========
        prepwizard_folder : str
            Path to the output folder from a prepwizard calculation
        """

        models = []
        failed_models = []
        for d in os.listdir(prepwizard_folder+'/output_models'):
            if os.path.isdir(prepwizard_folder+'/output_models/'+d):
                for f in os.listdir(prepwizard_folder+'/output_models/'+d):
                    if f.endswith('.log'):
                        with open(prepwizard_folder+'/output_models/'+d+'/'+f) as lf:
                            for l in lf:
                                if 'error' in l.lower():
                                    print('Error was found in log file: %s. Please check the calculation!' % f)
                                    model = f.replace('.log', '')
                                    failed_models.append(model)
                                    break

                    if f.endswith('.pdb'):
                        model = f.replace('.pdb', '')
                        models.append(model)
                        self.readModelFromPDB(model, prepwizard_folder+'/output_models/'+d+'/'+f,
                                              covalent_check=covalent_check, atom_mapping=atom_mapping)

        self.getModelsSequences()

        missing_models = set(self.models_names) - set(models)
        if missing_models != set():
            print('Missing models in prepwizard folder:')
            print('\t'+', '.join(missing_models))

        if return_missing:
            return missing_models
        if return_failed:
            return failed_models

    def analyseRosettaCalculation(self, rosetta_folder, atom_pairs=None, energy_by_residue=False,
                                  interacting_residues=False, query_residues=None, overwrite=False,
                                  protonation_states=False, decompose_bb_hb_into_pair_energies=False):
        """
        Analyse Rosetta calculation folder. The analysis reads the energies and calculate distances
        between atom pairs given. Optionally the analysis get the energy of each residue in each pose.
        Additionally, it can analyse the interaction between specific residues (query_residues option)and
        their neighbouring sidechains by mutating the neighbour residues to glycines.

        The atom pairs must be given in a dicionary with each key representing the name
        of a model and each value a list of the atom pairs to calculate in the format:
            {model_name: [((chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name)), ...], ...}

        The main analysis is stored at self.rosetta_data
        The energy by residue analysis is soterd at self.rosetta_ebr_data
        Sidechain interaction analysis is stored at self.rosetta_interacting_residues

        Data is also stored in csv files inside the Rosetta folder for easy retrieving the data if found:

        The main analysis is stored at ._rosetta_data.csv
        The energy by residue analysis is soterd at ._rosetta_energy_residue_data.csv
        Sidechain interaction analysis is stored at ._rosetta_interacting_residues_data.csv


        The overwrite option forces recalcualtion of the data.

        Parameters
        ==========
        rosetta_folder : str
            Path to the Rosetta Calculation Folder.
        atom_pairs : dict
            Pairs of atom to calculate for each model.
        energy_by_residue : bool
            Calculate energy by residue data?
        overwrite : bool
            Force the data calculation from the files.
        interacting_residues : str
            Calculate interacting energies between residues
        query_residues : list
            Residues to query neoghbour atoms. Leave None for all residues (not recommended, too slow!)
        decompose_bb_hb_into_pair_energies : bool
            Store backbone hydrogen bonds in the energy graph on a per-residue basis (this doubles the
            number of calculations, so is off by default).
        """

        # Write atom_pairs dictionary to json file
        if atom_pairs != None:
            with open(rosetta_folder+'/._atom_pairs.json', 'w') as jf:
                json.dump(atom_pairs, jf)

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(rosetta_folder, 'analyse_calculation.py', subfolder='pyrosetta')

        # Execute docking analysis
        os.chdir(rosetta_folder)

        analyse = True
        # Check if analysis files exists
        if os.path.exists('._rosetta_data.csv') and not overwrite:
            self.rosetta_data = pd.read_csv('._rosetta_data.csv')
            self.rosetta_data.set_index('description', inplace=True)
            analyse = False
            atom_pairs = None

        if energy_by_residue and not overwrite:
            if os.path.exists('._rosetta_energy_residue_data.csv'):
                self.rosetta_ebr_data = pd.read_csv('._rosetta_energy_residue_data.csv')
                self.rosetta_ebr_data.set_index(['description', 'chain', 'residue'], inplace=True)
            else:
                analyse = True

        if interacting_residues and not overwrite:
            if os.path.exists('._rosetta_interacting_residues_data.csv'):
                self.rosetta_interacting_residues = pd.read_csv('._rosetta_interacting_residues_data.csv')
                self.rosetta_interacting_residues.set_index(['description', 'chain', 'residue', 'neighbour chain', 'neighbour residue'], inplace=True)
            else:
                analyse = True

        if protonation_states and not overwrite:
            if os.path.exists('._rosetta_protonation_data.csv'):
                self.rosetta_protonation_states = pd.read_csv('._rosetta_protonation_data.csv')
                self.rosetta_protonation_states.set_index(['description', 'chain', 'residue'], inplace=True)
            else:
                analyse = True

        if analyse:
            command = 'python ._analyse_calculation.py . '
            if atom_pairs != None:
                command += '--atom_pairs ._atom_pairs.json '
            if energy_by_residue:
                command += '--energy_by_residue '
            if interacting_residues:
                command += '--interacting_residues '
                if query_residues != None:
                    command += '--query_residues '
                    command += ','.join([str(r) for r in query_residues])+' '
            if protonation_states:
                command += '--protonation_states '
            if decompose_bb_hb_into_pair_energies:
                command += '--decompose_bb_hb_into_pair_energies'

            try:
                os.system(command)
            except:
                os.chdir('..')
                raise ValueError('Rosetta calculation analysis failed. Check the ouput of the analyse_calculation.py script.')

            # Read the CSV file into pandas
            if not os.path.exists('._rosetta_data.csv'):
                os.chdir('..')
                raise ValueError('Rosetta analysis failed. Check the ouput of the analyse_calculation.py script.')

            # Read the CSV file into pandas
            self.rosetta_data = pd.read_csv('._rosetta_data.csv')
            self.rosetta_data.set_index('description', inplace=True)

            if energy_by_residue:
                if not os.path.exists('._rosetta_energy_residue_data.csv'):
                    raise ValueError('Rosetta energy by reisdue analysis failed. Check the ouput of the analyse_calculation.py script.')
                self.rosetta_ebr_data = pd.read_csv('._rosetta_energy_residue_data.csv')
                self.rosetta_ebr_data.set_index(['description', 'chain', 'residue'], inplace=True)

            if interacting_residues:
                if not os.path.exists('._rosetta_energy_residue_data.csv'):
                    raise ValueError('Rosetta interacting reisdues analysis failed. Check the ouput of the analyse_calculation.py script.')
                self.rosetta_interacting_residues = pd.read_csv('._rosetta_interacting_residues_data.csv')
                self.rosetta_interacting_residues.set_index(['description', 'chain', 'residue', 'neighbour chain', 'neighbour residue'], inplace=True)

            if protonation_states:
                self.rosetta_protonation_states = pd.read_csv('._rosetta_protonation_data.csv')
                self.rosetta_protonation_states.set_index(['description', 'chain', 'residue'], inplace=True)

        os.chdir('..')

    def getRosettaModelDistances(self, model):
        """
        Get all distances related to a model from the self.rosetta_data DataFrame.

        Parameters
        ==========
        model : str
            Model name

        Return
        ======
        distances : list
            Distances containing non-nan values for the model.

        """

        mask = []
        for pose in self.rosetta_data.index:
            model_base_name = '_'.join(pose.split('_')[:-1])
            if model == model_base_name:
                mask.append(True)
            else:
                mask.append(False)
        model_data = self.rosetta_data[mask]

        distances = []
        for d in model_data:
            if d.startswith('distance_'):
                if not model_data[d].dropna().empty:
                    distances.append(d)

        return distances

    def combineRosettaDistancesIntoMetric(self, metric_labels, overwrite=False):
        """
        Combine different equivalent distances contained in the self.distance_data
        attribute into specific named metrics. The function takes as input a
        dictionary (metric_distances) composed of inner dictionaries as follows:

            metric_labels = {
                metric_name = {
                    model = distances_list}}}

        The innermost distances_list object contains all equivalent distance names for
        a specific protein to be combined under the same metric_name column.

        The combination is done by taking the minimum value of all equivalent distances.

        Parmeters
        =========
        metric_labels : dict
            Dictionary defining which distances will be combined under a common name.
            (for details see above).
        """

        for name in metric_labels:
            if 'metric_'+name in self.rosetta_data.keys() and not overwrite:
                print('Combined metric %s already added. Give overwrite=True to recombine' % name)
            else:
                values = []
                models = []
                for model in self.rosetta_data.index:
                    base_name = '_'.join(model.split('_')[:-1])
                    if base_name not in models:
                        models.append(base_name)

                for model in list(models):
                    mask = []
                    for index in self.rosetta_data.index:
                        if model in index:
                            mask.append(True)
                        else:
                            mask.append(False)
                    model_data = self.rosetta_data[mask]
                    model_distances = metric_labels[name][model]

                    values += model_data[model_distances].min(axis=1).tolist()

                self.rosetta_data['metric_'+name] = values

    def rosettaFilterByProtonationStates(self, residue_states=None, inplace=False):
        """
        Filter the rosetta_data attribute based on the fufillment of protonation state conditions. Protonations states
        defintions must be given through the residue_states attribute. The input is a dictionary with model names as keys
        and as values lists of tuples with the following format: [((chain_id, residue_id), protonation_state), etc.]

        The function is currently implemented to only work with histidine residues.

        Parameters
        ==========
        residue_states : dict
            By model and residue definition of protonation states.
        inplace : bool
            Overwrites the self.rosetta_data by the filtered data frame.

        Returns
        =======
        filtered_data : pandas.DataFrame
            self.rosetta_data dataframe filterd by protonation states.
        """

        data = self.rosetta_protonation_states.reset_index()
        data.columns = [c.replace(' ', '_') for c in data.columns]

        filtered_models = []
        filtered_rows = []

        old_model = None
        histidines = []
        for index, row in data.iterrows():
            ti = time.time()
            model_tag = row.description

            # Empty hisitidine list
            if model_tag != old_model:

                # Check protonation states are in data
                keep_model = True
                if histidines != []:
                    model_base_name = '_'.join(model_tag.split('_')[:-1])
                    for rs in residue_states[model_base_name]:
                        if rs not in histidines:
                            keep_model = False

                # Store model
                if keep_model and histidines != []:
                    filtered_models.append(model_tag)

                histidines = []

            histidines.append(((row.chain, row.residue), (row.residue_state)))

            # Update current model as old
            old_model = model_tag

        # filter the rosetta_data attribute
        mask = []
        rosetta_data = self.rosetta_data.reset_index()
        for index, row in rosetta_data.iterrows():
            if row.description in filtered_models:
                mask.append(True)
            else:
                mask.append(False)

        filtered_data = self.rosetta_data[mask]
        if inplace:
            self.rosetta_data = filtered_data

        return filtered_data

    def loadMutantsAsNewModels(self, mutants_folder, filter_score_term='score', tags=None,
                               min_value=True, wat_to_hoh=True, keep_model_name=True):
        """
        Load the best energy models from a set of silent files inside a createMutants()
        calculation folder. The models are added to the list of models and do not replace
        any previous model already present in the library.

        Parameters
        ==========
        mutants_folder : str
            Path to folder where the Mutants output files are contained (see createMutants() function)
        filter_score_term : str
            Score term used to filter models
        tags : dict
            Tags to extract specific models from the mutant optimization
        """

        executable = 'extract_pdbs.linuxgccrelease'
        models = []

        # Check if params were given
        params = None
        if os.path.exists(mutants_folder+'/params'):
            params = mutants_folder+'/params'

        for d in os.listdir(mutants_folder+'/output_models'):
            if os.path.isdir(mutants_folder+'/output_models/'+d):
                for f in os.listdir(mutants_folder+'/output_models/'+d):
                    if f.endswith('.out'):
                        model = d
                        mutant = f.replace(model+'_', '').replace('.out', '')
                        scores = readSilentScores(mutants_folder+'/output_models/'+d+'/'+f)
                        if tags != None and mutant in tags:
                            print('Reading mutant model %s from the given tag %s' % (mutant, tags[mutant]))
                            best_model_tag = tags[mutant]
                        elif min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += ' -silent '+mutants_folder+'/output_models/'+d+'/'+f
                        if params != None:
                            command += ' -extra_res_path '+params+' '
                        command += ' -tags '+best_model_tag
                        os.system(command)

                        # Load mutants to the class
                        if keep_model_name:
                            mutant = model+'_'+mutant

                        self.models_names.append(mutant)
                        self.readModelFromPDB(mutant, best_model_tag+'.pdb', wat_to_hoh=wat_to_hoh)
                        os.remove(best_model_tag+'.pdb')
                        models.append(mutant)

        self.getModelsSequences()
        print('Added the following mutants from folder %s:' % mutants_folder)
        print('\t'+', '.join(models))

    def loadModelsFromRosettaOptimization(self, optimization_folder, filter_score_term='score',
                                          min_value=True, tags=None, wat_to_hoh=True,
                                          return_missing=False):
        """
        Load the best energy models from a set of silent files inside a specfic folder.
        Useful to get the best models from a relaxation run.

        Parameters
        ==========
        optimization_folder : str
            Path to folder where the Rosetta optimization files are contained
        filter_score_term : str
            Score term used to filter models
        relax_run : bool
            Is this a relax run?
        min_value : bool
            Grab the minimum score value. Set false to grab the maximum scored value.
        """

        executable = 'extract_pdbs.linuxgccrelease'
        models = []

        # Check if params were given
        params = None
        if os.path.exists(optimization_folder+'/params'):
            params = optimization_folder+'/params'

        for d in os.listdir(optimization_folder+'/output_models'):
            if os.path.isdir(optimization_folder+'/output_models/'+d):
                for f in os.listdir(optimization_folder+'/output_models/'+d):
                    if f.endswith('_relax.out'):
                        model = d
                        scores = readSilentScores(optimization_folder+'/output_models/'+d+'/'+f)
                        if tags != None and model in tags:
                            print('Reading model %s from the given tag %s' % (model, tags[model]))
                            best_model_tag = tags[model]
                        elif min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += ' -silent '+optimization_folder+'/output_models/'+d+'/'+f
                        if params != None:
                            command += ' -extra_res_path '+params+' '
                        command += ' -tags '+best_model_tag
                        os.system(command)
                        self.readModelFromPDB(model, best_model_tag+'.pdb', wat_to_hoh=wat_to_hoh)
                        os.remove(best_model_tag+'.pdb')
                        models.append(model)

        self.getModelsSequences()
        missing_models = set(self.models_names) - set(models)
        if missing_models != set():
            print('Missing models in relaxation folder:')
            print('\t'+', '.join(missing_models))
            if return_missing:
                return missing_models

    def loadModelsFromMissingLoopBuilding(self, job_folder, filter_score_term='score', min_value=True, param_files=None):
        """
        Load models from addMissingLoops() job calculation output.

        Parameters:
        job_folder : str
            Path to the addMissingLoops() calculation folder containing output.
        """

        # Get silent models paths
        executable = 'extract_pdbs.linuxgccrelease'
        output_folder = job_folder+'/output_models'
        models = []

        # Check if params were given
        params = None
        if os.path.exists(job_folder+'/params'):
            params = job_folder+'/params'

        # Check loops to rebuild from output folder structure
        for model in os.listdir(output_folder):
            model_folder = output_folder+'/'+model
            loop_models = {}
            for loop in os.listdir(model_folder):
                loop_folder = model_folder+'/'+loop
                for f in os.listdir(loop_folder):
                    # If rebuilded loops are found get best structures.
                    if f.endswith('.out'):
                        scores = readSilentScores(loop_folder+'/'+f)
                        best_model_tag = scores.idxmin()[filter_score_term]
                        if min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += ' -silent '+loop_folder+'/'+f
                        if params != None:
                            command += ' -extra_res_path '+params+' '
                        command += ' -tags '+best_model_tag
                        os.system(command)
                        loop = (int(loop.split('_')[0]), loop.split('_')[1])
                        loop_models[loop] = _readPDB(loop, best_model_tag+'.pdb')
                        os.remove(best_model_tag+'.pdb')
                        models.append(model)

            if len(loop_models) > 1:
                # Get original model chains
                model_chains = [*self.structures[model].get_chains()]

                # Create new structure, model and chains to add rebuilded segments
                structure = PDB.Structure.Structure(0)
                _model = PDB.Model.Model(0)
                chains = {}
                for model_chain in model_chains:
                    chains[model_chain.id] = PDB.Chain.Chain(model_chain.id)

                # Add missing loop segments to overall model
                current_residue = 0

                for loop in loop_models:
                    # Add loop remodel protocol
                    if len(loop[1]) == 1:
                        hanging_residues = 3
                    elif len(loop[1]) == 2:
                        hanging_residues = 2
                    else:
                        hanging_residues = 1
                    larger_loop_residue = loop[0]+len(loop[1])+1+hanging_residues
                    for i,residue in enumerate(loop_models[loop].get_residues()):
                        if i+1 > current_residue and i+1 <= larger_loop_residue:
                            chain_id = residue.get_parent().id
                            chains[chain_id].add(residue)
                            current_residue += 1

                # Load final model into the library
                for chain in chains:
                    _model.add(chains[chain])
                structure.add(_model)
                _saveStructureToPDB(structure, model+'.pdb')
            else:
                for loop in loop_models:
                    _saveStructureToPDB(loop_models[loop], model+'.pdb')

            self.readModelFromPDB(model, model+'.pdb')
            os.remove(model+'.pdb')

        missing_models = set(self.models_names) - set(models)
        if missing_models != set():
            print('Missing models in loop rebuild folder:')
            print('\t'+', '.join(missing_models))

    def loadModelsFromMembranePositioning(self, job_folder):
        """
        """
        for model in os.listdir(job_folder+'/output_models'):
            pdb_path = job_folder+'/output_models/'+model+'/'+model+'.pdb'
            self.readModelFromPDB(model, pdb_path)

    def saveModels(self, output_folder, keep_residues={}, models=None, convert_to_mae=False,
                   **keywords):
        """
        Save all models as PDBs into the output_folder.

        Parameters
        ==========
        output_folder : str
            Path to the output folder to store models.
        """
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if convert_to_mae:
            _copyScriptFile(output_folder, 'PDBtoMAE.py')
            script_name = '._PDBtoMAE.py'

        for model in self.models_names:

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            # Get residues to keep
            if model in keep_residues:
                kr = keep_residues[model]
            else:
                kr = []

            _saveStructureToPDB(self.structures[model],
                                output_folder+'/'+model+'.pdb',
                                keep_residues=kr,
                                **keywords)

            self._write_conect_lines(model, output_folder+'/'+model+'.pdb')

            if convert_to_mae:
                cwd = os.getcwd()
                os.chdir(output_folder)
                command = 'run ._PDBtoMAE.py'
                os.system(command)
                os.chdir(cwd)
                os.remove(output_folder+'/'+model+'.pdb')

    def removeModel(self, model):
        """
        Removes a specific model from this class

        Parameters
        ==========
        model : str
            Model name to remove
        """
        try:
            self.models_paths.pop(model)
            self.models_names.remove(model)
            self.structures.pop(model)
            self.sequences.pop(model)
        except:
            raise ValueError('Model  %s is not present' % model)

    def readTargetSequences(self, fasta_file):
        """
        Read the set of target sequences for the protein models
        """
        # Read sequences and store them in target_sequence attributes
        sequences = prepare_proteins.alignment.readFastaFile(fasta_file)
        for sequence in sequences:
            if sequence not in self.models_names:
                print('Given sequence name %s does not matches any protein model' % sequence)
            else:
                self.target_sequences[sequence] = sequences[sequence]

        missing_models = set(self.models_names) - set(self.target_sequences)
        if missing_models != set():
            print('Missing sequences in the given fasta file:')
            print('\t'+', '.join(missing_models))

    def compareSequences(self, sequences_file):
        """
        Compare models sequences to their given sequences and check for missing
        or changed sequence information.

        Parameters
        ==========
        sequences_file : str
            Path to the fasta file containing the sequences to compare. The model
            names must match.

        Returns
        =======
        sequence_differences : dict
            Dictionary containing missing or changed information.
        """

        if self.multi_chain:
            raise ValueError('PDBs contain multiple chains. Please select one chain.')
        self.readTargetSequences(sequences_file)

        # Iterate models to store sequence differences
        for model in self.models_names:

            if model not in self.target_sequences:
                message = 'Sequence for model %s not found in the given fasta file! ' % model
                message += 'Please make sure to include one sequence for each model '
                message += 'loaded into prepare proteins.'
                raise ValueError(message)

            # Create lists for missing information
            self.sequence_differences[model] = {}
            self.sequence_differences[model]['n_terminus'] = []
            self.sequence_differences[model]['mutations'] = []
            self.sequence_differences[model]['missing_loops'] = []
            self.sequence_differences[model]['c_terminus'] = []

            # Create a sequence alignement between current and target sequence
            to_align = {}
            to_align['current'] = self.sequences[model]
            to_align['target'] = self.target_sequences[model]
            msa = prepare_proteins.alignment.mafft.multipleSequenceAlignment(to_align)

            # Iterate the alignment to gather sequence differences
            p = 0
            n = True
            loop_sequence = ''
            loop_start = 0

            # Check for n-terminus, mutations and missing loops
            for i in range(msa.get_alignment_length()):
                csp = msa[0].seq[i]
                tsp = msa[1].seq[i]
                if csp != '-':
                    p += 1
                if csp == '-' and tsp != '-' and n:
                    self.sequence_differences[model]['n_terminus'].append(tsp)
                elif csp != '-' and tsp != '-':
                    n = False
                    if loop_sequence != '' and len(loop_sequence) > 1: # Ignore single-residue loops
                        self.sequence_differences[model]['missing_loops'].append((loop_start, loop_sequence))
                    loop_sequence = ''
                    loop_start = 0

                    if csp != tsp:
                        self.sequence_differences[model]['mutations'].append((p, tsp))

                elif csp == '-' and  tsp != '-' and p < len(to_align['current']):
                    if loop_start == 0:
                        loop_start = p
                    loop_sequence += tsp

            # Check for c-terminus
            for i in reversed(range(msa.get_alignment_length())):
                csp = msa[0].seq[i]
                tsp = msa[1].seq[i]
                if csp == '-' and tsp != '-':
                    self.sequence_differences[model]['c_terminus'].append(tsp)
                elif csp != '-' and tsp != '-':
                    break

            self.sequence_differences[model]['n_terminus'] = ''.join(self.sequence_differences[model]['n_terminus'])
            self.sequence_differences[model]['c_terminus'] = ''.join(reversed(self.sequence_differences[model]['c_terminus']))

        return self.sequence_differences

    def _write_conect_lines(self, model, pdb_file, atom_mapping=None):
        """
        Write stored conect lines for a particular model into the given PDB file.

        Parameters
        ==========
        model : str
            Model name
        pdb_file : str
            Path to PDB file to modify
        """

        def check_atom_in_atoms(atom, atoms):

            if atom not in atoms and atom_mapping != None and atom in atom_mapping:
                if isinstance(atom_mapping[atom], str):
                    atom = (atom[0], atom[1], atom_mapping[atom])
                elif isinstance(atom_mapping[atom], tuple) and len(atom_mapping[atom]) == 3:
                    atom = atom_mapping[atom]

            if atom not in atoms:
                residue_atoms = ' '.join([ac[-1] for ac in atoms if atom[1] == ac[1]])
                message = "Conect atom %s not found in %s's topology\n\n" % (atom, pdb_file)
                message += "Topology's residue %s atom names: %s" % (atom[1], residue_atoms)
                raise ValueError(message)

            return atom

        # Get atom indexes map
        atoms = self._getAtomIndexes(model, pdb_file, invert=True)

        # Check atoms not found in conects
        with open(pdb_file+'.tmp', 'w') as tmp:
            with open(pdb_file) as pdb:

                # write all lines but skip END line
                for line in pdb:
                    if not line.startswith('END'):
                        tmp.write(line)

                # Write new conect line mapping
                for entry in self.conects[model]:
                    line = 'CONECT'
                    for x in entry:
                        x = check_atom_in_atoms(x, atoms)
                        line += ' '+str(atoms[x])
                    line += '\n'
                    tmp.write(line)
            tmp.write('END\n')
        shutil.move(pdb_file+'.tmp', pdb_file)

    def _getChainSequence(self, chain):
        """
        Get the one-letter protein sequence of a Bio.PDB.Chain object.

        Parameters
        ----------
        chain : Bio.PDB.Chain
            Input chain to retrieve its sequence from.

        Returns
        -------
        sequence : str
            Sequence of the input protein chain.
        None
            If chain does not contain protein residues.
        """
        sequence = ''
        for r in chain:
            if r.id[0] == ' ': # Non heteroatom filter
                try:
                    sequence += PDB.Polypeptide.three_to_one(r.resname)
                except:
                    sequence += 'X'

        if sequence == '':
            return None
        else:
            return sequence

    def _checkCovalentLigands(self, model, pdb_file, atom_mapping=None, check_file=False):
        """
        """
        self.covalent[model] = [] # Store covalent residues
        if check_file:
            structure = _readPDB(model, pdb_file)
        else:
            structure = self.structures[model]

        # Iterate chains in model structure
        for c in structure[0]:

            indexes = [] # Store residue indexes
            hetero = [] # Store heteroatom residue indexes
            residues = [] # Store residues orderly (for later)
            for r in c:
                indexes.append(r.id[1])
                if r.id[0].startswith('H_'):
                    hetero.append(r.id[1])
                residues.append(r)

            # Check for individual and other gaps
            gaps2 = []  # Store individual gaps
            other_gaps = [] # Store other gaps
            for i in range(len(indexes)):
                if i > 0:
                    if indexes[i]-indexes[i-1] == 2:
                        gaps2.append((indexes[i-1], indexes[i]))
                    elif indexes[i]-indexes[i-1] != 1:
                        other_gaps.append(indexes[i])

            # Check if individual gaps can be filled with any residue in other_gaps
            for g2 in gaps2:
                for og in other_gaps:
                    if g2[1]-og == 1 and og-g2[0] == 1:
                        print('Found misplaced residue %s for model %s' % (og, model))
                        print('Possibly a covalent-link exists for this HETATM residue')
                        print('Sorting residues by their indexes... to disable pass covalent_check=False.')
                        self._sortStructureResidues(model, pdb_file, check_file=check_file,
                                                    atom_mapping=atom_mapping)
                        self.covalent[model].append(og)

            # Check if hetero-residue is found between two non-hetero residues
            for i,r in enumerate(residues):
                if r.id[1] in hetero:
                    if i+1  == len(residues):
                        continue
                    chain = r.get_parent()
                    pr = residues[i-1]
                    nr = residues[i+1]
                    if pr.get_parent().id == chain.id and nr.get_parent().id == chain.id:
                        if pr.id[0] == ' ' and nr.id[0] == ' ':
                            self.covalent[model].append(r.id[1])

    def _sortStructureResidues(self, model, pdb_file, atom_mapping=None, check_file=False):

        # Create new structure
        n_structure = PDB.Structure.Structure(0)

        # Create new model
        n_model = PDB.Model.Model(self.structures[model][0].id)

        if check_file:
            structure = _readPDB(model, pdb_file)
        else:
            structure = self.structures[model]

        # Iterate chains from old model
        for chain in structure[0]:
            n_chain = PDB.Chain.Chain(chain.id)

            # Gather residues
            residues = []
            for r in chain:
                residues.append(r)

            # Iterate residues orderly by their ID
            for r in sorted(residues, key=lambda x:x.id[1]):
                n_chain.add(r)

            n_model.add(n_chain)
        n_structure.add(n_model)

        _saveStructureToPDB(n_structure, pdb_file+'.tmp')
        self._write_conect_lines(model, pdb_file+'.tmp', atom_mapping=atom_mapping)
        shutil.move(pdb_file+'.tmp', pdb_file)
        n_structure = _readPDB(model, pdb_file)

        # Update structure model in library
        if not check_file:
            self.structures[model] = n_structure

    def _readPDBConectLines(self, pdb_file, model):
        """
        Read PDB file and get conect lines only
        """

        atoms = self._getAtomIndexes(model, pdb_file)
        conects = []
        # Read conect lines as dictionaries linking atoms
        with open(pdb_file) as pdbf:
            for l in pdbf:
                if l.startswith('CONECT'):
                    conects.append([atoms[int(x)] for x in l.split()[1:]])
        return conects

    def _getAtomIndexes(self, model, pdb_file, invert=False):

        # i = 0
        # old_r = None
        # new_r = None

        # Read PDB file
        atom_indexes = {}
        with open(pdb_file, 'r') as f:
            for l in f:
                if l.startswith('ATOM') or l.startswith('HETATM'):
                    ls = l.split()
                    index, name, chain, resid = (int(ls[1]), ls[2], ls[4], int(ls[5]))
                    atom_indexes[(chain, resid, name)] = index

        # Assign PDB indexes to each Bio.PDB atom
        atoms = {}
        for chain in self.structures[model][0]:
            for residue in chain:
                for atom in residue:
                    index = atom_indexes[(chain.id, residue.id[1], atom.name)]
                    # index = i
                    if invert:
                        atoms[_get_atom_tuple(atom)] = index
                    else:
                        atoms[index] = _get_atom_tuple(atom)
                    # i += 1
        return atoms

    def _getModelsPaths(self):
        """
        Get PDB models paths in the models_folder attribute

        Returns
        =======

        paths : dict
            Paths to all models
        """
        paths = {}
        for d in os.listdir(self.models_folder):
            if d.endswith('.pdb'):
                pdb_name = '.'.join(d.split('.')[:-1])
                paths[pdb_name] = self.models_folder+'/'+d

        return paths

    def __iter__(self):
        #returning __iter__ object
        self._iter_n = -1
        self._stop_inter = len(self.models_names)
        return self

    def __next__(self):
        self._iter_n += 1
        if self._iter_n < self._stop_inter:
            return self.models_names[self._iter_n]
        else:
            raise StopIteration

def readSilentScores(silent_file):
    """
    Read scores from a silent file into a Pandas DataFrame object.

    Parameters
    ==========
    silent_file : str
        Path to the silent file.

    Returns
    =======
    scores : Pandas.DataFrame
        Rosetta score for each model.
    """

    scores = {}
    terms = []
    with open(silent_file) as sf:
        for l in sf:
            if l.startswith('SCORE'):
                if terms == []:
                    terms = l.strip().split()
                    for t in terms:
                        scores[t] = []
                else:
                    for i,t in enumerate(terms):
                        try:
                            scores[t].append(float(l.strip().split()[i]))
                        except:
                            scores[t].append(l.strip().split()[i])
    scores = pd.DataFrame(scores)
    scores.pop('SCORE:')
    scores = pd.DataFrame(scores)
    scores = scores.set_index('description')
    scores = scores.sort_index()

    return scores

def _readPDB(name, pdb_file):
    """
    Read PDB file to a structure object
    """
    parser = PDB.PDBParser()
    structure = parser.get_structure(name, pdb_file)
    return structure

def _saveStructureToPDB(structure, output_file, remove_hydrogens=False,
                        remove_water=False, only_protein=False, keep_residues=[]):
    """
    Saves a structure into a PDB file

    Parameters
    ----------
    structure : list or Bio.PDB.Structure
        Structure to save
    remove_hydrogens : bool
        Remove hydrogen atoms from model?
    remove_water : bool
        Remove water residues from model?
    only_protein : bool
        Remove everything but the protein atoms?
    keep_residues : list
        List of residue indexes to keep when using the only_protein selector.
    """

    io = PDB.PDBIO()
    io.set_structure(structure)

    selector = None
    if remove_hydrogens:
        selector = _atom_selectors.notHydrogen()
    elif remove_water:
        selector = _atom_selectors.notWater()
    elif only_protein:
        selector = _atom_selectors.onlyProtein(keep_residues=keep_residues)
    if selector != None:
        io.save(output_file, selector)
    else:
        io.save(output_file)

def _copyScriptFile(output_folder, script_name, no_py=False, subfolder=None, hidden=True):
    """
    Copy a script file from the prepare_proteins package.

    Parameters
    ==========

    """
    # Get script
    path = "prepare_proteins/scripts"
    if subfolder != None:
        path = path+'/'+subfolder

    script_file = resource_stream(Requirement.parse("prepare_proteins"),
                                     path+'/'+script_name)
    script_file = io.TextIOWrapper(script_file)

    # Write control script to output folder
    if no_py == True:
        script_name = script_name[:-3]

    if hidden:
        output_path = output_folder+'/._'+script_name
    else:
        output_path = output_folder+'/'+script_name

    with open(output_path, 'w') as sof:
        for l in script_file:
            sof.write(l)

def _get_atom_tuple(atom):
    return (atom.get_parent().get_parent().id,
            atom.get_parent().id[1],
            atom.name)
