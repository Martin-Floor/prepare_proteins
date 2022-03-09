from . import alignment
from . import _atom_selectors
from . import rosettaScripts
from . import MD

import os
import shutil
import uuid
import itertools
import io
import subprocess
import json
from pkg_resources import resource_stream, Requirement

import numpy as np
from Bio import PDB
from Bio.PDB.DSSP import DSSP
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md

import prepare_proteins
import bsc_calculations

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

    def __init__(self, models_folder, get_sequences=True, get_ss=True, msa=False):
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
            Calculate a multiple sequence alignment for the models. Only works for
            single-chain structures at startup, othewise look for the calculateMSA()
            method.
        """

        self.models_folder = models_folder
        self.models_paths = self._getModelsPaths()
        self.models_names = [] # Store model names
        self.structures = {} # structures are stored here
        self.sequences = {} # sequences are stored here
        self.msa = None # multiple sequence alignment
        self.multi_chain = False
        self.ss = {} # secondary structure strings are stored here
        self.docking_data = None # secondary structure strings are stored here
        self.docking_ligands = {}
        self.sequence_differences = {} # Store missing/changed sequence information
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
                          elements=None, hetatom=True, water=False):
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
                # print(atom_names[i], coordinates[i], 0, 1.0, '  ', '%-4s' % atom_names[i], serial_number+i, elements[i)
                atom = PDB.Atom.Atom(atom_names[i], coordinates[i], 0, 1.0, ' ',
                                     '%-4s' % atom_names[i], serial_number+i, elements[i])
            else:
                atom = PDB.Atom.Atom(atom_names[i], coordinates[i], 0, 1.0, ' ',
                                     '%-4s' % atom_names[i], serial_number+i)
            residue.add(atom)
        chain[0].add(residue)

        return new_resid

    def readModelFromPDB(self, model, pdb_file, wat_to_hoh=False):
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

    def getResidueIndexFromMSAindex(self, msa_index):
        """
        Get the individual model residue indexes of a specific MSA positions

        Paramters
        =========
        msa_index : int
            Zero-based MSA index

        Returns
        =======
        residue_indexes : dict
            Residue indexes for each protein at the MSA position
        """

        residue_indexes = {}
        for model in self.models_names:
            residue_indexes[model] = 0

        for i in range(self.msa.get_alignment_length()):
            if i == msa_index:
                break
            for entry in self.msa:
                if entry.seq[i] != '-':
                    residue_indexes[entry.id] += 1
        return residue_indexes

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

    def alignModelsToReferencePDB(self, reference, output_folder, chain_indexes=None,
                                  trajectory_chain_indexes=None, reference_chain_indexes=None):
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
        """

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        reference = md.load(reference)
        for model in self.models_names:
            traj = md.load(self.models_paths[model])
            MD.alignTrajectoryBySequenceAlignment(traj, reference, chain_indexes=chain_indexes,
                                                  trajectory_chain_indexes=trajectory_chain_indexes)
            traj.save(output_folder+'/'+model+'.pdb')

    def setUpRosettaOptimization(self, relax_folder, nstruct=1000, relax_cycles=5,
                                 cst_files=None, mutations=False, models=None,
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
                raise ValuError('Mutations have been enabled but no sequence comparison\
has been carried out. Please run compareSequences() function before setting mutation=True.')

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Skip models not in the given list
            if models != None:
                if model not in models:
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
                    raise ValuError('Model %s is not in the cst_files dictionary!' % model)
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
            flags.add_relax_cst_options()

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
                # flags.addOption('relax:constrain_relax_to_start_coords', '')
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

    def addMissingLoops(self, job_folder, nstruct=1, sfxn='ref2015'):
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
            raise ValuError('No sequence comparison has been carried out. Please run \
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
                    loop_movers = rosettaScripts.loop_modeling.loopRebuild(xml, loop[0], loop[1],
                                                                           scorefxn=sfxn, hanging_residues=hanging_residues)
                    for m in loop_movers:
                        protocol.append(m)

                    # Add idealize step
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

                    # Write flags file
                    flags.write_flags(job_folder+'/flags/'+model+'_'+loop_name+'.flags')

                    # Create and append execution command
                    command = 'cd '+job_folder+'\n'
                    command += 'srun rosetta_scripts.mpi.linuxgccrelease @ '+'flags/'+model+'_'+loop_name+'.flags\n'
                    command += 'cd ..\n'

                    jobs.append(command)

        return jobs

    def setUpPrepwizardOptimization(self, prepare_folder, pH=7.0, epik_pH=False, samplewater=False,
                                    epik_pHt=False, remove_hydrogens=True, delwater_hbond_cutoff=False):
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
        self.saveModels(prepare_folder+'/input_models', remove_hydrogens=remove_hydrogens)

        # Copy control file to prepare folder
        _copySchrodingerControlFile(prepare_folder)

        # Generate jobs
        jobs = []
        for model in self.models_names:
            # Create model output folder
            output_folder = prepare_folder+'/output_models/'+model
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            command = 'cd '+output_folder+'\n'
            command += '"${SCHRODINGER}/utilities/prepwizard" '
            command += '../../input_models/'+model+'.pdb '
            command += model+'.pdb '
            command += '-fillsidechains '
            command += '-disulfides '
            if remove_hydrogens:
                command += '-rehtreat '
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
            command += '-JOBNAME prepare_'+model+' '
            command += '-HOST localhost:1\n'

            # Add control script command
            command += 'python3 ../../._schrodinger_control.py '
            command += model+'.log '
            command += '--job_type prepwizard\n'
            command += 'cd ../../..\n'
            jobs.append(command)

        return jobs

    def setUpDockingGrid(self, grid_folder, center_atoms, innerbox=(10,10,10),
                         outerbox=(30,30,30), useflexmae=True, peptide=False):
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
        self.saveModels(grid_folder+'/input_models')

        # Check that inner and outerbox values are given as integers
        for v in innerbox:
            if type(v) != int:
                raise ValuError('Innerbox values must be given as integers')
        for v in outerbox:
            if type(v) != int:
                raise ValuError('Outerbox values must be given as integers')

        # Copy control file to grid folder
        _copySchrodingerControlFile(grid_folder)

        # Create grid input files
        jobs = []
        for model in self.models_names:

            # Get coordinates of center residue
            chainid = center_atoms[model][0]
            resid = center_atoms[model][1]
            atom_name = center_atoms[model][2]

            for c in self.structures[model].get_chains():
                if c.id == chainid:
                    for r in c.get_residues():
                        if r.id[1] == resid:
                            for a in r.get_atoms():
                                if a.name == atom_name:
                                    x = a.coord[0]
                                    y = a.coord[1]
                                    z = a.coord[2]

            # Write grid input file
            with open(grid_folder+'/grid_inputs/'+model+'.in', 'w') as gif:
                gif.write('GRID_CENTER %.14f, %.14f, %.14f\n' % (x,y,z))
                gif.write('GRIDFILE '+model+'.zip\n')
                gif.write('INNERBOX %s, %s, %s\n' % innerbox)
                gif.write('OUTERBOX %s, %s, %s\n' % outerbox)
                gif.write('RECEP_FILE %s\n' % (model+'.mae'))
                if peptide:
                    gif.write('PEPTIDE True\n')
                if useflexmae:
                    gif.write('USEFLEXMAE YES\n')

            command = 'cd '+grid_folder+'/output_models\n'

            # Add convert PDB into mae format command
            command += '"$SCHRODINGER/utilities/structconvert" '
            command += '-ipdb ../input_models/'+model+'.pdb'+' '
            command += '-omae '+model+'.mae\n'

            # Add grid generation command
            command += '"${SCHRODINGER}/glide" '
            command += '../grid_inputs/'+model+'.in'+' '
            command += '-OVERWRITE '
            command += '-HOST localhost '
            command += '-TMPLAUNCHDIR\n'

            # Add control script command
            command += 'python3 ../._schrodinger_control.py '
            command += model+'.log '
            command += '--job_type grid\n'
            command += 'cd ../..\n'
            jobs.append(command)

        return jobs

    def setUpGlideDocking(self, docking_folder, grids_folder, ligands_folder,
                          poses_per_lig=100, precision='SP', use_ligand_charges=False,
                          energy_by_residue=False):
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

        # Copy control file to grid folder
        _copySchrodingerControlFile(docking_folder)

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
                command += '-TMPLAUNCHDIR\n'

                # Add control script command
                command += 'python3 ../../._schrodinger_control.py '
                command += grid+'_'+substrate+'.log '
                command += '--job_type docking\n'
                command += 'cd ../../..\n'
                jobs.append(command)

        return jobs

    def setUpSiteMapCalculation(self, job_folder, poses_folder, site_box=10, resolution='fine', overwrite=False):
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

        # Copy control file to grid folder
        _copySchrodingerControlFile(job_folder)
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
                        smi.write('PROTEIN ../../input_models/'+model+'/'+pose_name+'_ligand.mae\n')
                        smi.write('LIGMAE ../../input_models/'+model+'/'+pose_name+'_protein.mae\n')
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
                    command += '-TMPLAUNCHDIR\n'

                    # Add control script command
                    command += 'python3 ../../._schrodinger_control.py '
                    command += pose_name+'.log ' # Check
                    command += '--job_type sitemap\n'
                    command += 'cd ../../..\n'
                    jobs.append(command)
        return jobs

    def setUpCofactorParameterization(self, job_folder, cofactor_pdbs, include_terminal_rotamers=True,
                                      rotamer_resolution=10, charge_method='opls2005', forcefield='opls2005'):
        """
        Run PELE platform for ligand parameterization

        Parameters
        ==========
        job_folder : str
            Path to the job input folder
        cofactor_pdbs : list
            PDBs for the cofactors to parameterize
        """

        charge_methods = ['opls2005', 'am1bcc', 'gasteiger', 'dummy']
        if charge_method not in charge_methods:
            raise ValuError('Wrong charge method. Available: '+', '.join(charge_methods))

        # Create PELE job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if isinstance(cofactor_pdbs, str):
            cofactor_pdbs = [cofactor_pdbs]

        jobs = []
        for pdb in cofactor_pdbs:
            pdb_name = pdb.split('/')[-1].replace('.pdb', '')
            structure = _readPDB(pdb_name, pdb)

            if not os.path.exists(job_folder+'/'+pdb_name):
                os.mkdir(job_folder+'/'+pdb_name)

            # for residue in structure.get_residues():
            #     chain = residue.get_parent()
            #     chain.id  = 'L'
            #     residue.id = ('_H', 1, residue.id[2])

            # _saveStructureToPDB(structure, job_folder+'/'+pdb_name+'/'+pdb_name+'.pdb')
            shutil.copyfile(pdb, job_folder+'/'+pdb_name+'/'+pdb_name+'.pdb')

            # Create command
            command = 'cd '+job_folder+'/'+pdb_name+'\n'
            command += 'python -m peleffy.main '+pdb_name+'.pdb '
            # command += '--chain L '
            command += '-o "'+pdb_name+'_params" '
            command += '-r '+str(rotamer_resolution)+' '
            command += '-f '+str(forcefield)+' '
            if include_terminal_rotamers:
                command += '--include_terminal_rotamers '
            command += '-c '+charge_method+'\n'
            command += 'cd ../..'
            jobs.append(command)

        return jobs

    def setUpPELECalculation(self, pele_folder, models_folder, input_yaml, distances=None,
                             steps=100, debug=False, iterations=3, cpus=96, equilibration_steps=100,
                             separator='-', use_peleffy=True, usesrun=True, energy_by_residue=False,
                             analysis=False, energy_by_residue_type='all', peptide=False):
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

        Missing!
        """

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

                    ## Add dummy atom if peptide docking ### Strange fix!
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

                # Create YAML file
                for model in models:
                    protein, ligand = model
                    keywords = ['system', 'chain', 'resname', 'steps', 'iterations',
                                'cpus', 'equilibration', 'equilibration_steps', 'traj',
                                'usesrun', 'use_peleffy', 'debug']

                    with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input.yaml', 'w') as iyf:
                        if energy_by_residue:
                            # Use new PELE version with implemented energy_by_residue
                            iyf.write('pele_exec: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/bin/PELE-1.7.2_mpi"\n')
                            iyf.write('pele_data: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/Data"\n')
                            iyf.write('pele_documents: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/Documents/"\n')
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
                        iyf.write("equilibration: true\n")
                        iyf.write("equilibration_steps: "+str(equilibration_steps)+"\n")
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
                        # energy by residue is not implemented in PELE platform, therefore
                        # a scond script will modify the PELE.conf file to set up the energy
                        # by residue calculation.
                        if debug or energy_by_residue:
                            iyf.write("debug: true\n")

                        if distances != None:
                            iyf.write("atom_dist:\n")
                            for d in distances[protein][ligand]:
                                if isinstance(d[0], str):
                                    d1 = "- 'L:1:"+d[0]+"'\n"
                                else:
                                    d1 = "- '"+d[0][0]+":"+str(d[0][1])+":"+d[0][2]+"'\n"
                                if isinstance(d[1], str):
                                    d2 = "- 'L:1:"+d[1]+"'\n"
                                else:
                                    d2 = "- '"+d[1][0]+":"+str(d[1][1])+":"+d[1][2]+"'\n"
                                iyf.write(d1)
                                iyf.write(d2)

                        iyf.write('\n')
                        iyf.write("#Options gathered from "+input_yaml+'\n')

                        with open(input_yaml) as tyf:
                            for l in tyf:
                                if not l.startswith('#'):
                                    if l.split()[0].replace(':', '') not in keywords:
                                        iyf.write(l)

                    if energy_by_residue:
                        _copyScriptFile(pele_folder, 'addEnergyByResidueToPELEconf.py')
                        script_name = '._addEnergyByResidueToPELEconf.py'

                    # Create command
                    command = 'cd '+pele_folder+'/'+protein+'_'+ligand+'\n'
                    command += 'python -m pele_platform.main input.yaml\n'
                    if energy_by_residue:
                        command += 'python ../'+script_name+' output --energy_type '+energy_by_residue_type
                        if peptide:
                            command += '--peptide \n'
                        else:
                            command += '\n'
                        with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                            with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        l = 'restart: true\n'
                                    oyml.write(l)
                        command += 'python -m pele_platform.main input_restart.yaml\n'
                    command += 'cd ../..'
                    jobs.append(command)

        return jobs

    def analyseDocking(self, docking_folder, protein_atoms=None, atom_pairs=None, skip_chains=False, return_failed=False):
        """
        Missing
        """
        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(docking_folder, 'analyse_docking.py')
        script_path = docking_folder+'/._analyse_docking.py'

        # Write protein_atoms dictionary to json file
        if protein_atoms != None:
            with open(docking_folder+'/._protein_atoms.json', 'w') as jf:
                json.dump(protein_atoms, jf)

        # Write protein_atoms dictionary to json file
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

    def convertLigandPDBtoMae(self, ligands_folder):
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
        os.system('run ._PDBtoMAE.py')
        os.chdir(cwd)

    def addCatalyticDistance(self, exclude=None):
        """
        Add the catalytic distance to the docking data DataFrame by taking the minimum value of all non-nan columns.

        Parameters
        ==========
        exclude : list
            Columns to be excluded from the catalytic distance calculation.
        """

        if exclude != None:
            exclude += ['Score', 'RMSD'] + exclude
        else:
            exclude = ['Score', 'RMSD']

        catalytic_distances = {}
        for model in self.docking_ligands:
            catalytic_distances[model] = {}
            protein_series = self.docking_data[self.docking_data.index.get_level_values('Protein') == model]
            for ligand in self.docking_ligands[model]:
                catalytic_distances[model][ligand] = {}
                ligand_data = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                ligand_data = ligand_data.dropna(axis=1, how='all')
                labels = [label for label in ligand_data if label not in exclude]
                for i, row in ligand_data[labels].min(axis=1).iteritems():
                    catalytic_distances[model][ligand][i[2]] = row

        cd = []
        for i in self.docking_data.index:
            cd.append(catalytic_distances[i[0]][i[1]][i[2]])

        self.docking_data['Catalytic distance'] = cd

    def getBestDockingPoses(self, n_models=1, filter_by=None, filter_value=None,
                            filter_type='lower', return_failed=False):
        """
        Get best models based on the SCORE and (optionally) a specific column.
        The option filter_by allows to first filter all models by a specific filter_value
        and then select all models with the lowest score after the filter has been
        applied.

        Parameters
        ==========
        n_models : int
            The number of models to select for each protein + ligand docking.
        filter_by : str
            Name of the column to apply the filter.
        filter_value : float
            Threshold for the filter.
        filter_type : str
            If the models are lower or equal than (lower) or larger or equa than (larger)
            the specified filter_value.
        return_failed : bool
            Whether to return a list of the dockings without any models fulfilling
            the selection criteria. It is returned as a tuple alongside the filtered
            data frame.
        """
        best_poses = pd.DataFrame()
        bp = []
        failed = []
        for model in self.docking_ligands:
            protein_series = self.docking_data[self.docking_data.index.get_level_values('Protein') == model]
            for ligand in self.docking_ligands[model]:
                ligand_data = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                if filter_by != None:
                    if filter_value == None:
                        raise ValueError('When filtering by column you need to specify a filter_value.')
                    if filter_type == 'lower':
                        ligand_data = ligand_data[ligand_data[filter_by] <= filter_value]
                    elif filter_type == 'larger':
                        ligand_data = ligand_data[ligand_data[filter_by] >= filter_value]

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

    def extractDockingPoses(self, docking_data, docking_folder, output_folder):
        """
        Missing
        """
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
        command = 'run ._extract_docking.py ._docking_data.csv ../'+docking_folder
        os.system(command)

        # Remove docking data
        os.remove('._docking_data.csv')

        # move back to folder
        os.chdir('..')

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

    def loadModelsFromPrepwizardFolder(self, prepwizard_folder):
        """
        Read structures from a Schrodinger calculation.

        Parameters
        ==========
        prepwizard_folder : str
            Path to the output folder from a prepwizard calculation
        """

        models = []
        for d in os.listdir(prepwizard_folder):
            if os.path.isdir(prepwizard_folder+'/'+d):
                for f in os.listdir(prepwizard_folder+'/'+d):
                    if f.endswith('.pdb'):
                        model = f.replace('.pdb', '')
                        models.append(model)
                        self.readModelFromPDB(model, prepwizard_folder+'/'+d+'/'+f)
        missing_models = set(self.models_names) - set(models)
        if missing_models != set():
            print('Missing models in relaxation folder:')
            print('\t'+', '.join(missing_models))

    def loadModelsFromRosettaOptimization(self, optimization_folder, filter_score_term='score', min_value=True, tags=None):
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
                        self.readModelFromPDB(model, best_model_tag+'.pdb', wat_to_hoh=True)
                        os.remove(best_model_tag+'.pdb')
                        models.append(model)

        missing_models = set(self.models_names) - set(models)
        if missing_models != set():
            print('Missing models in relaxation folder:')
            print('\t'+', '.join(missing_models))

    def loadModelsFromMissingLoopBuilding(self, job_folder, filter_score_term='score', min_value=True,):
        """
        Load models from addMissingLoops() job calculation output.

        Parameters:
        job_folder : str
            Path to the addMissingLoops() calculation folder containing output.
        """

        # Get silent models paths
        executable = 'extract_pdbs.linuxgccrelease'
        output_folder = job_folder+'/output_models'

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
                        command += ' -tags '+best_model_tag
                        os.system(command)
                        loop = (int(loop.split('_')[0]), loop.split('_')[1])
                        loop_models[loop] = _readPDB(loop, best_model_tag+'.pdb')
                        os.remove(best_model_tag+'.pdb')

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
            os.remove(model+'.pdb')

    def loadModelsFromMembranePositioning(self, job_folder):
        """
        """
        for model in os.listdir(job_folder+'/output_models'):
            pdb_path = job_folder+'/output_models/'+model+'/'+model+'.pdb'
            self.readModelFromPDB(model, pdb_path)

    def saveModels(self, output_folder, keep_residues={}, models=None, **keywords):
        """
        Save all models as PDBs into the output_folder.

        Parameters
        ==========
        output_folder : str
            Path to the output folder to store models.
        """
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

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

    def removeModel(self, model):
        """
        Removes a specific model from this class

        Parameters
        ==========
        model : str
            Model name to remove
        """
        self.models_paths.pop(model)
        self.models_names.remove(model)
        self.structures.pop(model)
        self.sequences.pop(model)

    def compareSequences(self, sequences_file):
        """
        Compare models sequences to a set of different sequences and check missing
        or changed sequence information.

        Parameters
        ==========
        sequences_file : str
            Path to the sequences to compare

        Returns
        =======
        sequence_differences : dict
            Dictionary containing missing or changed information.
        """
        sequences = prepare_proteins.alignment.readFastaFile(sequences_file)

        # Iterate models to store sequence differences
        for model in self.models_names:

            # Create lists for missing information
            self.sequence_differences[model] = {}
            self.sequence_differences[model]['n_terminus'] = []
            self.sequence_differences[model]['mutations'] = []
            self.sequence_differences[model]['missing_loops'] = []
            self.sequence_differences[model]['c_terminus'] = []

            # Create a sequence alignement between current and target sequence
            to_align = {}
            to_align['current'] = self.sequences[model]
            to_align['target'] = sequences[model]
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
                    if loop_sequence != '':
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
    Read PDB file a structure object
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

def _copySchrodingerControlFile(output_folder):
    """
    Copy Schrodinger job control file to the specified folder
    """
    # Get control script
    control_script = resource_stream(Requirement.parse("prepare_proteins"),
                                     "prepare_proteins/_schrodinger_control.py")
    control_script = io.TextIOWrapper(control_script)

    # Write control script to output folder
    with open(output_folder+'/._schrodinger_control.py', 'w') as sof:
        for l in control_script:
            sof.write(l)

def _copyScriptFile(output_folder, script_name):
    """
    Copy a script file from the prepare_proteins package.

    Parameters
    ==========

    """
    # Get control script
    control_script = resource_stream(Requirement.parse("prepare_proteins"),
                                     "prepare_proteins/scripts/"+script_name)

    control_script = io.TextIOWrapper(control_script)

    # Write control script to output folder
    with open(output_folder+'/._'+script_name, 'w') as sof:
        for l in control_script:
            sof.write(l)
