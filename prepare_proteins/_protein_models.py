from . import alignment
from . import _atom_selectors
from . import rosettaScripts

import os
import shutil
import numpy as np
from Bio import PDB
from Bio.PDB.DSSP import DSSP

import pandas as pd

import uuid

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

    def __init__(self, models_folder):
        """
        Read PDB models as Bio.PDB structure objects.

        Parameters
        ==========
        models_folder : str
            Path to the folder containing the PDB models.
        """

        self.models_folder = models_folder
        self.models_paths = self._getModelsPaths()
        self.models_names = [] # Store model names
        self.structures = {} # structures are stored here
        self.sequences = {} # sequences are stored here
        self.msa = None # multiple sequence alignment
        self.multi_chain = False
        self.ss = {} # secondary structure strings are stored here

        # Read PDB structures into Biopython
        for model in sorted(self.models_paths):
            self.models_names.append(model)
            self.readModelFromPDB(model, self.models_paths[model])

        # Get sequence information based on stored structure objects
        self.getModelsSequences()

        # Calculate secondary structure inforamtion as strings
        self.calculateSecondaryStructure()

        # # Perform a multiple sequence aligment of models
        # self.calculateMSA()

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
        if coordinates.shape[0] != 3:
            raise ValueError('Coordinates must have shape (3,x). X=number of atoms in residue.')
        if len(coordinates.shape) > 1:
            if coordinates.shape[1] != len(atom_names):
                raise ValueError('Mismatch between the number of atom_names and coordinates.')
        if len(coordinates.shape) == 1:
                if len(atom_names) != 1:
                    raise ValueError('Mismatch between the number of atom_names and coordinates.')

        coordinates = coordinates.reshape(len(atom_names), 3)
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

    def readModelFromPDB(self, name, pdb_file):
        """
        Adds a model from a PDB file.

        Parameters
        ----------
        name : str
            Model name.
        pdb_file : str
            Path to the pdb file.

        Returns
        -------
        structure : Bio.PDB.Structure
            Structure object.
        """
        parser = PDB.PDBParser()
        self.structures[name] = parser.get_structure(name, pdb_file)
        self.models_paths[name] = pdb_file

        return self.structures[name]

    def getModelsSequences(self):
        """
        Get sequence information for all stored models.

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

    def calculateMSA(self):
        """
        Calculate a Multiple Sequence Alignment from the current models' sequences.

        Returns
        =======
        alignment : Bio.AlignIO
            Multiple sequence alignment in Biopython format.
        """
        self.msa = alignment.mafft.multipleSequenceAlignment(self.sequences)

        return self.msa

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
                self._saveStructureToPDB(self.structures[model], structure_path)

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
                remove.append(chain)

        model = [*self.structures[model].get_models()][0]
        for chain in remove:
            model.detach_child(chain.id)

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

    def setUpRosettaOptimization(self, relax_folder, nstruct=1, relax_cycles=5,
                                 output_folder='output_models', cst_files=None):
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
        if not os.path.exists(relax_folder+'/'+output_folder):
            os.mkdir(relax_folder+'/'+output_folder)

        # Save all models
        self.saveModels(relax_folder+'/input_models')

        # Create flags files
        jobs = []
        for model in self.models_names:
            # Create xml minimization protocol
            xml = rosettaScripts.xmlScript()

            # Create all-atom score function
            if cst_files != None:
                sfxn = rosettaScripts.scorefunctions.new_scorefunction('ref2015_cst',
                                                                       weights_file='ref2015_cst')
            else:
                sfxn = rosettaScripts.scorefunctions.new_scorefunction('ref2015',
                                                                       weights_file='ref2015')
            xml.addScorefunction(sfxn)

            # Create fastrelax mover
            relax = rosettaScripts.movers.fastRelax(repeats=relax_cycles, scorefxn=sfxn)
            xml.addMover(relax)

            if cst_files != None:
                if model not in cst_files:
                    raise ValuError('Model %s is not in the cst_files dictionary!' % model)
                set_cst = rosettaScripts.movers.constraintSetMover(add_constraints=True,
                                                                   cst_file='../'+cst_files[model])
                xml.addMover(set_cst)
                xml.setProtocol([set_cst, relax])
            else:
                xml.setProtocol([relax])

            xml.addOutputScorefunction(sfxn)
            xml.write_xml(relax_folder+'/xml/'+model+'_relax.xml')

            # Create options for minimization protocol
            flags = rosettaScripts.flags('xml/'+model+'_relax.xml',
                                         nstruct=nstruct, s='input_models/'+model+'.pdb',
                                         output_silent_file=output_folder+'/'+model+'_relax.out')

            flags.add_relax_options()
            flags.write_flags(relax_folder+'/flags/'+model+'_relax.flags')

            command = 'cd '+relax_folder+'\n'
            command += 'srun rosetta_scripts.mpi.linuxgccrelease @ '+'flags/'+model+'_relax.flags\n'
            command += 'cd ..\n'
            jobs.append(command)

        return jobs

    def setUpPrepwizardOptimization(self, prepare_folder, pH=7.0):
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
        self.saveModels(prepare_folder+'/input_models')

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
            command += '-rehtreat '
            command += '-epik_pH '+str(pH)+' '
            command += '-epik_pHt 2.0 '
            command += '-propka_pH '+str(pH)+' '
            command += '-f 2005 '
            command += '-rmsd 0.3 '
            command += '-samplewater '
            command += '-delwater_hbond_cutoff 3 '
            command += '-JOBNAME prepare_'+model+' '
            command += '-HOST localhost:1\n'
            command += 'cd ../../..\n'
            jobs.append(command)
        return jobs

    def loadModelsFromPrepwizardFolder(self, prepwizard_folder):
        """
        Read structures from a Schrodinger calculation.

        Parameters
        ==========
        prepwizard_folder : str
            Path to the output folder from a prepwizard calculation
        """

        for d in os.listdir(prepwizard_folder):
            for f in os.listdir(prepwizard_folder+'/'+d):
                if f.endswith('.pdb'):
                    model = f.replace('.pdb', '')
                    self.readModelFromPDB(model, prepwizard_folder+'/'+d+'/'+f)

    def loadFromSilentFolder(self, silent_folder, filter_score_term='score', min_value=True, relax_run=True):
        """
        Load the best energy models from a set of silent files inside a specfic folder.
        Useful to get the best models from a relaxation run.

        Parameters
        ==========
        silent_folder : str
            Path to folde where a silent file for each model is stored
        filter_score_term : str
            Score term used to filter models
        relax_run : bool
            Is this a relax run?
        min_value : bool
            Grab the minimum score value. Set false to grab the maximum scored value.
        """

        executable = 'extract_pdbs.linuxgccrelease'
        # set prefix according to calculation output type
        if relax_run:
            suffix = '_relax'
        else:
            suffix = ''

        for d in os.listdir(silent_folder):
            if d.endswith(suffix+'.out'):
                model = d.replace(suffix+'.out', '')
                scores = readSilentScores(silent_folder+'/'+d)
                if min_value:
                    best_model_tag = scores.idxmin()[filter_score_term]
                else:
                    best_model_tag = scores.idxmxn()[filter_score_term]
                command = executable
                command += ' -silent '+silent_folder+'/'+d
                command += ' -tags '+best_model_tag
                os.system(command)
                self.readModelFromPDB(model, best_model_tag+'.pdb')
                os.remove(best_model_tag+'.pdb')

    def saveModels(self, output_folder, keep_residues={}, **keywords):
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
            if model in keep_residues:
                kr = keep_residues[model]
            else:
                kr = []
            self._saveStructureToPDB(self.structures[model],
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

    def _saveStructureToPDB(self, structure, output_file, remove_hydrogens=False,
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
