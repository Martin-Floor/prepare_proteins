from . import alignment
from . import _atom_selectors
from . import rosettaScripts

import os
import shutil
import numpy as np
from Bio import PDB
from Bio.PDB.DSSP import DSSP

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
        Contains the Bio.PDB structure object to each model.

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
        for model in self.models_paths:
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
        multi_chain = False
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
                multi_chain = True

        return self.sequences

        # If any model has more than one chain set multi_chain to True.
        if multi_chain:
            self.multi_chain = True

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

    def setUpRosettaOptimization(self, minimization_folder, nstruct=1, relax_cycles=5,
                                 output_folder='output_models', cst_files=None):
        """
        Set up minimizations using Rosetta FastRelax protocol.
        """

        # Create minimization job folders
        if not os.path.exists(minimization_folder):
            os.mkdir(minimization_folder)
        if not os.path.exists(minimization_folder+'/input_models'):
            os.mkdir(minimization_folder+'/input_models')
        if not os.path.exists(minimization_folder+'/flags'):
            os.mkdir(minimization_folder+'/flags')
        if not os.path.exists(minimization_folder+'/xml'):
            os.mkdir(minimization_folder+'/xml')
        if not os.path.exists(minimization_folder+'/'+output_folder):
            os.mkdir(minimization_folder+'/'+output_folder)

        # Save all models
        self.saveModels(minimization_folder+'/input_models')

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
            xml.write_xml(minimization_folder+'/xml/'+model+'_relax.xml')

            # Create options for minimization protocol
            flags = rosettaScripts.flags('xml/'+model+'_relax.xml',
                                         nstruct=nstruct, s='input_models/'+model+'.pdb',
                                         output_silent_file=output_folder+'/'+model+'_relax.out')

            flags.add_relax_options()
            flags.write_flags(minimization_folder+'/flags/'+model+'_relax.flags')

            command = 'cd '+minimization_folder+'\n'
            command += 'srun rosetta_scripts.mpi.linuxgccrelease @ '+'flags/'+model+'_relax.flags\n'
            command += 'cd ..\n'
            jobs.append(command)

        return jobs

    def saveModels(self, output_folder):
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
            self._saveStructureToPDB(self.structures[model],
                                output_folder+'/'+model+'.pdb')

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
                            remove_water=False, only_protein=False):
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
        """

        io = PDB.PDBIO()
        io.set_structure(structure)

        selector = None
        if remove_hydrogens:
            selector = _atom_selectors.notHydrogen()
        elif remove_water:
            selector = _atom_selectors.notWater()
        elif only_protein:
            selector = _atom_selectors.onlyProtein()

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
