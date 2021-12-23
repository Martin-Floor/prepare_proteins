from . import alignment
from . import _atom_selectors

import os
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
