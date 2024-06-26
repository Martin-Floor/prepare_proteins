import fileinput
import gc
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import warnings

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from Bio import PDB, BiopythonWarning
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import aa3
from pkg_resources import Requirement, resource_listdir, resource_stream
from scipy.spatial import distance_matrix

import prepare_proteins

from . import MD, _atom_selectors, alignment, rosettaScripts


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

    def __init__(
        self,
        models_folder,
        get_sequences=True,
        get_ss=False,
        msa=False,
        verbose=False,
        only_models=None,
        exclude_models=None,
        ignore_biopython_warnings=False,
        collect_memory_every=None,
    ):
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

        if ignore_biopython_warnings:
            warnings.simplefilter("ignore", BiopythonWarning)

        if only_models == None:
            only_models = []

        elif isinstance(only_models, str):
            only_models = [only_models]

        elif not isinstance(only_models, (list, tuple, set)):
            raise ValueError(
                "You must give models as a list or a single model as a string!"
            )

        if exclude_models == None:
            exclude_models = []

        elif isinstance(exclude_models, str):
            exclude_models = [exclude_models]

        elif not isinstance(exclude_models, (list, tuple, set)):
            raise ValueError(
                "You must give excluded models as a list or a single model as a string!"
            )

        self.models_folder = models_folder
        if isinstance(self.models_folder, dict):
            self.models_paths = self.models_folder
        elif isinstance(self.models_folder, str):
            self.models_paths = self._getModelsPaths(
                only_models=only_models, exclude_models=exclude_models
            )
        self.models_names = []  # Store model names
        self.structures = {}  # structures are stored here
        self.sequences = {}  # sequences are stored here
        self.target_sequences = {}  # Final sequences are stored here
        self.msa = None  # multiple sequence alignment
        self.multi_chain = False
        self.ss = {}  # secondary structure strings are stored here
        self.docking_data = None  # secondary structure strings are stored here
        self.docking_distances = {}
        self.docking_angles = {}
        self.docking_metric_type = {}
        self.docking_ligands = {}
        self.rosetta_data = None  # Rosetta data is stored here
        self.sequence_differences = {}  # Store missing/changed sequence information
        self.conects = {}  # Store the conection inforamtion for each model
        self.covalent = {}  # Store covalent residues

        self.distance_data = {}
        self.models_data = {}

        # Read PDB structures into Biopython
        collect_memory = False
        for i, model in enumerate(sorted(self.models_paths)):

            if verbose:
                print("Reading model: %s" % model)

            if collect_memory_every and i % collect_memory_every == 0:
                collect_memory = True
            else:
                collect_memory = False

            self.models_names.append(model)
            self.readModelFromPDB(
                model,
                self.models_paths[model],
                add_to_path=True,
                collect_memory=collect_memory,
            )

        if get_sequences:
            # Get sequence information based on stored structure objects
            self.getModelsSequences()

        if get_ss:
            # Calculate secondary structure inforamtion as strings
            self.calculateSecondaryStructure()

        # # Perform a multiple sequence aligment of models
        if msa:
            if self.multichain:
                print(
                    "MSA cannot be calculated at startup when multichain models \
are given. See the calculateMSA() method for selecting which chains will be algined."
                )
            else:
                self.calculateMSA()

    def addResidueToModel(
        self,
        model,
        chain_id,
        resname,
        atom_names,
        coordinates,
        new_resid=None,
        elements=None,
        hetatom=True,
        water=False,
    ):
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
            raise ValueError("The input model was not found.")

        # Check chain ID
        chain = [
            chain
            for chain in self.structures[model].get_chains()
            if chain_id == chain.id
        ]
        if len(chain) != 1:
            print("Chain ID %s was not found in the selected model." % chain_id)
            print("Creating a new chain with ID %s" % chain_id)
            new_chain = PDB.Chain.Chain(chain_id)
            for m in self.structures[model]:
                m.add(new_chain)
            chain = [
                chain
                for chain in self.structures[model].get_chains()
                if chain_id == chain.id
            ]

        # Check coordinates correctness
        if coordinates.shape == ():
            if np.isnan(coordinates):
                raise ValueError("Given Coordinate in nan!")
        elif np.isnan(coordinates.any()):
            raise ValueError("Some given Coordinates are nan!")
        if coordinates.shape[1] != 3:
            raise ValueError(
                "Coordinates must have shape (x,3). X=number of atoms in residue."
            )
        if len(coordinates.shape) > 1:
            if coordinates.shape[0] != len(atom_names):
                raise ValueError(
                    "Mismatch between the number of atom_names and coordinates."
                )
        if len(coordinates.shape) == 1:
            if len(atom_names) != 1:
                raise ValueError(
                    "Mismatch between the number of atom_names and coordinates."
                )

        # Create new residue
        if new_resid == None:
            try:
                new_resid = max([r.id[1] for r in chain[0].get_residues()]) + 1
            except:
                new_resid = 1

        rt_flag = " "  # Define the residue type flag for complete the residue ID.
        if hetatom:
            rt_flag = "H"
        if water:
            rt_flag = "W"
        residue = PDB.Residue.Residue((rt_flag, new_resid, " "), resname, " ")

        # Add new atoms to residue
        try:
            serial_number = max([a.serial_number for a in chain[0].get_atoms()]) + 1
        except:
            serial_number = 1
        for i, atnm in enumerate(atom_names):
            if elements:
                atom = PDB.Atom.Atom(
                    atom_names[i],
                    coordinates[i],
                    0,
                    1.0,
                    " ",
                    "%-4s" % atom_names[i],
                    serial_number + i,
                    elements[i],
                )
            else:
                atom = PDB.Atom.Atom(
                    atom_names[i],
                    coordinates[i],
                    0,
                    1.0,
                    " ",
                    "%-4s" % atom_names[i],
                    serial_number + i,
                )
            residue.add(atom)
        chain[0].add(residue)

        return new_resid

    def removeModelAtoms(self, model, atoms_list):
        """
        Remove specific atoms of a model. Atoms to delete are given as a list of tuples.
        Each tuple contains three positions specifying (chain_id, residue_id, atom_name).

        Paramters
        =========
        model : str
            model ID
        atom_lists : list
            Specifies the list of atoms to delete for the particular model.
        """

        def removeAtomInConects(self, model, atom):
            """
            Function for removing conect lines involving the deleted atom.
            """
            to_remove = []
            for conect in self.conects[model]:
                if atom in conect:
                    to_remove.append(conect)
            for conect in to_remove:
                self.conects[model].remove(conect)

        for remove_atom in atoms_list:
            for chain in self.structures[model].get_chains():
                if chain.id == remove_atom[0]:
                    for residue in chain:
                        if residue.id[1] == remove_atom[1]:
                            for atom in residue:
                                if atom.name == remove_atom[2]:
                                    print(
                                        "Removing atom: "
                                        + str(remove_atom)
                                        + " from model "
                                        + model
                                    )
                                    residue.detach_child(atom.id)
                                    removeAtomInConects(self, model, remove_atom)

    def removeModelResidues(self, model, residues_list):
        """
        Remove a group of residues from the model structure.

        Paramters
        =========
        model : str
            model ID
        residues_list : list
            Specifies the list of resdiues to delete for the particular model.
        """

        # Get all atoms for residues to remove them
        atoms_to_remove = []
        for residue in self.structures[model].get_residues():
            chain = residue.get_parent().id
            if (chain, residue.id[1]) in residues_list:
                for atom in residue:
                    atoms_to_remove.append((chain, residue.id[1], atom.name))

        if atoms_to_remove == []:
            raise ValueError("No atoms were found for the specified residue!")

        self.removeModelAtoms(model, atoms_to_remove)

    def removeAtomFromConectLines(self, residue_name, atom_name, verbose=True):
        """
        Remove the given (atom_name) atoms from all the connect lines involving
        the given (residue_name) residues.
        """

        # Match all the atoms with the given residue and atom name

        for model in self:
            resnames = {}
            for chain in self.structures[model].get_chains():
                for residue in chain:
                    resnames[(chain.id, residue.id[1])] = residue.resname

            conects = []
            count = 0
            for conect in self.conects[model]:
                new_conect = []
                for atom in conect:
                    if resnames[atom[:-1]] != residue_name and atom[-1] != atom_name:
                        new_conect.append(atom)
                    else:
                        count += 1
                if new_conect == []:
                    continue
                conects.append(new_conect)
            self.conects[model] = conects
            if verbose:
                print(f"Removed {count} from conect lines of model {model}")

    def removeCaps(self, models=None, remove_ace=True, remove_nma=True):
        """
        Remove caps from models.
        """

        for model in self:

            if models and model not in models:
                continue

            for chain in self.structures[model].get_chains():

                st_residues = [r for r in chain if r.resname in aa3]

                ACE = None
                NMA = None
                NT = None
                CT = None

                for residue in self.structures[model].get_residues():
                    if residue.resname == "ACE":
                        ACE = residue
                    elif residue.resname == "NMA":
                        NMA = residue

                for i, residue in enumerate(chain):

                    if (
                        ACE
                        and residue.id[1] == ACE.id[1] + 1
                        and residue.resname in aa3
                    ):
                        NT = residue
                    elif not ACE and i == 0:
                        NT = residue
                    if NMA and residue.id[1] == NMA.id[1] and residue.resname in aa3:
                        CT = residue
                    elif not NMA and i == len(st_residues) - 1:
                        CT = residue

                # Remove termini
                if ACE and remove_ace:
                    for a in ACE:
                        self.removeAtomFromConectLines("ACE", a, verbose=False)
                    chain.detach_child(ACE.id)

                if NMA and remove_nma:
                    for a in NMA:
                        self.removeAtomFromConectLines("NMA", a, verbose=False)
                    chain.detach_child(NMA.id)

    def addOXTAtoms(self):
        """
        Add missing OXT atoms for terminal residues when missing
        """

        # Define internal coordinates for OXT
        oxt_c_distance = 1.251
        oxt_c_o_angle = 124.222
        oxt_c_o_ca = 179.489

        for model in self:
            for chain in self.structures[model].get_chains():
                residues = [r for r in chain if r.id[0] == " "]
                if residues == []:
                    continue
                last_atoms = {a.name: a for a in residues[-1]}

                if "OXT" not in last_atoms:

                    oxt_coord = _computeCartesianFromInternal(
                        last_atoms["C"].coord,
                        last_atoms["O"].coord,
                        last_atoms["CA"].coord,
                        1.251,
                        124.222,
                        179.489,
                    )
                    serial_number = max([a.serial_number for a in residues[-1]]) + 1
                    oxt = PDB.Atom.Atom(
                        "OXT", oxt_coord, 0, 1.0, " ", "OXT ", serial_number + 1, "O"
                    )
                    residues[-1].add(oxt)

    def readModelFromPDB(
        self,
        model,
        pdb_file,
        wat_to_hoh=False,
        covalent_check=True,
        atom_mapping=None,
        add_to_path=False,
        conect_update=True,
        collect_memory=False,
    ):
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

        if model not in self.models_names:
            self.models_names.append(model)

        self.structures[model] = _readPDB(model, pdb_file)

        if wat_to_hoh:
            for residue in self.structures[model].get_residues():
                if residue.resname == "WAT":
                    residue.resname = "HOH"

        if model not in self.conects or self.conects[model] == []:
            # Read conect lines
            self.conects[model] = self._readPDBConectLines(pdb_file, model)

        # Check covalent ligands
        if covalent_check:
            self._checkCovalentLigands(model, pdb_file, atom_mapping=atom_mapping)

        # Update conect lines
        if conect_update:
            self.conects[model] = self._readPDBConectLines(pdb_file, model)

        if add_to_path:
            self.models_paths[model] = pdb_file

        if collect_memory:
            gc.collect()  # Collect memory

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

    def renumberModels(self):
        """
        Renumber every PDB chain residues from 1 onward.
        """

        for model in self:
            for c in self.structures[model].get_chains():
                for i, r in enumerate(c):
                    r.id = (r.id[0], i + 1, r.id[2])

    def calculateMSA(self, extra_sequences=None, chains=None):
        """
        Calculate a Multiple Sequence Alignment from the current models' sequences.

        Returns
        =======
        alignment : Bio.AlignIO
            Multiple sequence alignment in Biopython format.
        chains : dict
            Dictionary specifying which chain to use for each model
        """

        # If only a single ID is given use it for all models
        if isinstance(chains, str):
            cd = {}
            for model in self:
                cd[model] = chains
            chains = cd

        for model in self:
            if isinstance(self.sequences[model], dict) and chains == None:
                raise ValueError(
                    "There are multiple chains in model %s. Specify which \
chain to use for each model with the chains option."
                    % model
                )

        if chains != None:
            sequences = {}
            for model in self.models_names:
                if isinstance(self.sequences[model], dict):
                    sequences[model] = self.sequences[model][chains[model]]
                else:
                    sequences[model] = self.sequences[model]

            if isinstance(extra_sequences, dict):
                for s in extra_sequences:
                    sequences[s] = extra_sequences[s]
        else:
            sequences = self.sequences.copy()

        if isinstance(extra_sequences, dict):
            for s in extra_sequences:
                sequences[s] = extra_sequences[s]

        self.msa = alignment.mafft.multipleSequenceAlignment(sequences, stderr=False)

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
                conserved.append((i, list(positions[i])[0]))

        return conserved

    def getStructurePositionsFromMSAindexes(self, msa_indexes, msa=None, models=None):
        """
        Get the individual model residue structure positions of a set of MSA indexes
        Paramters
        =========
        msa_indexes : list
            Zero-based MSA indexes
        Returns
        =======
        residue_indexes : dict
            Residue indexes for each protein at the MSA positions
        """

        if isinstance(msa_indexes, int):
            msa_indexes = [msa_indexes]

        if models == None:
            models = []
        elif isinstance(models, str):
            models = [models]

        # If msa not given get the class msa attribute
        if msa == None:
            msa = self.msa

        positions = {}
        residue_ids = {}

        # Gather dictionary between sequence position and residue PDB index
        for model in self.models_names:
            if models != [] and model not in models:
                continue

            positions[model] = 0
            residue_ids[model] = {}
            for i, r in enumerate(self.structures[model].get_residues()):
                residue_ids[model][i + 1] = r.id[1]

        # Gather sequence indexes for the given MSA index
        sequence_positions = {}
        for i in range(msa.get_alignment_length()):
            # Count structure positions
            for entry in msa:

                if entry.id not in self.models_names:
                    continue
                sequence_positions.setdefault(entry.id, [])

                if entry.seq[i] != "-":
                    positions[entry.id] += 1

            # Get residue positions matching the MSA indexes
            if i in msa_indexes:
                for entry in msa:
                    if entry.id not in self.models_names:
                        continue

                    if entry.seq[i] == "-":
                        sequence_positions[entry.id].append(None)
                    else:
                        sequence_positions[entry.id].append(
                            residue_ids[entry.id][positions[entry.id]]
                        )

        return sequence_positions

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
                structure_path = "." + str(uuid.uuid4()) + ".pdb"
                _saveStructureToPDB(self.structures[model], structure_path)

            dssp = DSSP(self.structures[model][0], structure_path)
            if _save_structure:
                os.remove(structure_path)
            ss = []
            for k in dssp.keys():
                ss.append(dssp[k][2])
            ss = "".join(ss)
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
                print("From model %s Removing chain %s" % (model, chain.id))
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
            raise ValueError(
                "removeTerminalUnstructuredRegions() function only supports single chain models"
            )

        # Calculate residues to be removed
        for model in self.models_names:

            # Get N-terminal residues to remove based on secondary structure.
            remove_indexes = []
            for i, r in enumerate(self.ss[model]):
                if r == "-":
                    remove_indexes.append(i)
                else:
                    break

            if len(remove_indexes) > n_hanging:
                remove_indexes = remove_indexes[:-n_hanging]
            else:
                remove_indexes = []

            # Get C-terminal residues to remove based on secondary structure.
            remove_C = []
            for i, r in enumerate(self.ss[model][::-1]):
                if r == "-":
                    remove_C.append(i)
                else:
                    break
            if len(remove_C) > n_hanging:
                remove_C = remove_C[:-n_hanging]
            else:
                remove_C = []

            for x in remove_C:
                remove_indexes.append(len(self.ss[model]) - 1 - x)

            # Sort indexes
            remove_indexes = sorted(remove_indexes)

            # Get residues to remove from models structures
            remove_this = []
            for c in self.structures[model].get_chains():
                for i, r in enumerate(c.get_residues()):
                    if i in remove_indexes:
                        remove_this.append(r)
                chain = c

            # Remove residues
            for r in remove_this:
                chain.detach_child(r.id)

        self.getModelsSequences()
        self.calculateSecondaryStructure(_save_structure=True)

    def removeTerminiByConfidenceScore(
        self, confidence_threshold=70.0, keep_up_to=5, verbose=True
    ):
        """
        Remove terminal regions with low confidence scores from models.

        confidence_threshold : float
            AlphaFold confidence threshold to consider residues as having a low score.
        keep_up_to : int
            If any terminal region is no larger than this value it will be kept.
        """

        remove_models = set()
        ## Warning only single chain implemented
        for model in self.models_names:

            atoms = [a for a in self.structures[model].get_atoms()]
            bfactors = [a.bfactor for a in atoms]

            if np.average(bfactors) == 0:
                if verbose:
                    print(
                        "Warning: model %s has no atom with the selected confidence!"
                        % model
                    )
                remove_models.add(model)
                continue

            n_terminus = set()
            something = False
            for a in atoms:
                if a.bfactor < confidence_threshold:
                    n_terminus.add(a.get_parent().id[1])
                else:
                    something = True
                    break

            c_terminus = set()

            for a in reversed(atoms):
                if a.bfactor < confidence_threshold:
                    c_terminus.add(a.get_parent().id[1])
                else:
                    something = True
                    break

            if not something:
                if verbose and model not in remove_models:
                    print(
                        "Warning: model %s has no atom with the selected confidence!"
                        % model
                    )
                remove_models.add(model)
                continue

            n_terminus = sorted(list(n_terminus))
            c_terminus = sorted(list(c_terminus))

            if len(n_terminus) <= keep_up_to:
                n_terminus = []
            if len(c_terminus) <= keep_up_to:
                c_terminus = []

            remove_this = []
            for c in self.structures[model].get_chains():
                for r in c.get_residues():
                    if r.id[1] in n_terminus or r.id[1] in c_terminus:
                        remove_this.append(r)
                chain = c
                # Remove residues
                for r in remove_this:
                    chain.detach_child(r.id)

        for model in remove_models:
            self.removeModel(model)

        self.getModelsSequences()
        # self.calculateSecondaryStructure(_save_structure=True)

        # Missing save models and reload them to take effect.

    def removeNotAlignedRegions(
        self,
        ref_structure,
        max_ca_ca=5.0,
        remove_low_confidence_unaligned_loops=False,
        confidence_threshold=50.0,
        min_loop_length=10,
    ):
        """
        Remove models regions that not aligned with the given reference structure. The mapping is based on
        the current structural alignment of the reference and the target models. The termini are removed
        if they don't align with the reference structure. Internal loops that do not align with the given
        reference structure can optionally be removed if they have a confidence score lower than the one
        defined as threshold.

        Parameters
        ==========
        ref_structure : str or Bio.PDB.Structure.Structure
            Path to the input PDB or model name to use as reference. Otherwise a Bio.PDB.Structure object
            can be given.
        max_ca_ca : float
            Maximum CA-CA distance for two residues to be considered aligned.
        remove_low_confidence_unaligned_loops : bool
            Remove not aligned loops with low confidence from the model
        confidence_threshold : float
            Threshold to consider a loop region not algined as low confidence for their removal.
        min_loop_length : int
            Length of the internal unaligned region to be considered a loop.

        Returns
        =======

        """

        # Check input structure input
        if isinstance(ref_structure, str):
            if ref_structure.endswith(".pdb"):
                ref_structure = prepare_proteins._readPDB("ref", ref_structure)
            else:
                if ref_structure in self.models_names:
                    ref_structure = self.structures[ref_structure]
                else:
                    raise ValueError("Reference structure was not found in models")
        elif not isinstance(ref_structure, PDB.Structure.Structure):
            raise ValueError(
                "ref_structure should be a  Bio.PDB.Structure.Structure or string object"
            )

        # Iterate models
        for i, model in enumerate(self):

            # Get structurally aligned residues to reference structure
            aligned_residues = _getAlignedResiduesBasedOnStructuralAlignment(
                ref_structure, self.structures[model]
            )

            ### Remove unaligned termini ###

            # Get structurally aligned residues to reference structure
            target_residues = [
                r for r in self.structures[model].get_residues() if r.id[0] == " "
            ]

            n_terminus = set()
            for ar, tr in zip(aligned_residues, target_residues):
                if ar == "-":
                    n_terminus.add(tr.id[1])
                else:
                    break

            c_terminus = set()
            for ar, tr in reversed(list(zip(aligned_residues, target_residues))):
                if ar == "-":
                    c_terminus.add(tr.id[1])
                else:
                    break

            n_terminus = sorted(list(n_terminus))
            c_terminus = sorted(list(c_terminus))

            #         ### Remove unaligned low confidence loops ###
            #         if remove_low_confidence_unaligned_loops:
            #             loops = []
            #             loop = []
            #             loops_to_remove = []
            #             for ar,tr in zip(aligned_residues, target_residues):
            #                 if ar == '-':
            #                     loop.append(tr)
            #                 else:
            #                     loops.append(loop)
            #                     loop = []

            #             for loop in loops:
            #                 if len(loop) >= min_loop_length:
            #                     low_confidence_residues = []
            #                     for r in loop:
            #                         for a in r:
            #                             if a.bfactor < confidence_threshold:
            #                                 low_confidence_residues.append(r)
            #                                 break
            #                     if len(low_confidence_residues) > min_loop_length:
            #                         for r in low_confidence_residues:
            #                             loops_to_remove.append(r.id[1])
            #         print(model, ','.join([str(x) for x in loops_to_remove]))

            remove_this = []
            for c in self.structures[model].get_chains():
                for r in c.get_residues():
                    if (
                        r.id[1] in n_terminus or r.id[1] in c_terminus
                    ):  # or r.id[1] in low_confidence_residues:
                        remove_this.append(r)
                chain = c
                # Remove residues
                for r in remove_this:
                    chain.detach_child(r.id)

        self.getModelsSequences()

    def alignModelsToReferencePDB(
        self,
        reference,
        output_folder,
        chain_indexes=None,
        trajectory_chain_indexes=None,
        reference_chain_indexes=None,
        aligment_mode="aligned",
        verbose=False,
        reference_residues=None,
    ):
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

        Returns
        =======
        rmsd : tuple
            A tuple containing the RMSD in Angstroms and the number of alpha-carbon
            atoms over which it was calculated.
        """

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        reference = md.load(reference)
        rmsd = {}

        # Check chain indexes input
        if isinstance(trajectory_chain_indexes, list):
            tci = {}
            for model in self.models_names:
                tci[model] = trajectory_chain_indexes
            trajectory_chain_indexes = tci

        for model in self.models_names:

            if verbose:
                print("Saving model: %s" % model)

            traj = md.load(self.models_paths[model])
            if trajectory_chain_indexes is None:
                rmsd[model] = MD.alignTrajectoryBySequenceAlignment(
                    traj,
                    reference,
                    chain_indexes=chain_indexes,
                    trajectory_chain_indexes=trajectory_chain_indexes,
                    reference_chain_indexes=reference_chain_indexes,
                    aligment_mode=aligment_mode,
                    reference_residues=reference_residues,
                )
            else:
                rmsd[model] = MD.alignTrajectoryBySequenceAlignment(
                    traj,
                    reference,
                    chain_indexes=chain_indexes,
                    trajectory_chain_indexes=trajectory_chain_indexes[model],
                    reference_chain_indexes=reference_chain_indexes,
                    aligment_mode=aligment_mode,
                    reference_residues=reference_residues,
                )

            # Get bfactors
            bfactors = np.array([a.bfactor for a in self.structures[model].get_atoms()])

            # Correct B-factors outside the -10 to 100 range accepted ny mdtraj
            bfactors = np.where(bfactors >= 100.0, 99.99, bfactors)
            bfactors = np.where(bfactors <= -10.0, -9.99, bfactors)

            traj.save(output_folder + "/" + model + ".pdb", bfactors=bfactors)

        return rmsd

    def positionLigandsAtCoordinate(
        self,
        coordinate,
        ligand_folder,
        output_folder,
        separator="-",
        overwrite=True,
        only_models=None,
        only_ligands=None,
    ):
        """
        Position a set of ligands into specific protein coordinates.

        Parameters
        ==========
        coordinate : tuple or dict
            New desired coordinates of the ligand
        ligand_folder : str
            Path to the ligands folder to store ligand molecules
        output_folder : str
            Path to the output folder to store models
        overwrite : bool
            Overwrite if structure file already exists.
        """

        # Create output directory
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if isinstance(only_models, str):
            only_models = [only_models]

        if isinstance(only_ligands, str):
            only_ligands = [only_ligands]

        # Copy script file to output directory
        _copyScriptFile(output_folder, "positionLigandAtCoordinate.py")

        for l in os.listdir(ligand_folder):
            if l.endswith(".mae"):
                ln = l.replace(".mae", "")
            elif l.endswith(".pdb"):
                ln = l.replace(".pdb", "")
            else:
                continue

            if not isinstance(only_ligands, type(None)):
                if ln not in only_ligands:
                    continue

            for model in self:

                if not isinstance(only_models, type(None)):
                    if model not in only_models:
                        continue

                self.docking_ligands.setdefault(model, [])
                self.docking_ligands[model].append(ln)

                if not os.path.exists(output_folder + "/" + model):
                    os.mkdir(output_folder + "/" + model)

                if (
                    os.path.exists(
                        output_folder
                        + "/"
                        + model
                        + "/"
                        + model
                        + separator
                        + ln
                        + separator
                        + "0.pdb"
                    )
                    and not overwrite
                ):
                    continue

                _saveStructureToPDB(
                    self.structures[model],
                    output_folder + "/" + model + "/" + model + separator + ln + ".pdb",
                )
                command = (
                    "run python3 " + output_folder + "/._positionLigandAtCoordinate.py "
                )
                command += (
                    output_folder + "/" + model + "/" + model + separator + ln + ".pdb "
                )
                command += ligand_folder + "/" + l + " "
                if isinstance(coordinate, dict):
                    coordinate_string = (
                        '"' + ",".join([str(x) for x in coordinate[model]]) + '"'
                    )
                elif isinstance(coordinate, tuple) and len(coordinate) == 3:
                    coordinate_string = (
                        '"' + ",".join([str(x) for x in coordinate]) + '"'
                    )
                else:
                    raise ValueError(
                        "coordinate needs to be a 3-element tuple of integers or dict."
                    )
                if "-" in coordinate_string:
                    coordinate_string = coordinate_string.replace("-", "\-")
                command += coordinate_string
                command += ' --separator "' + separator + '" '
                command += " --pele_poses\n"
                os.system(command)

    def createMetalConstraintFiles(
        self, job_folder, sugars=False, params_folder=None, models=None
    ):
        """
        Create metal constraint files.

        Parameters
        ==========
        job_folder : str
            Folder path where to place the constraint files.
        sugars : bool
            Use carbohydrate aware Rosetta PDB reading.
        params_folder : str
            Path to a folder containing a set of params file to be employed.
        models : list
            Only consider models inside the given list.
        """

        # Create mutation job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")

        output_folder = job_folder + "/cst_files"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Copy models to input folder
        self.saveModels(job_folder + "/input_models", models=models)

        # Copy embeddingToMembrane.py script
        _copyScriptFile(job_folder, "createMetalConstraints.py", subfolder="pyrosetta")

        jobs = []
        for model in self:

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            command = "cd " + output_folder + "\n"
            command += "python ../._createMetalConstraints.py "
            command += "../input_models/" + model + ".pdb "
            command += "metal_" + model + ".cst "
            if sugars:
                command += "--sugars "
            if params_folder != None:
                command += "--params_folder " + params_folder + " "
            command += "\ncd " + "../" * len(output_folder.split("/")) + "\n"
            jobs.append(command)

        return jobs

    def createMutants(
        self,
        job_folder,
        mutants,
        nstruct=100,
        relax_cycles=0,
        cst_optimization=True,
        executable="rosetta_scripts.mpi.linuxgccrelease",
        sugars=False,
        param_files=None,
        mpi_command="slurm",
        cpus=None,
    ):
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
        sugars : bool
            Use carbohydrate aware Rosetta optimization
        """

        mpi_commands = ["slurm", "openmpi", None]
        if mpi_command not in mpi_commands:
            raise ValueError(
                "Wrong mpi_command it should either: " + " ".join(mpi_commands)
            )

        if mpi_command == "openmpi" and not isinstance(cpus, int):
            raise ValueError("You must define the number of CPU")

        # Create mutation job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")
        if not os.path.exists(job_folder + "/flags"):
            os.mkdir(job_folder + "/flags")
        if not os.path.exists(job_folder + "/xml"):
            os.mkdir(job_folder + "/xml")
        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Save considered models
        considered_models = list(mutants.keys())
        self.saveModels(job_folder + "/input_models", models=considered_models)

        jobs = []

        # Create all-atom score function
        score_fxn_name = "ref2015"
        sfxn = rosettaScripts.scorefunctions.new_scorefunction(
            score_fxn_name, weights_file=score_fxn_name
        )

        # Create and append execution command
        if mpi_command == None:
            mpi_command = ""
        elif mpi_command == "slurm":
            mpi_command = "srun "
        elif mpi_command == "openmpi":
            mpi_command = "mpirun -np " + str(cpus) + " "
        else:
            mpi_command = mpi_command + " "

        for model in self.models_names:

            # Skip models not in given mutants
            if model not in considered_models:
                continue

            if not os.path.exists(job_folder + "/output_models/" + model):
                os.mkdir(job_folder + "/output_models/" + model)

            # Iterate each mutant
            for mutant in mutants[model]:

                # Create xml mutation (and minimization) protocol
                xml = rosettaScripts.xmlScript()
                protocol = []

                # Add score function
                xml.addScorefunction(sfxn)

                for m in mutants[model][mutant]:
                    mutate = rosettaScripts.movers.mutate(
                        name="mutate_" + str(m[0]),
                        target_residue=m[0],
                        new_residue=PDB.Polypeptide.one_to_three(m[1]),
                    )
                    xml.addMover(mutate)
                    protocol.append(mutate)

                if relax_cycles:
                    # Create fastrelax mover
                    relax = rosettaScripts.movers.fastRelax(
                        repeats=relax_cycles, scorefxn=sfxn
                    )
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
                xml.write_xml(job_folder + "/xml/" + model + "_" + mutant + ".xml")

                # Create options for minimization protocol
                flags = rosettaScripts.flags(
                    "../../xml/" + model + "_" + mutant + ".xml",
                    nstruct=nstruct,
                    s="../../input_models/" + model + ".pdb",
                    output_silent_file=model + "_" + mutant + ".out",
                )

                # Add relaxation with constraints options and write flags file
                if cst_optimization and relax_cycles:
                    flags.add_relax_cst_options()
                else:
                    flags.add_relax_options()

                # Add path to params files
                if param_files != None:
                    if not os.path.exists(job_folder + "/params"):
                        os.mkdir(job_folder + "/params")
                    if isinstance(param_files, str):
                        param_files = [param_files]
                    for param in param_files:
                        param_name = param.split("/")[-1]
                        shutil.copyfile(param, job_folder + "/params/" + param_name)
                    flags.addOption("in:file:extra_res_path", "../../params")

                if sugars:
                    flags.addOption("include_sugars")
                    flags.addOption("alternate_3_letter_codes", "pdb_sugar")
                    flags.addOption("write_glycan_pdb_codes")
                    flags.addOption("auto_detect_glycan_connections")
                    flags.addOption("maintain_links")

                flags.write_flags(
                    job_folder + "/flags/" + model + "_" + mutant + ".flags"
                )

                command = "cd " + job_folder + "/output_models/" + model + "\n"
                command += (
                    mpi_command
                    + executable
                    + " @ "
                    + "../../flags/"
                    + model
                    + "_"
                    + mutant
                    + ".flags\n"
                )
                command += "cd ../../..\n"
                jobs.append(command)

        return jobs

    def createDisulfureBond(
        self,
        job_folder,
        cys_dic,
        nstruct=100,
        relax_cycles=0,
        cst_optimization=True,
        executable="rosetta_scripts.mpi.linuxgccrelease",
        param_files=None,
        mpi_command="slurm",
        cpus=None,
        remove_existing=False,
        repack=True,
        scorefxn="ref2015",
    ):
        """
        Create Disulfure bonds from protein models. Cysteine residues must be given as a nested dictionary
        with each protein as the first key and the name of the particular mutant as the second key.
        The value of each inner dictionary is a list containing only one element string with the cisteine pairs with : separator.
        It is recommended to use absolute pose positions.

        The mover is ForceDisulfides

        Parameters
        ==========
        job_folder : str
            Folder path where to place the mutation job.
        cys_dic : dict
            Dictionary specify the cysteine bonds to generate. # It is better to use the pose numbering to avoid
            problems with the chains. !!!!! IMPORTANT !!!!!! Adjust first the positions in a previous function
            as in the case of mutate residue.

        relax_cycles : int
            Apply this number of relax cycles (default:0, i.e., no relax).
        nstruct : int
            Number of structures to generate when relaxing mutant
        param_files : list
            Params file to use when reading model with Rosetta.
        """

        # Check both residues are cysteine

        mpi_commands = ["slurm", "openmpi", None]
        if mpi_command not in mpi_commands:
            raise ValueError(
                "Wrong mpi_command it should either: " + " ".join(mpi_commands)
            )

        if mpi_command == "openmpi" and not isinstance(cpus, int):
            raise ValueError("")

        # Create mutation job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")
        if not os.path.exists(job_folder + "/flags"):
            os.mkdir(job_folder + "/flags")
        if not os.path.exists(job_folder + "/xml"):
            os.mkdir(job_folder + "/xml")
        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Save considered models
        considered_models = list(cys_dic.keys())
        self.saveModels(job_folder + "/input_models", models=considered_models)

        jobs = []

        # Create all-atom score function
        score_fxn_name = "ref2015"
        sfxn = rosettaScripts.scorefunctions.new_scorefunction(
            score_fxn_name, weights_file=score_fxn_name
        )

        # Create and append execution command
        if mpi_command == None:
            mpi_command = ""
        elif mpi_command == "slurm":
            mpi_command = "srun "
        elif mpi_command == "openmpi":
            mpi_command = "mpirun -np " + str(cpus) + " "
        else:
            mpi_command = mpi_command + " "

        for model in self.models_names:

            # Skip models not in given cys_dic
            if model not in considered_models:
                continue

            if not os.path.exists(job_folder + "/output_models/" + model):
                os.mkdir(job_folder + "/output_models/" + model)

            # Iterate each mutant
            for mutant in cys_dic[model]:

                # Create xml mutation (and minimization) protocol
                xml = rosettaScripts.xmlScript()
                protocol = []

                # Add score function
                xml.addScorefunction(sfxn)

                for m in cys_dic[model][mutant]:
                    disulfide = rosettaScripts.movers.ForceDisulfides(
                        name="disulfide_" + str(m[0]),
                        disulfides=m[0],
                        remove_existing=remove_existing,
                        repack=repack,
                        scorefxn=scorefxn,
                    )
                    xml.addMover(disulfide)
                    protocol.append(disulfide)

                if relax_cycles:
                    # Create fastrelax mover
                    relax = rosettaScripts.movers.fastRelax(
                        repeats=relax_cycles, scorefxn=sfxn
                    )
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
                xml.write_xml(job_folder + "/xml/" + model + "_" + mutant + ".xml")

                # Create options for minimization protocol
                flags = rosettaScripts.flags(
                    "../../xml/" + model + "_" + mutant + ".xml",
                    nstruct=nstruct,
                    s="../../input_models/" + model + ".pdb",
                    output_silent_file=model + "_" + mutant + ".out",
                )

                # Add relaxation with constraints options and write flags file
                if cst_optimization and relax_cycles:
                    flags.add_relax_cst_options()
                else:
                    flags.add_relax_options()

                # Add path to params files
                if param_files != None:
                    if not os.path.exists(job_folder + "/params"):
                        os.mkdir(job_folder + "/params")
                    if isinstance(param_files, str):
                        param_files = [param_files]
                    for param in param_files:
                        param_name = param.split("/")[-1]
                        shutil.copyfile(param, job_folder + "/params/" + param_name)
                    flags.addOption("in:file:extra_res_path", "../../params")

                flags.write_flags(
                    job_folder + "/flags/" + model + "_" + mutant + ".flags"
                )

                command = "cd " + job_folder + "/output_models/" + model + "\n"
                command += (
                    mpi_command
                    + executable
                    + " @ "
                    + "../../flags/"
                    + model
                    + "_"
                    + mutant
                    + ".flags\n"
                )
                command += "cd ../../..\n"
                jobs.append(command)

        return jobs

    def setUpRosettaOptimization(
        self,
        relax_folder,
        nstruct=1000,
        relax_cycles=5,
        cst_files=None,
        mutations=False,
        models=None,
        cst_optimization=True,
        membrane=False,
        membrane_thickness=15,
        param_files=None,
        patch_files=None,
        parallelisation="srun",
        executable="rosetta_scripts.mpi.linuxgccrelease",
        cpus=None,
        skip_finished=True,
        null=False,
        cartesian=False,
        extra_flags=None,
        sugars=False,
        symmetry=False,
        rosetta_path=None,
        ca_constraint=False,
        ligand_chain=None,
    ):
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
        if not os.path.exists(relax_folder + "/input_models"):
            os.mkdir(relax_folder + "/input_models")
        if not os.path.exists(relax_folder + "/flags"):
            os.mkdir(relax_folder + "/flags")
        if not os.path.exists(relax_folder + "/xml"):
            os.mkdir(relax_folder + "/xml")
        if not os.path.exists(relax_folder + "/output_models"):
            os.mkdir(relax_folder + "/output_models")
        if symmetry:
            if not os.path.exists(relax_folder + "/symmetry"):
                os.mkdir(relax_folder + "/symmetry")

        if parallelisation not in ["mpirun", "srun"]:
            raise ValueError("Are you sure about your parallelisation type?")

        if parallelisation == "mpirun" and cpus == None:
            raise ValueError("You must setup the number of cpus when using mpirun")
        if parallelisation == "srun" and cpus != None:
            raise ValueError(
                "CPUs can only be set up when using mpirun parallelisation!"
            )

        if cst_optimization and nstruct > 100:
            print(
                "WARNING: A large number of structures (%s) is not necessary when running constrained optimizations!"
                % nstruct
            )
            print("Consider running 100 or less structures.")

        # Save all models
        self.saveModels(relax_folder + "/input_models", models=models)

        if symmetry and rosetta_path == None:
            raise ValueError(
                "To run relax with symmetry absolute rosetta path must be given to run make_symmdef_file.pl script."
            )

        if symmetry:
            for m in self.models_names:

                # Skip models not in the given list
                if models != None:
                    if model not in models:
                        continue

                ref_chain = symmetry[m][0]
                sym_chains = " ".join(symmetry[m][1:])

                os.system(
                    rosetta_path
                    + "/main/source/src/apps/public/symmetry/make_symmdef_file.pl -p "
                    + relax_folder
                    + "/input_models/"
                    + m
                    + ".pdb -a "
                    + ref_chain
                    + " -i "
                    + sym_chains
                    + " > "
                    + relax_folder
                    + "/symmetry/"
                    + m
                    + ".symm"
                )

        # Check that sequence comparison has been done before adding mutational steps
        if mutations:
            if self.sequence_differences == {}:
                raise ValueError(
                    "Mutations have been enabled but no sequence comparison\
has been carried out. Please run compareSequences() function before setting mutation=True."
                )

        # Check if other cst files have been given.
        if ca_constraint:
            if not cst_files:
                cst_files = {}

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            if not os.path.exists(relax_folder + "/output_models/" + model):
                os.mkdir(relax_folder + "/output_models/" + model)

            if skip_finished:
                # Check if model has already been calculated and finished
                score_file = (
                    relax_folder + "/output_models/" + model + "/" + model + "_relax.sc"
                )
                if os.path.exists(score_file):
                    scores = _readRosettaScoreFile(score_file)
                    if scores.shape[0] >= nstruct:
                        continue

            if ca_constraint:
                if not os.path.exists(relax_folder + "/cst_files"):
                    os.mkdir(relax_folder + "/cst_files")

                if not os.path.exists(relax_folder + "/cst_files/" + model):
                    os.mkdir(relax_folder + "/cst_files/" + model)

                cst_file = (
                    relax_folder + "/cst_files/" + model + "/" + model + "_CA.cst"
                )
                _createCAConstraintFile(self.structures[model], cst_file)

                cst_files.setdefault(model, [])
                cst_files[model].append(cst_file)

            # Create xml minimization protocol
            xml = rosettaScripts.xmlScript()
            protocol = []

            # Create membrane scorefucntion
            if membrane:
                # Create all-atom score function
                weights_file = "mpframework_smooth_fa_2012"
                if cartesian:
                    weights_file += "_cart"

                sfxn = rosettaScripts.scorefunctions.new_scorefunction(
                    "mpframework_smooth_fa_2012", weights_file=weights_file
                )

                # Add constraint weights to membrane score function
                if cst_files != None:
                    reweights = (
                        ("chainbreak", 1.0),
                        ("coordinate_constraint", 1.0),
                        ("atom_pair_constraint", 1.0),
                        ("angle_constraint", 1.0),
                        ("dihedral_constraint", 1.0),
                        ("res_type_constraint", 1.0),
                        ("metalbinding_constraint", 1.0),
                    )

                    for rw in reweights:
                        sfxn.addReweight(rw[0], rw[1])

            # Create all-atom scorefucntion
            else:
                score_fxn_name = "ref2015"

                if cartesian:
                    score_fxn_name += "_cart"

                # Check if constraints are given
                if cst_files != None:
                    score_fxn_name += "_cst"

                # Create all-atom score function
                sfxn = rosettaScripts.scorefunctions.new_scorefunction(
                    score_fxn_name, weights_file=score_fxn_name
                )
            xml.addScorefunction(sfxn)

            # Detect symmetry if specified
            if symmetry:
                # detect_symmetry = rosettaScripts.movers.DetectSymmetry(subunit_tolerance=1, plane_tolerance=1)
                setup_symmetry = rosettaScripts.movers.SetupForSymmetry(
                    definition="../../symmetry/" + model + ".symm"
                )
                xml.addMover(setup_symmetry)
                protocol.append(setup_symmetry)

            # Create mutation movers if needed
            if mutations:
                if self.sequence_differences[model]["mutations"] != {}:
                    for m in self.sequence_differences[model]["mutations"]:
                        mutate = rosettaScripts.movers.mutate(
                            name="mutate_" + str(m[0]),
                            target_residue=m[0],
                            new_residue=PDB.Polypeptide.one_to_three(m[1]),
                        )
                        xml.addMover(mutate)
                        protocol.append(mutate)

            # Add constraint mover if constraint file is given.
            if cst_files != None:
                if model not in cst_files:
                    raise ValueError(
                        "Model %s is not in the cst_files dictionary!" % model
                    )

                if isinstance(cst_files[model], str):
                    cst_files[model] = [cst_files[model]]

                if not os.path.exists(relax_folder + "/cst_files"):
                    os.mkdir(relax_folder + "/cst_files")

                if not os.path.exists(relax_folder + "/cst_files/" + model):
                    os.mkdir(relax_folder + "/cst_files/" + model)

                for cst_file in cst_files[model]:

                    cst_file_name = cst_file.split("/")[-1]

                    if not os.path.exists(
                        relax_folder + "/cst_files/" + model + "/" + cst_file_name
                    ):
                        shutil.copyfile(
                            cst_file,
                            relax_folder + "/cst_files/" + model + "/" + cst_file_name,
                        )

                    set_cst = rosettaScripts.movers.constraintSetMover(
                        add_constraints=True,
                        cst_file="../../cst_files/" + model + "/" + cst_file_name,
                    )
                xml.addMover(set_cst)
                protocol.append(set_cst)

            if membrane:
                add_membrane = rosettaScripts.rosetta_MP.movers.addMembraneMover()
                xml.addMover(add_membrane)
                protocol.append(add_membrane)

                init_membrane = (
                    rosettaScripts.rosetta_MP.movers.membranePositionFromTopologyMover()
                )
                xml.addMover(init_membrane)
                protocol.append(init_membrane)

            # Create fastrelax mover
            relax = rosettaScripts.movers.fastRelax(repeats=relax_cycles, scorefxn=sfxn)
            xml.addMover(relax)

            if not null:
                protocol.append(relax)

                # Set protocol
                xml.setProtocol(protocol)

            # Add scorefunction output
            xml.addOutputScorefunction(sfxn)

            # Write XMl protocol file
            xml.write_xml(relax_folder + "/xml/" + model + "_relax.xml")

            if symmetry:
                input_model = model + "_INPUT.pdb"
            else:
                input_model = model + ".pdb"

            # Create options for minimization protocol
            flags = rosettaScripts.flags(
                "../../xml/" + model + "_relax.xml",
                nstruct=nstruct,
                s="../../input_models/" + input_model,
                output_silent_file=model + "_relax.out",
            )

            # Add extra flags
            if extra_flags != None:
                for o in extra_flags:
                    if isinstance(o, tuple):
                        flags.addOption(*o)
                    else:
                        flags.addOption(o)

            # Add relaxation with constraints options and write flags file
            if cst_optimization:
                flags.add_relax_cst_options()
            else:
                flags.add_relax_options()

            # Add path to params files
            if param_files != None:

                if not os.path.exists(relax_folder + "/params"):
                    os.mkdir(relax_folder + "/params")

                if isinstance(param_files, str):
                    param_files = [param_files]

                patch_line = ""
                for param in param_files:
                    param_name = param.split("/")[-1]
                    shutil.copyfile(param, relax_folder + "/params/" + param_name)
                    if not param_name.endswith(".params"):
                        patch_line += "../../params/" + param_name + " "
                flags.addOption("in:file:extra_res_path", "../../params")
                if patch_line != "":
                    flags.addOption("in:file:extra_patch_fa", patch_line)

            if membrane:
                flags.addOption("mp::setup::spans_from_structure", "true")
                flags.addOption("relax:constrain_relax_to_start_coords")

            if sugars:
                flags.addOption("include_sugars")
                flags.addOption("alternate_3_letter_codes", "pdb_sugar")
                flags.addOption("write_glycan_pdb_codes")
                flags.addOption("auto_detect_glycan_connections")
                flags.addOption("maintain_links")

            flags.write_flags(relax_folder + "/flags/" + model + "_relax.flags")

            # Create and append execution command
            command = "cd " + relax_folder + "/output_models/" + model + "\n"
            if parallelisation == "mpirun":
                if cpus == 1:
                    command += (
                        executable + " @ " + "../../flags/" + model + "_relax.flags\n"
                    )
                else:
                    command += (
                        "mpirun -np "
                        + str(cpus)
                        + " "
                        + executable
                        + " @ "
                        + "../../flags/"
                        + model
                        + "_relax.flags\n"
                    )
            else:
                command += (
                    "srun "
                    + executable
                    + " @ "
                    + "../../flags/"
                    + model
                    + "_relax.flags\n"
                )
            command += "cd ../../..\n"
            jobs.append(command)

        return jobs

    def setUpMembranePositioning(self, job_folder, membrane_thickness=15, models=None):
        """ """
        # Create minimization job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)
        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")
        if not os.path.exists(job_folder + "/flags"):
            os.mkdir(job_folder + "/flags")
        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Save all models
        self.saveModels(job_folder + "/input_models", models=models)

        # Copy embeddingToMembrane.py script
        _copyScriptFile(job_folder, "embeddingToMembrane.py")

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            if not os.path.exists(job_folder + "/output_models/" + model):
                os.mkdir(job_folder + "/output_models/" + model)

            flag_file = job_folder + "/flags/mp_span_from_pdb_" + model + ".flags"
            with open(flag_file, "w") as flags:
                flags.write("-mp::thickness " + str(membrane_thickness) + "\n")
                flags.write("-s model.pdb\n")
                flags.write("-out:path:pdb .\n")

            # flag_file = job_folder+'/flags/mp_transform_'+model+'.flags'
            # with open(flag_file, 'w') as flags:
            #     flags.write('-s ../../input_models/'+model+'.pdb\n')
            #     flags.write('-mp:transform:optimize_embedding true\n')
            #     flags.write('-mp:setup:spanfiles '+model+'.span\n')
            #     flags.write('-out:no_nstruct_label\n')

            command = "cd " + job_folder + "/output_models/" + model + "\n"
            command += "cp ../../input_models/" + model + ".pdb model.pdb\n"
            command += (
                "mp_span_from_pdb.linuxgccrelease @ ../../flags/mp_span_from_pdb_"
                + model
                + ".flags\n"
            )
            # command += 'rm model.pdb \n'
            command += "mv model.pdb " + model + ".pdb\n"
            command += "mv model.span " + model + ".span\n"
            # command += 'mp_transform.linuxgccrelease @ ../../flags/mp_transform_'+model+'.flags\n'
            # command += 'python ../../._embeddingToMembrane.py'+' '+model+'.pdb\n'
            command += "cd ../../..\n"
            jobs.append(command)

        return jobs

    def addMissingLoops(
        self, job_folder, nstruct=1, sfxn="ref2015", param_files=None, idealize=True
    ):
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
        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")
        if not os.path.exists(job_folder + "/flags"):
            os.mkdir(job_folder + "/flags")
        if not os.path.exists(job_folder + "/xml"):
            os.mkdir(job_folder + "/xml")
        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Save all models
        self.saveModels(job_folder + "/input_models")

        # Check that sequence comparison has been done before checking missing loops
        if self.sequence_differences == {}:
            raise ValueError(
                "No sequence comparison has been carried out. Please run \
compareSequences() function before adding missing loops."
            )

        # Create flags files
        jobs = []
        for model in self.models_names:

            # Check that model has missing loops
            if self.sequence_differences[model]["missing_loops"] != []:

                missing_loops = self.sequence_differences[model]["missing_loops"]

                for loop in missing_loops:

                    loop_name = str(loop[0]) + "_" + str(loop[1])

                    if not os.path.exists(job_folder + "/output_models/" + model):
                        os.mkdir(job_folder + "/output_models/" + model)

                    if not os.path.exists(
                        job_folder + "/output_models/" + model + "/" + loop_name
                    ):
                        os.mkdir(
                            job_folder + "/output_models/" + model + "/" + loop_name
                        )

                    # Create xml minimization protocol
                    xml = rosettaScripts.xmlScript()
                    protocol = []

                    # Create score function

                    scorefxn = rosettaScripts.scorefunctions.new_scorefunction(
                        sfxn, weights_file=sfxn
                    )
                    # Add loop remodel protocol
                    if len(loop[1]) == 1:
                        hanging_residues = 3
                    elif len(loop[1]) == 2:
                        hanging_residues = 2
                    else:
                        hanging_residues = 1
                    loop_movers = rosettaScripts.loop_modeling.loopRebuild(
                        xml,
                        loop[0],
                        loop[1],
                        scorefxn=sfxn,
                        hanging_residues=hanging_residues,
                    )
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
                    xml.write_xml(
                        job_folder + "/xml/" + model + "_" + loop_name + ".xml"
                    )

                    # Create options for minimization protocol
                    output_silent = (
                        "output_models/"
                        + model
                        + "/"
                        + loop_name
                        + "/"
                        + model
                        + "_"
                        + loop_name
                        + ".out"
                    )
                    flags = rosettaScripts.flags(
                        "xml/" + model + "_" + loop_name + ".xml",
                        nstruct=nstruct,
                        s="input_models/" + model + ".pdb",
                        output_silent_file=output_silent,
                    )

                    # Add path to params files
                    if param_files != None:
                        if not os.path.exists(job_folder + "/params"):
                            os.mkdir(job_folder + "/params")

                        if isinstance(param_files, str):
                            param_files = [param_files]
                        for param in param_files:
                            param_name = param.split("/")[-1]
                            shutil.copyfile(param, job_folder + "/params/" + param_name)
                        flags.addOption("in:file:extra_res_path", "params")

                    # Write flags file
                    flags.write_flags(
                        job_folder + "/flags/" + model + "_" + loop_name + ".flags"
                    )

                    # Create and append execution command
                    command = "cd " + job_folder + "\n"
                    command += (
                        "srun rosetta_scripts.mpi.linuxgccrelease @ "
                        + "flags/"
                        + model
                        + "_"
                        + loop_name
                        + ".flags\n"
                    )
                    command += "cd ..\n"

                    jobs.append(command)

        return jobs

    def setUpPrepwizardOptimization(
        self,
        prepare_folder,
        pH=7.0,
        epik_pH=False,
        samplewater=False,
        models=None,
        epik_pHt=False,
        remove_hydrogens=False,
        delwater_hbond_cutoff=False,
        fill_loops=False,
        protonation_states=None,
        noepik=False,
        mae_input=False,
        noprotassign=False,
        use_new_version=False,
        replace_symbol=None,
        captermini=False,
        keepfarwat=False,
        skip_finished=False,
        **kwargs,
    ):
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
        if not os.path.exists(prepare_folder + "/input_models"):
            os.mkdir(prepare_folder + "/input_models")
        if not os.path.exists(prepare_folder + "/output_models"):
            os.mkdir(prepare_folder + "/output_models")

        # Save all input models
        self.saveModels(
            prepare_folder + "/input_models",
            convert_to_mae=mae_input,
            remove_hydrogens=remove_hydrogens,
            replace_symbol=replace_symbol,
            models=models,
        )  # **kwargs)

        # Generate jobs
        jobs = []
        for model in self.models_names:

            if models != None and model not in models:
                continue

            if replace_symbol:
                model_name = model.replace(replace_symbol[0], replace_symbol[1])
            else:
                model_name = model

            output_path = (
                prepare_folder
                + "/output_models/"
                + model_name
                + "/"
                + model_name
                + ".pdb"
            )
            if skip_finished and os.path.exists(output_path):
                continue

            if fill_loops:
                if model not in self.target_sequences:
                    raise ValueError(
                        "Target sequence for model %s was not given. First\
make sure of reading the target sequences with the function readTargetSequences()"
                        % model
                    )
                sequence = {}
                sequence[model] = self.target_sequences[model]
                fasta_file = prepare_folder + "/input_models/" + model_name + ".fasta"
                alignment.writeFastaFile(sequence, fasta_file)

            # Create model output folder
            output_folder = prepare_folder + "/output_models/" + model_name
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            if fill_loops:
                command = "cd " + prepare_folder + "/input_models/\n"
                command += "pwd=$(pwd)\n"
                command += "cd ../output_models/" + model_name + "\n"
            else:
                command = "cd " + output_folder + "\n"

            command += '"${SCHRODINGER}/utilities/prepwizard" '
            if mae_input:
                command += "../../input_models/" + model_name + ".mae "
            else:
                command += "../../input_models/" + model_name + ".pdb "
            command += model_name + ".pdb "
            command += "-fillsidechains "
            command += "-disulfides "
            if keepfarwat:
                command += "-keepfarwat "
            if fill_loops:
                command += "-fillloops "
                command += '-fasta_file "$pwd"/' + model_name + ".fasta "
            if captermini:
                command += "-captermini "
            if remove_hydrogens:
                command += "-rehtreat "
            if noepik:
                command += "-noepik "
            if noprotassign:
                command += "-noprotassign "
            else:
                if epik_pH:
                    command += "-epik_pH " + str(pH) + " "
                if epik_pHt:
                    command += "-epik_pHt " + str(epik_pHt) + " "
            command += "-propka_pH " + str(pH) + " "
            command += "-f 2005 "
            command += "-rmsd 0.3 "
            if samplewater:
                command += "-samplewater "
            if delwater_hbond_cutoff:
                command += "-delwater_hbond_cutoff " + str(delwater_hbond_cutoff) + " "

            if not isinstance(protonation_states, type(None)):
                for ps in protonation_states[model]:
                    if use_new_version:
                        command += "-force " + str(ps[0]) + " " + str(ps[1]) + " "
                    else:
                        command += "-force " + str(ps[0]) + " " + str(ps[1]) + " "

            command += "-JOBNAME " + model_name + " "
            command += "-HOST localhost:1 "
            command += "-WAIT\n"
            command += "cd ../../..\n"
            jobs.append(command)

        return jobs

    def setUpDockingGrid(
        self,
        grid_folder,
        center_atoms,
        innerbox=(10, 10, 10),
        outerbox=(30, 30, 30),
        useflexmae=True,
        peptide=False,
        mae_input=True,
        cst_positions=None,
        models=None,
        skip_finished=False,
    ):
        """
        Setup grid calculation for each model.

        Parameters
        ==========
        grid_folder : str
            Path to grid calculation folder
        center_atoms : tuple
            Atoms to center the grid box.
        cst_positions : dict
            atom and radius for cst position for each model:
            cst_positions = {
            model : ((chain_id, residue_index, atom_name), radius), ...
            }
        """

        # Create grid job folders
        if not os.path.exists(grid_folder):
            os.mkdir(grid_folder)

        if not os.path.exists(grid_folder + "/input_models"):
            os.mkdir(grid_folder + "/input_models")

        if not os.path.exists(grid_folder + "/grid_inputs"):
            os.mkdir(grid_folder + "/grid_inputs")

        if not os.path.exists(grid_folder + "/output_models"):
            os.mkdir(grid_folder + "/output_models")

        # Save all input models
        self.saveModels(grid_folder + "/input_models", convert_to_mae=mae_input)

        # Check that inner and outerbox values are given as integers
        for v in innerbox:
            if type(v) != int:
                raise ValueError("Innerbox values must be given as integers")
        for v in outerbox:
            if type(v) != int:
                raise ValueError("Outerbox values must be given as integers")

        # Create grid input files
        jobs = []
        for model in self.models_names:

            if models != None:
                if model not in models:
                    continue

            # Check if output grid exists
            output_path = grid_folder + "/output_models/" + model + ".zip"
            if skip_finished and os.path.exists(output_path):
                continue

            if all([isinstance(x, (float, int)) for x in center_atoms[model]]):
                x = float(center_atoms[model][0])
                y = float(center_atoms[model][1])
                z = float(center_atoms[model][2])

            else:
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

            if cst_positions != None:

                cst_x = {}
                cst_y = {}
                cst_z = {}

                # Convert to a list of only one position cst is given
                if isinstance(cst_positions[model], tuple):
                    cst_positions[model] = [cst_positions[model]]

                for i, position in enumerate(cst_positions[model]):

                    # Get coordinates of center residue
                    chainid = position[0][0]
                    resid = position[0][1]
                    atom_name = position[0][2]

                    for c in self.structures[model].get_chains():
                        if c.id == chainid:
                            for r in c.get_residues():
                                if r.id[1] == resid:
                                    for a in r.get_atoms():
                                        if a.name == atom_name:
                                            cst_x[i + 1] = a.coord[0]
                                            cst_y[i + 1] = a.coord[1]
                                            cst_z[i + 1] = a.coord[2]

            # Check if any atom center was found.
            if x == None:
                raise ValueError("Given atom center not found for model %s" % model)

            # Check if any atom center was found.
            if cst_positions and cst_x == {}:
                raise ValueError("Given atom constraint not found for model %s" % model)

            # Write grid input file
            with open(grid_folder + "/grid_inputs/" + model + ".in", "w") as gif:
                gif.write(
                    "GRID_CENTER %.14f, %.14f, %.14f\n"
                    % (
                        x,
                        y,
                        z,
                    )
                )
                gif.write("GRIDFILE " + model + ".zip\n")
                gif.write("INNERBOX %s, %s, %s\n" % innerbox)
                gif.write("OUTERBOX %s, %s, %s\n" % outerbox)

                if cst_positions != None:
                    for i, position in enumerate(cst_positions[model]):
                        gif.write(
                            'POSIT_CONSTRAINTS "position'
                            + str(i + 1)
                            + ' %.14f %.14f %.14f %.14f",\n'
                            % (cst_x[i + 1], cst_y[i + 1], cst_z[i + 1], position[-1])
                        )

                if mae_input:
                    gif.write("RECEP_FILE %s\n" % ("../input_models/" + model + ".mae"))
                else:
                    gif.write("RECEP_FILE %s\n" % ("../input_models/" + model + ".pdb"))
                if peptide:
                    gif.write("PEPTIDE True\n")
                if useflexmae:
                    gif.write("USEFLEXMAE YES\n")

            command = "cd " + grid_folder + "/output_models\n"

            # Add grid generation command
            command += '"${SCHRODINGER}/glide" '
            command += "../grid_inputs/" + model + ".in" + " "
            command += "-OVERWRITE "
            command += "-HOST localhost "
            command += "-TMPLAUNCHDIR "
            command += "-WAIT\n"
            command += "cd ../..\n"

            jobs.append(command)

        return jobs

    def setUpGlideDocking(
        self,
        docking_folder,
        grids_folder,
        ligands_folder,
        models=None,
        poses_per_lig=100,
        precision="SP",
        use_ligand_charges=False,
        energy_by_residue=False,
        use_new_version=False,
        cst_fragments=None,
        skip_finished=None,
    ):
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
        cst_fragments: dict
            Dictionary with a tuple composed by 3 elements: first, the fragment of the ligand
            to which you want to make the bias (in SMARTS pattern nomenclature), the number of feature,
            if you only have 1 constraint on the grid and in the ligand, it will be 1, and True/False
            depending on if you want to include that constraint inside the grid constraint (True) or
            exclude it (False).
            {'model': {'ligand': ('[fragment]', feature, True) ...
            }
        """

        # Create docking job folders
        if not os.path.exists(docking_folder):
            os.mkdir(docking_folder)

        if not os.path.exists(docking_folder + "/input_models"):
            os.mkdir(docking_folder + "/input_models")

        if not os.path.exists(docking_folder + "/output_models"):
            os.mkdir(docking_folder + "/output_models")

        # Save all input models
        self.saveModels(docking_folder + "/input_models")

        # Read paths to grid files
        grids_paths = {}
        for f in os.listdir(grids_folder + "/output_models"):
            if f.endswith(".zip"):
                name = f.replace(".zip", "")
                grids_paths[name] = grids_folder + "/output_models/" + f

        # Read paths to substrates
        substrates_paths = {}
        for f in os.listdir(ligands_folder):
            if f.endswith(".mae"):
                name = f.replace(".mae", "")
                substrates_paths[name] = ligands_folder + "/" + f

        # Set up docking jobs
        jobs = []
        for grid in grids_paths:

            # Skip if models are given and not in models
            if models != None:
                if grid not in models:
                    continue

            # Create ouput folder
            if not os.path.exists(docking_folder + "/output_models/" + grid):
                os.mkdir(docking_folder + "/output_models/" + grid)

            for substrate in substrates_paths:

                output_path = docking_folder+"/output_models/"+grid+'/'+grid+"_"+ substrate+'_pv.maegz'
                if skip_finished and os.path.exists(output_path):
                    continue

                # Create glide dock input
                with open(
                    docking_folder
                    + "/output_models/"
                    + grid
                    + "/"
                    + grid
                    + "_"
                    + substrate
                    + ".in",
                    "w",
                ) as dif:
                    dif.write("GRIDFILE GRID_PATH/" + grid + ".zip\n")
                    dif.write("LIGANDFILE ../../../%s\n" % substrates_paths[substrate])
                    dif.write("POSES_PER_LIG %s\n" % poses_per_lig)
                    if use_ligand_charges:
                        dif.write("LIG_MAECHARGES true\n")
                    dif.write("PRECISION %s\n" % precision)
                    if energy_by_residue:
                        dif.write("WRITE_RES_INTERACTION true\n")
                    # Constraints
                    if cst_fragments != None:

                        if isinstance(cst_fragments[grid][substrate], tuple):
                            cst_fragments[grid][substrate] = [
                                cst_fragments[grid][substrate]
                            ]

                        for i, fragment in enumerate(cst_fragments[grid][substrate]):
                            dif.write("[CONSTRAINT_GROUP:1]\n")
                            dif.write("\tUSE_CONS position1:" + str(i + 1) + ",\n")
                            dif.write("\tNREQUIRED_CONS ALL\n")
                            dif.write("[FEATURE:" + str(i + 1) + "]\n")
                            dif.write(
                                '\tPATTERN1 "' + fragment[0] + " " + str(fragment[1])
                            )
                            if fragment[2]:
                                dif.write(" include")
                            dif.write('"\n')

                # Create commands
                command = "cd " + docking_folder + "/output_models/" + grid + "\n"

                # Schrodinger has problem with relative paths to the grid files
                # This is a quick fix for that (not elegant, but works).
                command += "cwd=$(pwd)\n"
                grid_folder = "/".join(grids_paths[grid].split("/")[:-1])
                command += "cd ../../../%s\n" % grid_folder
                command += "gd=$(pwd)\n"
                command += "cd $cwd\n"
                command += 'sed -i "s@GRID_PATH@$gd@" %s \n' % (
                    grid + "_" + substrate + ".in"
                )

                # Add docking command
                command += '"${SCHRODINGER}/glide" '
                command += grid + "_" + substrate + ".in" + " "
                command += "-OVERWRITE "
                command += "-adjust "
                command += "-HOST localhost:1 "
                command += "-TMPLAUNCHDIR "
                command += "-WAIT\n"
                command += "cd ../../..\n"
                jobs.append(command)

        return jobs

    def setUpSiteMapForModels(
        self,
        job_folder,
        target_residues,
        site_box=10,
        enclosure=0.5,
        maxvdw=1.1,
        resolution="fine",
        reportsize=100,
        overwrite=False,
        maxdist=8.0,
        sidechain=True,
        only_models=None,
        replace_symbol=None,
        write_conect_lines=True,
    ):
        """
        Generates a SiteMap calculation for model poses (no ligand) near specified residues.
        Parameters
        ==========
        job_folder : str
            Path to the calculation folder
        target_residues : dict
            Dictionary per model with a list of lists of residues (chain_id, residues) for which
            to calculate sitemap pockets.
        replace_symbol : str
            Symbol to replace for saving the models
        """

        # Create site map job folders
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")

        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Copy script to generate protein and ligand mae inputs, separately.
        _copyScriptFile(job_folder, "prepareForSiteMap.py")
        script_path = job_folder + "/._prepareForSiteMap.py"

        # Save all input models
        self.saveModels(
            job_folder + "/input_models",
            write_conect_lines=write_conect_lines,
            replace_symbol=replace_symbol,
        )

        # Create input files
        jobs = []
        for model in self.models_names:

            # Skip models not in only_models list
            if only_models != None:
                if model not in only_models:
                    continue

            if replace_symbol:
                model_name = model.replace(replace_symbol[0], replace_symbol[1])
            else:
                model_name = model

            # Create an output folder for each model
            output_folder = job_folder + "/output_models/" + model_name
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            # Generate input protein files
            input_protein = job_folder + "/input_models/" + model_name + ".pdb"

            input_mae = (
                job_folder
                + "/output_models/"
                + model_name
                + "/"
                + model_name
                + "_protein.mae"
            )
            if not os.path.exists(input_mae) or overwrite:
                command = "run " + script_path + " "
                command += input_protein + " "
                command += output_folder + " "
                command += "--protein_only "
                os.system(command)

            if not isinstance(target_residues, dict):
                raise ValueError("Problem: target_residues must be a dictionary!")

            if model not in target_residues:
                raise ValueError(
                    f"Problem: model {model} not found in target_residues dictionary!"
                )

            elif isinstance(target_residues[model], (str, tuple)):
                target_residues[model] = [target_residues[model]]

            elif isinstance(target_residues[model][0], (str, tuple)):
                target_residues[model] = [target_residues[model]]

            for residue_selection in target_residues[model]:

                label = ""
                for r in residue_selection:
                    label += "".join([str(x) for x in r]) + "_"
                label = label[:-1]

                # Create folder
                if not os.path.exists(output_folder + "/" + label):
                    os.mkdir(output_folder + "/" + label)

                # Add site map command
                command = (
                    "cd "
                    + job_folder
                    + "/output_models/"
                    + model_name
                    + "/"
                    + label
                    + "\n"
                )
                command += '"${SCHRODINGER}/sitemap" '
                command += "-j " + model_name + " "
                command += "-prot ../" + model_name + "_protein.mae" + " "
                command += "-sitebox " + str(site_box) + " "
                command += "-resolution " + str(resolution) + " "
                command += "-keepvolpts yes "
                command += "-keeplogs yes "
                # command += '-maxdist '+str(maxdist)+' '
                # command += '-enclosure '+str(enclosure)+' '
                # command += '-maxvdw '+str(maxvdw)+' '
                command += "-reportsize " + str(reportsize) + " "

                command += '-siteasl "'
                for r in residue_selection:
                    if isinstance(r, tuple) and len(r) == 2:
                        command += (
                            "(chain.name "
                            + str(r[0])
                            + " and res.num {"
                            + str(r[1])
                            + "} "
                        )
                    elif isinstance(r, str) and len(r) == 1:
                        command += '"(chain.name ' + str(r[0]) + " "
                    else:
                        raise ValueError("Incorrect residue definition!")
                    if sidechain:
                        command += "and not (atom.pt ca,c,n,h,o)) or "
                    else:
                        command += ") or "
                command = command[:-4] + '" '
                command += "-HOST localhost:1 "
                command += "-TMPLAUNCHDIR "
                command += "-WAIT\n"
                command += "cd ../../../..\n"
                jobs.append(command)

        return jobs

    def setUpSiteMapForLigands(
        self, job_folder, poses_folder, site_box=10, resolution="fine", overwrite=False
    ):
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

        if not os.path.exists(job_folder + "/input_models"):
            os.mkdir(job_folder + "/input_models")

        if not os.path.exists(job_folder + "/output_models"):
            os.mkdir(job_folder + "/output_models")

        # Copy script to generate protein and ligand mae inputs, separately.
        _copyScriptFile(job_folder, "prepareForSiteMap.py")
        script_path = job_folder + "/._prepareForSiteMap.py"

        # Create input files
        jobs = []
        for model in os.listdir(poses_folder):
            if not os.path.isdir(poses_folder + "/" + model):
                continue
            if not os.path.exists(job_folder + "/input_models/" + model):
                os.mkdir(job_folder + "/input_models/" + model)
            if not os.path.exists(job_folder + "/output_models/" + model):
                os.mkdir(job_folder + "/output_models/" + model)

            for pose in os.listdir(poses_folder + "/" + model):
                if pose.endswith(".pdb"):
                    pose_name = pose.replace(".pdb", "")

                    # Generate input protein and ligand files
                    input_ligand = (
                        job_folder
                        + "/input_models/"
                        + model
                        + "/"
                        + pose_name
                        + "_ligand.mae"
                    )
                    input_protein = (
                        job_folder
                        + "/input_models/"
                        + model
                        + "/"
                        + pose_name
                        + "_protein.mae"
                    )
                    if (
                        not os.path.exists(input_ligand)
                        or not os.path.exists(input_protein)
                        or overwrite
                    ):
                        command = "run " + script_path + " "
                        command += poses_folder + "/" + model + "/" + pose + " "
                        command += job_folder + "/input_models/" + model
                        os.system(command)

                    # Write Site Map input file
                    with open(
                        job_folder
                        + "/output_models/"
                        + model
                        + "/"
                        + pose_name
                        + ".in",
                        "w",
                    ) as smi:
                        smi.write(
                            "PROTEIN ../../input_models/"
                            + model
                            + "/"
                            + pose_name
                            + "_protein.mae\n"
                        )
                        smi.write(
                            "LIGMAE ../../input_models/"
                            + model
                            + "/"
                            + pose_name
                            + "_ligand.mae\n"
                        )
                        smi.write("SITEBOX " + str(site_box) + "\n")
                        smi.write("RESOLUTION " + resolution + "\n")
                        smi.write("REPORTSIZE 100\n")
                        smi.write("KEEPVOLPTS yes\n")
                        smi.write("KEEPLOGS yes\n")

                    # Add site map command
                    command = "cd " + job_folder + "/output_models/" + model + "\n"
                    command += '"${SCHRODINGER}/sitemap" '
                    command += pose_name + ".in" + " "
                    command += "-HOST localhost:1 "
                    command += "-TMPLAUNCHDIR "
                    command += "-WAIT\n"
                    command += "cd ../../..\n"
                    jobs.append(command)
        return jobs

    def analyseSiteMapCalculation(
        self,
        sitemap_folder,
        failed_value=0,
        verbose=True,
        output_models=None,
        replace_symbol=None,
    ):
        """
        Extract score values from a site map calculation.
         Parameters
         ==========
         sitemap_folder : str
             Path to the site map calculation folder. See
             setUpSiteMapForModels()
        failed_value : None, float or int
            The value to put in the columns of failed siteMap calculations.
        output_models : str
            Folder to combine models with sitemap points

        Returns
        =======
        sitemap_data : pandas.DataFrame
            Site map pocket information.
        """

        def parseVolumeInfo(log_file):
            """
            Parse eval log file for site scores.
            Parameters
            ==========
            eval_log : str
                Eval log file from sitemap output
            Returns
            =======
            pocket_data : dict
                Scores for the given pocket
            """
            with open(log_file) as lf:
                c = False
                for l in lf:
                    if l.startswith("SiteScore"):
                        c = True
                        labels = l.split()
                        continue
                    if c:
                        values = [float(x) for x in l.split()]
                        pocket_data = {x: y for x, y in zip(labels, values)}
                        c = False
            return pocket_data

        def checkIfCompleted(log_file):
            """
            Check log file for calculation completition.
            Parameters
            ==========
            log_file : str
                Path to the standard sitemap log file.
            Returns
            =======
            completed : bool
                Did the simulation end correctly?
            """
            with open(log_file) as lf:
                for l in lf:
                    if "SiteMap successfully completed" in l:
                        return True
            return False

        def checkIfFound(log_file):
            """
            Check log file for found sites.
            Parameters
            ==========
            log_file : str
                Path to the standard sitemap log file.
            Returns
            =======
            found : bool
                Did the simulation end correctly?
            """
            found = True
            with open(log_file) as lf:
                for l in lf:
                    if "No sites found" in l or "no sites were found" in l:
                        found = False
            return found

        if (
            replace_symbol
            and not isinstance(replace_symbol, tuple)
            and not len(replace_symbol) == 2
        ):
            raise ValueError(
                "replace_symbol must be a tuple: (old_symbol,  new_symbol)"
            )

        sitemap_data = {}
        sitemap_data["Model"] = []
        sitemap_data["Pocket"] = []

        input_folder = sitemap_folder + "/input_models"
        output_folder = sitemap_folder + "/output_models"

        if output_models:
            if not os.path.exists(output_models):
                os.mkdir(output_models)

        for model in os.listdir(output_folder):

            if replace_symbol:
                model_name = model.replace(replace_symbol[1], replace_symbol[0])
            else:
                model_name = model

            for r in os.listdir(output_folder + "/" + model):

                # Check if chain or residue was given
                if len(r) == 1:
                    pocket_type = "chain"
                else:
                    pocket_type = "residue"

                if os.path.isdir(output_folder + "/" + model + "/" + r):
                    log_file = (
                        output_folder + "/" + model + "/" + r + "/" + model + ".log"
                    )
                    if os.path.exists(log_file):
                        completed = checkIfCompleted(log_file)
                    else:
                        if verbose:
                            message = (
                                "Log file for model %s and "
                                + pocket_type
                                + " %s was not found!\n" % (model, r)
                            )
                            message += "It seems the calculation has not run yet..."
                            print(message)
                        continue

                    if not completed:
                        if verbose:
                            print(
                                "There was a problem with model %s and "
                                + pocket_type
                                + " %s" % (model, r)
                            )
                        continue
                    else:
                        found = checkIfFound(log_file)
                        if not found:
                            if verbose:
                                print(
                                    "No sites were found for model %s and "
                                    + pocket_type
                                    + " %s" % (model, r)
                                )
                            continue

                    pocket = r
                    pocket_data = parseVolumeInfo(log_file)

                    sitemap_data["Model"].append(model_name)
                    sitemap_data["Pocket"].append(pocket)

                    for l in pocket_data:
                        sitemap_data.setdefault(l, [])
                        sitemap_data[l].append(pocket_data[l])

                    if output_models:
                        print("Storing Volume Points models at %s" % output_models)
                        input_file = input_folder + "/" + model + ".pdb"
                        volpoint_file = (
                            output_folder
                            + "/"
                            + model
                            + "/"
                            + r
                            + "/"
                            + model
                            + "_site_1_volpts.pdb"
                        )
                        if os.path.exists(volpoint_file):

                            istruct = _readPDB(model + "_input", input_file)
                            imodel = [x for x in istruct.get_models()][0]
                            vstruct = _readPDB(model + "_volpts", volpoint_file)
                            vpt_chain = PDB.Chain.Chain("V")
                            for r in vstruct.get_residues():
                                vpt_chain.add(r)
                            imodel.add(vpt_chain)

                            _saveStructureToPDB(
                                istruct, output_models + "/" + model_name + "_vpts.pdb"
                            )
                        else:
                            print(
                                "Volume points PDB not found for model %s and residue %s"
                                % (m, r)
                            )

        sitemap_data = pd.DataFrame(sitemap_data)
        sitemap_data.set_index(["Model", "Pocket"], inplace=True)

        return sitemap_data

    def definePocketResiduesWithSiteMap(
        self,
        volpts_models,
        distance_to_points=2.5,
        only_models=None,
        output_file=None,
        overwrite=False,
        replace_symbol=None,
    ):
        """
        Calculates the active site residues based on the volume points from a sitemap
        calcualtion. The models should be written with the option output_models from
        the analiseSiteMapCalculation() function.

        Parameters
        ==========
        volpts_models : str
            Path to the folder where models containing the sitemap volume points residues.
        only_models : (str, list)
            Specific models to be processed, if None all the models loaded in this class
            will be used
        distance_to_points : float
            The distance to consider a residue in contact with the volume point atoms.
        output_file : str
            Path the json output file to store the residue data
        overwrite : bool
            Overwrite json file if found? (essentially, calculate all again)
        """

        if output_file == None:
            raise ValueError("An ouput file name must be given")
        if not output_file.endswith(".json"):
            output_file = output_file + ".json"

        if (
            replace_symbol
            and not isinstance(replace_symbol, tuple)
            and not len(replace_symbol) == 2
        ):
            raise ValueError(
                "replace_symbol must be a tuple: (old_symbol,  new_symbol)"
            )

        if not os.path.exists(output_file) or overwrite:

            residues = {}
            for model in self:

                # Skip models not in only_models list
                if only_models != None:
                    if model not in only_models:
                        continue

                if replace_symbol:
                    model_name = model.replace(replace_symbol[0], replace_symbol[1])
                else:
                    model_name = model

                # Check if the volume points model file exists
                volpts_file = volpts_models + "/" + model_name + "_vpts.pdb"
                if not os.path.exists(volpts_file):
                    print(
                        "Model %s not found in the volume points folder %s!"
                        % (model, volpts_models)
                    )

                traj = md.load(volpts_file)
                protein = traj.topology.select("protein and not resname vpt")
                vpts = traj.topology.select("resname vpt")
                n = md.compute_neighbors(
                    traj, distance_to_points / 10, vpts, haystack_indices=protein
                )
                residues[model] = list(
                    set(
                        [
                            traj.topology.atom(i).residue.resSeq
                            for i in n[0]
                            if traj.topology.atom(i).is_sidechain
                        ]
                    )
                )

            with open(output_file, "w") as jf:
                json.dump(residues, jf)

        else:
            with open(output_file) as jf:
                residues = json.load(jf)

        for model in residues:
            residues[model] = np.array(list(residues[model]))

        return residues

    def getInContactResidues(
        self,
        residue_selection,
        distance_threshold=2.5,
        sidechain_selection=False,
        return_residues=False,
        only_protein=False,
        sidechain=False,
        backbone=False,
    ):
        """
        Get residues in close contact to a residue selection
        """

        in_contact = {}
        for model in self:

            # Get structure coordinates
            structure = self.structures[model]
            selected_coordinates = _getStructureCoordinates(
                structure,
                sidechain=sidechain_selection,
                only_residues=residue_selection[model],
            )
            selected_atoms = _getStructureCoordinates(
                structure,
                sidechain=sidechain_selection,
                only_residues=residue_selection[model],
                return_atoms=True,
            )

            other_coordinates = _getStructureCoordinates(
                structure,
                sidechain=sidechain,
                exclude_residues=residue_selection[model],
            )
            other_atoms = _getStructureCoordinates(
                structure,
                sidechain=sidechain,
                exclude_residues=residue_selection[model],
                return_atoms=True,
            )

            if selected_coordinates.size == 0:
                raise ValueError(
                    f"Problem matching the given residue selection for model {model}"
                )

            # Compute the distance matrix between the two set of coordinates
            M = distance_matrix(selected_coordinates, other_coordinates)
            in_contact[model] = np.array(other_atoms)[
                np.argwhere(M <= distance_threshold)[:, 1]
            ]
            in_contact[model] = [tuple(a) for a in in_contact[model]]

            # Only return tuple residues
            if return_residues:
                residues = []
                for atom in in_contact[model]:
                    residues.append(tuple(atom[:2]))
                in_contact[model] = list(set(residues))

        return in_contact

    def setUpLigandParameterization(
        self,
        job_folder,
        ligands_folder,
        charge_method=None,
        only_ligands=None,
        rotamer_resolution=10,
    ):
        """
        Run PELE platform for ligand parameterization
        Parameters
        ==========
        job_folder : str
            Path to the job input folder
        ligands_folder : str
            Path to the folder containing the ligand molecules in PDB format.
        """

        charge_methods = ["gasteiger", "am1bcc", "OPLS"]
        if charge_method == None:
            charge_method = "OPLS"

        if charge_method not in charge_methods:
            raise ValueError(
                "The charge method should be one of: " + str(charge_methods)
            )

        # Create PELE job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        # Copy script to generate protein and ligand mae inputs, separately.
        _copyScriptFile(job_folder, "peleffy_ligand.py")

        jobs = []
        for ligand in os.listdir(ligands_folder):

            extension = ligand.split(".")[-1]

            if extension == "pdb":
                ligand_name = ligand.replace("." + extension, "")

                # Only process ligands given in only_ligands list
                if only_ligands != None:
                    if ligand_name not in only_ligands:
                        continue

                # structure = _readPDB(ligand_name, ligands_folder+'/'+ligand)
                if not os.path.exists(job_folder + "/" + ligand_name):
                    os.mkdir(job_folder + "/" + ligand_name)

                # _saveStructureToPDB(structure, job_folder+'/'+pdb_name+'/'+pdb_name+extension)
                shutil.copyfile(
                    ligands_folder + "/" + ligand,
                    job_folder
                    + "/"
                    + ligand_name
                    + "/"
                    + ligand_name
                    + "."
                    + extension,
                )

                # Create command
                command = "cd " + job_folder + "/" + ligand_name + "\n"
                command += (
                    "python  ../._peleffy_ligand.py "
                    + ligand_name
                    + "."
                    + extension
                    + " "
                )
                command += "--rotamer_resolution " + str(rotamer_resolution) + " "
                command += "\n"
                command += "cd ../..\n"
                jobs.append(command)

        return jobs

    def _setUpCovalentLigandParameterization(
        self, model, residue_index, base_aa, output_folder=""
    ):
        """
        Add a step of parameterization for a covalent residue in a specific model.

        Parameters
        ==========
        model : str
            Model name
        resname : str
            Name of the covalent residue
        base_aa : dict
            Three letter identity of the aminoacid upon which the ligand is covalently bound.
            One entry in the dictionary for residue name, i.e., base_aa={'FAD':'CYS', 'NAD':'ASP', etc.}
        output_folder : str
            Output folder where to put the ligand PDB file.
        """

        def getAtomsIndexes(pdb_file):

            atoms_indexes = {}
            with open(pdb_file) as pdb:
                for l in pdb:
                    if l.startswith("ATOM") or l.startswith("HETATM"):
                        index, name, chain, resid = (
                            int(l[7:12]),
                            l[12:17].strip(),
                            l[21],
                            int(l[22:27]),
                        )
                        atoms_indexes[(chain, resid, name)] = index
            return atoms_indexes

        def addConectLines(model, pdb_file):

            # Add conect lines to ligand structure
            atoms_indexes = getAtomsIndexes(pdb_file)
            with open(pdb_file) as pdb:
                for l in pdb:
                    if l.startswith("ATOM") or l.startswith("HETATM"):
                        index, name, chain, resid = (
                            int(l[7:12]),
                            l[12:17].strip(),
                            l[21],
                            int(l[22:27]),
                        )
                        atoms_indexes[(chain, resid, name)] = index

            with open(pdb_file + ".tmp", "w") as tmp:
                with open(pdb_file) as pdb:
                    for l in pdb:
                        if l.startswith("ATOM") or l.startswith("HETATM"):
                            if not l.startswith("END"):
                                tmp.write(l)

                    # count = 0
                    for conect in self.conects[model]:
                        if conect[0] in atoms_indexes:
                            line = "CONECT"
                            for atom in conect:
                                if atom in atoms_indexes:
                                    line += "%5s" % atoms_indexes[atom]
                            line += "\n"
                            tmp.write(line)
                            # count += 1

                tmp.write("END\n")
            shutil.move(pdb_file + ".tmp", pdb_file)

        # Define atom names
        c_atom = "C"
        n_atom = "N"
        o_atom = "O"

        ### Create covalent-ligand-only PDB
        cov_structure = PDB.Structure.Structure(0)
        cov_model = PDB.Model.Model(0)
        cov_chain = None
        for r in self.structures[model].get_residues():
            if r.id[1] == residue_index:
                resname = r.resname
                if resname not in base_aa:
                    message = "Residue %s not found in the base_aa dictionary!"
                    message += "Please give the base of the aminoacid with the 'base_aa' keyword"
                    raise ValueError(message)

                cov_residue = r
                cov_chain = PDB.Chain.Chain(r.get_parent().id)
                cov_chain.add(r)
                break

        if cov_chain == None:
            raise ValueError(
                "Residue %s not found in model %s structure" % (resname, model)
            )

        cov_model.add(cov_chain)
        cov_structure.add(cov_model)
        _saveStructureToPDB(cov_structure, output_folder + "/" + resname + ".pdb")
        addConectLines(model, output_folder + "/" + resname + ".pdb")

        # Get atoms to which append hydrogens
        indexes = getAtomsIndexes(output_folder + "/" + resname + ".pdb")
        selected_atoms = []
        for atom in indexes:
            if atom[-1] == c_atom:
                c_atom = str(indexes[atom])
                selected_atoms.append(c_atom)
            elif atom[-1] == n_atom:
                n_atom = str(indexes[atom])
                selected_atoms.append(n_atom)
            elif atom[-1] == o_atom:
                o_atom = str(indexes[atom])
        selected_atoms = ",".join(selected_atoms)

        # Set C-O bond as double bond to secure single hydrogen addition to C atom
        add_bond = str(c_atom) + "," + str(o_atom) + ",2"

        _copyScriptFile(output_folder, "addHydrogens.py")

        ### Add hydrogens to PDB structure
        print("Replacing covalent bonds with hydrogens at %s residue..." % resname)
        command = "run python3 " + output_folder + "/._addHydrogens.py "
        command += output_folder + "/" + resname + ".pdb "
        command += output_folder + "/" + resname + ".pdb "
        command += "--indexes " + selected_atoms + " "
        command += "--add_bond " + add_bond + " "
        command += "--covalent"
        os.system(command)

        # Copy file to avoid the faulty preparation...
        shutil.copyfile(
            output_folder + "/" + resname + ".pdb",
            output_folder + "/" + resname + "_p.pdb",
        )

    def setUpPELECalculation(
        self,
        pele_folder,
        models_folder,
        input_yaml,
        box_centers=None,
        distances=None,
        angles=None,
        constraints=None,
        ligand_index=1,
        box_radius=10,
        steps=100,
        debug=False,
        iterations=5,
        cpus=96,
        equilibration_steps=100,
        ligand_energy_groups=None,
        separator="-",
        use_peleffy=True,
        usesrun=True,
        energy_by_residue=False,
        ebr_new_flag=False,
        ninety_degrees_version=False,
        analysis=False,
        energy_by_residue_type="all",
        peptide=False,
        equilibration_mode="equilibrationLastSnapshot",
        spawning="independent",
        continuation=False,
        equilibration=True,
        skip_models=None,
        skip_ligands=None,
        extend_iterations=False,
        only_models=None,
        only_ligands=None,
        only_combinations=None,
        ligand_templates=None,
        seed=12345,
        log_file=False,
        nonbonded_energy=None,
        nonbonded_energy_type="all",
        nonbonded_new_flag=False,
        covalent_setup=False,
        covalent_base_aa=None,
        membrane_residues=None,
        bias_to_point=None,
        com_bias1=None,
        com_bias2=None,
        epsilon=0.5,
        rescoring=False,
        ligand_equilibration_cst=True,
        regional_metrics=None,
        regional_thresholds=None,
        max_regional_iterations=None,
        regional_energy_bias="Binding Energy",
        regional_best_fraction=0.2,
        constraint_level=1,
        restore_input_coordinates=True
    ):
        """
        Generates a PELE calculation for extracted poses. The function reads all the
        protein ligand poses and creates input for a PELE platform set up run.

        Constraints must be given for positional constraints as:
        {(model1_name,ligand_name):[(springConstant,(chain_id, residue_id, atom_name)), ...], (model1_name,ligand_name):...}
        And for distance constraints as:
        {(model1_name,ligand_name):[(springConstant,distance,(chain1_id, residue1_id, atom_name1),(chain2_id, residue2_id, atom_name2)) ...], (model1_name,ligand_name):...}


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

        # Flag for checking if continuation was given as True
        if continuation:
            continue_all = True
        else:
            continue_all = False

        energy_by_residue_types = ["all", "lennard_jones", "sgb", "electrostatic"]
        if energy_by_residue_type not in energy_by_residue_types:
            raise ValueError(
                "%s not found. Try: %s"
                % (energy_by_residue_type, energy_by_residue_types)
            )

        spawnings = [
            "independent",
            "inverselyProportional",
            "epsilon",
            "variableEpsilon",
            "independentMetric",
            "UCB",
            "FAST",
            "ProbabilityMSM",
            "MetastabilityMSM",
            "IndependentMSM",
            "regional",
        ]

        methods = ["rescoring"]

        if spawning != None and spawning not in spawnings:
            message = "Spawning method %s not found." % spawning
            message = "Allowed options are: " + str(spawnings)
            raise ValueError(message)

        regional_spawning = False
        if spawning == "regional":

            # Check for required inputs
            if not isinstance(regional_metrics, dict):
                raise ValueError(
                    "For the regional spawning you must define the regional_metrics dictionary."
                )
            if not isinstance(regional_thresholds, dict):
                raise ValueError(
                    "For the regional spawning you must define the regional_thresholds dictionary."
                )

            if regional_energy_bias not in ["Total Energy", "Binding Energy"]:
                raise ValueError(
                    'You must give either "Total Energy" or "Binding Energy" to bias the regional spawning simulation!'
                )

            regional_spawning = True
            spawning = "independent"

        if isinstance(membrane_residues, type(None)):
            membrane_residues = {}

        if isinstance(bias_to_point, type(None)):
            bias_to_point = {}

        # Check bias_to_point input
        if isinstance(bias_to_point, (list, tuple)):
            d = {}
            for model in self:
                d[model] = bias_to_point
            bias_to_point = d

        if not isinstance(bias_to_point, dict):
            raise ValueError("bias_to_point should be a dictionary or a list.")

        # Check COM distance bias inputs
        if isinstance(com_bias1, type(None)):
            com_bias1 = {}

        if isinstance(com_bias2, type(None)):
            com_bias2 = {}

        if com_bias1 != {} and com_bias2 == {} or com_bias1 == {} and com_bias2 != {}:
            raise ValueError(
                "You must give both COM atom groups to apply a COM distance bias."
            )

        # Create PELE job folder
        if not os.path.exists(pele_folder):
            os.mkdir(pele_folder)

        # Use to find the relative location of general scripts
        rel_path_to_root = "../"
        if regional_spawning:
            rel_path_to_root = "../" * 2
            _copyScriptFile(pele_folder, "regionalSpawning.py")

        # Read docking poses information from models_folder and create pele input
        # folders.
        jobs = []
        for d in os.listdir(models_folder):
            if os.path.isdir(models_folder + "/" + d):
                models = {}
                ligand_pdb_name = {}
                for f in os.listdir(models_folder + "/" + d):

                    fs = f.split(separator)
                    protein = fs[0]
                    ligand = fs[1]
                    pose = fs[2].replace(".pdb", "")

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

                    if only_combinations and (protein, ligand) not in only_combinations:
                        continue

                    # Create PELE job folder for each docking
                    protein_ligand = protein + separator + ligand
                    protein_ligand_folder = pele_folder + "/" + protein_ligand
                    if not os.path.exists(protein_ligand_folder):
                        os.mkdir(protein_ligand_folder)

                    if regional_spawning:

                        # Create metrics dictionaries
                        reg_met = {}
                        for m in regional_metrics:
                            if protein not in regional_metrics[m]:
                                raise ValueError(
                                    f"Protein {protein} was not found in the regional_metrics dictionary for metric {m}"
                                )

                            if ligand not in regional_metrics[m][protein]:
                                raise ValueError(
                                    f"Ligand {ligand} was not found in the regional_metrics dictionary for protein {protein} and metric {m}"
                                )

                            # Check if distance_ and angle_ prefix were given
                            reg_met[m] = []
                            for v in regional_metrics[m][protein][ligand]:
                                if "-" in v:
                                    v = v.replace("-", "_")
                                if not v.startswith("distance_") and not v.startswith(
                                    "angle_"
                                ):
                                    if len(v.split("_")) == 2:
                                        v = "distance_" + v
                                    elif len(v.split("_")) == 3:
                                        v = "angle_" + v
                                reg_met[m].append(v)

                        with open(protein_ligand_folder + "/metrics.json", "w") as jf:
                            json.dump(reg_met, jf)

                        # Check regional thresholds format
                        for m in regional_thresholds:
                            rm = regional_thresholds[m]

                            incorrect = False
                            if not isinstance(rm, (int, float)) and not isinstance(
                                rm, tuple
                            ):
                                incorrect = True
                            elif isinstance(rm, tuple) and len(rm) != 2:
                                incorrect = True
                            elif isinstance(rm, tuple) and (
                                not isinstance(rm[0], (int, float))
                                or not isinstance(rm[1], (int, float))
                            ):
                                incorrect = True
                            if incorrect:
                                raise ValueError(
                                    "The regional thresholds should be floats or two-elements tuples of floats"
                                )  # Review this check for more complex region definitions

                        with open(
                            protein_ligand_folder + "/metrics_thresholds.json", "w"
                        ) as jf:
                            json.dump(regional_thresholds, jf)

                        protein_ligand_folder = protein_ligand_folder + "/0"
                        if not os.path.exists(protein_ligand_folder):
                            os.mkdir(protein_ligand_folder)

                    structure = _readPDB(
                        protein_ligand, models_folder + "/" + d + "/" + f
                    )

                    # Change water names if any
                    for residue in structure.get_residues():
                        if residue.id[0] == "W":
                            residue.resname = "HOH"

                        if residue.get_parent().id == "L":
                            ligand_pdb_name[ligand] = residue.resname

                    ## Add dummy atom if peptide docking ### Strange fix =)
                    if peptide:
                        for chain in structure.get_chains():
                            if chain.id == "L":
                                # Create new residue
                                new_resid = (
                                    max([r.id[1] for r in chain.get_residues()]) + 1
                                )
                                residue = PDB.Residue.Residue(
                                    ("H", new_resid, " "), "XXX", " "
                                )
                                serial_number = (
                                    max([a.serial_number for a in chain.get_atoms()])
                                    + 1
                                )
                                atom = PDB.Atom.Atom(
                                    "X",
                                    [0, 0, 0],
                                    0,
                                    1.0,
                                    " ",
                                    "%-4s" % "X",
                                    serial_number + 1,
                                    "H",
                                )
                                residue.add(atom)
                                chain.add(residue)

                    _saveStructureToPDB(structure, protein_ligand_folder + "/" + f)
                    self._write_conect_lines(
                        protein, protein_ligand_folder + "/" + f, check_file=True
                    )

                    if (protein, ligand) not in models:
                        models[(protein, ligand)] = []
                    models[(protein, ligand)].append(f)

                # If templates are given for ligands
                templates = {}
                if ligand_templates != None:

                    # Create templates folder
                    if not os.path.exists(pele_folder + "/templates"):
                        os.mkdir(pele_folder + "/templates")

                    for ligand in os.listdir(ligand_templates):

                        if not os.path.isdir(ligand_templates + "/" + ligand):
                            continue

                        # Create ligand template folder
                        if not os.path.exists(pele_folder + "/templates/" + ligand):
                            os.mkdir(pele_folder + "/templates/" + ligand)

                        templates[ligand] = []
                        for f in os.listdir(ligand_templates + "/" + ligand):
                            if f.endswith(".rot.assign") or f.endswith("z"):

                                # Copy template files
                                shutil.copyfile(
                                    ligand_templates + "/" + ligand + "/" + f,
                                    pele_folder + "/templates/" + ligand + "/" + f,
                                )

                                templates[ligand].append(f)

                # Create YAML file
                for model in models:
                    protein, ligand = model
                    protein_ligand = protein + separator + ligand
                    protein_ligand_folder = pele_folder + "/" + protein_ligand
                    if regional_spawning:
                        protein_ligand_folder += "/0"

                    keywords = [
                        "system",
                        "chain",
                        "resname",
                        "steps",
                        "iterations",
                        "atom_dist",
                        "analyse",
                        "cpus",
                        "equilibration",
                        "equilibration_steps",
                        "traj",
                        "working_folder",
                        "usesrun",
                        "use_peleffy",
                        "debug",
                        "box_radius",
                        "box_center",
                        "equilibration_mode",
                        "seed",
                        "spawning",
                        "constraint_level",
                    ]

                    # Generate covalent parameterization setup
                    if not covalent_setup:
                        if protein in self.covalent and self.covalent[protein] != []:
                            print(
                                "WARNING: Covalent bound ligands were found. Consider giving covalent_setup=True"
                            )
                    else:
                        if covalent_base_aa == None:
                            message = (
                                "You must give the base AA upon which each covalently"
                            )
                            message += "attached ligand is bound. E.g., covalent_base_aa=base_aa={'FAD':'CYS', 'NAD':'ASP', etc.}"
                            raise ValueError(message)

                        if protein in self.covalent:
                            for index in self.covalent[protein]:
                                output_folder = protein_ligand_folder + "/output/"
                                if not os.path.exists(output_folder + "/ligand"):
                                    os.makedirs(output_folder + "/ligand")
                                self._setUpCovalentLigandParameterization(
                                    protein,
                                    index,
                                    covalent_base_aa,
                                    output_folder=output_folder + "/ligand",
                                )

                                # Copy covalent parameterization script
                                _copyScriptFile(
                                    output_folder, "covalentLigandParameterization.py"
                                )

                                # Define covalent parameterization command
                                skip_covalent_residue = [
                                    r.resname
                                    for r in self.structures[protein].get_residues()
                                    if r.id[1] == index
                                ][0]
                                covalent_command = "cd output\n"
                                covalent_command += (
                                    "python "
                                    + rel_path_to_root
                                    + "._covalentLigandParameterization.py ligand/"
                                    + skip_covalent_residue
                                    + ".pdb "
                                    + skip_covalent_residue
                                    + " "
                                    + covalent_base_aa[skip_covalent_residue]
                                    + "\n"
                                )

                                # Copy modify processed script
                                _copyScriptFile(
                                    output_folder, "modifyProcessedForCovalentPELE.py"
                                )
                                cov_residues = ",".join(
                                    [str(x) for x in self.covalent[protein]]
                                )
                                covalent_command += (
                                    "python "
                                    + rel_path_to_root
                                    + "._modifyProcessedForCovalentPELE.py "
                                    + cov_residues
                                    + " \n"
                                )
                                covalent_command += "mv DataLocal/Templates/OPLS2005/Protein/templates_generated/* DataLocal/Templates/OPLS2005/Protein/\n"
                                covalent_command += "cd ..\n"

                    # Write input yaml
                    with open(protein_ligand_folder + "/" + "input.yaml", "w") as iyf:
                        if energy_by_residue or nonbonded_energy != None:
                            # Use new PELE version with implemented energy_by_residue
                            iyf.write('pele_exec: "/gpfs/projects/bsc72/PELE++/mnv/1.8.1b1/bin/PELE_mpi"\n')
                            iyf.write('pele_data: "/gpfs/projects/bsc72/PELE++/mnv/1.8.1b1/Data"\n')
                            iyf.write('pele_documents: "/gpfs/projects/bsc72/PELE++/mnv/1.8.1b1/Documents/"\n')
                        elif ninety_degrees_version:
                            # Use new PELE version with implemented 90 degrees fix
                            print('paths of PELE version should be changed')
                            iyf.write(
                                'pele_exec: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/bin/PELE-1.8_mpi"\n'
                            )
                            iyf.write(
                                'pele_data: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Data"\n'
                            )
                            iyf.write(
                                'pele_documents: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Documents/"\n'
                            )
                        if len(models[model]) > 1:
                            equilibration_mode = "equilibrationCluster"
                            iyf.write("system: '*.pdb'\n")
                        else:
                            iyf.write("system: '" + " ".join(models[model]) + "'\n")
                        iyf.write("chain: 'L'\n")
                        if peptide:
                            iyf.write("resname: 'XXX'\n")
                            iyf.write("skip_ligand_prep:\n")
                            iyf.write(" - 'XXX'\n")
                        else:
                            iyf.write("resname: '" + ligand_pdb_name[ligand] + "'\n")
                        iyf.write("steps: " + str(steps) + "\n")
                        iyf.write("iterations: " + str(iterations) + "\n")
                        iyf.write("cpus: " + str(cpus) + "\n")
                        if equilibration:
                            iyf.write("equilibration: true\n")
                            iyf.write(
                                "equilibration_mode: '" + equilibration_mode + "'\n"
                            )
                            iyf.write(
                                "equilibration_steps: "
                                + str(equilibration_steps)
                                + "\n"
                            )
                        else:
                            iyf.write("equilibration: false\n")
                        if spawning != None:
                            iyf.write("spawning: '" + str(spawning) + "'\n")
                        if constraint_level:
                            iyf.write(
                                "constraint_level: '" + str(constraint_level) + "'\n"
                            )
                        if rescoring:
                            iyf.write("rescoring: true\n")

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

                        if covalent_setup:
                            iyf.write("skip_ligand_prep:\n")
                            iyf.write(' - "' + skip_covalent_residue + '"\n')

                        if ligand in templates:
                            iyf.write("templates:\n")
                            iyf.write(' - "LIGAND_TEMPLATE_PATH_ROT"\n')
                            iyf.write(' - "LIGAND_TEMPLATE_PATH_Z"\n')
                            iyf.write("skip_ligand_prep:\n")
                            iyf.write(' - "' + ligand_pdb_name[ligand] + '"\n')

                        iyf.write("box_radius: " + str(box_radius) + "\n")
                        if isinstance(box_centers, type(None)) and peptide:
                            raise ValueError(
                                "You must give per-protein box_centers when docking peptides!"
                            )
                        if not isinstance(box_centers, type(None)):
                            if (
                                not all(
                                    isinstance(x, float) for x in box_centers[model]
                                )
                                and not all(
                                    isinstance(x, int) for x in box_centers[model]
                                )
                                and not all(
                                    isinstance(x, np.float32)
                                    for x in box_centers[model]
                                )
                            ):
                                # get coordinates from tuple
                                coordinates = None
                                for chain in self.structures[model[0]].get_chains():
                                    if chain.id == box_centers[model][0]:
                                        for r in chain:
                                            if r.id[1] == box_centers[model][1]:
                                                for atom in r:
                                                    if (
                                                        atom.name
                                                        == box_centers[model][2]
                                                    ):
                                                        coordinates = atom.coord
                                if isinstance(coordinates, type(None)):
                                    raise ValueError(
                                        f"Atom {box_centers[model]} was not found for protein {model[0]}"
                                    )
                            else:
                                coordinates = box_centers[model]

                            box_center = ""
                            for coord in coordinates:
                                # if not isinstance(coord, float):
                                #    raise ValueError('Box centers must be given as a (x,y,z) tuple or list of floats.')
                                box_center += "  - " + str(float(coord)) + "\n"
                            iyf.write("box_center: \n" + box_center)

                        # energy by residue is not implemented in PELE platform, therefore
                        # a scond script will modify the PELE.conf file to set up the energy
                        # by residue calculation.
                        if any(
                            [
                                debug,
                                energy_by_residue,
                                peptide,
                                nonbonded_energy != None,
                                membrane_residues,
                                bias_to_point,
                                com_bias1,
                                ligand_equilibration_cst,
                                regional_spawning,
                                constraint_level,
                            ]
                        ):
                            iyf.write("debug: true\n")

                        if distances != None:
                            iyf.write("atom_dist:\n")
                            for d in distances[protein][ligand]:
                                if isinstance(d[0], str):
                                    d1 = (
                                        "- 'L:" + str(ligand_index) + ":" + d[0] + "'\n"
                                    )
                                else:
                                    d1 = (
                                        "- '"
                                        + d[0][0]
                                        + ":"
                                        + str(d[0][1])
                                        + ":"
                                        + d[0][2]
                                        + "'\n"
                                    )
                                if isinstance(d[1], str):
                                    d2 = (
                                        "- 'L:" + str(ligand_index) + ":" + d[1] + "'\n"
                                    )
                                else:
                                    d2 = (
                                        "- '"
                                        + d[1][0]
                                        + ":"
                                        + str(d[1][1])
                                        + ":"
                                        + d[1][2]
                                        + "'\n"
                                    )
                                iyf.write(d1)
                                iyf.write(d2)

                        if constraints != None:
                            iyf.write("external_constraints:\n")
                            for c in constraints[(protein, ligand)]:
                                if len(c) == 2:
                                    line = (
                                        "- '"
                                        + str(c[0])
                                        + "-"
                                        + str(c[1][0])
                                        + ":"
                                        + str(c[1][1])
                                        + ":"
                                        + str(c[1][2])
                                        + "'\n"
                                    )  # cst_force and atom_index for positional cst
                                elif len(c) == 4:
                                    line = (
                                        "- '"
                                        + str(c[0])
                                        + "-"
                                        + str(c[1])
                                        + "-"
                                        + str(c[2][0])
                                        + ":"
                                        + str(c[2][1])
                                        + ":"
                                        + str(c[2][2])
                                        + "-"
                                        + str(c[3][0])
                                        + ":"
                                        + str(c[3][1])
                                        + ":"
                                        + str(c[3][2])
                                        + "'\n"
                                    )  # cst_force, distance, atom_index1, atom_index2 for distance cst
                                else:
                                    raise ValueError(
                                        "Constraint for protein "
                                        + protein
                                        + " with ligand "
                                        + ligand
                                        + " are not defined correctly."
                                    )
                                iyf.write(line)

                        if seed:
                            iyf.write("seed: " + str(seed) + "\n")

                        if log_file:
                            iyf.write("log: true\n")

                        iyf.write("\n")
                        iyf.write("#Options gathered from " + input_yaml + "\n")

                        with open(input_yaml) as tyf:
                            for l in tyf:
                                if l.startswith("#"):
                                    continue
                                elif l.startswith("-"):
                                    continue
                                elif l.strip() == "":
                                    continue
                                if l.split()[0].replace(":", "") not in keywords:
                                    iyf.write(l)

                    if energy_by_residue:
                        _copyScriptFile(pele_folder, "addEnergyByResidueToPELEconf.py")
                        ebr_script_name = "._addEnergyByResidueToPELEconf.py"
                        if not isinstance(ligand_energy_groups, type(None)):
                            if not isinstance(ligand_energy_groups, dict):
                                raise ValueError(
                                    "ligand_energy_groups, must be given as a dictionary"
                                )
                            with open(
                                protein_ligand_folder + "/ligand_energy_groups.json",
                                "w",
                            ) as jf:
                                json.dump(ligand_energy_groups[ligand], jf)

                    if protein in membrane_residues:
                        _copyScriptFile(pele_folder, "addMembraneConstraints.py")
                        mem_res_script = (
                            "._addMembraneConstraints.py"  # I have added the _
                        )

                    if nonbonded_energy != None:
                        _copyScriptFile(
                            pele_folder, "addAtomNonBondedEnergyToPELEconf.py"
                        )
                        nbe_script_name = "._addAtomNonBondedEnergyToPELEconf.py"
                        if not isinstance(nonbonded_energy, dict):
                            raise ValueError(
                                "nonbonded_energy, must be given as a dictionary"
                            )
                        with open(
                            protein_ligand_folder + "/nonbonded_energy_atoms.json", "w"
                        ) as jf:
                            json.dump(nonbonded_energy[protein][ligand], jf)

                    if protein in bias_to_point:
                        _copyScriptFile(pele_folder, "addBiasToPoint.py")
                        btp_script = "._addBiasToPoint.py"

                    if protein in com_bias1:
                        _copyScriptFile(pele_folder, "addComDistancesBias.py")
                        cbs_script = "._addComDistancesBias.py"

                    if peptide:
                        _copyScriptFile(pele_folder, "modifyPelePlatformForPeptide.py")
                        peptide_script_name = "._modifyPelePlatformForPeptide.py"

                    if ligand_equilibration_cst:
                        _copyScriptFile(
                            pele_folder, "addLigandConstraintsToPELEconf.py"
                        )
                        equilibration_script_name = (
                            "._addLigandConstraintsToPELEconf.py"
                        )
                        _copyScriptFile(pele_folder, "changeAdaptiveIterations.py")
                        adaptive_script_name = "._changeAdaptiveIterations.py"

                    if restore_input_coordinates:
                        _copyScriptFile(
                            pele_folder, 'restoreChangedCoordinates.py'
                        )
                        restore_coordinates_script_name = "._restoreChangedCoordinates.py"

                    # Create command
                    command = "cd " + protein_ligand_folder + "\n"

                    # Add commands to write template folder absolute paths
                    if ligand in templates:
                        command += "export CWD=$(pwd)\n"
                        command += "cd ../templates/" + ligand + "\n"
                        command += "export TMPLT_DIR=$(pwd)\n"
                        command += "cd $CWD\n"
                        for tf in templates[ligand]:
                            if continuation:
                                yaml_file = "input_restart.yaml"
                            else:
                                yaml_file = "input.yaml"
                            if tf.endswith(".assign"):
                                command += (
                                    "sed -i s,LIGAND_TEMPLATE_PATH_ROT,$TMPLT_DIR/"
                                    + tf
                                    + ",g "
                                    + yaml_file
                                    + "\n"
                                )
                            elif tf.endswith("z"):
                                command += (
                                    "sed -i s,LIGAND_TEMPLATE_PATH_Z,$TMPLT_DIR/"
                                    + tf
                                    + ",g "
                                    + yaml_file
                                    + "\n"
                                )

                    if not continuation:
                        command += "python -m pele_platform.main input.yaml\n"

                        if regional_spawning:
                            continuation = True

                        if angles:
                            # Copy individual angle definitions to each protein and ligand folder
                            if protein in angles and ligand in angles[protein]:
                                with open(
                                    protein_ligand_folder + "/._angles.json", "w"
                                ) as jf:
                                    json.dump(angles[protein][ligand], jf)

                            # Copy script to add angles to pele.conf
                            _copyScriptFile(pele_folder, "addAnglesToPELEConf.py")
                            command += (
                                "python "
                                + rel_path_to_root
                                + "._addAnglesToPELEConf.py output "
                            )
                            command += "._angles.json "
                            command += (
                                "output/input/"
                                + protein_ligand
                                + separator
                                + pose
                                + "_processed.pdb\n"
                            )
                            continuation = True

                        if constraint_level:
                            # Copy script to add angles to pele.conf
                            _copyScriptFile(
                                pele_folder, "correctPositionalConstraints.py"
                            )
                            command += (
                                "python "
                                + rel_path_to_root
                                + "._correctPositionalConstraints.py output "
                            )
                            command += (
                                "output/input/"
                                + protein_ligand
                                + separator
                                + pose
                                + "_processed.pdb\n"
                            )
                            continuation = True

                        if energy_by_residue:
                            command += (
                                "python "
                                + rel_path_to_root
                                + ebr_script_name
                                + " output --energy_type "
                                + energy_by_residue_type
                                + "--new_version "
                                + new_version
                            )
                            if isinstance(ligand_energy_groups, dict):
                                command += (
                                    " --ligand_energy_groups ligand_energy_groups.json"
                                )
                                command += " --ligand_index " + str(ligand_index)
                            if ebr_new_flag:
                                command += " --new_version "
                            if peptide:
                                command += " --peptide \n"
                                command += (
                                    "python "
                                    + rel_path_to_root
                                    + peptide_script_name
                                    + " output "
                                    + " ".join(models[model])
                                    + "\n"
                                )
                            else:
                                command += "\n"

                        if protein in membrane_residues:
                            command += (
                                "python " + rel_path_to_root + mem_res_script + " "
                            )
                            command += "output "  # I think we should change this for a variable
                            command += "--membrane_residues "
                            command += (
                                ",".join([str(x) for x in membrane_residues[protein]])
                                + "\n"
                            )  # 1,2,3,4,5
                            continuation = True

                        if protein in bias_to_point:
                            command += "python " + rel_path_to_root + btp_script + " "
                            command += "output "  # I think we should change this for a variable
                            command += (
                                "point_"
                                + ",".join([str(x) for x in bias_to_point[protein]])
                                + " "
                            )
                            command += "--epsilon " + str(epsilon) + "\n"
                            continuation = True

                        if protein in com_bias1 and ligand in com_bias1[protein]:
                            # Write both COM groups as json files
                            with open(
                                protein_ligand_folder + "/._com_group1.json", "w"
                            ) as jf:
                                json.dump(com_bias1[protein][ligand], jf)

                            with open(
                                protein_ligand_folder + "/._com_group2.json", "w"
                            ) as jf:
                                json.dump(com_bias2[protein][ligand], jf)

                            command += "python " + rel_path_to_root + cbs_script + " "
                            command += "output "  # I think we should change this for a variable
                            command += "._com_group1.json "
                            command += "._com_group2.json "
                            command += "--epsilon " + str(epsilon) + "\n"
                            continuation = True

                        if covalent_setup:
                            command += covalent_command
                            continuation = True

                        if ligand_equilibration_cst:

                            # Copy input_yaml for equilibration
                            oyml = open(
                                protein_ligand_folder + "/input_equilibration.yaml", "w"
                            )
                            debug_line = False
                            restart_line = False
                            with open(protein_ligand_folder + "/input.yaml") as iyml:
                                for l in iyml:
                                    if "debug: true" in l:
                                        debug_line = True
                                        oyml.write("restart: true\n")
                                        oyml.write("adaptive_restart: true\n")
                                        continue
                                    elif "restart: true" in l:
                                        restart_line = True
                                    elif l.startswith("iterations:"):
                                        l = "iterations: 1\n"
                                    elif l.startswith("steps:"):
                                        l = "steps: 1\n"
                                    oyml.write(l)
                                if not debug_line and not restart_line:
                                    oyml.write("restart: true\n")
                                    oyml.write("adaptive_restart: true\n")
                            oyml.close()

                            # Add commands for adding ligand constraints
                            command += "cp output/pele.conf output/pele.conf.backup\n"
                            command += (
                                "cp output/adaptive.conf output/adaptive.conf.backup\n"
                            )

                            # Modify pele.conf to add ligand constraints
                            command += (
                                "python "
                                + rel_path_to_root
                                + equilibration_script_name
                                + " "
                            )
                            command += "output "  # I think we should change this for a variable
                            if (
                                isinstance(ligand_equilibration_cst, (int, float))
                                and ligand_equilibration_cst != 1.0
                            ):
                                command += "--constraint_value " + str(
                                    float(ligand_equilibration_cst)
                                )
                            command += "\n"

                            # Modify adaptive.conf to remove simulation steps
                            command += (
                                "python "
                                + rel_path_to_root
                                + adaptive_script_name
                                + " "
                            )
                            command += "output "  # I think we should change this for a variable
                            command += "--iterations 1 "
                            command += "--steps 1\n"

                            # Launch equilibration
                            command += "python -m pele_platform.main input_equilibration.yaml\n"

                            # Recover conf files
                            command += "cp output/pele.conf.backup output/pele.conf\n"
                            command += (
                                "cp output/adaptive.conf.backup output/adaptive.conf\n"
                            )
                            # Add commands for launching equilibration only

                            continuation = True

                        if restore_input_coordinates:
                            command += "python "+ rel_path_to_root+restore_coordinates_script_name+" "
                            command += f+' '
                            command += "output/input/"+f.replace('.pdb', '_processed.pdb')+'\n'

                    if continuation:
                        debug_line = False
                        restart_line = False
                        # Copy input_yaml for equilibration
                        oyml = open(protein_ligand_folder + "/input_restart.yaml", "w")
                        debug_line = False
                        restart_line = False
                        with open(protein_ligand_folder + "/input.yaml") as iyml:
                            for l in iyml:
                                if "debug: true" in l:
                                    debug_line = True
                                    oyml.write("restart: true\n")
                                    oyml.write("adaptive_restart: true\n")
                                    continue
                                elif "restart: true" in l:
                                    restart_line = True
                                oyml.write(l)
                            if not debug_line and not restart_line:
                                oyml.write("restart: true\n")
                                oyml.write("adaptive_restart: true\n")
                        oyml.close()

                        if extend_iterations:
                            _copyScriptFile(pele_folder, "extendAdaptiveIteartions.py")
                            extend_script_name = "._extendAdaptiveIteartions.py"
                            command += (
                                "python "
                                + rel_path_to_root
                                + extend_script_name
                                + " output "
                                + str(iterations)
                                + "\n"
                            )
                        if not energy_by_residue:
                            command += (
                                "python -m pele_platform.main input_restart.yaml\n"
                            )

                        if (
                            any(
                                [
                                    membrane_residues,
                                    bias_to_point,
                                    com_bias1,
                                    ligand_equilibration_cst,
                                    angles,
                                    regional_spawning,
                                    constraint_level,
                                ]
                            )
                            and not continue_all
                        ):
                            continuation = False
                            debug = False

                    elif peptide:
                        command += (
                            "python "
                            + rel_path_to_root
                            + peptide_script_name
                            + " output "
                            + " ".join(models[model])
                            + "\n"
                        )
                        with open(
                            protein_ligand_folder + "/" + "input_restart.yaml", "w"
                        ) as oyml:
                            with open(
                                protein_ligand_folder + "/" + "input.yaml"
                            ) as iyml:
                                for l in iyml:
                                    if "debug: true" in l:
                                        l = "restart: true\n"
                                    oyml.write(l)
                        if nonbonded_energy == None:
                            command += (
                                "python -m pele_platform.main input_restart.yaml\n"
                            )

                    elif extend_iterations and not continuation:
                        raise ValueError(
                            "extend_iterations must be used together with the continuation keyword"
                        )

                    if nonbonded_energy != None:
                        command += (
                            "python "
                            + rel_path_to_root
                            + nbe_script_name
                            + " output --energy_type "
                            + nonbonded_energy_type
                        )
                        command += " --target_atoms nonbonded_energy_atoms.json"
                        protein_chain = [
                            c for c in self.structures[protein].get_chains() if c != "L"
                        ][0]
                        command += " --protein_chain " + protein_chain.id
                        if ebr_new_flag or nonbonded_new_flag:
                            command += " --new_version"
                        command += "\n"

                        if not os.path.exists(
                            protein_ligand_folder + "/" + "input_restart.yaml"
                        ):
                            with open(
                                protein_ligand_folder + "/" + "input_restart.yaml", "w"
                            ) as oyml:
                                with open(
                                    protein_ligand_folder + "/" + "input.yaml"
                                ) as iyml:
                                    for l in iyml:
                                        if "debug: true" in l:
                                            l = "restart: true\n"
                                        oyml.write(l)
                        command += "python -m pele_platform.main input_restart.yaml\n"

                    # Remove debug line from input.yaml for covalent setup (otherwise the Data folder is not copied!)
                    if covalent_setup:
                        with open(
                            protein_ligand_folder + "/" + "input.yaml.tmp", "w"
                        ) as oyf:
                            with open(
                                protein_ligand_folder + "/" + "input.yaml"
                            ) as iyf:
                                for l in iyf:
                                    if not "debug: true" in l:
                                        oyf.write(l)
                        shutil.move(
                            protein_ligand_folder + "/" + "input.yaml.tmp",
                            protein_ligand_folder + "/" + "input.yaml",
                        )

                    if regional_spawning:
                        command += "cd ../\n"
                        command += "python ../._regionalSpawning.py "
                        command += "metrics.json "
                        command += "metrics_thresholds.json "
                        command += "--separator " + separator + " "
                        command += '--energy_bias "' + regional_energy_bias + '" '
                        command += (
                            "--regional_best_fraction "
                            + str(regional_best_fraction)
                            + " "
                        )
                        if max_regional_iterations:
                            command += (
                                "--max_iterations " + str(max_regional_iterations) + " "
                            )
                        if angles:
                            command += "--angles "
                        if restore_input_coordinates:
                            command += '--restore_coordinates '
                        command += "\n"

                    command += "cd ../../"
                    jobs.append(command)

        return jobs

    def setUpMDSimulations(
        self,
        md_folder,
        sim_time,
        nvt_time=2,
        npt_time=0.2,
        equilibration_dt=2,
        production_dt=2,
        temperature=298.15,
        frags=1,
        local_command_name=None,
        remote_command_name="${GMXBIN}",
        ff="amber99sb-star-ildn",
        ligand_chains=None,
        ion_chains=None,
        replicas=1,
        charge=None,
        system_output="System",
        models=None,
    ):
        """
        Sets up MD simulations for each model. The current state only allows to set
        up simulations using the Gromacs software.

        If the input pdb has additional non aa residues besides ligand (ions,HETATMs,...)
        they should be separated in individual chains.

        TODO:
        - Test with multiple protein chains
        - Test with all combos (no ligand, ions+no ligand,ligand+no ions...) (done)
        - Test with peptide/already parameterised ligands
        - Test with multiple ligands
        - Test with input waters
        - Test with different ions

        Issues:
        - CA constraints (implemented)
        - H constraints (not done in tutorial but i think it makes sense)
        - Temperature coupling groups (ions with protein or solvent)(implemented)
        - continuation (implemented)
        - velocitites initialized (topol outside, can be done velocities generated in nvt)
        - Ion_chain_I         1 (in topol shouldnt it be 2?)

        Parameters
        ==========
        md_folder : str
            Path to the job folder where the MD input files are located.
        sim_time : int
            Simulation time in ns
        frags : int
            Number of fragments to divide the simulation.
        program : str
            Program to execute simulation.
        command : str
            Command to call program.
        ff : str
            Force field to use for simulation.
        """
        # This should not be a variable i leave to avoid errors with earlier versions
        remote_command_name = "${GMXBIN}"

        if isinstance(models, str):
            models = [models]

        # Create MD job folders
        if not os.path.exists(md_folder):
            os.mkdir(md_folder)
        if not os.path.exists(md_folder + "/scripts"):
            os.mkdir(md_folder + "/scripts")
        if not os.path.exists(md_folder + "/FF"):
            os.mkdir(md_folder + "/FF")
        if not os.path.exists(md_folder + "/FF/" + ff + ".ff"):
            os.mkdir(md_folder + "/FF/" + ff + ".ff")
        if not os.path.exists(md_folder + "/input_models"):
            os.mkdir(md_folder + "/input_models")
        if not os.path.exists(md_folder + "/output_models"):
            os.mkdir(md_folder + "/output_models")

        if local_command_name == None:
            possible_command_names = ["gmx", "gmx_mpi"]
            command_name = None
            for command in possible_command_names:
                if shutil.which(command) != None:
                    command_name = command
            if command_name == None:
                raise ValueError(
                    "Gromacs executable is required for the setup and was not found. The following executable names were tested: "
                    + ",".join(possible_command_names)
                )
        else:
            command_name = local_command_name

        if ligand_chains != None:
            if isinstance(ligand_chains, str):
                ligand_chains = [ligand_chains]
            if not os.path.exists(md_folder + "/ligand_params"):
                os.mkdir(md_folder + "/ligand_params")

        # Save all input models
        self.saveModels(md_folder + "/input_models")

        # Copy script files
        for file in resource_listdir(
            Requirement.parse("prepare_proteins"),
            "prepare_proteins/scripts/md/gromacs/mdp",
        ):
            if not file.startswith("__"):
                _copyScriptFile(
                    md_folder + "/scripts/",
                    file,
                    subfolder="md/gromacs/mdp",
                    no_py=False,
                    hidden=False,
                )

        for file in resource_listdir(
            Requirement.parse("prepare_proteins"),
            "prepare_proteins/scripts/md/gromacs/ff/" + ff,
        ):
            if not file.startswith("__"):
                _copyScriptFile(
                    md_folder + "/FF/" + ff + ".ff",
                    file,
                    subfolder="md/gromacs/ff/" + ff,
                    no_py=False,
                    hidden=False,
                )

        # Replace parameters in the mdp file with given arguments
        for line in fileinput.input(md_folder + "/scripts/em.mdp", inplace=True):
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        for line in fileinput.input(md_folder + "/scripts/md.mdp", inplace=True):
            if "TIME_INTEGRATOR" in line:
                line = line.replace("TIME_INTEGRATOR", str(production_dt / 1000))
                if equilibration_dt > 2:
                    print(
                        "WARNING: you have selected a time integrator higher than 2 femtoseconds. Constraints have been automatically changed to all bonds. This may affect the accuracy of your simulation."
                    )
                    cst = "all-bonds"
                else:
                    cst = "h-bonds"

            if "BOND_CONSTRAINTS" in line:
                line = line.replace("BOND_CONSTRAINTS", cst)
            if "NUMBER_OF_STEPS" in line:
                line = line.replace(
                    "NUMBER_OF_STEPS",
                    str(int((sim_time * (1e6 / production_dt)) / frags)),
                )
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)

            sys.stdout.write(line)

        for line in fileinput.input(md_folder + "/scripts/nvt.mdp", inplace=True):
            if "TIME_INTEGRATOR" in line:
                line = line.replace("TIME_INTEGRATOR", str(equilibration_dt / 1000))
                if equilibration_dt > 2:
                    print(
                        "WARNING: you have selected a time integrator higher than 2 femtoseconds. Constraints have been automatically changed to all bonds. This may affect the accuracy of your simulation."
                    )
                    cst = "all-bonds"
                else:
                    cst = "h-bonds"

            if "BOND_CONSTRAINTS" in line:
                line = line.replace("BOND_CONSTRAINTS", cst)
            if "NUMBER_OF_STEPS" in line:
                line = line.replace(
                    "NUMBER_OF_STEPS", str(int(nvt_time * (1e6 / equilibration_dt)))
                )
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)

            sys.stdout.write(line)

        for line in fileinput.input(md_folder + "/scripts/npt.mdp", inplace=True):
            if "TIME_INTEGRATOR" in line:
                line = line.replace("TIME_INTEGRATOR", str(equilibration_dt / 1000))
                if equilibration_dt > 2:
                    print(
                        "WARNING: you have selected a time integrator higher than 2 femtoseconds. Constraints have been automatically changed to all bonds. This may affect the accuracy of your simulation."
                    )
                    cst = "all-bonds"
                else:
                    cst = "h-bonds"

            if "BOND_CONSTRAINTS" in line:
                line = line.replace("BOND_CONSTRAINTS", cst)

            if "NUMBER_OF_STEPS" in line:
                line = line.replace(
                    "NUMBER_OF_STEPS", str(int(npt_time * (1e6 / equilibration_dt)))
                )
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        # Setup jobs for each model
        jobs = []
        for model in self.models_names:

            if models and model not in models:
                continue

            # Create additional folders
            if not os.path.exists(md_folder + "/input_models/" + model):
                os.mkdir(md_folder + "/input_models/" + model)

            if not os.path.exists(md_folder + "/output_models/" + model):
                os.mkdir(md_folder + "/output_models/" + model)

            for i in range(replicas):
                if not os.path.exists(
                    md_folder + "/output_models/" + model + "/" + str(i)
                ):
                    os.mkdir(md_folder + "/output_models/" + model + "/" + str(i))

            parser = PDB.PDBParser()
            structure = parser.get_structure(
                "protein", md_folder + "/input_models/" + model + ".pdb"
            )

            # Parse structures to set correct histidine protonation
            gmx_codes = []

            # Get ion residues
            if ion_chains == None:
                ion_chains = []
            ion_residues = []

            for mdl in structure:
                for chain in mdl:
                    for residue in chain:
                        if chain.get_id() in ion_chains:
                            ion_residues.append(residue.id[1])
                        HD1 = False
                        HE2 = False
                        if residue.resname == "HIS":
                            for atom in residue:
                                if atom.name == "HD1":
                                    HD1 = True
                                if atom.name == "HE2":
                                    HE2 = True
                        if HD1 != False or HE2 != False:
                            if HD1 == True and HE2 == False:
                                number = 0
                            if HD1 == False and HE2 == True:
                                number = 1
                            if HD1 == True and HE2 == True:
                                number = 2
                            gmx_codes.append(number)
            his_pro = str(gmx_codes)[1:-1].replace(",", "")

            # Setup ligand parametrisation
            if ligand_chains != None:
                ligand_res = _getLigandParameters(
                    structure,
                    ligand_chains,
                    md_folder + "/input_models/" + model,
                    md_folder + "/ligand_params",
                    charge=charge,
                )
            else:
                shutil.copyfile(
                    md_folder + "/input_models/" + model + ".pdb",
                    md_folder + "/input_models/" + model + "/protein.pdb",
                )

            # Generate commands
            for i in range(replicas):
                command = "cd " + md_folder + "\n"
                command += "export GMXLIB=$(pwd)/FF" + "\n"
                # Set up commands
                # Define setup gmx commands to be run locally in order to get correct indexes
                command_local = command
                command_local += (
                    "mkdir output_models/" + model + "/" + str(i) + "/topol" + "\n"
                )
                command_local += (
                    "cp input_models/"
                    + model
                    + "/protein.pdb output_models/"
                    + model
                    + "/"
                    + str(i)
                    + "/topol/protein.pdb"
                    + "\n"
                )
                if ligand_chains != None:
                    command_local += (
                        "cp ligand_params/atomtypes.itp output_models/"
                        + model
                        + "/"
                        + str(i)
                        + "/topol/atomtypes.itp"
                        + "\n"
                    )
                    for ligand_name in ligand_res.values():
                        command_local += (
                            "cp -r ligand_params/"
                            + ligand_name
                            + "/"
                            + ligand_name
                            + ".acpype output_models/"
                            + model
                            + "/"
                            + str(i)
                            + "/topol/"
                            + "\n"
                        )
                command_local += (
                    "cd output_models/" + model + "/" + str(i) + "/topol" + "\n"
                )
                command_local += (
                    "echo "
                    + his_pro
                    + " | "
                    + command_name
                    + " pdb2gmx -f protein.pdb -o prot.pdb -p topol.top -his -ignh -ff "
                    + ff
                    + " -water tip3p -vsite hydrogens"
                    + "\n"
                )

                # Replace name for crystal waters if there are any
                # command_local += 'sed -i -e \'s/SOL/CWT/g\' topol.top \n'

                if ligand_chains != None:
                    lig_files = ""
                    for ligand_name in ligand_res.values():
                        lig_files += (
                            " ../../../../input_models/"
                            + model
                            + "/"
                            + ligand_name
                            + ".pdb "
                        )
                    command_local += (
                        "grep -h ATOM prot.pdb " + lig_files + " >| complex.pdb" + "\n"
                    )
                    command_local += (
                        command_name + " editconf -f complex.pdb -o complex.gro" + "\n"
                    )
                    line = ""
                    line += '#include "atomtypes.itp"\\n'
                    for ligand_name in ligand_res.values():
                        line += (
                            '#include "'
                            + ligand_name
                            + ".acpype\/"
                            + ligand_name
                            + '_GMX.itp"\\n'
                        )

                        line += "#ifdef POSRES\\n"
                        
                        line += (
                            '#include "'
                            + ligand_name
                            + ".acpype\/posre_"
                            + ligand_name
                            + '.itp"\\n'
                        )
                        line += "#endif\\n"

                    line += "'"

                    local_path = (os.getcwd() + "/" + md_folder + "/FF").replace(
                        "/", "\/"
                    )
                    print(line)
                    command_local += (
                        "sed -i '/^#include \""
                        + local_path
                        + "\/"
                        + ff
                        + '.ff\/forcefield.itp"*/a '
                        + line
                        + " topol.top"
                        + "\n"
                    )
                    print(command_local)
                    for ligand_name in ligand_res.values():
                        command_local += (
                            "sed -i -e '$a"
                            + ligand_name.ljust(20)
                            + "1"
                            + "' topol.top"
                            + "\n"
                        )

                else:
                    command_local += (
                        command_name + " editconf -f prot.pdb -o complex.gro" + "\n"
                    )

                command_local += (
                    command_name
                    + " editconf -f complex.gro -o prot_box.gro -c -d 1.0 -bt octahedron"
                    + "\n"
                )
                command_local += (
                    command_name
                    + " solvate -cp prot_box.gro -cs spc216.gro -o prot_solv.gro -p topol.top"
                    + "\n"
                )

                group_dics = {}
                command_local += (
                    'echo "q"| '
                    + command_name
                    + " make_ndx -f  prot_solv.gro -o index.ndx"
                    + "\n"
                )

                # Run local commands
                with open("tmp.sh", "w") as f:
                    f.write(command_local)
                subprocess.run("bash tmp.sh", shell=True)
                os.remove("tmp.sh")

                # Read complex index
                group_dics["complex"] = _readGromacsIndexFile(
                    md_folder
                    + "/"
                    + "output_models/"
                    + model
                    + "/"
                    + str(i)
                    + "/topol"
                    + "/index.ndx"
                )

                # generate tmp index to check for crystal waters
                os.system('echo "q"| '+command_name+' make_ndx -f  '+md_folder+'/output_models/'+model+'/'+str(i)+'/topol/complex.gro -o '+md_folder+'/output_models/'+model+'/'+str(i)+'/topol/tmp_index.ndx'+'\n')
                group_dics['tmp_index'] = _readGromacsIndexFile(md_folder+'/'+'output_models/'+model+'/'+str(i)+'/topol'+'/tmp_index.ndx')

                if 'Water' in group_dics['tmp_index']:
                    reading = False
                    crystal_waters_ndx_lines = '[ CrystalWaters ]\n'
                    for line in open(md_folder+'/'+'output_models/'+model+'/'+str(i)+'/topol'+'/tmp_index.ndx'):
                        if '[' in line and reading:
                            reading = False
                        elif '[ Water ]' in line:
                            reading = True
                        elif reading:
                            crystal_waters_ndx_lines += line

                    with open(md_folder+'/'+'output_models/'+model+'/'+str(i)+'/topol'+'/index.ndx','a') as f:
                        f.write(crystal_waters_ndx_lines)

                    os.system('echo \''+group_dics['complex']['Water']+' & !'+str(len(group_dics['complex']))+'\\nq\' | '+command_name+' make_ndx -f  '+md_folder+'/output_models/'+model+'/'+str(i)+'/topol/prot_solv.gro -o '+md_folder+'/output_models/'+model+'/'+str(i)+'/topol/index.ndx'+' -n '+md_folder+'/output_models/'+model+'/'+str(i)+'/topol/index.ndx'+'\n')
                    os.system('echo \'del '+group_dics['complex']['SOL']+'\n name '+str(len(group_dics['complex']))+' SOL\\nq\' | '+command_name+' make_ndx -f  '+md_folder+'/output_models/'+model+'/'+str(i)+'/topol/prot_solv.gro -o '+md_folder+'/output_models/'+model+'/'+str(i)+'/topol/index.ndx'+' -n '+md_folder+'/output_models/'+model+'/'+str(i)+'/topol/index.ndx'+'\n')

                    # Update group_dics
                    group_dics['complex'] = _readGromacsIndexFile(md_folder+'/'+'output_models/'+model+'/'+str(i)+'/topol'+'/index.ndx')

                sol_group = 'SOL'

                # With the index info now add the ions (now we can select the SOL :D)
                command_local = command
                command_local += (
                    "cd output_models/" + model + "/" + str(i) + "/topol" + "\n"
                )
                # command_local += 'sed -i -e \'s/SOL/Water_\&_!CrystalWaters/g\' topol.top \n'
                command_local += (
                    command_name
                    + " grompp -f ../../../../scripts/ions.mdp -c prot_solv.gro -p topol.top -o prot_ions.tpr -maxwarn 1"
                    + "\n"
                )
                command_local += (
                    "echo "
                    + group_dics["complex"][sol_group]
                    + " | "
                    + command_name
                    + " genion -s prot_ions.tpr -o prot_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.1 -n index.ndx"
                    + "\n"
                )
                command_local += (
                    'echo "q"| '
                    + command_name
                    + " make_ndx -f  prot_ions.gro -o index.ndx"
                    + "\n"
                )

                # command_local += 'sed -i -e \'s/CWT/SOL/g\' topol.top \n'

                # Run local commands
                with open("tmp.sh", "w") as f:
                    f.write(command_local)
                subprocess.run("bash tmp.sh", shell=True)
                os.remove("tmp.sh")

                group_dics["complex"] = _readGromacsIndexFile(
                    md_folder
                    + "/"
                    + "output_models/"
                    + model
                    + "/"
                    + str(i)
                    + "/topol"
                    + "/index.ndx"
                )

                if ligand_chains != None or ion_residues != []:
                    # If we have ligands or ions we must do more stuff
                    command_local = command
                    command_local += (
                        "cd output_models/" + model + "/" + str(i) + "/topol" + "\n"
                    )
                    # Generate ligand index and Protein_Ligand selector
                    lig_selector = ""
                    if ligand_chains != None:
                        for ligand_name in ligand_res.values():
                            command_local += (
                                'echo -e "0 & ! a H*\\nq"| '
                                + command_name
                                + " make_ndx -f  "
                                + ligand_name
                                + ".acpype/"
                                + ligand_name
                                + "_GMX.gro -o "
                                + ligand_name
                                + "_index.ndx"
                                + "\n"
                            )
                            lig_selector += group_dics["complex"][ligand_name] + "|"

                    # Generate Protein_ProteinIons selector and Water_and_Ions_and_notProteinIons selector
                    ion_selector = ""
                    water_and_solventions_selector = ""
                    if ion_residues != []:
                        for r in ion_residues:
                            ion_selector += "r " + str(r) + "|"
                            water_and_solventions_selector += " ! r " + str(r) + " &"

                    selector_line = ""
                    # If we have both lig and ions we need:
                    #  - selector of prot_ion_lig for first tc groups
                    #  - selector of water_and_solvent_ions for second tc group
                    #  - selector for protein_ion for constraints
                    if lig_selector != "" and ion_selector != "":
                        selector_line += (
                            group_dics["complex"]["Protein"]
                            + "|"
                            + ion_selector[:-1]
                            + "|"
                            + lig_selector[:-1]
                            + "\\n"
                        )
                        selector_line += (
                            group_dics["complex"]["Protein"]
                            + "|"
                            + ion_selector
                            + "\\n"
                        )
                        selector_line += (
                            group_dics["complex"][sol_group]
                            + " | "
                            + group_dics["complex"]["Ion"]
                            + " & "
                            + water_and_solventions_selector[:-1]
                            + "\\n"
                        )
                    # If only ion we dont need prot_ion_lig (we use same for tc group and constraint)
                    elif ion_selector != "":
                        selector_line += (
                            group_dics["complex"]["Protein"]
                            + "|"
                            + ion_selector
                            + "\\n"
                        )
                        selector_line += (
                            group_dics["complex"][sol_group]
                            + " | "
                            + group_dics["complex"]["Ion"]
                            + " & "
                            + water_and_solventions_selector[:-1]
                            + "\\n"
                        )
                    # If not ions we only need prot_lig for tc group and can use water_and_not_ions for the other
                    elif lig_selector != "":
                        selector_line += (
                            group_dics["complex"]["Protein"]
                            + "|"
                            + lig_selector
                            + "\\n"
                        )

                    print(selector_line)
                    command_local += (
                        'echo -e "'
                        + selector_line
                        + 'q"| '
                        + command_name
                        + " make_ndx -f  prot_ions.gro -o index.ndx"
                        + "\n"
                    )

                    # Run local commands
                    with open("tmp.sh", "w") as f:
                        f.write(command_local)
                    subprocess.run("bash tmp.sh", shell=True)
                    os.remove("tmp.sh")

                    # Update complex indexes and add ligand index
                    group_dics["complex"] = _readGromacsIndexFile(
                        md_folder
                        + "/"
                        + "output_models/"
                        + model
                        + "/"
                        + str(i)
                        + "/topol"
                        + "/index.ndx"
                    )

                    if ligand_chains != None:
                        for ligand_name in ligand_res.values():
                            group_dics[ligand_name] = _readGromacsIndexFile(
                                md_folder
                                + "/"
                                + "output_models/"
                                + model
                                + "/"
                                + str(i)
                                + "/topol"
                                + "/"
                                + ligand_name
                                + "_index.ndx"
                            )

                # sed -i  "s#/home/miguel/Nextprot/casein_design/KAP/svn_muts/test_mds_delet/FF#$path/FF#g" output_models/KAP_ALAR_17_S47_H111-ND1_D115_0012_0012_0276/0/topol/topol.top
                command += "cd output_models/" + model + "/" + str(i) + "\n"

                # CHANGE PATH NAMES OF FF IN TOPOL FILE
                local_path = os.getcwd() + "/" + md_folder + "/FF"
                command += (
                    'sed -i  "s#' + local_path + '#$GMXLIB#g" topol/topol.top' + "\n"
                )

                # Energy minimization
                if not os.path.exists(
                    md_folder
                    + "/output_models/"
                    + model
                    + "/"
                    + str(i)
                    + "/em/prot_em.tpr"
                ):
                    command += "mkdir em" + "\n"
                    command += "cd em" + "\n"
                    command += (
                        remote_command_name
                        + " grompp -f ../../../../scripts/em.mdp -c ../topol/prot_ions.gro -p ../topol/topol.top -o prot_em.tpr"
                        + "\n"
                    )
                    command += remote_command_name + " mdrun -v -deffnm prot_em" + "\n"
                    command += "cd .." + "\n"

                # NVT equilibration
                if not os.path.exists(
                    md_folder
                    + "/output_models/"
                    + model
                    + "/"
                    + str(i)
                    + "/nvt/prot_nvt.tpr"
                ):
                    command += "mkdir nvt" + "\n"
                    command += "cd nvt" + "\n"
                    command += "cp -r ../../../../scripts/nvt.mdp ." + "\n"

                    tc_grps1 = ["Protein"]
                    if ion_residues != []:
                        tc_grps2 = "SOL_Ion"
                        for r in ion_residues:
                            tc_grps1.append("r_" + str(r))
                            tc_grps2 += "_&_!r_" + str(r)
                    else:
                        tc_grps2 = "Water_and_ions"

                    if ligand_chains != None:
                        tc_grps1.extend(ligand_res.values())

                    command += (
                        "sed -i  '/tc-grps/c\\tc-grps = "
                        + "_".join(tc_grps1)
                        + " "
                        + tc_grps2
                        + "' nvt.mdp"
                        + "\n"
                    )

                    if ligand_chains != None:
                        for ligand_name in ligand_res.values():
                            command += (
                                "echo "
                                + group_dics[ligand_name]["System_&_!H*"]
                                + " | "
                                + remote_command_name
                                + " genrestr -f ../topol/"
                                + ligand_name
                                + ".acpype/"
                                + ligand_name
                                + "_GMX.gro -n ../topol/"
                                + ligand_name
                                + "_index.ndx -o ../topol/"
                                + ligand_name
                                + ".acpype/posre_"
                                + ligand_name
                                + ".itp -fc 1000 1000 1000"
                                + "\n"
                            )

                    if ion_residues != []:
                        grp_name = "Protein"
                        for r in ion_residues:
                            grp_name += "_r_" + str(r)
                        sel = group_dics["complex"][grp_name]
                    else:
                        sel = group_dics["complex"]["Protein"]
                    command += (
                        "echo "
                        + sel
                        + " | "
                        + remote_command_name
                        + " genrestr -f ../topol/prot_ions.gro -o ../topol/posre.itp -fc 1000 1000 1000 -n ../topol/index.ndx\n"
                    )

                    command += (
                        remote_command_name
                        + " grompp -f nvt.mdp -c ../em/prot_em.gro -p ../topol/topol.top -o prot_nvt.tpr -r ../em/prot_em.gro -n ../topol/index.ndx\n"
                    )
                    command += remote_command_name + " mdrun -v -deffnm prot_nvt" + "\n"
                    command += "cd .." + "\n"

                # NPT equilibration
                FClist = ("550", "300", "170", "90", "50", "30", "15", "10", "5")
                if not os.path.exists(
                    md_folder + "/output_models/" + model + "/" + str(i) + "/npt"
                ):
                    command += "mkdir npt" + "\n"
                command += "cd npt" + "\n"

                tc_grps1 = ["Protein"]
                if ion_residues != []:
                    tc_grps2 = "SOL_Ion"
                    for r in ion_residues:
                        tc_grps1.append("r_" + str(r))
                        tc_grps2 += "_&_!r_" + str(r)
                else:
                    tc_grps2 = "Water_and_ions"

                if ligand_chains != None:
                    tc_grps1.extend(ligand_res.values())

                command += "cp -r ../../../../scripts/npt.mdp ." + "\n"
                command += (
                    "sed -i  '/tc-grps/c\\tc-grps = "
                    + "_".join(tc_grps1)
                    + " "
                    + tc_grps2
                    + "' npt.mdp"
                    + "\n"
                )

                if ion_residues != []:
                    grp_name = "Protein"
                    for r in ion_residues:
                        grp_name += "_r_" + str(r)
                    sel = group_dics["complex"][grp_name]
                else:
                    sel = group_dics["complex"]["Protein"]

                for i in range(len(FClist) + 1):
                    if not os.path.exists(
                        md_folder
                        + "/output_models/"
                        + model
                        + "/"
                        + str(i)
                        + "/npt/prot_npt_"
                        + str(i + 1)
                        + ".tpr"
                    ):
                        if i == 0:
                            command += (
                                remote_command_name
                                + " grompp -f npt.mdp -c ../nvt/prot_nvt.gro -t ../nvt/prot_nvt.cpt -p ../topol/topol.top -o prot_npt_1.tpr -r ../nvt/prot_nvt.gro -n ../topol/index.ndx\n"
                            )
                            command += (
                                remote_command_name
                                + " mdrun -v -deffnm prot_npt_"
                                + str(i + 1)
                                + "\n"
                            )
                        else:
                            if ligand_chains != None:
                                for ligand_name in ligand_res.values():
                                    command += (
                                        "echo "
                                        + group_dics[ligand_name]["System_&_!H*"]
                                        + " | "
                                        + remote_command_name
                                        + " genrestr -f ../topol/"
                                        + ligand_name
                                        + ".acpype/"
                                        + ligand_name
                                        + "_GMX.gro -n ../topol/"
                                        + ligand_name
                                        + "_index.ndx -o ../topol/"
                                        + ligand_name
                                        + ".acpype/posre_"
                                        + ligand_name
                                        + ".itp  -fc "
                                        + FClist[i - 1]
                                        + " "
                                        + FClist[i - 1]
                                        + " "
                                        + FClist[i - 1]
                                        + "\n"
                                    )

                            command += (
                                "echo "
                                + sel
                                + " | "
                                + remote_command_name
                                + " genrestr -f ../topol/prot_ions.gro -o ../topol/posre.itp  -fc "
                                + FClist[i - 1]
                                + " "
                                + FClist[i - 1]
                                + " "
                                + FClist[i - 1]
                                + " -n ../topol/index.ndx\n"
                            )

                            command += (
                                remote_command_name
                                + " grompp -f npt.mdp -c prot_npt_"
                                + str(i)
                                + ".gro -t prot_npt_"
                                + str(i)
                                + ".cpt -p ../topol/topol.top -o prot_npt_"
                                + str(i + 1)
                                + ".tpr -r prot_npt_"
                                + str(i)
                                + ".gro -n ../topol/index.ndx\n"
                            )
                            command += (
                                remote_command_name
                                + " mdrun -v -deffnm prot_npt_"
                                + str(i + 1)
                                + "\n"
                            )
                command += "cd .." + "\n"

                # Production run
                if not os.path.exists(
                    md_folder + "/output_models/" + model + "/" + str(i) + "/md"
                ):
                    command += "mkdir md" + "\n"
                command += "cd md" + "\n"

                tc_grps1 = ["Protein"]
                if ion_residues != []:
                    tc_grps2 = "SOL_Ion"
                    for r in ion_residues:
                        tc_grps1.append("r_" + str(r))
                        tc_grps2 += "_&_!r_" + str(r)
                else:
                    tc_grps2 = "Water_and_ions"

                if ligand_chains != None:
                    tc_grps1.extend(ligand_res.values())

                command += "cp -r ../../../../scripts/md.mdp ." + "\n"
                command += (
                    "sed -i  '/tc-grps/c\\tc-grps = "
                    + "_".join(tc_grps1)
                    + " "
                    + tc_grps2
                    + "' md.mdp"
                    + "\n"
                )

                for i in range(1, frags + 1):
                    if not os.path.exists(
                        md_folder
                        + "/output_models/"
                        + model
                        + "/"
                        + str(i)
                        + "/md/prot_md_"
                        + str(i)
                        + ".xtc"
                    ):
                        if i == 1:
                            command += (
                                remote_command_name
                                + " grompp -f md.mdp -c ../npt/prot_npt_"
                                + str(len(FClist) + 1)
                                + ".gro  -t ../npt/prot_npt_"
                                + str(len(FClist) + 1)
                                + ".cpt -p ../topol/topol.top -o prot_md_"
                                + str(i)
                                + ".tpr -n ../topol/index.ndx"
                                + "\n"
                            )
                            command += (
                                remote_command_name
                                + " mdrun -v -deffnm prot_md_"
                                + str(i)
                                + "\n"
                            )
                        else:
                            command += (
                                remote_command_name
                                + " grompp -f md.mdp -c prot_md_"
                                + str(i - 1)
                                + ".gro -t prot_md_"
                                + str(i - 1)
                                + ".cpt -p ../topol/topol.top -o prot_md_"
                                + str(i)
                                + ".tpr -n ../topol/index.ndx"
                                + "\n"
                            )
                            command += (
                                remote_command_name
                                + " mdrun -v -deffnm prot_md_"
                                + str(i)
                                + "\n"
                            )
                    else:
                        if os.path.exists(
                            md_folder
                            + "/output_models/"
                            + model
                            + "/"
                            + str(i)
                            + "/md/prot_md_"
                            + str(i)
                            + "_prev.cpt"
                        ):
                            command += (
                                remote_command_name
                                + " mdrun -v -deffnm prot_md_"
                                + str(i)
                                + " -cpi prot_md_"
                                + str(i)
                                + "_prev.cpt"
                                + "\n"
                            )

                jobs.append(command)

        return jobs

    def getTrajectoryPaths(self, path, step="md", traj_name="prot_md_cat_noPBC.xtc"):
        """ """
        output_paths = []
        for folder in os.listdir(path + "/output_models/"):
            if folder in self.models_names:
                traj_path = path + "/output_models/" + folder + "/" + step
                output_paths.append(traj_path + "/" + traj_name)

        return output_paths

    def removeBoundaryConditions(self, path, command, step="md", remove_water=False):
        """
        Remove boundary conditions from gromacs simulation trajectory file

        Parameters
        ==========
        path : str
            Path to the job folder where the MD outputs files are located.
        command : str
            Command to call program.
        """
        for folder in os.listdir(path + "/output_models/"):
            if folder in self.models_names:
                traj_path = path + "/output_models/" + folder + "/" + step
                for file in os.listdir(traj_path):
                    if (
                        file.endswith(".xtc")
                        and not file.endswith("_noPBC.xtc")
                        and not os.path.exists(
                            traj_path + "/" + file.split(".")[0] + "_noPBC.xtc"
                        )
                    ):
                        if remove_water == True:
                            option = "14"
                        else:
                            option = "0"
                        os.system(
                            "echo "
                            + option
                            + " | "
                            + command
                            + " trjconv -s "
                            + traj_path
                            + "/"
                            + file.split(".")[0]
                            + ".tpr -f "
                            + traj_path
                            + "/"
                            + file
                            + " -o "
                            + traj_path
                            + "/"
                            + file.split(".")[0]
                            + "_noPBC.xtc -pbc mol -ur compact"
                        )

                if not os.path.exists(traj_path + "/prot_md_cat_noPBC.xtc"):
                    os.system(
                        command
                        + " trjcat -f "
                        + traj_path
                        + "/*_noPBC.xtc -o "
                        + traj_path
                        + "/prot_md_cat_noPBC.xtc -cat"
                    )

                ### md_1 or npt_10

                if (
                    not os.path.exists(
                        "/".join(traj_path.split("/")[:-1])
                        + "/npt/prot_npt_10_no_water.gro"
                    )
                    and remove_water == True
                ):
                    os.system(
                        "echo 1 | gmx editconf -ndef -f "
                        + "/".join(traj_path.split("/")[:-1])
                        + "/npt/prot_npt_10.gro -o "
                        + "/".join(traj_path.split("/")[:-1])
                        + "/npt/prot_npt_10_no_water.gro"
                    )

    def analyseDocking(
        self,
        docking_folder,
        protein_atoms=None,
        angles=None,
        atom_pairs=None,
        skip_chains=False,
        return_failed=False,
        ignore_hydrogens=False,
        separator="-",
        overwrite=True,
        output_folder='.analysis',
    ):
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
        separator : str
            Symbol to use for separating protein from ligand names. Should not be found in any model or ligand name.
        overwrite : bool
            Rerun analysis.
        """

        # Create analysis folder
        if not os.path.exists(docking_folder + '/'+output_folder):
            os.mkdir(docking_folder + '/'+output_folder)

        # Create analysis folder
        if not os.path.exists(docking_folder + '/'+output_folder+"/atom_pairs"):
            os.mkdir(docking_folder + '/'+output_folder+"/atom_pairs")

        # Create analysis folder
        if angles:
            if not os.path.exists(docking_folder + '/'+output_folder+"/angles"):
                os.mkdir(docking_folder + '/'+output_folder+"/angles")

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        prepare_proteins._copyScriptFile(
            docking_folder + '/'+output_folder, "analyse_docking.py"
        )
        script_path = docking_folder + '/'+output_folder+"/._analyse_docking.py"

        # Write protein_atoms dictionary to json file
        if protein_atoms:
            with open(docking_folder + '/'+output_folder+"/._protein_atoms.json", "w") as jf:
                json.dump(protein_atoms, jf)

        # Write atom_pairs dictionary to json file
        if atom_pairs:
            with open(docking_folder + '/'+output_folder+"/._atom_pairs.json", "w") as jf:
                json.dump(atom_pairs, jf)

        # Write angles dictionary to json file
        if angles:
            with open(docking_folder + '/'+output_folder+"/._angles.json", "w") as jf:
                json.dump(angles, jf)

        command = (
            "run "
            + docking_folder
            + '/'+output_folder+"/._analyse_docking.py "
            + docking_folder
        )
        if atom_pairs:
            command += (
                " --atom_pairs " + docking_folder + '/'+output_folder+"/._atom_pairs.json"
            )
        elif protein_atoms:
            command += (
                " --protein_atoms " + docking_folder + '/'+output_folder+"/._protein_atoms.json"
            )
        if angles:
            command += " --angles " + docking_folder + '/'+output_folder+"/._angles.json"
        if skip_chains:
            command += " --skip_chains"
        if return_failed:
            command += " --return_failed"
        if ignore_hydrogens:
            command += " --ignore_hydrogens"
        command += " --separator " + separator
        command += " --only_models " + ",".join(self.models_names)
        if overwrite:
            command += " --overwrite "

        os.system(command)

        # Read the CSV file into pandas
        if not os.path.exists(docking_folder + '/'+output_folder+"/docking_data.csv"):
            raise ValueError(
                "Docking analysis failed. Check the ouput of the analyse_docking.py script."
            )

        self.docking_data = pd.read_csv(docking_folder + '/'+output_folder+"/docking_data.csv")
        # Create multiindex dataframe
        self.docking_data.set_index(["Protein", "Ligand", "Pose"], inplace=True)
        # Force de definition of the MultiIndex
        self.docking_data.index = pd.MultiIndex.from_tuples(
            self.docking_data.index, names=["Protein", "Ligand", "Pose"]
        )

        for f in os.listdir(docking_folder + '/'+output_folder+"/atom_pairs"):
            model = f.split(separator)[0]
            ligand = f.split(separator)[1].split(".")[0]

            # Read the CSV file into pandas
            self.docking_distances.setdefault(model, {})
            self.docking_distances[model][ligand] = pd.read_csv(
                docking_folder + '/'+output_folder+"/atom_pairs/" + f
            )
            self.docking_distances[model][ligand].set_index(
                ["Protein", "Ligand", "Pose"], inplace=True
            )

            self.docking_ligands.setdefault(model, [])
            if ligand not in self.docking_ligands[model]:
                self.docking_ligands[model].append(ligand)

        if angles:
            for f in os.listdir(docking_folder + '/'+output_folder+"/atom_pairs"):
                model = f.split(separator)[0]
                ligand = f.split(separator)[1].split(".")[0]

                # Read the CSV file into pandas
                self.docking_angles.setdefault(model, {})
                self.docking_angles[model][ligand] = pd.read_csv(
                    docking_folder + '/'+output_folder+"/angles/" + f
                )
                self.docking_angles[model][ligand].set_index(
                    ["Protein", "Ligand", "Pose"], inplace=True
                )

        if return_failed:
            with open(docking_folder + '/'+output_folder+"/._failed_dockings.json") as jifd:
                failed_dockings = json.load(jifd)
            return failed_dockings

    def convertLigandPDBtoMae(self, ligands_folder, change_ligand_name=True):
        """
        Convert ligand PDBs into MAE files.

        Parameters
        ==========
        ligands_folder : str
            Path to the folder where ligands are in PDB format
        """

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(ligands_folder, "PDBtoMAE.py")
        script_name = "._PDBtoMAE.py"

        cwd = os.getcwd()
        os.chdir(ligands_folder)
        command = "run ._PDBtoMAE.py"
        if change_ligand_name:
            command += " --change_ligand_name"
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
        _copyScriptFile(ligands_folder, "MAEtoPDB.py")
        script_name = "._MAEtoPDB.py"

        cwd = os.getcwd()
        os.chdir(ligands_folder)
        os.system("run ._MAEtoPDB.py")
        os.chdir(cwd)

    def getDockingDistances(self, model, ligand):
        """
        Get the distances related to a model and ligand docking.
        """
        distances = []

        if model not in self.docking_distances:
            return None

        if ligand not in self.docking_distances[model]:
            return None

        for d in self.docking_distances[model][ligand]:
            distances.append(d)

        if distances != []:
            return distances
        else:
            return None

    def calculateModelsDistances(self, atom_pairs, verbose=False):
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

        ### Add all label entries to dictionary
        for model in self.structures:
            self.distance_data[model] = {}
            self.distance_data[model]["model"] = []

            for d in atom_pairs[model]:
                # Generate label for distance
                label = "distance_"
                label += "_".join([str(x) for x in d[0]]) + "-"
                label += "_".join([str(x) for x in d[1]])
                self.distance_data[model].setdefault(label, [])

        for model in self.structures:

            if verbose:
                print("Calculating distances for model %s" % model)

            self.distance_data[model]["model"].append(model)

            # Get atoms in atom_pairs as dictionary
            atoms = {}
            for d in atom_pairs[model]:
                for t in d:
                    atoms.setdefault(t[0], {})
                    atoms[t[0]].setdefault(t[1], [])
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
                                    coordinates[chain.id][r.id[1]][
                                        atom.name
                                    ] = atom.coord

            # Calculate atom distances
            for d in atom_pairs[model]:

                # Generate label for distance
                label = "distance_"
                label += "_".join([str(x) for x in d[0]]) + "-"
                label += "_".join([str(x) for x in d[1]])

                # Calculate distance
                atom1 = d[0]
                atom2 = d[1]

                if atom1[2] not in coordinates[atom1[0]][atom1[1]]:
                    raise ValueError(
                        "Atom name %s was not found in residue %s of chain %s"
                        % (atom1[2], atom1[1], atom1[0])
                    )
                if atom2[2] not in coordinates[atom2[0]][atom2[1]]:
                    raise ValueError(
                        "Atom name %s was not found in residue %s of chain %s"
                        % (atom2[2], atom2[1], atom2[0])
                    )

                coord1 = coordinates[atom1[0]][atom1[1]][atom1[2]]
                coord2 = coordinates[atom2[0]][atom2[1]][atom2[2]]
                value = np.linalg.norm(coord1 - coord2)

                # Add data to dataframe
                self.distance_data[model][label].append(value)

            self.distance_data[model] = pd.DataFrame(
                self.distance_data[model]
            ).reset_index()
            self.distance_data[model].set_index("model", inplace=True)

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

        model_data = self.distance_data[model]
        distances = []
        for d in model_data:
            if "distance_" in d:
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

        if isinstance(self.models_data, dict) and self.models_data == {}:
            self.models_data["model"] = [m for m in self]

        for name in metric_distances:

            if "metric_" + name in self.models_data.keys() and not overwrite:
                print(
                    "Combined metric %s already added. Give overwrite=True to recombine"
                    % name
                )
            else:
                values = []
                models = []

                for model in self:
                    model_data = self.distance_data[model]
                    model_distances = metric_distances[name][model]
                    values += model_data[model_distances].min(axis=1).tolist()

                self.models_data["metric_" + name] = values

        self.models_data = pd.DataFrame(self.models_data)
        print(self.models_data)
        self.models_data.set_index("model", inplace=True)

        return self.models_data

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
        self.protonation_states["model"] = []
        self.protonation_states["chain"] = []
        self.protonation_states["residue"] = []
        self.protonation_states["name"] = []
        self.protonation_states["state"] = []

        # Iterate all models' structures
        for model in self.models_names:
            structure = self.structures[model]
            for r in structure.get_residues():

                # Skip if a list of residues is given per model
                if residues != None:
                    if (r.get_parent().id, r.id[1]) not in residues[model]:
                        continue

                # Get Histidine protonation states
                if r.resname == "HIS":
                    atoms = [a.name for a in r]
                    self.protonation_states["model"].append(model)
                    self.protonation_states["chain"].append(r.get_parent().id)
                    self.protonation_states["residue"].append(r.id[1])
                    self.protonation_states["name"].append(r.resname)
                    if "HE2" in atoms and "HD1" in atoms:
                        self.protonation_states["state"].append("HIP")
                    elif "HD1" in atoms:
                        self.protonation_states["state"].append("HID")
                    elif "HE2" in atoms:
                        self.protonation_states["state"].append("HIE")

        # Convert dictionary to Pandas
        self.protonation_states = pd.DataFrame(self.protonation_states)
        self.protonation_states.set_index(["model", "chain", "residue"], inplace=True)

        return self.protonation_states

    def combineDockingDistancesIntoMetrics(self, catalytic_labels, overwrite=False):
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
            if "metric_" + name in self.docking_data.keys() and not overwrite:
                print(
                    "Combined metric %s already added. Give overwrite=True to recombine"
                    % name
                )
            else:
                values = []
                for model in self.docking_data.index.levels[0]:

                    # Check whether model is found in docking distances
                    if model not in self.docking_distances:
                        continue

                    model_series = self.docking_data[
                        self.docking_data.index.get_level_values("Protein") == model
                    ]

                    for ligand in self.docking_data.index.levels[1]:

                        # Check whether ligand is found in model's docking distances
                        if ligand not in self.docking_distances[model]:
                            continue

                        ligand_series = model_series[
                            model_series.index.get_level_values("Ligand") == ligand
                        ]

                        # Check input metric
                        # Check how metrics will be combined
                        distance_metric = False
                        angle_metric = False
                        for x in catalytic_labels[name][model][ligand]:
                            if len(x.split("-")) == 2:
                                distance_metric = True
                            elif len(x.split("-")) == 3:
                                angle_metric = True

                        if distance_metric and angle_metric:
                            raise ValueError(
                                f"Metric {m} combines distances and angles which is not supported."
                            )

                        if distance_metric:
                            distances = catalytic_labels[name][model][ligand]
                            distance_values = (
                                self.docking_distances[model][ligand][distances]
                                .min(axis=1)
                                .tolist()
                            )
                            assert ligand_series.shape[0] == len(distance_values)
                            values += distance_values
                            self.docking_metric_type[name] = "distance"
                        elif angle_metric:
                            angles = catalytic_labels[name][model][ligand]
                            if len(angles) > 1:
                                raise ValueError(
                                    "Combining more than one angle into a metric is not currently supported."
                                )
                            angle_values = (
                                self.docking_angles[model][ligand][angles]
                                .min(axis=1)
                                .tolist()
                            )
                            assert ligand_series.shape[0] == len(angle_values)
                            values += angle_values
                            self.docking_metric_type[name] = "angle"

                self.docking_data["metric_" + name] = values

    def getBestDockingPoses(
        self,
        filter_values,
        n_models=1,
        return_failed=False,
        exclude_models=None,
        exclude_ligands=None,
        exclude_pairs=None,
    ):
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
        exclude_models : list
            List of models to be excluded from the selection.
        exclude_ligands : list
            List of ligands to be excluded from the selection.
        exclude_pairs : list
            List of pair tuples (model, ligand) to be excluded from the selection.

        """

        if exclude_models == None:
            exclude_models = []
        if exclude_ligands == None:
            exclude_ligands = []
        if exclude_pairs == None:
            exclude_pairs = []

        best_poses = pd.DataFrame()
        bp = []
        failed = []
        for model in self.docking_ligands:

            if model in exclude_models:
                continue

            protein_series = self.docking_data[
                self.docking_data.index.get_level_values("Protein") == model
            ]

            for ligand in self.docking_ligands[model]:

                if ligand in exclude_ligands:
                    continue

                if (model, ligand) in exclude_pairs:
                    continue

                ligand_data = protein_series[
                    protein_series.index.get_level_values("Ligand") == ligand
                ]
                for metric in filter_values:

                    if metric not in ["Score", "RMSD"]:
                        # Add prefix if not given
                        if not metric.startswith("metric_"):
                            metric_label = "metric_" + metric
                        else:
                            metric_label = metric

                        # Filter values according to the type of threshold given
                        if isinstance(filter_values[metric], (float, int)):
                            ligand_data = ligand_data[
                                ligand_data[metric_label] <= filter_values[metric]
                            ]
                        elif isinstance(filter_values[metric], (tuple, list)):
                            ligand_data = ligand_data[
                                ligand_data[metric_label] >= filter_values[metric][0]
                            ]
                            ligand_data = ligand_data[
                                ligand_data[metric_label] <= filter_values[metric][1]
                            ]
                    else:
                        metric_label = metric
                        ligand_data = ligand_data[
                            ligand_data[metric_label] < filter_values[metric]
                        ]

                if ligand_data.empty:
                    failed.append((model, ligand))
                    continue
                if ligand_data.shape[0] < n_models:
                    print(
                        "WARNING: less than %s models available for docking %s + %s"
                        % (n_models, model, ligand)
                    )
                for i in ligand_data["Score"].nsmallest(n_models).index:
                    bp.append(i)

        if return_failed:
            return failed, self.docking_data[self.docking_data.index.isin(bp)]
        return self.docking_data[self.docking_data.index.isin(bp)]

    def getBestDockingPosesIteratively(
        self, metrics, ligands=None, distance_step=0.1, angle_step=1.0, fixed=None
    ):

        # Create a list for fixed metrics
        if not fixed:
            fixed = []
        elif isinstance(fixed, str):
            fixed = [fixed]

        if set(metrics.keys()) - set(fixed) == set():
            raise ValueError("You must leave at least one metric not fixed")

        metrics = metrics.copy()

        extracted = []
        selected_indexes = []

        # Define all protein and ligand combinations with docking data
        protein_and_ligands = set([x[:2] for x in self.docking_data.index])

        extracted = set()  # Save extracted models
        selected_indexes = []
        while len(extracted) < len(protein_and_ligands):

            # Get best poses with current thresholds
            best_poses = self.getBestDockingPoses(metrics, n_models=1)

            # Save indexes of best models
            selected_protein_ligands = set()
            for index in best_poses.index:
                if (
                    index[:2] not in extracted
                ):  # Omit selected models in previous iterations
                    selected_indexes.append(index)
                    selected_protein_ligands.add(index[:2])

            # Store models extracted at this iteration
            for pair in selected_protein_ligands:
                extracted.add(pair)

            # Get docking data for missing entries
            mask = []
            for index in self.docking_data.index:
                if index[:2] in (protein_and_ligands - extracted):
                    mask.append(True)
                else:
                    mask.append(False)

            remaining_data = self.docking_data[np.array(mask)]

            # Compute metric acceptance for each metric for all missing pairs
            if not remaining_data.empty:
                metric_acceptance = {}
                for metric in metrics:
                    if not metric.startswith("metric_"):
                        metric_label = "metric_" + metric
                    else:
                        metric_label = metric
                    if isinstance(metrics[metric], (float, int)):
                        metric_acceptance[metric] = remaining_data[
                            remaining_data[metric_label] <= metrics[metric]
                        ].shape[0]
                    elif isinstance(metrics[metric], (tuple, list)):
                        metric_filter = remaining_data[
                            metrics[metric][0] <= remaining_data[metric_label]
                        ]
                        metric_acceptance[metric] = metric_filter[
                            metric_filter[metric_label] <= metrics[metric][1]
                        ].shape[0]

                lowest_metric = [
                    m
                    for m, a in sorted(metric_acceptance.items(), key=lambda x: x[1])
                    if m not in fixed
                ][0]
                lowest_metric_doc = lowest_metric.replace("metric_", "")
                if self.docking_metric_type[lowest_metric_doc] == "distance":
                    step = distance_step
                if self.docking_metric_type[lowest_metric_doc] == "angle":
                    step = angle_step

                if isinstance(metrics[lowest_metric], (float, int)):
                    metrics[lowest_metric] += step

                # Change to list to allow item assignment
                if isinstance(metrics[lowest_metric], tuple):
                    metrics[lowest_metric] = list(metrics[lowest_metric])

                if isinstance(metrics[lowest_metric], list):
                    metrics[lowest_metric][0] -= step
                    metrics[lowest_metric][1] += step

        # Get rows with the selected indexes
        mask = self.docking_data.index.isin(selected_indexes)
        best_poses = self.docking_data[mask]

        return best_poses

    def extractDockingPoses(
        self,
        docking_data,
        docking_folder,
        output_folder,
        separator="-",
        only_extract_new=True,
        covalent_check=True,
        remove_previous=False,
    ):
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
        only_extract_new : bool
            Only extract models not present in the output_folder
        remove_previous : bool
            Remove all content in the output folder
        """

        # Check the separator is not in model or ligand names
        for model in self.docking_ligands:
            if separator in str(model):
                raise ValueError(
                    "The separator %s was found in model name %s. Please use a different separator symbol."
                    % (separator, model)
                )
            for ligand in self.docking_ligands[model]:
                if separator in ligand:
                    raise ValueError(
                        "The separator %s was found in ligand name %s. Please use a different separator symbol."
                        % (separator, ligand)
                    )

        # Remove output_folder
        if os.path.exists(output_folder):
            if remove_previous:
                shutil.rmtree(output_folder)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        else:
            # Gather already extracted models
            if only_extract_new:
                extracted_models = set()
                for model in os.listdir(output_folder):
                    if not os.path.isdir(output_folder + "/" + model):
                        continue
                    for f in os.listdir(output_folder + "/" + model):
                        if f.endswith(".pdb"):
                            m, l = f.split(separator)[:2]
                            extracted_models.add((m, l))

                # Filter docking data to not include the already extracted models
                extracted_indexes = []
                for i in docking_data.index:
                    if i[:2] in extracted_models:
                        extracted_indexes.append(i)
                docking_data = docking_data[~docking_data.index.isin(extracted_indexes)]
                if docking_data.empty:
                    print("All models were already extracted!")
                    print("Set only_extract_new=False to extract them again!")
                    return
                else:
                    print(f"{len(extracted_models)} models were already extracted!")
                    print(f"Extracting {docking_data.shape[0]} new models")

        # Copy analyse docking script (it depends on schrodinger so we leave it out.)
        _copyScriptFile(output_folder, "extract_docking.py")
        script_path = output_folder + "/._extract_docking.py"

        # Move to output folder
        os.chdir(output_folder)

        # Save given docking data to csv
        dd = docking_data.reset_index()
        dd.to_csv("._docking_data.csv", index=False)

        # Execute docking analysis
        command = (
            "run ._extract_docking.py ._docking_data.csv ../"
            + docking_folder
            + " --separator "
            + separator
        )
        os.system(command)

        # Remove docking data
        os.remove("._docking_data.csv")

        # move back to folder
        os.chdir("..")

        # Check models for covalent residues
        for protein in os.listdir(output_folder):
            if not os.path.isdir(output_folder + "/" + protein):
                continue
            for f in os.listdir(output_folder + "/" + protein):
                if covalent_check:
                    self._checkCovalentLigands(
                        protein,
                        output_folder + "/" + protein + "/" + f,
                        check_file=True,
                    )

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
            raise ValueError("has no docking data")

        if isinstance(data_frame, type(None)):
            data_frame = self.docking_data

        protein_series = data_frame[
            data_frame.index.get_level_values("Protein") == protein
        ]
        ligand_series = protein_series[
            protein_series.index.get_level_values("Ligand") == ligand
        ]

        return ligand_series

    def plotDocking(
        self,
        protein,
        ligand,
        x="RMSD",
        y="Score",
        z=None,
        colormap="Blues_r",
        output_folder=None,
        extension=".png",
        dpi=200,
    ):

        if output_folder != None:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

        protein_series = self.docking_data[
            self.docking_data.index.get_level_values("Protein") == protein
        ]
        if protein_series.empty:
            print("Model %s not found in Docking data" % protein)
            return None
        ligand_series = protein_series[
            protein_series.index.get_level_values("Ligand") == ligand
        ]
        if ligand_series.empty:
            print(
                "Ligand %s not found in Docking data for protein %s" % (ligand, protein)
            )
            return None

        fig, ax = plt.subplots()
        if z != None:
            ligand_series.plot(kind="scatter", x=x, y=y, c=z, colormap=colormap, ax=ax)
        else:
            ligand_series.plot(kind="scatter", x=x, y=y, ax=ax)

        plt.title(protein + " + " + ligand)
        if output_folder != None:
            plt.savefig(
                output_folder + "/" + protein + "_" + ligand + extension, dpi=dpi
            )
            plt.close()

    def loadModelsFromPrepwizardFolder(
        self,
        prepwizard_folder,
        return_missing=False,
        return_failed=False,
        covalent_check=True,
        models=None,
        atom_mapping=None,
        conect_update=False,
        replace_symbol=None,
        collect_memory_every=None,
    ):
        """
        Read structures from a Schrodinger calculation.

        Parameters
        ==========
        prepwizard_folder : str
            Path to the output folder from a prepwizard calculation
        """

        if (
            replace_symbol
            and not isinstance(replace_symbol, tuple)
            and len(replace_symbol) != 2
        ):
            raise ValueError("replace_symbol must be a tuple: (old_symbol, new_symbol)")

        all_models = []
        failed_models = []
        load_count = 0  # For collect memory
        collect_memory = False
        for d in os.listdir(prepwizard_folder + "/output_models"):
            if os.path.isdir(prepwizard_folder + "/output_models/" + d):
                for f in os.listdir(prepwizard_folder + "/output_models/" + d):
                    if f.endswith(".log"):
                        with open(
                            prepwizard_folder + "/output_models/" + d + "/" + f
                        ) as lf:
                            for l in lf:
                                if "error" in l.lower():
                                    print(
                                        "Error was found in log file: %s. Please check the calculation!"
                                        % f
                                    )
                                    model = f.replace(".log", "")

                                    if replace_symbol:
                                        model = model.replace(
                                            replace_symbol[1], replace_symbol[0]
                                        )

                                    failed_models.append(model)
                                    break

                    if f.endswith(".pdb"):
                        model = f.replace(".pdb", "")

                        if replace_symbol:
                            model = model.replace(replace_symbol[1], replace_symbol[0])

                        # skip models not loaded into the library
                        if model not in self.models_names:
                            continue

                        # Skip models not in the given models list
                        if models != None and model not in models:
                            continue

                        if (
                            collect_memory_every
                            and load_count % collect_memory_every == 0
                        ):
                            collect_memory = True
                        else:
                            collect_memory = False

                        all_models.append(model)
                        self.readModelFromPDB(
                            model,
                            prepwizard_folder + "/output_models/" + d + "/" + f,
                            covalent_check=covalent_check,
                            atom_mapping=atom_mapping,
                            conect_update=conect_update,
                            collect_memory=collect_memory,
                        )
                        load_count += 1

        self.getModelsSequences()
        missing_models = set(self.models_names) - set(all_models)
        if missing_models != set():
            print("Missing models in prepwizard folder:")
            print("\t" + ", ".join(missing_models))

        if return_missing:
            return missing_models
        if return_failed:
            return failed_models

    def analyseRosettaCalculation(
        self,
        rosetta_folder,
        atom_pairs=None,
        energy_by_residue=False,
        interacting_residues=False,
        query_residues=None,
        overwrite=False,
        protonation_states=False,
        decompose_bb_hb_into_pair_energies=False,
        binding_energy=False,
        cpus=None,
        return_jobs=False,
        verbose=False,
    ):
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
        binding_energy : str
            Comma-separated list of chains for which calculate the binding energy.
        """

        if not os.path.exists(rosetta_folder):
            raise ValueError(
                'The Rosetta calculation folder: "%s" does not exists!' % rosetta_folder
            )

        # Write atom_pairs dictionary to json file
        if atom_pairs != None:
            with open(rosetta_folder + "/._atom_pairs.json", "w") as jf:
                json.dump(atom_pairs, jf)

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(rosetta_folder, "analyse_calculation.py", subfolder="pyrosetta")

        # Execute docking analysis
        command = (
            "python "
            + rosetta_folder
            + "/._analyse_calculation.py "
            + rosetta_folder
            + " "
        )

        if binding_energy:
            command += "--binding_energy " + binding_energy + " "
        if atom_pairs != None:
            command += "--atom_pairs " + rosetta_folder + "/._atom_pairs.json "
        if return_jobs:
            command += "--models MODEL "
        if energy_by_residue:
            command += "--energy_by_residue "
        if interacting_residues:
            command += "--interacting_residues "
            if query_residues != None:
                command += "--query_residues "
                command += ",".join([str(r) for r in query_residues]) + " "
        if protonation_states:
            command += "--protonation_states "
        if decompose_bb_hb_into_pair_energies:
            command += "--decompose_bb_hb_into_pair_energies "
        if cpus != None:
            command += "--cpus " + str(cpus) + " "
        if verbose:
            command += "--verbose "
        command += "\n"

        # Compile individual models for each job
        if return_jobs:
            commands = []
            for m in self:
                commands.append(command.replace("MODEL", m))

            print("Returning jobs for running the analysis in parallel.")
            print(
                "After jobs have finished, rerun this function removing return_jobs=True!"
            )
            return commands

        else:
            try:
                os.system(command)
            except:
                os.chdir("..")
                raise ValueError(
                    "Rosetta calculation analysis failed. Check the ouput of the analyse_calculation.py script."
                )

        # Compile dataframes into rosetta_data attributes
        self.rosetta_data = []
        self.rosetta_distances = {}
        self.rosetta_ebr = []
        self.rosetta_neighbours = []
        self.rosetta_protonation = []
        binding_energy_df = []

        analysis_folder = rosetta_folder + '/'+output_folder
        for model in self:

            # Read scores
            scores_folder = analysis_folder + "/scores"
            scores_csv = scores_folder + "/" + model + ".csv"
            if os.path.exists(scores_csv):
                self.rosetta_data.append(pd.read_csv(scores_csv))

            # Read binding energies
            be_folder = analysis_folder + "/binding_energy"
            be_csv = be_folder + "/" + model + ".csv"
            if os.path.exists(be_csv):
                binding_energy_df.append(pd.read_csv(be_csv))

            # Read distances
            distances_folder = analysis_folder + "/distances"
            distances_csv = distances_folder + "/" + model + ".csv"
            if os.path.exists(distances_csv):
                self.rosetta_distances[model] = pd.read_csv(distances_csv)
                self.rosetta_distances[model].set_index(["Model", "Pose"], inplace=True)

            # Read energy-by-residue data
            ebr_folder = analysis_folder + "/ebr"
            erb_csv = ebr_folder + "/" + model + ".csv"
            if os.path.exists(erb_csv):
                self.rosetta_ebr.append(pd.read_csv(erb_csv))

            # Read interacting neighbours data
            neighbours_folder = analysis_folder + "/neighbours"
            neighbours_csv = neighbours_folder + "/" + model + ".csv"
            if os.path.exists(neighbours_csv):
                self.rosetta_neighbours.append(pd.read_csv(neighbours_csv))

            # Read protonation data
            protonation_folder = analysis_folder + "/protonation"
            protonation_csv = protonation_folder + "/" + model + ".csv"
            if os.path.exists(protonation_csv):
                self.rosetta_protonation.append(pd.read_csv(protonation_csv))

        if self.rosetta_data == []:
            raise ValueError("No rosetta output was found in %s" % rosetta_folder)

        self.rosetta_data = pd.concat(self.rosetta_data)
        self.rosetta_data.set_index(["Model", "Pose"], inplace=True)

        if binding_energy:

            binding_energy_df = pd.concat(binding_energy_df)
            binding_energy_df.set_index(["Model", "Pose"], inplace=True)

            # Add interface scores to rosetta_data
            for score in binding_energy_df:
                index_value_map = {}
                for i, v in binding_energy_df.iterrows():
                    index_value_map[i] = v[score]

                values = []
                for i in self.rosetta_data.index:
                    values.append(index_value_map[i])

                self.rosetta_data[score] = values

        if energy_by_residue and self.rosetta_ebr != []:
            self.rosetta_ebr = pd.concat(self.rosetta_ebr)
            self.rosetta_ebr.set_index(
                ["Model", "Pose", "Chain", "Residue"], inplace=True
            )
        else:
            self.rosetta_ebr = None

        if interacting_residues and self.rosetta_neighbours != []:
            self.rosetta_neighbours = pd.concat(self.rosetta_neighbours)
            self.rosetta_neighbours.set_index(
                ["Model", "Pose", "Chain", "Residue"], inplace=True
            )
        else:
            self.rosetta_neighbours = None

        if protonation_states and self.rosetta_protonation != []:
            self.rosetta_protonation = pd.concat(self.rosetta_protonation)
            self.rosetta_protonation.set_index(
                ["Model", "Pose", "Chain", "Residue"], inplace=True
            )
        else:
            self.rosetta_protonation = None

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

        distances = []
        for d in self.rosetta_distances[model]:
            if d.startswith("distance_"):
                distances.append(d)

        return distances

    def combineRosettaDistancesIntoMetric(
        self, metric_labels, overwrite=False, rosetta_data=None, rosetta_distances=None
    ):
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

        if isinstance(rosetta_data, type(None)):
            rosetta_data = self.rosetta_data

        if isinstance(rosetta_distances, type(None)):
            rosetta_distances = self.rosetta_distances

        for name in metric_labels:
            if "metric_" + name in rosetta_data.keys() and not overwrite:
                print(
                    "Combined metric %s already added. Give overwrite=True to recombine"
                    % name
                )

            else:
                values = []
                for model in rosetta_data.index.levels[0]:
                    model_distances = rosetta_distances[model]
                    md = model_distances[metric_labels[name][model]]
                    values += md.min(axis=1).tolist()

                rosetta_data["metric_" + name] = values

    def getBestRosettaModels(
        self, filter_values, n_models=1, return_failed=False, exclude_models=None
    ):
        """
        Get best rosetta models based on their best "total_score" and a set of metrics
        with specified thresholds. The filter thresholds must be provided with a dictionary
        using the metric names as keys and the thresholds as the values.

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
        exclude_models : list
            List of models to be excluded from the best poses selection.
        """

        if exclude_models == None:
            exclude_models = []

        best_poses = pd.DataFrame()
        bp = []
        failed = []
        for model in self.rosetta_data.index.levels[0]:

            if model in exclude_models:
                continue

            model_data = self.rosetta_data[
                self.rosetta_data.index.get_level_values("Model") == model
            ]
            for metric in filter_values:
                if not metric.startswith("metric_"):
                    metric_label = "metric_" + metric
                else:
                    metric_label = metric
                model_data = model_data[
                    model_data[metric_label] < filter_values[metric]
                ]
                if model_data.empty:
                    if model not in failed:
                        failed.append(model)
                    continue
                if model_data.shape[0] < n_models:
                    print(
                        "WARNING: less than %s models passed the filter %s + %s"
                        % (n_models, model, ligand)
                    )
                for i in model_data["score"].nsmallest(n_models).index:
                    bp.append(i)

        if return_failed:
            return failed, self.rosetta_data[self.rosetta_data.index.isin(bp)]
        return self.rosetta_data[self.rosetta_data.index.isin(bp)]

    def getBestRosettaModelsIteratively(
        self, metrics, min_threshold=3.5, max_threshold=5.0, step_size=0.1
    ):
        """
        Extract the best rosetta poses by iterating the metrics thresholds from low values to high values.
        At each iteration the poses are filtered by the current metric threshold and the lowest scoring poses
        are selected. Further iterations at higher metric thresholds are applied to those model that do not
        had poses passing all the metric filters. A current limitation of the method is that at each iteration
        it uses the same theshold value for all the metrics.

        Parameters
        ==========
        metrics : list
            A list of the metrics to be used as filters
        min_threshold : float
            The lowest threshold to apply for filtering poses by the metric values.
        """

        extracted = []
        selected_indexes = []

        # Iterate the threshold values to be employed as filters
        for t in np.arange(min_threshold, max_threshold + (step_size / 10), step_size):

            # Get models already selected
            excluded_models = [m for m in extracted]

            # Append metric_ prefic if needed
            filter_values = {
                (m if m.startswith("metric_") else "metric_" + m): t for m in metrics
            }

            # Filter poses at the current threshold
            best_poses = self.getBestRosettaModels(
                filter_values, n_models=1, exclude_models=excluded_models
            )

            # Save selected indexes not saved in previous iterations
            for row in best_poses.index:
                if row[0] not in extracted:
                    selected_indexes.append(row)
                if row[0] not in extracted:
                    extracted.append(row[0])

        best_poses = self.rosetta_data[self.rosetta_data.index.isin(selected_indexes)]

        return best_poses

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
        data.columns = [c.replace(" ", "_") for c in data.columns]

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
                    model_base_name = "_".join(model_tag.split("_")[:-1])
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

    def loadMutantsAsNewModels(
        self,
        mutants_folder,
        filter_score_term="score",
        tags=None,
        min_value=True,
        wat_to_hoh=True,
        keep_model_name=True,
        only_mutants=None,
    ):
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

        executable = "extract_pdbs.linuxgccrelease"
        models = []

        if only_mutants == None:
            only_mutants = []

        if isinstance(only_mutants, str):
            only_mutants = [only_mutants]

        # Check if params were given
        params = None
        if os.path.exists(mutants_folder + "/params"):
            params = mutants_folder + "/params"

        for d in os.listdir(mutants_folder + "/output_models"):
            if os.path.isdir(mutants_folder + "/output_models/" + d):
                for f in os.listdir(mutants_folder + "/output_models/" + d):
                    if f.endswith(".out"):

                        model = d
                        mutant = f.replace(model + "_", "").replace(".out", "")

                        # Read only given mutants
                        if only_mutants != []:
                            if (
                                mutant not in only_mutants
                                and model + "_" + mutant not in only_mutants
                            ):
                                continue

                        scores = readSilentScores(
                            mutants_folder + "/output_models/" + d + "/" + f
                        )
                        if tags != None and mutant in tags:
                            print(
                                "Reading mutant model %s from the given tag %s"
                                % (mutant, tags[mutant])
                            )
                            best_model_tag = tags[mutant]
                        elif min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += (
                            " -silent "
                            + mutants_folder
                            + "/output_models/"
                            + d
                            + "/"
                            + f
                        )
                        if params != None:
                            command += " -extra_res_path " + params + " "
                        command += " -tags " + best_model_tag
                        os.system(command)

                        # Load mutants to the class
                        if keep_model_name:
                            mutant = model + "_" + mutant

                        # self.models_names.append(mutant)
                        self.readModelFromPDB(
                            mutant, best_model_tag + ".pdb", wat_to_hoh=wat_to_hoh
                        )
                        os.remove(best_model_tag + ".pdb")
                        models.append(mutant)

        self.getModelsSequences()
        print("Added the following mutants from folder %s:" % mutants_folder)
        print("\t" + ", ".join(models))

    def loadModelsFromRosettaOptimization(
        self,
        optimization_folder,
        filter_score_term="score",
        min_value=True,
        tags=None,
        wat_to_hoh=True,
        return_missing=False,
        sugars=False,
        conect_update=False,
    ):
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
        tags : dict
            The tag of a specific pose to be loaded for the given model. Each model
            must have a single tag in the tags dictionary. If a model is not found
            in the tags dictionary, normal processing will follow to select
            the loaded pose.
        wat_to_hoh : bool
            Change water names from WAT to HOH when loading.
        return_missing : bool
            Return missing models from the optimization_folder.
        """

        executable = "extract_pdbs.linuxgccrelease"
        models = []

        # Check if params were given
        params = None
        if os.path.exists(optimization_folder + "/params"):
            params = optimization_folder + "/params"
            patch_line = ""
            for p in os.listdir(params):
                if not p.endswith(".params"):
                    patch_line += params + "/" + p + " "

        for d in os.listdir(optimization_folder + "/output_models"):
            if os.path.isdir(optimization_folder + "/output_models/" + d):
                for f in os.listdir(optimization_folder + "/output_models/" + d):
                    if f.endswith("_relax.out"):
                        model = d

                        # skip models not loaded into the library
                        if model not in self.models_names:
                            continue

                        scores = readSilentScores(
                            optimization_folder + "/output_models/" + d + "/" + f
                        )
                        if tags != None and model in tags:
                            print(
                                "Reading model %s from the given tag %s"
                                % (model, tags[model])
                            )
                            best_model_tag = tags[model]
                        elif min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += (
                            " -silent "
                            + optimization_folder
                            + "/output_models/"
                            + d
                            + "/"
                            + f
                        )
                        if params != None:
                            command += " -extra_res_path " + params
                            if patch_line != "":
                                command += " -extra_patch_fa " + patch_line
                        command += " -tags " + best_model_tag
                        if sugars:
                            command += " -include_sugars"
                            command += " -alternate_3_letter_codes pdb_sugar"
                            command += " -write_glycan_pdb_codes"
                            command += " -auto_detect_glycan_connections"
                            command += " -maintain_links"
                        os.system(command)
                        self.readModelFromPDB(
                            model,
                            best_model_tag + ".pdb",
                            wat_to_hoh=wat_to_hoh,
                            conect_update=conect_update,
                        )
                        os.remove(best_model_tag + ".pdb")
                        models.append(model)

        self.getModelsSequences()

        missing_models = set(self.models_names) - set(models)
        if missing_models != set():
            print("Missing models in relaxation folder:")
            print("\t" + ", ".join(missing_models))
            if return_missing:
                return missing_models

    def loadModelsFromMissingLoopBuilding(
        self, job_folder, filter_score_term="score", min_value=True, param_files=None
    ):
        """
        Load models from addMissingLoops() job calculation output.

        Parameters:
        job_folder : str
            Path to the addMissingLoops() calculation folder containing output.
        """

        # Get silent models paths
        executable = "extract_pdbs.linuxgccrelease"
        output_folder = job_folder + "/output_models"
        models = []

        # Check if params were given
        params = None
        if os.path.exists(job_folder + "/params"):
            params = job_folder + "/params"

        # Check loops to rebuild from output folder structure
        for model in os.listdir(output_folder):
            model_folder = output_folder + "/" + model
            loop_models = {}
            for loop in os.listdir(model_folder):
                loop_folder = model_folder + "/" + loop
                for f in os.listdir(loop_folder):
                    # If rebuilded loops are found get best structures.
                    if f.endswith(".out"):
                        scores = readSilentScores(loop_folder + "/" + f)
                        best_model_tag = scores.idxmin()[filter_score_term]
                        if min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += " -silent " + loop_folder + "/" + f
                        if params != None:
                            command += " -extra_res_path " + params + " "
                        command += " -tags " + best_model_tag
                        os.system(command)
                        loop = (int(loop.split("_")[0]), loop.split("_")[1])
                        loop_models[loop] = _readPDB(loop, best_model_tag + ".pdb")
                        os.remove(best_model_tag + ".pdb")
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
                    larger_loop_residue = loop[0] + len(loop[1]) + 1 + hanging_residues
                    for i, residue in enumerate(loop_models[loop].get_residues()):
                        if i + 1 > current_residue and i + 1 <= larger_loop_residue:
                            chain_id = residue.get_parent().id
                            chains[chain_id].add(residue)
                            current_residue += 1

                # Load final model into the library
                for chain in chains:
                    _model.add(chains[chain])
                structure.add(_model)
                _saveStructureToPDB(structure, model + ".pdb")
            else:
                for loop in loop_models:
                    _saveStructureToPDB(loop_models[loop], model + ".pdb")

            self.readModelFromPDB(model, model + ".pdb")
            os.remove(model + ".pdb")

        missing_models = set(self.models_names) - set(models)
        if missing_models != set():
            print("Missing models in loop rebuild folder:")
            print("\t" + ", ".join(missing_models))

    def loadModelsFromMembranePositioning(self, job_folder):
        """ """
        for model in os.listdir(job_folder + "/output_models"):
            pdb_path = job_folder + "/output_models/" + model + "/" + model + ".pdb"
            self.readModelFromPDB(model, pdb_path)

    def saveModels(
        self,
        output_folder,
        keep_residues={},
        models=None,
        convert_to_mae=False,
        write_conect_lines=True,
        replace_symbol=None,
        **keywords,
    ):
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
            _copyScriptFile(output_folder, "PDBtoMAE.py")
            script_name = "._PDBtoMAE.py"

        if replace_symbol:
            if not isinstance(replace_symbol, tuple) or len(replace_symbol) != 2:
                raise ValueError(
                    "replace_symbol must be a tuple (old_symbol, new_symbol)"
                )

        for model in self.models_names:

            if replace_symbol:
                model_name = model.replace(replace_symbol[0], replace_symbol[1])
            else:
                model_name = model

            # Skip models not in the given list
            if models != None:
                if model not in models:
                    continue

            # Get residues to keep
            if model in keep_residues:
                kr = keep_residues[model]
            else:
                kr = []

            _saveStructureToPDB(
                self.structures[model],
                output_folder + "/" + model_name + ".pdb",
                keep_residues=kr,
                **keywords,
            )

            if "remove_hydrogens" in keywords:
                if keywords["remove_hydrogens"] == True:
                    check_file = True
                    hydrogens = False
                else:
                    check_file = False
                    hydrogens = True
            else:
                check_file = False
                hydrogens = True

            if write_conect_lines:
                self._write_conect_lines(
                    model,
                    output_folder + "/" + model_name + ".pdb",
                    check_file=check_file,
                    hydrogens=hydrogens,
                )

        if convert_to_mae:
            command = "cd " + output_folder + "\n"
            command += "run ._PDBtoMAE.py\n"
            command += "cd ../\n"
            os.system(command)

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
            raise ValueError("Model  %s is not present" % model)

    def readTargetSequences(self, fasta_file):
        """
        Read the set of target sequences for the protein models
        """
        # Read sequences and store them in target_sequence attributes
        sequences = prepare_proteins.alignment.readFastaFile(fasta_file)
        for sequence in sequences:
            if sequence not in self.models_names:
                print(
                    "Given sequence name %s does not matches any protein model"
                    % sequence
                )
            else:
                self.target_sequences[sequence] = sequences[sequence]

        missing_models = set(self.models_names) - set(self.target_sequences)
        if missing_models != set():
            print("Missing sequences in the given fasta file:")
            print("\t" + ", ".join(missing_models))

    def compareSequences(self, sequences_file, chain=None):
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

        if self.multi_chain and chain == None:
            raise ValueError("PDBs contain multiple chains. Please select one chain.")

        self.readTargetSequences(sequences_file)

        # Iterate models to store sequence differences
        for model in self.models_names:

            if model not in self.target_sequences:
                message = (
                    "Sequence for model %s not found in the given fasta file! " % model
                )
                message += "Please make sure to include one sequence for each model "
                message += "loaded into prepare proteins."
                raise ValueError(message)

            # Create lists for missing information
            self.sequence_differences[model] = {}
            self.sequence_differences[model]["n_terminus"] = []
            self.sequence_differences[model]["mutations"] = []
            self.sequence_differences[model]["missing_loops"] = []
            self.sequence_differences[model]["c_terminus"] = []

            # Create a sequence alignement between current and target sequence
            to_align = {}
            if chain:
                to_align["current"] = self.sequences[model][chain]
            else:
                to_align["current"] = self.sequences[model]
            to_align["target"] = self.target_sequences[model]
            msa = prepare_proteins.alignment.mafft.multipleSequenceAlignment(
                to_align, stderr=False, stdout=False
            )

            # Iterate the alignment to gather sequence differences
            p = 0
            n = True
            loop_sequence = ""
            loop_start = 0

            # Check for n-terminus, mutations and missing loops
            for i in range(msa.get_alignment_length()):
                csp = msa[0].seq[i]
                tsp = msa[1].seq[i]
                if csp != "-":
                    p += 1
                if csp == "-" and tsp != "-" and n:
                    self.sequence_differences[model]["n_terminus"].append(tsp)
                elif csp != "-" and tsp != "-":
                    n = False
                    if (
                        loop_sequence != "" and len(loop_sequence) > 1
                    ):  # Ignore single-residue loops
                        self.sequence_differences[model]["missing_loops"].append(
                            (loop_start, loop_sequence)
                        )
                    loop_sequence = ""
                    loop_start = 0

                    if csp != tsp:
                        self.sequence_differences[model]["mutations"].append((p, tsp))

                elif csp == "-" and tsp != "-" and p < len(to_align["current"]):
                    if loop_start == 0:
                        loop_start = p
                    loop_sequence += tsp

            # Check for c-terminus
            for i in reversed(range(msa.get_alignment_length())):
                csp = msa[0].seq[i]
                tsp = msa[1].seq[i]
                if csp == "-" and tsp != "-":
                    self.sequence_differences[model]["c_terminus"].append(tsp)
                elif csp != "-" and tsp != "-":
                    break

            self.sequence_differences[model]["n_terminus"] = "".join(
                self.sequence_differences[model]["n_terminus"]
            )
            self.sequence_differences[model]["c_terminus"] = "".join(
                reversed(self.sequence_differences[model]["c_terminus"])
            )

        return self.sequence_differences

    def _write_conect_lines(
        self, model, pdb_file, atom_mapping=None, check_file=False, hydrogens=True
    ):
        """
        Write stored conect lines for a particular model into the given PDB file.

        Parameters
        ==========
        model : str
            Model name
        pdb_file : str
            Path to PDB file to modify
        """

        def check_atom_in_atoms(atom, atoms, atom_mapping):
            if atom_mapping != None:
                atom_mapping = atom_mapping[model]

            if atom not in atoms and atom_mapping != None and atom in atom_mapping:
                if isinstance(atom_mapping[atom], str):
                    atom = (atom[0], atom[1], atom_mapping[atom])
                elif (
                    isinstance(atom_mapping[atom], tuple)
                    and len(atom_mapping[atom]) == 3
                ):
                    atom = atom_mapping[atom]

            if atom not in atoms:
                residue_atoms = " ".join([ac[-1] for ac in atoms if atom[1] == ac[1]])
                message = "Conect atom %s not found in %s's topology\n\n" % (
                    atom,
                    pdb_file,
                )
                message += "Topology's residue %s atom names: %s" % (
                    atom[1],
                    residue_atoms,
                )
                raise ValueError(message)

            return atom

        # Get atom indexes map
        atoms = self._getAtomIndexes(
            model, pdb_file, invert=True, check_file=check_file
        )

        # Check atoms not found in conects
        with open(pdb_file + ".tmp", "w") as tmp:
            with open(pdb_file) as pdb:

                # write all lines but skip END line
                for line in pdb:
                    if not line.startswith("END"):
                        tmp.write(line)

                # Write new conect line mapping
                for entry in self.conects[model]:
                    line = "CONECT"
                    for x in entry:
                        if not hydrogens:
                            type_index = x[2].find(next(filter(str.isalpha, x[2])))
                            if x[2][type_index] != "H":
                                x = check_atom_in_atoms(
                                    x, atoms, atom_mapping=atom_mapping
                                )
                                line += "%5s" % atoms[x]
                        else:
                            x = check_atom_in_atoms(x, atoms, atom_mapping=atom_mapping)
                            line += "%5s" % atoms[x]

                    line += "\n"
                    tmp.write(line)
            tmp.write("END\n")
        shutil.move(pdb_file + ".tmp", pdb_file)

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
        sequence = ""
        for r in chain:

            filter = False
            if r.resname in ["HIE", "HID", "HIP"]:
                resname = "HIS"
            elif r.resname == "CYX":
                resname = "CYS"
            elif r.resname == "ASH":
                resname = "ASP"
            elif r.resname == "GLH":
                resname = "GLU"
            else:
                # Leave out HETATMs
                if r.id[0] != " ":
                    filter = True
                resname = r.resname

            if not filter:  # Non heteroatom filter
                try:
                    sequence += PDB.Polypeptide.three_to_one(resname)
                except:
                    sequence += "X"

        if sequence == "":
            return None
        else:
            return sequence

    def _checkCovalentLigands(
        self, model, pdb_file, atom_mapping=None, check_file=False
    ):
        """ """
        self.covalent[model] = []  # Store covalent residues
        if check_file:
            structure = _readPDB(model, pdb_file)
        else:
            structure = self.structures[model]

        # Iterate chains in model structure
        for c in structure[0]:

            indexes = []  # Store residue indexes
            hetero = []  # Store heteroatom residue indexes
            residues = []  # Store residues orderly (for later)
            for r in c:
                indexes.append(r.id[1])
                if r.id[0].startswith("H_"):
                    hetero.append(r.id[1])
                residues.append(r)

            # Check for individual and other gaps
            gaps2 = []  # Store individual gaps
            other_gaps = []  # Store other gaps
            for i in range(len(indexes)):
                if i > 0:
                    if indexes[i] - indexes[i - 1] == 2:
                        gaps2.append((indexes[i - 1], indexes[i]))
                    elif indexes[i] - indexes[i - 1] != 1:
                        other_gaps.append(indexes[i])

            # Check if individual gaps can be filled with any residue in other_gaps
            for g2 in gaps2:
                for og in other_gaps:
                    if g2[1] - og == 1 and og - g2[0] == 1:

                        if check_file:
                            print(
                                "Found misplaced residue %s for file %s"
                                % (og, pdb_file)
                            )
                        else:
                            print(
                                "Found misplaced residue %s for model %s" % (og, model)
                            )

                        print("Possibly a covalent-link exists for this HETATM residue")
                        print(
                            "Sorting residues by their indexes... to disable pass covalent_check=False."
                        )

                        self._sortStructureResidues(
                            model,
                            pdb_file,
                            check_file=check_file,
                            atom_mapping=atom_mapping,
                        )
                        self.covalent[model].append(og)

            # Check if hetero-residue is found between two non-hetero residues
            for i, r in enumerate(residues):
                if r.id[1] in hetero and r.resname not in ["HIP", "HID", "HIE"]:
                    if i + 1 == len(residues):
                        continue
                    chain = r.get_parent()
                    pr = residues[i - 1]
                    nr = residues[i + 1]
                    if (
                        pr.get_parent().id == chain.id
                        and nr.get_parent().id == chain.id
                    ):
                        if pr.id[0] == " " and nr.id[0] == " ":
                            self.covalent[model].append(r.id[1])

    def _sortStructureResidues(
        self, model, pdb_file, atom_mapping=None, check_file=False
    ):

        # Create new structure
        n_structure = PDB.Structure.Structure(0)

        # Create new model
        n_model = PDB.Model.Model(self.structures[model][0].id)

        if check_file:
            structure = _readPDB(model, pdb_file)
        else:
            structure = self.structures[model]

        # Iterate chains from old model
        model = [m for m in structure][0]
        for chain in model:
            n_chain = PDB.Chain.Chain(chain.id)

            # Gather residues
            residues = []
            for r in chain:
                residues.append(r)

            # Iterate residues orderly by their ID
            for r in sorted(residues, key=lambda x: x.id[1]):
                n_chain.add(r)

            n_model.add(n_chain)
        n_structure.add(n_model)

        _saveStructureToPDB(n_structure, pdb_file + ".tmp")
        self._write_conect_lines(
            model, pdb_file + ".tmp", atom_mapping=atom_mapping, check_file=check_file
        )
        shutil.move(pdb_file + ".tmp", pdb_file)
        n_structure = _readPDB(model, pdb_file)

        # Update structure model in library
        if not check_file:
            self.structures[model] = n_structure

    def _readPDBConectLines(self, pdb_file, model, only_hetatoms=False):
        """
        Read PDB file and get conect lines only
        """
        # Get atom indexes by tuple and objects
        atoms = self._getAtomIndexes(model, pdb_file)
        if only_hetatoms:
            atoms_objects = self._getAtomIndexes(model, pdb_file, return_objects=True)
        conects = []
        # Read conect lines as dictionaries linking atoms
        with open(pdb_file) as pdbf:
            for l in pdbf:
                if l.startswith("CONECT"):
                    l = l.replace("CONECT", "")
                    l = l.strip("\n").rstrip()
                    num = len(l) / 5
                    new_l = [int(l[i * 5 : (i * 5) + 5]) for i in range(int(num))]
                    if only_hetatoms:
                        het_atoms = [
                            (
                                True
                                if atoms_objects[int(x)].get_parent().id[0] != " "
                                else False
                            )
                            for x in new_l
                        ]
                        if True not in het_atoms:
                            continue
                    conects.append([atoms[int(x)] for x in new_l])
        return conects

    def _getAtomIndexes(
        self, model, pdb_file, invert=False, check_file=False, return_objects=False
    ):

        # Read PDB file
        atom_indexes = {}
        with open(pdb_file, "r") as f:
            for l in f:
                if l.startswith("ATOM") or l.startswith("HETATM"):
                    index, name, chain, resid = (
                        int(l[6:11]),
                        l[12:16].strip(),
                        l[21],
                        int(l[22:26]),
                    )
                    atom_indexes[(chain, resid, name)] = index

        if check_file:
            structure = _readPDB(model, pdb_file)
        else:
            structure = self.structures[model]

        # Assign PDB indexes to each Bio.PDB atom
        atoms = {}
        for chain in structure[0]:
            for residue in chain:
                for atom in residue:

                    # Get atom PDB index
                    index = atom_indexes[(chain.id, residue.id[1], atom.name)]

                    # Return atom objects instead of tuples
                    if return_objects:
                        _atom = atom
                    else:
                        _atom = _get_atom_tuple(atom)

                    # Invert the returned dictionary
                    if invert:
                        atoms[_atom] = index
                    else:
                        atoms[index] = _atom
        return atoms

    def _getModelsPaths(self, only_models=None, exclude_models=None):
        """
        Get PDB models paths in the models_folder attribute

        Returns
        =======

        paths : dict
            Paths to all models
        """

        paths = {}
        for d in os.listdir(self.models_folder):
            if d.endswith(".pdb"):
                pdb_name = ".".join(d.split(".")[:-1])

                if only_models != []:
                    if pdb_name not in only_models:
                        continue

                if exclude_models != []:
                    if pdb_name in exclude_models:
                        continue

                paths[pdb_name] = self.models_folder + "/" + d

        return paths

    def __iter__(self):
        # returning __iter__ object
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
            if l.startswith("SCORE"):
                if terms == []:
                    terms = l.strip().split()
                    for t in terms:
                        scores[t] = []
                else:
                    for i, t in enumerate(terms):
                        try:
                            scores[t].append(float(l.strip().split()[i]))
                        except:
                            scores[t].append(l.strip().split()[i])
    scores = pd.DataFrame(scores)
    scores.pop("SCORE:")
    scores = pd.DataFrame(scores)
    scores = scores.set_index("description")
    scores = scores.sort_index()

    return scores


def _readPDB(name, pdb_file):
    """
    Read PDB file to a structure object
    """
    parser = PDB.PDBParser()
    structure = parser.get_structure(name, pdb_file)
    return structure


def _saveStructureToPDB(
    structure,
    output_file,
    remove_hydrogens=False,
    remove_water=False,
    only_protein=False,
    keep_residues=[],
):
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

def _copyScriptFile(
    output_folder, script_name, no_py=False, subfolder=None, hidden=True
):
    """
    Copy a script file from the prepare_proteins package.

    Parameters
    ==========

    """
    # Get script
    path = "prepare_proteins/scripts"
    if subfolder != None:
        path = path + "/" + subfolder

    script_file = resource_stream(
        Requirement.parse("prepare_proteins"), path + "/" + script_name
    )
    script_file = io.TextIOWrapper(script_file)

    # Write control script to output folder
    if no_py == True:
        script_name = script_name.replace(".py", "")

    if hidden:
        output_path = output_folder + "/._" + script_name
    else:
        output_path = output_folder + "/" + script_name

    with open(output_path, "w") as sof:
        for l in script_file:
            sof.write(l)


def _computeCartesianFromInternal(coord2, coord3, coord4, distance, angle, torsion):
    """
    Compute the cartesian coordinates for the i atom based on internal coordinates
    of other three atoms (j, k, l).

    Parameters
    ==========
    coord1 : numpy.ndarray shape=(3,)
        Coordinate of the j atom bound to the i atom
    coord2 : numpy.ndarray shape=(3,)
        Coordinate of the k atom bound to the j atom
    coord3 : numpy.ndarray shape=(3,)
        Coordinate of the l atom bound to the k atom
    distance : float
        Distance between the i and j atoms in angstroms
    angle : float
        Angle between the i, j, and k atoms in degrees
    torsion : float
        Torsion between the i, j, k, l atoms in degrees

    Returns
    =======
    coord1 : float
        Coordinate of the i atom

    """

    torsion = torsion * np.pi / 180.0  # Convert to radians
    angle = angle * np.pi / 180.0  # Convert to radians

    v1 = coord2 - coord3
    v2 = coord2 - coord4

    n = np.cross(v1, v2)
    nn = np.cross(v1, n)

    n /= np.linalg.norm(n)
    nn /= np.linalg.norm(nn)

    n *= -np.sin(torsion)
    nn *= np.cos(torsion)

    v3 = n + nn
    v3 /= np.linalg.norm(v3)
    v3 *= distance * np.sin(angle)

    v1 /= np.linalg.norm(v1)
    v1 *= distance * np.cos(angle)

    coord1 = coord2 + v3 - v1

    return coord1


def _get_atom_tuple(atom):
    return (atom.get_parent().get_parent().id, atom.get_parent().id[1], atom.name)


def _getStructureCoordinates(
    structure,
    as_dict=False,
    return_atoms=False,
    only_protein=False,
    sidechain=False,
    backbone=False,
    only_residues=None,
    exclude_residues=None,
):
    """
    Get the coordinates for each atom in the structure.
    """

    if as_dict:
        if return_atoms:
            raise ValueError("as_dict and return_atoms are not compatible!")
        coordinates = {}
    else:
        coordinates = []

    for atom in structure.get_atoms():
        residue = atom.get_parent()
        chain = residue.get_parent()
        residue_tuple = (chain.id, residue.id[1])
        atom_tuple = (chain.id, residue.id[1], atom.name)

        if exclude_residues and residue_tuple in exclude_residues:
            continue

        if only_residues and residue_tuple not in only_residues:
            continue

        if only_protein or sidechain or backbone:
            if residue.id[0] != " ":
                continue

        if sidechain:
            if atom.name in ["N", "CA", "C", "O"]:
                continue

        elif backbone:
            if atom.name not in ["N", "CA", "C", "O"]:
                continue

        if as_dict:
            coordinates[atom_tuple] = atom.coord
        elif return_atoms:
            coordinates.append(atom_tuple)
        else:
            coordinates.append(atom.coord)

    if not as_dict:
        coordinates = np.array(coordinates)

    return coordinates


def _readRosettaScoreFile(score_file, indexing=False):
    """
    Generates an iterator from the poses in the silent file

    Arguments:
    ==========
    silent_file : (str)
        Path to the input silent file.

    Returns:
    ========
        Generator object for the poses in the silen file.
    """
    with open(score_file) as sf:
        lines = [x for x in sf.readlines() if x.startswith("SCORE:")]
        score_terms = lines[0].split()
        scores = {}
        for line in lines[1:]:
            for i, score in enumerate(score_terms):
                if score not in scores:
                    scores[score] = []
                try:
                    scores[score].append(float(line.split()[i]))
                except:
                    scores[score].append(line.split()[i])

    scores.pop("SCORE:")
    for s in scores:
        if s == "description":
            models = []
            poses = []
            for x in scores[s]:
                if x == "description":
                    continue
                model, pose = "_".join(x.split("_")[:-1]), x.split("_")[-1]
                models.append(model)
                poses.append(int(pose))
            continue
        scores[s] = np.array(scores[s])
    scores.pop("description")
    scores["Model"] = np.array(models)
    scores["Pose"] = np.array(poses)

    # Sort all values based on pose number
    for s in scores:
        scores[s] = [x for _, x in sorted(zip(scores["Pose"], scores[s]))]

    scores = pd.DataFrame(scores)

    if indexing:
        scores = scores.set_index(["Model", "Pose"])

    return scores


def _getAlignedResiduesBasedOnStructuralAlignment(
    ref_struct, target_struct, max_ca_ca=5.0
):
    """
    Return a sequence string with aligned residues based on a structural alignment. All residues
    not structurally aligned are returned as '-'.

    Parameters
    ==========
    ref_struct : str
        Reference structure
    target_struct : str
        Target structure
    max_ca_ca : float
        Maximum CA-CA distance to be considered aligned

    Returns
    =======
    aligned_residues : str
        Full-length sequence string containing only aligned residues.
    """

    # Get sequences
    r_sequence = "".join(
        [
            PDB.Polypeptide.three_to_one(r.resname)
            for r in ref_struct.get_residues()
            if r.id[0] == " "
        ]
    )
    t_sequence = "".join(
        [
            PDB.Polypeptide.three_to_one(r.resname)
            for r in target_struct.get_residues()
            if r.id[0] == " "
        ]
    )

    # Get alpha-carbon coordinates
    r_ca_coord = np.array([a.coord for a in ref_struct.get_atoms() if a.name == "CA"])
    t_ca_coord = np.array(
        [a.coord for a in target_struct.get_atoms() if a.name == "CA"]
    )

    # Map residues based on a CA-CA distances
    D = distance_matrix(t_ca_coord, r_ca_coord)  # Calculate CA-CA distance matrix
    D = np.where(D <= max_ca_ca, D, np.inf)  # < Cap matrix to max_ca_ca

    # Map residues to the closest CA-CA distance
    mapping = {}

    # Start mapping from closest distance avoiding double assignments
    while not np.isinf(D).all():
        i, j = np.unravel_index(D.argmin(), D.shape)
        mapping[i] = j
        D[i].fill(np.inf)  # This avoids double assignments
        D[:, j].fill(np.inf)  # This avoids double assignments

    # Create alignment based on the structural alignment mapping
    aligned_residues = []
    for i, r in enumerate(t_sequence):
        if i in mapping:
            aligned_residues.append(r)
        else:
            aligned_residues.append("-")

    # Join list to get aligned sequences
    aligned_residues = "".join(aligned_residues)

    return aligned_residues


def _createCAConstraintFile(structure, cst_file, sd=1.0):
    """
    Create a cst file restraining all alpha carbons to their input coordinates
    with an harmonic constraint.

    Parameters
    ==========
    structure : Bio.PDB.Structure.Structure
        Biopython structure object
    cst_file : str
        Path to the output constraint file
    sd : float
        Standard deviation parameter of the HARMONIC constraint.
    """

    cst_file = open(cst_file, "w")
    ref_res = None

    for r in structure.get_residues():

        if r.id[0] != " ":
            continue

        res, chain = r.id[1], r.get_parent().id

        if not ref_res:
            ref_res, ref_chain = res, chain

        # Update reference if chain changes
        if ref_chain != chain:
            ref_res, ref_chain = res, chain

        for atom in r.get_atoms():
            if atom.name == "CA":
                ca_atom = atom

        ca_coordinte = list(ca_atom.coord)

        cst_line = "CoordinateConstraint "
        cst_line += "CA " + str(res) + chain + " "
        cst_line += " CA " + str(ref_res) + ref_chain + " "
        cst_line += str(" ".join([str("%.4f" % c) for c in ca_coordinte])) + " "
        cst_line += "HARMONIC 0 " + str(sd) + "\n"
        cst_file.write(cst_line)

    cst_file.close()


def _getLigandParameters(
    structure, ligand_chains, struct_path, params_path, charge=None
):

    class chainSelect(PDB.Select):
        def accept_chain(self, chain):
            if chain.get_id() in ligand_chains:
                return False
            else:
                return True

    if charge == None:
        charge = {}

    ligand_res = {}

    # Get ligand residues from structure
    for mdl in structure:
        for chain in mdl:
            for residue in chain:
                if chain.get_id() in ligand_chains:
                    ligand_res[chain.get_id()] = residue.resname

    if ligand_res == {}:
        raise ValueError("Ligand was not found at chains %s" % str(ligand_chains))

    io = PDB.PDBIO()
    pdb_chains = list(structure.get_chains())
    num_chains = len(pdb_chains)

    if num_chains < 2:
        raise ValueError(
            "Input pdb "
            + model
            + " has only one chain. Protein and ligand should be separated in individual chains."
        )

    io.set_structure(structure)
    io.save(struct_path + "/protein.pdb", chainSelect())

    # Get ligand coords in input file
    ligand_coords = {}
    for chain in pdb_chains:
        if chain.get_id() in ligand_chains:
            ligand_coords[chain.get_id()] = [a.coord for a in chain.get_atoms()]
            io.set_structure(chain)
            io.save(struct_path + "/" + ligand_res[chain.get_id()] + ".pdb")

    # Get ligand parameters
    print(ligand_res)
    for it, lig_chain in enumerate(ligand_res):
        ligand_name = ligand_res[lig_chain]
        if ligand_name not in os.listdir(params_path):
            os.mkdir(params_path + "/" + ligand_name)
            shutil.copyfile(
                struct_path + "/" + ligand_name + ".pdb",
                params_path + "/" + ligand_name + "/" + ligand_name + ".pdb",
            )
            os.chdir(params_path + "/" + ligand_name)

            # Call acpype
            print("Parameterizing ligand %s" % ligand_name)
            command = "acpype -i " + ligand_name + ".pdb"
            if ligand_name in charge:
                command += " -n " + str(charge[ligand_name])
            os.system(command)

            f = open(ligand_name + ".acpype/" + ligand_name + "_GMX.itp")
            lines = f.readlines()
            atomtypes_lines = []
            new_lines = []
            atomtypes = False
            atoms = False
            for i, l in enumerate(lines):
                if atomtypes:
                    if l.startswith("[ moleculetype ]"):
                        new_lines.append(l)
                        atomtypes = False
                    else:
                        # print(l[:-1])
                        spl = l.split()
                        if spl != []:
                            spl[0] = ligand_name + spl[0]
                            spl[1] = ligand_name + spl[1]
                            atomtypes_lines.append(" ".join(spl))
                elif atoms:
                    if l.startswith("[ bonds ]"):
                        new_lines.append(l)
                        atoms = False
                    else:
                        spl = l.split()
                        if spl != []:
                            spl[1] = ligand_name + spl[1]
                            new_lines.append(" ".join(spl) + "\n")
                else:
                    new_lines.append(l)

                if l.startswith(";name"):
                    if lines[i - 1].startswith("[ atomtypes ]"):
                        atomtypes = True

                elif l.startswith(";"):
                    if lines[i - 1].startswith("[ atoms ]"):
                        atoms = True

            if it == 0:
                write_type = "w"
            else:
                write_type = "a"
            with open("../atomtypes.itp", write_type) as f:
                if it == 0:
                    f.write("[ atomtypes ]\n")
                for line in atomtypes_lines:
                    f.write(line + "\n")

            with open(ligand_name + ".acpype/" + ligand_name + "_GMX.itp", "w") as f:
                for line in new_lines:
                    if not line.startswith("[ atomtypes ]"):
                        f.write(line)

            os.chdir("../../..")

        # Apply ligand coords in input file to parametrized structure.
        parser = PDB.PDBParser()
        ligand_structure = parser.get_structure(
            "ligand",
            params_path
            + "/"
            + ligand_name
            + "/"
            + ligand_name
            + ".acpype/"
            + ligand_name
            + "_NEW.pdb",
        )
        for i, atom in enumerate(ligand_structure.get_atoms()):
            atom.coord = ligand_coords[lig_chain][i]
        io.set_structure(ligand_structure)
        io.save(struct_path + "/" + ligand_name + ".pdb")

    return ligand_res


def _readGromacsIndexFile(file):

    f = open(file, "r")
    groups = [
        x.replace("[", "").replace("]", "").replace("\n", "").strip()
        for x in f.readlines()
        if x.startswith("[")
    ]

    group_dict = {}
    for i, g in enumerate(groups):
        group_dict[g] = str(i)

    return group_dict
