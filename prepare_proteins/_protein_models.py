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
import copy
import re
import pkg_resources

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from Bio import PDB, BiopythonWarning
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import aa3
from ipywidgets import interactive_output, VBox, IntSlider, Checkbox, interact, fixed, Dropdown, FloatSlider, FloatRangeSlider
from pkg_resources import Requirement, resource_listdir, resource_stream
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

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
        only_hetatm_conects=False,
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
        self.rosetta_docking_data = None  # secondary structure strings are stored here
        self.rosetta_docking_distances = {}
        self.rosetta_docking_angles = {}
        self.rosetta_docking_metric_type = {}
        self.rosetta_docking_ligands = {}
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
                only_hetatoms=only_hetatm_conects,
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

    def addCappingGroups(self, rosetta_style_caps=False, prepwizard_style_caps=False,
                         openmm_style_caps=False, stdout=False, stderr=False,
                         conect_update=True, only_hetatoms=False):

        if sum([bool(rosetta_style_caps), bool(prepwizard_style_caps), bool(openmm_style_caps)]) > 1:
            raise ValueError('You must give only on cap style option!')

        # Manage stdout and stderr
        if stdout:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if stderr:
            stderr = None
        else:
            stderr = subprocess.DEVNULL

        if not os.path.exists('_capping_'):
            os.mkdir('_capping_')

        if not os.path.exists('_capping_/input_models'):
            os.mkdir('_capping_/input_models')

        if not os.path.exists('_capping_/output_models'):
            os.mkdir('_capping_/output_models')

        self.saveModels('_capping_/input_models')

        _copyScriptFile('_capping_', "addCappingGroups.py")
        command =  'run python3 _capping_/._addCappingGroups.py '
        command += '_capping_/input_models/ '
        command += '_capping_/output_models/ '
        if rosetta_style_caps:
            command += '--rosetta_style_caps '
        elif prepwizard_style_caps:
            command += '--prepwizard_style_caps '
        elif openmm_style_caps:
            command += '--openmm_style_caps '

        subprocess.run(command, shell=True, stdout=stdout, stderr=stderr)

        for f in os.listdir('_capping_/output_models'):
            model = f.replace('.pdb', '')
            self.readModelFromPDB(model, '_capping_/output_models/'+f, conect_update=conect_update, only_hetatoms=only_hetatoms)
        shutil.rmtree('_capping_')

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
        only_hetatoms=False,
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

        if model not in self.conects or self.conects[model] == [] and conect_update:
            # Read conect lines
            self.conects[model] = self._readPDBConectLines(pdb_file, model, only_hetatoms=only_hetatoms)

        # Check covalent ligands
        if covalent_check:
            self._checkCovalentLigands(model, pdb_file, atom_mapping=atom_mapping)

        # Update conect lines
        if conect_update:
            self.conects[model] = self._readPDBConectLines(pdb_file, model, only_hetatoms=only_hetatoms)

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

    def renumberModels(self, by_chain=True):
        """
        Renumber every PDB chain residues from 1 onward.
        """

        for m in self:
            structure = PDB.Structure.Structure(0)
            pdb_model = PDB.Model.Model(0)
            for model in self.structures[m]:
                i = 0
                for c in model:
                    if by_chain:
                        i = 0
                    residues = [r for r in c]
                    chain_copy = PDB.Chain.Chain(c.id)
                    for r in residues:
                        new_id = (r.id[0], i + 1, r.id[2])
                        c.detach_child(r.id)
                        r.id = new_id
                        chain_copy.add(r)
                        i += 1
                    pdb_model.add(chain_copy)

            structure.add(pdb_model)
            self.structures[m] = structure

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

    import mdtraj as md

    def calculateSecondaryStructure(self, simplified=True):
        """
        Calculate secondary structure information for each model using MDTraj.

        Parameters
        ==========
        simplified : bool, default=False
            If True, reduces the DSSP codes to:
            - H (helix) → "H"
            - E (sheet) → "E"
            - Everything else → "C" (coil)

        frame : int, default=0
            Frame index to extract the secondary structure from (MD simulations).

        dssp : str, default='score'
            The DSSP algorithm to use. Options:
            - 'score' : Uses MDTraj’s built-in DSSP scoring method.
            - 'sander' : Uses Sander DSSP method.
            - 'mkdssp' : Calls external DSSP executable (if available).

        Returns
        =======
        ss : dict
            Contains the secondary structure strings for each model.
        """

        for model in self.models_names:

            # Load structure into MDTraj
            traj = md.load(self.models_paths[model])

            # Compute secondary structure
            self.ss[model] = md.compute_dssp(traj, simplified=simplified)[0]

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
        self,
        confidence_threshold=70.0,
        keep_up_to=5,
        lr=None,
        ur=None,
        renumber=False,
        verbose=True,
        output=None):
        """
        Remove terminal regions with low confidence scores and optionally trim residues by range.

        Parameters:
            confidence_threshold : float
                AlphaFold confidence threshold to consider residues as having a low score.
            keep_up_to : int
                If any terminal region is no larger than this value it will be kept.
            lr : dict, optional
                Dictionary specifying the lower range of residue indices to keep per model.
            ur : dict, optional
                Dictionary specifying the upper range of residue indices to keep per model.
            renumber : bool
                Whether to renumber residues after trimming.
            verbose : bool
                Whether to print warnings and updates.
            output : str, optional
                File path to save the modified structure.
        """
        remove_models = set()
        for model in self.models_names:
            atoms = [a for a in self.structures[model].get_atoms()]
            bfactors = [a.bfactor for a in atoms]

            if np.average(bfactors) == 0:
                if verbose:
                    print(
                        f"Warning: model {model} has no atom with the selected confidence!"
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
                        f"Warning: model {model} has no atom with the selected confidence!"
                    )
                remove_models.add(model)
                continue

            n_terminus = sorted(list(n_terminus))
            c_terminus = sorted(list(c_terminus))

            if len(n_terminus) <= keep_up_to:
                n_terminus = []
            if len(c_terminus) <= keep_up_to:
                c_terminus = []

            model_lr = lr.get(model, None) if lr else None
            model_ur = ur.get(model, None) if ur else None

            for c in self.structures[model].get_chains():
                remove_this = []
                for r in c.get_residues():
                    if (
                        r.id[1] in n_terminus
                        or r.id[1] in c_terminus
                        or (
                            model_lr is not None
                            and model_ur is not None
                            and r.id[1] not in range(model_lr, model_ur + 1)
                        )
                    ):
                        remove_this.append(r)
                chain = c
                for r in remove_this:
                    chain.detach_child(r.id)

            if renumber:
                for c in self.structures[model].get_chains():
                    for i, r in enumerate(c):
                        r.id = (r.id[0], i + 1, r.id[2])

        for model in remove_models:
            self.removeModel(model)

        if output:
            for model in self.models_names:
                io.set_structure(self.structures[model])
                io.save(output)

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

                if not isinstance(mutants[model][mutant], list):
                    raise ValueError('Mutations for a particular variant should be given as a list of tuples!')

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
        hoh_to_wat=True,
        pdb_output=False,
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

        # Convert any water to WAT name
        if hoh_to_wat:
            for model in self:
                for r in self.structures[model].get_residues():
                    if r.id[0] == 'W':
                        r.resname = 'WAT'

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
                    relax_folder + "/output_models/" + model + "/" + model + "_relax.out"
                )
                if os.path.exists(score_file):
                    scores = _readRosettaScoreFile(score_file, skip_empty=True)
                    if not isinstance(scores, type(None)) and scores.shape[0] >= nstruct:
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
            if pdb_output:
                output_silent_file = None
            else:
                output_silent_file = model + "_relax.out"

            flags = rosettaScripts.flags(
                "../../xml/" + model + "_relax.xml",
                nstruct=nstruct,
                s="../../input_models/" + input_model,
                output_silent_file=output_silent_file
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

                for r in self.structures[model].get_residues():
                    if r.resname == 'NMA':
                        _copyScriptFile(relax_folder+"/params", 'NMA.params', subfolder='rosetta_params', path='prepare_proteins', hidden=False)

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
            command += "-f OPLS_2005 "
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
        exclude_models=None,
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

        if isinstance(models, str):
            models = [models]

        if isinstance(exclude_models, str):
            exclude_models = [exclude_models]

        # Save all input models
        self.saveModels(grid_folder + "/input_models", convert_to_mae=mae_input, models=models)

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

            if models and model not in models:
                continue

            if exclude_models and model in exclude_models:
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
        only_ligands=None,
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

        if isinstance(only_ligands, str):
            only_ligands = [only_ligands]

        if isinstance(models, str):
            models = [models]

        # Create docking job folders
        if not os.path.exists(docking_folder):
            os.mkdir(docking_folder)

        if not os.path.exists(docking_folder + "/input_models"):
            os.mkdir(docking_folder + "/input_models")

        if not os.path.exists(docking_folder + "/output_models"):
            os.mkdir(docking_folder + "/output_models")

        # Save all input models
        self.saveModels(docking_folder + "/input_models", models=models)

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

                if only_ligands and name not in only_ligands:
                    continue

                substrates_paths[name] = ligands_folder + "/" + f

        # Set up docking jobs
        jobs = []
        for grid in grids_paths:

            # Skip if models are given and not in models
            if models:
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

    def setUpRosettaDocking(self, docking_folder, ligands_pdb_folder=None, ligands_sdf_folder=None,
                            param_files=None,
                            coordinates=None, smiles_file=None, sdf_file=None, docking_protocol='repack',
                            high_res_cycles=None, high_res_repack_every_Nth=None, num_conformers=50,
                            prune_rms_threshold=0.5, max_attempts=1000, rosetta_home=None, separator='-',
                            use_exp_torsion_angle_prefs=True, use_basic_knowledge=True, only_scorefile=False,
                            enforce_chirality=True, skip_conformers_if_found=False, grid_width=10.0,
                            n_jobs=1, python2_executable='python2.7', store_initial_placement=False,
                            pdb_output=False, distances=None, angles=None, parallelisation="srun",
                            executable='rosetta_scripts.mpi.linuxgccrelease', ligand_chain='B', nstruct=100):

        """
        Set up docking calculations using Rosetta.

        Parameters:
        -----------
        docking_folder : str
            Path to the folder where docking results will be stored.
        ligands_pdb_folder : str, optional
            Path to the folder containing ligand PDB files. Mutually exclusive with `ligands_sdf_folder`, `smiles_file`, and `sdf_file`.
        ligands_sdf_folder : str, optional
            Path to the folder containing ligand SDF files. Mutually exclusive with `ligands_pdb_folder`, `smiles_file`, and `sdf_file`.
        smiles_file : str, optional
            Path to a file containing ligand SMILES strings. Mutually exclusive with `ligands_pdb_folder`, `ligands_sdf_folder`, and `sdf_file`.
        sdf_file : str, optional
            Path to an SDF file containing ligands. Mutually exclusive with `ligands_pdb_folder`, `ligands_sdf_folder`, and `smiles_file`.
        coordinates : (list, tuple, dict), optional
            The per-model coordinates dictionary to position the ligand (it can be multiple coordinates).
            If a list or tuples of coordinates is given, then they will be used for all the models.
        docking_protocol : str, optional
            Docking protocol to use. Available options are 'repack', 'mcm', and 'custom'. Default is 'repack'.
        high_res_cycles : int, optional
            Number of cycles for high-resolution docking when using the 'custom' protocol.
            Must be provided if 'custom' protocol is selected.
        high_res_repack_every_Nth : int, optional
            Repack frequency for high-resolution docking when using the 'custom' protocol.
            Must be provided if 'custom' protocol is selected.
        num_conformers : int, optional
            Number of conformers to generate for each ligand. Default is 50.
        prune_rms_threshold : float, optional
            RMSD threshold for pruning similar conformers. Default is 0.5 Å.
        max_attempts : int, optional
            Maximum number of attempts for embedding conformers. Default is 1000.
        rosetta_home : str, optional
            Path to the Rosetta home directory.
        separator : str, optional
            Separator character to use in file names. Default is '-'.
        use_exp_torsion_angle_prefs : bool, optional
            Use experimental torsion angle preferences in embedding. Default is True.
        use_basic_knowledge : bool, optional
            Use basic knowledge such as planarity of aromatic rings. Default is True.
        enforce_chirality : bool, optional
            Enforce correct chiral centers during embedding. Default is True.
        skip_conformers_if_found : bool, optional
            Skip generating conformers if they are already found. Default is False.
        grid_width : float, optional
            Width of the scoring grid. Default is 10.0.
        n_jobs : int, optional
            Number of parallel jobs to run. Default is 1.
        python2_executable : str, optional
            Path to the Python 2 executable. Default is 'python2.7'.
        executable : str, optional
            Path to the Rosetta scripts executable. Default is 'rosetta_scripts.mpi.linuxgccrelease'.
        ligand_chain : str, optional
            Chain identifier for the ligand. Default is 'B'.
        nstruct : int, optional
            Number of output structures to generate. Default is 100.

        Raises:
        -------
        ValueError
            If an invalid docking protocol is specified or if required parameters for the
            'custom' protocol are not provided, or if mutually exclusive inputs are given.

        Notes:
        ------
        This function prepares the necessary folders and XML script for running Rosetta docking
        simulations. It sets up score functions, ligand areas, interface builders, movemap builders,
        scoring grids, and movers based on the specified protocol. It also generates conformers for
        each ligand using RDKit and writes them to an SDF file with the PDB file name included if
        input is from PDB files, SMILES, or an SDF file.

        The generated XML script is saved in the specified docking folder.
        """

        def generate_conformers(mol, num_conformers=50, prune_rms_threshold=0.5, max_attempts=1000,
                                use_exp_torsion_angle_prefs=True, use_basic_knowledge=True,
                                enforce_chirality=True):
            """
            Generate and optimize conformers for a molecule.

            Parameters:
            -----------
            mol : rdkit.Chem.Mol
                The RDKit molecule for which to generate conformers.
            num_conformers : int, optional
                Number of conformers to generate. Default is 50.
            prune_rms_threshold : float, optional
                RMSD threshold for pruning similar conformers. Default is 0.5 Å.
            max_attempts : int, optional
                Maximum number of attempts for embedding conformers. Default is 1000.
            use_exp_torsion_angle_prefs : bool, optional
                Use experimental torsion angle preferences in embedding. Default is True.
            use_basic_knowledge : bool, optional
                Use basic knowledge such as planarity of aromatic rings. Default is True.
            enforce_chirality : bool, optional
                Enforce correct chiral centers during embedding. Default is True.

            Returns:
            --------
            rdkit.Chem.Mol
                RDKit molecule with embedded conformers.
            """
            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Ensure aromaticity is detected
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)

            # Embed multiple conformers
            params = AllChem.ETKDGv3()
            params.numThreads = 0  # Use all available threads
            params.pruneRmsThresh = prune_rms_threshold
            params.maxAttempts = max_attempts
            params.useExpTorsionAnglePrefs = use_exp_torsion_angle_prefs
            params.useBasicKnowledge = use_basic_knowledge
            params.enforceChirality = enforce_chirality

            conformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

            # Optimize each conformer using UFF
            for conf_id in conformers:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

            return mol

        def write_molecule_to_sdf(mol, output_sdf_path):
            """
            Write the molecule with conformers to an SDF file with explicit aromatic bonds.

            Parameters:
            -----------
            mol : rdkit.Chem.Mol
                The RDKit molecule with conformers.
            output_sdf_path : str
                Path to the output SDF file.
            """
            sdf_writer = Chem.SDWriter(output_sdf_path)
            sdf_writer.SetKekulize(False)  # Ensure aromatic bonds are not Kekulized
            for conf_id in range(mol.GetNumConformers()):
                sdf_writer.write(mol, confId=conf_id)
            sdf_writer.close()

        def process_ligand_file(ligand_file_path, base_name, conformers_input_folder):
            """
            Process a single ligand file (PDB, SDF, or SMILES) to generate conformers and write to SDF.

            Parameters:
            -----------
            ligand_file_path : str
                Path to the ligand file.
            base_name : str
                Base name of the ligand.
            conformers_input_folder : str
                Folder to store the output SDF file.
            """
            output_sdf_path = os.path.join(conformers_input_folder, f"{base_name}.sdf")

            # Determine file type and load molecule
            if ligand_file_path.endswith('.pdb'):
                mol = Chem.MolFromPDBFile(ligand_file_path, removeHs=False)
            elif ligand_file_path.endswith('.sdf'):
                mol_supplier = Chem.SDMolSupplier(ligand_file_path, removeHs=False)
                mol = mol_supplier[0]
            else:
                raise ValueError(f"Unsupported file type: {ligand_file_path}")

            if mol is None:
                raise ValueError(f"Could not read file: {ligand_file_path}")

            # Generate conformers
            mol_with_conformers = generate_conformers(mol, num_conformers=num_conformers,
                                                      prune_rms_threshold=prune_rms_threshold,
                                                      max_attempts=max_attempts,
                                                      use_exp_torsion_angle_prefs=use_exp_torsion_angle_prefs,
                                                      use_basic_knowledge=use_basic_knowledge,
                                                      enforce_chirality=enforce_chirality)

            # Set the molecule name
            mol_with_conformers.SetProp("_Name", base_name)

            # Write to SDF file
            write_molecule_to_sdf(mol_with_conformers, output_sdf_path)
            return base_name

        def make_param_file(input_sdf_path, ligand_name, output_dir):
            """
            Generate a parameter file for a ligand.

            Parameters:
            -----------
            input_sdf_path : str
                Path to the input SDF file containing conformers.
            ligand_name : str
                Name of the ligand.
            output_dir : str
                Directory to save the generated parameter file.

            Returns:
            --------
            str
                Path to the generated parameter file.
            """
            full_dir = os.path.join(output_dir, ligand_name)
            os.makedirs(full_dir, exist_ok=True)

            param_file_command = (
                f"{rosetta_home}/main/source/scripts/python/public/molfile_to_params.py -n {ligand_name} "
                f"--long-names --clobber --conformers-in-one-file --mm-as-virt {input_sdf_path}"
            )

            output_param_path = f"{ligand_name}.params"
            output_pdb_path = f"{ligand_name}.pdb"
            output_conformer_path = f"{ligand_name}_conformers.pdb"

            subprocess.call(param_file_command, shell=True)

            try:
                shutil.move(os.path.join(os.getcwd(), output_param_path), os.path.join(full_dir, output_param_path))
            except IOError:
                print(f"Something went wrong with {ligand_name}")
                return None
            shutil.move(os.path.join(os.getcwd(), output_pdb_path), os.path.join(full_dir, output_pdb_path))
            try:
                shutil.move(os.path.join(os.getcwd(), output_conformer_path), os.path.join(full_dir, output_conformer_path))
            except IOError:
                print(f"No conformers for {ligand_name}")
            return os.path.join(full_dir, output_param_path)

        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as e:
            raise ImportError("RDKit is not installed. Please install it to use the setUpRosettaDocking function.")

        if angles:
            raise ValueError('Angles has not being implemented. Rosetta scripts do not have an easy function to compute angles. Perhaps with CreateAngleConstraint mover?')

        # Check for mutually exclusive inputs
        if sum([bool(ligands_pdb_folder), bool(ligands_sdf_folder), bool(smiles_file), bool(sdf_file)]) != 1:
            raise ValueError('Specify exactly one of ligands_pdb_folder, ligands_sdf_folder, smiles_file, or sdf_file.')

        # Check given separator compatibility
        for model in self:
            if separator in model:
                raise ValueError(f'The given separator {separator} was found in model {model}. Please use a different one.')

        # Uniform coordinates format
        if isinstance(coordinates, (list, tuple)):
            if isinstance(coordinates[0], (list, tuple)) and len(coordinates[0]) == 3:
                tmp = {model: coordinates for model in self}
                coordinates = tmp
            elif len(coordinates) == 3 and isinstance(coordinates[0], (int, float)):
                tmp = {model: [coordinates] for model in self}
                coordinates = tmp
            else:
                raise ValueError('Check your given coordinates format')
        elif not isinstance(coordinates, dict):
            raise ValueError('Check your given coordinates format')

        # Check given protocol
        available_protocols = ['repack', 'mcm', 'custom']
        if docking_protocol not in available_protocols:
            raise ValueError(f'Invalid protocol. Available protocols are: {available_protocols}')

        if docking_protocol == 'custom' and (not high_res_cycles or not high_res_repack_every_Nth):
            raise ValueError('You must provide high_res_cycles and high_res_repack_every_Nth for the custom protocol.')

        # Create docking job folders
        os.makedirs(docking_folder, exist_ok=True)
        xml_folder = os.path.join(docking_folder, 'xml')
        conformers_input_folder = os.path.join(docking_folder, 'conformers')
        ligand_params_folder = os.path.join(docking_folder, 'ligand_params')
        input_models_folder = os.path.join(docking_folder, 'input_models')
        flags_folder = os.path.join(docking_folder, 'flags')
        output_folder = os.path.join(docking_folder, 'output_models')

        for folder in [xml_folder, conformers_input_folder, ligand_params_folder, input_models_folder,
                       flags_folder, output_folder]:
            os.makedirs(folder, exist_ok=True)

        # Write model structures
        self.saveModels(input_models_folder)

        # Create score functions
        ligand_soft_rep = rosettaScripts.scorefunctions.new_scorefunction('ligand_soft_rep',
                                                                          weights_file='ligand_soft_rep')
        ligand_soft_rep.addReweight('fa_elec', weight=0.42)
        ligand_soft_rep.addReweight('hbond_bb_sc', weight=1.3)
        ligand_soft_rep.addReweight('hbond_sc', weight=1.3)
        ligand_soft_rep.addReweight('rama', weight=0.2)

        ligand_hard_rep = rosettaScripts.scorefunctions.new_scorefunction('ligand_hard_rep',
                                                                          weights_file='ligand')
        ligand_hard_rep.addReweight('fa_intra_rep', weight=0.004)
        ligand_hard_rep.addReweight('fa_elec', weight=0.42)
        ligand_hard_rep.addReweight('hbond_bb_sc', weight=1.3)
        ligand_hard_rep.addReweight('hbond_sc', weight=1.3)
        ligand_hard_rep.addReweight('rama', weight=0.2)

        # Create ligand areas
        docking_sidechain = rosettaScripts.ligandArea('docking_sidechain',
                                                       chain=ligand_chain,
                                                       cutoff=6.0,
                                                       add_nbr_radius=True,
                                                       all_atom_mode=True,
                                                       minimize_ligand=10)

        final_sidechain = rosettaScripts.ligandArea('final_sidechain',
                                                     chain=ligand_chain,
                                                     cutoff=6.0,
                                                     add_nbr_radius=True,
                                                     all_atom_mode=True)

        final_backbone = rosettaScripts.ligandArea('final_backbone',
                                                    chain=ligand_chain,
                                                    cutoff=7.0,
                                                    add_nbr_radius=True,
                                                    all_atom_mode=True,
                                                    calpha_restraints=0.3)

        # Set interface builders
        side_chain_for_docking = rosettaScripts.interfaceBuilder('side_chain_for_docking',
                                                                 ligand_areas=docking_sidechain)
        side_chain_for_final = rosettaScripts.interfaceBuilder('side_chain_for_final',
                                                               ligand_areas=final_sidechain)
        backbone = rosettaScripts.interfaceBuilder('final_backbone',
                                                   ligand_areas=final_backbone,
                                                   extension_window=3)

        # Set movemap builders
        docking = rosettaScripts.movemapBuilder('docking', sc_interface=side_chain_for_docking)
        final = rosettaScripts.movemapBuilder('final', sc_interface=side_chain_for_final,
                                              bb_interface=backbone, minimize_water=True)

        # Set up scoring grid
        vdw = rosettaScripts.scoringGrid.classicGrid('vdw', weight=1.0)

        ### Create docking movers

        # Write coordinates
        xyz = []
        for model in coordinates:
            for coordinate in coordinates[model]:
                model_xyz = {}
                model_xyz['file_name'] = '../../input_models/'+model+'.pdb'
                model_xyz['x'] = coordinate[0]
                model_xyz['y'] = coordinate[1]
                model_xyz['z'] = coordinate[2]
                xyz.append(model_xyz)
        with open(docking_folder+'/coordinates.json', 'w') as jf:
            json.dump(xyz, jf)

        # Create start from mover for initial ligand positioning
        startFrom = rosettaScripts.movers.startFrom(chain=ligand_chain)
        startFrom.addFile('../../coordinates.json')

        if store_initial_placement:
            store_initial_placement = rosettaScripts.movers.dumpPdb(name='store_initial_placement', file_name='initial_placement.pdb')

        # Create transform mover
        transform = rosettaScripts.movers.transform(chain=ligand_chain, box_size=5.0, move_distance=0.1, angle=5,
                                                    cycles=500, repeats=1, temperature=5, initial_perturb=5.0)

        # Create high-resolution docking mover according to the employed docking protocol
        if docking_protocol == 'repack':
            cycles = 1
            repack_every_Nth = 1
        elif docking_protocol == 'mcm':
            cycles = 6
            repack_every_Nth = 3
        elif docking_protocol == 'custom':
            cycles = high_res_cycles
            repack_every_Nth = high_res_repack_every_Nth

        highResDocker = rosettaScripts.movers.highResDocker(cycles=cycles,
                                                            repack_every_Nth=repack_every_Nth,
                                                            scorefxn=ligand_soft_rep,
                                                            movemap_builder=docking)

        # Create final minimization mover
        finalMinimizer = rosettaScripts.movers.finalMinimizer(scorefxn=ligand_hard_rep, movemap_builder=final)

        # Add mover for reporting metrics
        interfaceScoreCalculator = rosettaScripts.movers.interfaceScoreCalculator(chains=ligand_chain,
                                                                                  scorefxn=ligand_hard_rep,
                                                                                  compute_grid_scores=False)

        ### Create combined protocols

        # Create low-resolution mover
        low_res_dock_movers = [transform]
        low_res_dock = rosettaScripts.movers.parsedProtocol('low_res_dock', low_res_dock_movers)

        # Create high-resolution mover
        high_res_dock_movers = [highResDocker, finalMinimizer]
        high_res_dock = rosettaScripts.movers.parsedProtocol('high_res_dock', high_res_dock_movers)

        # Create reporting combined mover
        reporting_movers = [interfaceScoreCalculator]
        reporting = rosettaScripts.movers.parsedProtocol('reporting', reporting_movers)

        # Convert ligand input to conformers
        ligands = []

        # Handle PDB ligand files
        if ligands_pdb_folder:
            for ligand_file in os.listdir(ligands_pdb_folder):
                if not ligand_file.endswith('.pdb'):
                    continue
                ligand_file_path = os.path.join(ligands_pdb_folder, ligand_file)
                base_name = os.path.splitext(ligand_file)[0]
                ligands.append(process_ligand_file(ligand_file_path, base_name, conformers_input_folder))

        # Handle SDF ligand files
        elif ligands_sdf_folder:
            for ligand_file in os.listdir(ligands_sdf_folder):
                if not ligand_file.endswith('.sdf'):
                    continue
                ligand_file_path = os.path.join(ligands_sdf_folder, ligand_file)
                base_name = os.path.splitext(ligand_file)[0]
                ligands.append(process_ligand_file(ligand_file_path, base_name, conformers_input_folder))

        # Handle SMILES input
        elif smiles_file:
            with open(smiles_file, 'r') as f:
                for line in f:
                    smi, name = line.strip().split()
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        raise ValueError(f"Could not parse SMILES: {smi}")
                    mol = Chem.AddHs(mol)
                    mol.SetProp("_Name", name)
                    output_sdf_path = os.path.join(conformers_input_folder, f"{name}.sdf")
                    ligands.append(name)
                    mol_with_conformers = generate_conformers(mol, num_conformers=num_conformers,
                                                              prune_rms_threshold=prune_rms_threshold,
                                                              max_attempts=max_attempts,
                                                              use_exp_torsion_angle_prefs=use_exp_torsion_angle_prefs,
                                                              use_basic_knowledge=use_basic_knowledge,
                                                              enforce_chirality=enforce_chirality)
                    write_molecule_to_sdf(mol_with_conformers, output_sdf_path)

        # Handle SDF input
        elif sdf_file:
            suppl = Chem.SDMolSupplier(sdf_file)
            for mol in suppl:
                if mol is None:
                    continue
                name = mol.GetProp("_Name")
                output_sdf_path = os.path.join(conformers_input_folder, f"{name}.sdf")
                ligands.append(name)
                mol_with_conformers = generate_conformers(mol, num_conformers=num_conformers,
                                                          prune_rms_threshold=prune_rms_threshold,
                                                          max_attempts=max_attempts,
                                                          use_exp_torsion_angle_prefs=use_exp_torsion_angle_prefs,
                                                          use_basic_knowledge=use_basic_knowledge,
                                                          enforce_chirality=enforce_chirality)
                write_molecule_to_sdf(mol_with_conformers, output_sdf_path)

        # Add path to params files
        if param_files != None:

            if not os.path.exists(docking_folder + "/params"):
                os.mkdir(docking_folder + "/params")

            if isinstance(param_files, str):
                param_files = [param_files]

            for param in param_files:
                param_name = param.split("/")[-1]
                shutil.copyfile(param, docking_folder + "/params/" + param_name)

        # Process each ligand
        jobs = []
        for ligand in ligands:

            # Add chain mover
            ligand_pdb = '../../ligand_params/'+ligand+'/'+ligand+'.pdb'
            addLigand = rosettaScripts.movers.addChain(name='addLigand', update_PDBInfo=True, file_name=ligand_pdb)

            # make params
            input_sdf_path = os.path.join(conformers_input_folder, f"{ligand}.sdf")
            output_dir = ligand_params_folder
            make_param_file(input_sdf_path, ligand, output_dir)

            # Process each model
            for model in self:

                # Create xml ligand docking
                xml = rosettaScripts.xmlScript()

                # Add score functions
                xml.addScorefunction(ligand_soft_rep)
                xml.addScorefunction(ligand_hard_rep)

                # Add ligand areas
                xml.addLigandArea(docking_sidechain)
                xml.addLigandArea(final_sidechain)
                xml.addLigandArea(final_backbone)

                # Add interface builders
                xml.addInterfaceBuilder(side_chain_for_docking)
                xml.addInterfaceBuilder(side_chain_for_final)
                xml.addInterfaceBuilder(backbone)

                # Add movemap builders
                xml.addMovemapBuilder(docking)
                xml.addMovemapBuilder(final)

                # Add scoring grid
                xml.addScoringGrid(vdw, ligand_chain=ligand_chain, width=grid_width)

                # Add doking movers
                xml.addMover(addLigand)
                xml.addMover(startFrom)
                if store_initial_placement:
                    xml.addMover(store_initial_placement)
                xml.addMover(transform)
                xml.addMover(highResDocker)
                xml.addMover(finalMinimizer)

                # Add scoring movers
                xml.addMover(interfaceScoreCalculator)

                # Add compund movers
                xml.addMover(low_res_dock)
                xml.addMover(high_res_dock)
                xml.addMover(reporting)

                # Set up protocol
                protocol = []
                protocol.append(addLigand)
                protocol.append(startFrom)
                if store_initial_placement:
                    protocol.append(store_initial_placement)
                protocol.append(low_res_dock)
                protocol.append(high_res_dock)
                protocol.append(reporting)

                # Add distance filters
                if distances:

                    # Define ligand residue as the consecutive from the last residue
                    for r in self.structures[model].get_residues():
                        r
                    ligand_residue = r.id[1]+1

                    for atoms in distances[model][ligand]:

                        if isinstance(atoms[0], str):
                            atoms = ((ligand_chain, ligand_residue, atoms[0]), atoms[1])
                        if isinstance(atoms[1], str):
                            atoms = (atoms[0], (ligand_chain, ligand_residue, atoms[1]))

                        label = "distance_"
                        label += "_".join([str(x) for x in atoms[0]]) + "-"
                        label += "_".join([str(x) for x in atoms[1]])

                        d = rosettaScripts.filters.atomicDistance(name=label,
                            residue1=str(atoms[0][1])+atoms[0][0], atomname1=atoms[0][2],
                            residue2=str(atoms[1][1])+atoms[1][0], atomname2=atoms[1][2],
                            distance=5.0, confidence=0.0)
                        xml.addFilter(d)
                        protocol.append(d)

    #             # Add angle filters
    #             if angles:
    #                 for atoms in angles[model][ligand]:
    #                     label = "angle_"
    #                     label += "_".join([str(x) for x in atoms[0]]) + "-"
    #                     label += "_".join([str(x) for x in atoms[1]]) + "-"
    #                     label += "_".join([str(x) for x in atoms[2]])
    #                     a = rosettaScripts.filters.atomicAngle(name=label, # There is no atomicAngle function in rosetta scripts
    #                         residue1=atoms[0][0]+str(atoms[0][1]), atomname1=atoms[0][2],
    #                         residue2=atoms[1][0]+str(atoms[1][1]), atomname2=atoms[1][2],
    #                         residue3=atoms[2][0]+str(atoms[2][1]), atomname3=atoms[2][2],
    #                         angle=120.0, confidence=0.0)
    #                     xml.addFilter(a)
    #                     protocol.append(a)

                # Set protocol
                xml.setProtocol(protocol)

                # Write XML protocol file
                xml_output = os.path.join(xml_folder, f'{docking_protocol}{separator}{ligand}{separator}{model}.xml')
                xml.write_xml(xml_output)

                if not only_scorefile and not pdb_output:
                    output_silent_file = f'{model}{separator}{ligand}.out'
                else:
                    output_silent_file = None

                # Create flags files
                flags = rosettaScripts.options.flags(f'../../xml/{docking_protocol}{separator}{ligand}{separator}{model}.xml',
                                                     nstruct=nstruct, s='../../input_models/'+model+'.pdb',
                                                     output_silent_file=output_silent_file,
                                                     output_score_file=f'{model}{separator}{ligand}.sc')
                if only_scorefile:
                    flags.addOption('out:file:score_only')

                ligand_params = '../../ligand_params/'+ligand+'/'+ligand+'.params'
                flags.addOption('extra_res_fa', ligand_params)
                flags.add_ligand_docking_options()
                if param_files:
                    flags.addOption("in:file:extra_res_path", "../../params")
                flags_output = os.path.join(flags_folder, f'{docking_protocol}{separator}{ligand}{separator}{model}.flags')
                flags.write_flags(flags_output)

                # Create output folder and execute commands
                model_output_folder = os.path.join(output_folder, model+separator+ligand)
                os.makedirs(model_output_folder, exist_ok=True)
                command =  'cd '+model_output_folder+'\n'
                if parallelisation:
                    command += parallelisation+' '
                command += executable+' '
                command += '@ ../../flags/'+f'{docking_protocol}{separator}{ligand}{separator}{model}.flags '
                command += '\n'
                command += 'cd ../../..'
                jobs.append(command)

        return jobs

    def generatePointsAroundAtoms(self, atom_tuples, radii, num_points_per_radius, threshold_distance, verbose=False):
        """
        Generates equidistant points around atoms on the surfaces of spheres with different radii using the Fibonacci lattice method.
        Only stores points that are farther than a threshold distance from any heavy atom.

        :param atom_tuples: Dictionary with model names as keys and atom tuples (chain_id, residue_id, atom_name) as values
        :param radii: List of radii of the spheres
        :param num_points_per_radius: Number of points to generate on each sphere
        :param threshold_distance: Minimum distance from any heavy atom to store a point
        :param verbose: If True, print detailed debug information
        :return: Dictionary of accumulated points around atoms by model
        """
        all_model_points = {}

        for model, atom_tuple in atom_tuples.items():
            structure = self.structures[model]
            chain_id, residue_id, atom_name = atom_tuple
            target_atom = None
            heavy_atom_coords = []

            # Find the target atom and collect all heavy atom coordinates
            for chain in structure.get_chains():
                if chain.id == chain_id:
                    for residue in chain.get_residues():
                        for atom in residue.get_atoms():
                            if atom.element != 'H':  # Exclude hydrogen atoms
                                heavy_atom_coords.append(atom.coord)
                            if residue.id[1] == residue_id and atom.name == atom_name:
                                target_atom = atom

            if not target_atom:
                raise ValueError(f"Target atom not found in the structure for model {model}")

            target_atom_coord = target_atom.coord
            all_points = []

            # Convert heavy atom coordinates to numpy array for distance calculations
            heavy_atom_coords = np.array(heavy_atom_coords)
            if verbose:
                print(f"Model: {model}, Heavy atom coordinates shape: {heavy_atom_coords.shape}")

            for radius in radii:
                points = []
                # Generate equidistant points on the surface of the sphere using Fibonacci lattice
                indices = np.arange(0, num_points_per_radius, dtype=float) + 0.5
                phi = np.arccos(1 - 2 * indices / num_points_per_radius)
                theta = np.pi * (1 + 5**0.5) * indices

                if verbose:
                    print(f"Model: {model}, Generated points before filtering for radius {radius}:")
                for i in range(num_points_per_radius):
                    x = radius * np.sin(phi[i]) * np.cos(theta[i])
                    y = radius * np.sin(phi[i]) * np.sin(theta[i])
                    z = radius * np.cos(phi[i])
                    point = target_atom_coord + np.array([x, y, z])
                    if verbose:
                        print(point)

                    # Check distance to all heavy atom points
                    if heavy_atom_coords.size > 0:
                        distances = cdist([point], heavy_atom_coords)
                        if verbose:
                            print(f"Model: {model}, Distances for point {i} at radius {radius}: {distances}")
                        if np.all(distances > threshold_distance):
                            points.append(point)
                    else:
                        points.append(point)

                if verbose:
                    print(f"Model: {model}, Points after filtering for radius {radius}: {points}")
                all_points.extend(points)

            if verbose:
                print(f"Model: {model}, Accumulated points: {all_points}")
            all_model_points[model] = all_points

        return all_model_points

    def visualizePointsOnStructure(self, points_by_model, output_folder, chain_id='A', residue_name="SPH", verbose=False):
        def addResidueToStructure(structure, chain_id, resname, atom_names, coordinates, new_resid=None, elements=None, hetatom=True, water=False):
            """
            Add a new residue with given atoms to the structure.
            """
            model = structure[0]
            chain = next((chain for chain in model.get_chains() if chain_id == chain.id), None)
            if not chain:
                if verbose:
                    print(f"Chain ID {chain_id} not found. Creating a new chain.")
                chain = Chain.Chain(chain_id)
                model.add(chain)

            if coordinates.ndim == 1:
                coordinates = np.array([coordinates])

            if coordinates.shape[1] != 3 or len(coordinates) != len(atom_names):
                if verbose:
                    print(f"Atom names: {atom_names}")
                    print(f"Coordinates: {coordinates}")
                raise ValueError("Mismatch between atom names and coordinates.")

            new_resid = new_resid or max((r.id[1] for r in chain.get_residues()), default=0) + 1
            rt_flag = 'H' if hetatom else 'W' if water else ' '
            residue = PDB.Residue.Residue((rt_flag, new_resid, ' '), resname, ' ')

            serial_number = max((a.serial_number for a in chain.get_atoms()), default=0) + 1
            for i, atnm in enumerate(atom_names):
                atom = PDB.Atom.Atom(atnm, coordinates[i], 0, 1.0, " ", f"{atnm: <4}", serial_number + i, elements[i] if elements else "H")
                residue.add(atom)
            chain.add(residue)
            return new_resid

        def addPointsToStructure(structure, points, chain_id, residue_name="SPH"):
            """
            Add points as a new residue to the structure.
            """
            atom_names = [f"SP{i+1}" for i in range(len(points))]
            coordinates = np.array(points)
            elements = ['H'] * len(points)
            if verbose:
                print(f"Number of points: {len(points)}")
                print(f"Number of atom names: {len(atom_names)}")
            addResidueToStructure(structure, chain_id, residue_name, atom_names, coordinates, elements=elements, hetatom=True)
            return structure

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for model, points in points_by_model.items():
            structure = copy.deepcopy(self.structures[model])  # Make a copy of the structure

            # Add points to structure
            structure = addPointsToStructure(structure, points, chain_id, residue_name)

            # Define output file path
            output_file = f"{output_folder}/{model}.pdb"

            # Save the modified structure to a PDB file
            _saveStructureToPDB(structure, output_file)

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
        sitemap_folder,
        distance_to_points=2.5,
        only_models=None,
        output_file=None,
        overwrite=False,
        replace_symbol=None,
        sidechain_only=False,
    ):
        """
        Calculates the active site residues based on the volume points from a sitemap
        calcualtion. The models should be written with the option output_models from
        the analiseSiteMapCalculation() function.

        Parameters
        ==========
        sitemap_folder : str
            Path to the folder where sitemap calculation, containing the sitemap volume points residues, is located.
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

        def merge_pdbs_one_model(pdb1, pdb2, output_pdb):
            parser = PDB.PDBParser(QUIET=True)
            io = PDB.PDBIO()

            # Load both PDB structures
            struct1 = parser.get_structure("struct1", pdb1)
            struct2 = parser.get_structure("struct2", pdb2)

            # Create a new structure and a new model (single frame)
            merged_struct = PDB.Structure.Structure("merged")
            merged_model = PDB.Model.Model(0)  # Single model with ID 0

            # Add all chains from the first structure's first model
            for chain in struct1[0]:
                merged_model.add(chain)

            # Add all chains from the second structure's first model
            for chain in struct2[0]:
                # Optionally, rename chain IDs if there's a conflict:
                # chain.id = chain.id + "_2"
                merged_model.add(chain)

            # Add the merged model to the new structure
            merged_struct.add(merged_model)

            # Write the merged structure to file
            io.set_structure(merged_struct)
            io.save(output_pdb)

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

                # Get input PDB
                input_pdb = f'{sitemap_folder}/input_models/{model}.pdb'

                if not os.path.exists(input_pdb):
                    raise ValueError(f'Input file {input_pdb} not found!')

                residues.setdefault(model, {})

                # Check if the volume points model file exists
                output_path = f'{sitemap_folder}/output_models/{model}'

                for resp in os.listdir(output_path):

                    if not bool(re.search(r'\d+$', resp)):
                        continue

                    resp_path = output_path+"/"+resp
                    volpts_file = None
                    for pdb in os.listdir(resp_path):
                        if pdb.endswith('volpts.pdb'):
                            volpts_file =  resp_path+"/"+pdb

                    if not volpts_file:
                        print(
                            "Model %s not found in the volume points folder %s!"
                            % (model, resp_path)
                        )
                        continue

                    # Combine input and volpts pdbs
                    tmp_file = volpts_file.replace('.pdb', '.tmp.pdb')
                    merge_pdbs_one_model(input_pdb, volpts_file, tmp_file)

                    # Compute neighbours
                    traj = md.load(tmp_file)
                    os.remove(tmp_file)

                    if sidechain_only:
                        protein = traj.topology.select("protein and sidechain and not resname vpt")
                    else:
                        protein = traj.topology.select("protein and not resname vpt")

                    vpts = traj.topology.select("resname vpt")
                    n = md.compute_neighbors(
                        traj, distance_to_points / 10, vpts, haystack_indices=protein
                    )
                    residues[model][resp] = list(
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
            for residue in residues[model]:
                residues[model][residue] = np.array(list(residues[model][residue]))

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
        com_residue_pairs={},
        epsilon=0.5,
        rescoring=False,
        ligand_equilibration_cst=True,
        regional_metrics=None,
        regional_thresholds=None,
        regional_combinations=None,
        regional_exclusions=None,
        max_regional_iterations=None,
        regional_energy_bias="Binding Energy",
        regional_best_fraction=0.2,
        constraint_level=1,
        restore_input_coordinates=False,
        skip_connect_rewritting=False
    ):
        """
        Generates a PELE calculation for extracted poses. The function reads all the
        protein-ligand poses and creates input files for a PELE simulation setup.

        Parameters
        ----------
        pele_folder : str
            Path to the folder where PELE calculations will be located.
        models_folder : str
            Path to the folder containing input docking poses.
        input_yaml : str
            Path to the input YAML file used as a template for all runs.
        box_centers : dict, optional
            Dictionary specifying the centers of the simulation box for each model.
        distances : dict, optional
            Distance constraints for the simulation. Format:
            {model_name: [(chain1, residue1, atom1), (chain2, residue2, atom2)]}.
        angles : dict, optional
            Angular constraints for the simulation.
        constraints : dict, optional
            Positional and distance constraints for each model and ligand.
        ligand_index : int, optional, default=1
            Index of the ligand in the structure file.
        box_radius : float, optional, default=10
            Radius of the simulation box.
        steps : int, optional, default=100
            Number of simulation steps per iteration.
        debug : bool, optional, default=False
            If True, enables debug mode for the simulation.
        iterations : int, optional, default=5
            Number of iterations for the simulation.
        cpus : int, optional, default=96
            Number of CPUs to use for parallelization.
        equilibration_steps : int, optional, default=100
            Number of equilibration steps before the production run.
        ligand_energy_groups : dict, optional
            Additional groups to consider for energy by residue reports.
        separator : str, optional, default="-"
            Separator used in filenames between protein and ligand identifiers.
        use_peleffy : bool, optional, default=True
            If True, PELEffy will be used for ligand parameterization.
        usesrun : bool, optional, default=True
            If True, the PELE simulation will use srun for job submission.
        energy_by_residue : bool, optional, default=False
            If True, energy by residue will be calculated.
        ebr_new_flag : bool, optional, default=False
            If True, uses the new version of the energy by residue calculation.
        ninety_degrees_version : bool, optional, default=False
            If True, uses the 90 degrees version of PELE.
        analysis : bool, optional, default=False
            If True, enables analysis mode after the simulation.
        energy_by_residue_type : str, optional, default="all"
            Type of energy to be calculated per residue. Options: 'all', 'lennard_jones', 'sgb', 'electrostatic'.
        peptide : bool, optional, default=False
            If True, treats the system as a peptide for specific setup steps.
        equilibration_mode : str, optional, default="equilibrationLastSnapshot"
            Mode used for equilibration: "equilibrationLastSnapshot" or "equilibrationCluster".
        spawning : str, optional, default="independent"
            Spawning method used for adaptive sampling. Options include 'independent', 'epsilon', 'variableEpsilon', etc.
        continuation : bool, optional, default=False
            If True, continues from the previous run instead of starting fresh.
        equilibration : bool, optional, default=True
            If True, performs equilibration before the production run.
        skip_models : list, optional
            List of protein models to skip from the simulation.
        skip_ligands : list, optional
            List of ligands to skip from the simulation.
        extend_iterations : bool, optional, default=False
            If True, extends the number of iterations beyond the default.
        only_models : list, optional
            List of protein models to include in the simulation.
        only_ligands : list, optional
            List of ligands to include in the simulation.
        only_combinations : list, optional
            List of protein-ligand combinations to include.
        ligand_templates : str, optional
            Path to custom ligand templates for parameterization.
        seed : int, optional, default=12345
            Random seed for the simulation.
        log_file : bool, optional, default=False
            If True, enables logging to a file.
        nonbonded_energy : dict, optional
            Dictionary specifying nonbonded energy atoms for specific protein-ligand pairs.
        nonbonded_energy_type : str, optional, default="all"
            Type of nonbonded energy to calculate. Options: 'all', 'lennard_jones', 'electrostatic'.
        nonbonded_new_flag : bool, optional, default=False
            If True, uses the new version of nonbonded energy calculation.
        covalent_setup : bool, optional, default=False
            If True, sets up the simulation for covalently bound ligands.
        covalent_base_aa : dict, optional
            Dictionary specifying the amino acid residue involved in covalent binding for each ligand.
        membrane_residues : dict, optional
            Dictionary specifying membrane residues to apply specific constraints.
        bias_to_point : dict, optional
            Dictionary specifying biasing points in the system for specific models.
        com_bias1 : dict, optional
            First group of atoms for center-of-mass biasing.
        com_bias2 : dict, optional
            Second group of atoms for center-of-mass biasing.
        epsilon : float, optional, default=0.5
            Epsilon value used for biasing the center-of-mass distance.
        rescoring : bool, optional, default=False
            If True, performs a rescoring calculation.
        ligand_equilibration_cst : bool, optional, default=True
            If True, applies constraints during ligand equilibration.
        regional_metrics : dict, optional
            Metrics for regional spawning in adaptive sampling.
        regional_thresholds : dict, optional
            Thresholds for regional spawning metrics.
        max_regional_iterations : int, optional
            Maximum number of iterations for regional spawning.
        regional_energy_bias : str, optional, default="Binding Energy"
            Bias metric for regional spawning, either 'Total Energy' or 'Binding Energy'.
        regional_best_fraction : float, optional, default=0.2
            Fraction of the best-performing states selected in regional spawning.
        constraint_level : int, optional, default=1
            Level of constraints applied during the simulation (0 for none, 1 for basic).
        restore_input_coordinates : bool, optional, default=False
            If True, restores the original coordinates after PELE processing (not working)

        Returns
        -------
        list
            A list of shell commands to run the PELE jobs.

        Detailed input examples
        -----------------------
        com_group_1 : list of lists
            A list containing atom definitions for the first group of atoms used in center-of-mass (COM) distance calculations.
            Each atom is defined as a list with the following format:

            [
                [chain_id, residue_id, atom_name],
                [chain_id, residue_id, atom_name],
                ...
            ]

            Where:
            - chain_id : str
                The ID of the chain where the atom is located (e.g., 'A', 'B').
            - residue_id : int
                The residue number where the atom is located (e.g., 100).
            - atom_name : str
                The name of the atom (e.g., 'CA' for alpha carbon, 'O' for oxygen).

            Example:
            [
                ["A", 100, "CA"],
                ["A", 102, "CB"],
                ["B", 150, "O"]
            ]

            In this example, the group includes:
            - The alpha carbon atom (CA) of residue 100 in chain 'A'.
            - The beta carbon atom (CB) of residue 102 in chain 'A'.
            - The oxygen atom (O) of residue 150 in chain 'B'.

        com_group_2 : list of lists
            The format is identical to com_group_1 but represents the second group of atoms.
            These two groups are used to calculate the center-of-mass distance between them.
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

            if (regional_combinations or regional_exclusions) and not (regional_combinations and regional_exclusions):
                raise ValueError('You must give both, regional_combinations and regional_exclusions not just one of them.')

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

        if isinstance(com_residue_pairs, type(None)):
            com_residue_pairs = {}

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
        models = {}
        ligand_pdb_name = {}
        pose_number = {}
        for d in os.listdir(models_folder):
            if os.path.isdir(models_folder + "/" + d):
                for f in os.listdir(models_folder + "/" + d):

                    fs = f.split(separator)
                    protein = fs[0]
                    ligand = fs[1]
                    pose = fs[2].replace(".pdb", "")
                    pose_number[(protein, ligand)] = pose

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
                        metric_types = {}
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
                                        prefix = "distance"
                                    elif len(v.split("_")) == 3:
                                        prefix = "angle"
                                    v =  prefix+'_'+v
                                else:
                                    prefix = v.split('_')[0]
                                reg_met[m].append(v)
                                metric_types.setdefault(m, prefix)

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

                        if regional_combinations:

                            # Collect all unique metrics from combinations
                            unique_metrics = set()
                            for new_metric, metrics in regional_combinations.items():
                                metric_metric_types = [metric_types[m] for m in reg_met]
                                if len(set(metric_metric_types)) != 1:
                                    raise ValueError('For regional spawning, you are attempting to combine different metric types (e.g., distances and angles) is not allowed.')
                                unique_metrics.update(metrics)
                                metrics_list = list(unique_metrics)

                            # Ensure all required metric columns were given in the regional metrics list
                            missing_columns = set(metrics_list) - set(regional_metrics.keys())
                            if missing_columns:
                                raise ValueError(f"Missing combination metrics in regional metrics: {missing_columns}")

                            # Check all exclusion metrics are defined in the combinations metrics
                            excusion_metrics = []

                            for exclusion in regional_exclusions:
                                if isinstance(regional_exclusions, list):
                                    excusion_metrics += [x for x in exclusion]
                                elif isinstance(regional_exclusions, dict):
                                    excusion_metrics += [x for x in regional_exclusions[exclusion]]

                            missing_columns = set(excusion_metrics) - set(metrics_list)
                            if missing_columns:
                                raise ValueError(f"Missing exclusion metrics in combination metrics: {missing_columns}")

                            with open(
                                protein_ligand_folder + "/regional_combinations.json", "w"
                            ) as jf:
                                json.dump(regional_combinations, jf)

                            with open(
                                protein_ligand_folder + "/regional_exclusions.json", "w"
                            ) as jf:
                                json.dump(regional_exclusions, jf)

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

                    if skip_connect_rewritting:
                        print(f'The structure {f} has pre-defined CONECT lines probably from extractPELEPoses() function. Skipping saving structure and re-writting them. Directly copying structure to PELE folder..')
                        # Specify the source file path
                        source_file = f'{models_folder}/{d}/{f}'
                        # Specify the destination file path
                        destination_folder = f'{protein_ligand_folder}/{f}'
                        # Perform the copy operation
                        print(f'Copying the structure {f} from source folder: {models_folder}/{d}/{f} to destination_folder: {protein_ligand_folder}')
                        shutil.copyfile(source_file, destination_folder)

                    else:
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
            pose = pose_number[model]
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
                        "constraint_level: " + str(constraint_level) + "\n"
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
                        com_residue_pairs,
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

            if protein in com_residue_pairs:
                _copyScriptFile(pele_folder, "addComDistances.py")
                cds_script = "._addComDistances.py"

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

                if protein in com_residue_pairs and ligand in com_residue_pairs[protein]:
                    # Write COM groups as json file
                    with open(
                        protein_ligand_folder + "/._com_groups.json", "w"
                    ) as jf:
                        json.dump(com_residue_pairs[protein][ligand], jf)

                    command += "python " + rel_path_to_root + cds_script + " "
                    command += "output "  # I think we should change this for a variable
                    command += "._com_groups.json\n"
                    continuation = True

                if covalent_setup:
                    command += covalent_command
                    continuation = True

                if restore_input_coordinates:
                    command += "python "+ rel_path_to_root+restore_coordinates_script_name+" "
                    command += "output/input/"+protein_ligand+separator+pose+".pdb "
                    command += "output/input/"+protein_ligand+separator+pose+"_processed.pdb\n"
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
                    continuation = True

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
                    _copyScriptFile(pele_folder, "changeAdaptiveIterations.py")
                    extend_script_name = "._changeAdaptiveIterations.py"
                    command += (
                        "python "
                        + rel_path_to_root
                        + extend_script_name
                        + " output "  # I think we should change this for a variable
                        + "--iterations "
                        + str(iterations)+' '
                        + "--steps "
                        + str(steps)+' '
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
                if regional_combinations:
                    command += "--combinations regional_combinations.json "
                    command += "--exclusions regional_exclusions.json "
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

    def setUpMDSimulations(self, md_folder, sim_time, nvt_time=2, npt_time=0.2, equilibration_dt=2, production_dt=2,
                           temperature=298.15, frags=1, local_command_name=None, remote_command_name="${GMXBIN}",
                           ff="amber99sb-star-ildn", ligand_chains=None, ion_chains=None, replicas=1, charge=None,
                           system_output="System", models=None, overwrite=False, remove_backups=False,constantph=False):
        """
        Sets up MD simulations for each model. The current state only allows to set
        up simulations using the Gromacs software.

        If the input pdb has additional non aa residues besides ligand (ions, HETATMs, ...)
        they should be separated in individual chains.

        Parameters:
        ==========
        md_folder : str
            Path to the job folder where the MD input files are located.
        sim_time : int
            Simulation time in ns
        nvt_time : float
            Time for NVT equilibration in ns
        npt_time : float
            Time for NPT equilibration in ns
        equilibration_dt : float
            Time step for equilibration in fs
        production_dt : float
            Time step for production in fs
        temperature : float
            Simulation temperature in K
        frags : int
            Number of fragments to divide the simulation.
        local_command_name : str
            Local command name for Gromacs.
        remote_command_name : str
            Remote command name for Gromacs.
        ff : str
            Force field to use for simulation.
        ligand_chains : list
            List of ligand chains.
        ion_chains : list
            List of ion chains.
        replicas : int
            Number of replicas.
        charge : int
            Charge of the system.
        system_output : str
            Output system name.
        models : list
            List of models.
        overwrite : bool
            Whether to overwrite existing files.
        remove_backups : bool
            Whether to remove backup files generated by Gromacs.
        """

        def _copyScriptFile(dest_folder, file, subfolder=None):
            source = resource_stream(Requirement.parse("prepare_proteins"), f"prepare_proteins/scripts/{subfolder}/{file}")
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            with open(os.path.join(dest_folder, file), 'wb') as dest_file:
                shutil.copyfileobj(source, dest_file)

        def _readGromacsIndexFile(file):
            with open(file, 'r') as f:
                groups = [x.replace('[', '').replace(']', '').replace('\n', '').strip() for x in f.readlines() if x.startswith('[')]
            return {g: str(i) for i, g in enumerate(groups)}

        def _getLigandParameters(structure, ligand_chains, struct_path, params_path, charge=None):
            class chainSelect(PDB.Select):
                def accept_chain(self, chain):
                    return chain.get_id() not in ligand_chains

            charge = charge or {}
            ligand_res = {chain.get_id(): residue.resname for mdl in structure for chain in mdl for residue in chain if chain.get_id() in ligand_chains}
            if not ligand_res:
                raise ValueError(f"Ligand was not found at chains {str(ligand_chains)}")

            io = PDB.PDBIO()
            pdb_chains = list(structure.get_chains())
            if len(pdb_chains) < 2:
                raise ValueError("Input pdb has only one chain. Protein and ligand should be separated in individual chains.")

            io.set_structure(structure)
            io.save(f"{struct_path}/protein.pdb", chainSelect())

            ligand_coords = {chain.get_id(): [a.coord for a in chain.get_atoms()] for chain in pdb_chains if chain.get_id() in ligand_chains}
            for chain_id, coords in ligand_coords.items():
                io.set_structure([chain for chain in pdb_chains if chain.get_id() == chain_id][0])
                io.save(f"{struct_path}/{ligand_res[chain_id]}.pdb")

            lig_counter = 0
            for lig_chain, ligand_name in ligand_res.items():
                lig_counter += 1
                if ligand_name not in os.listdir(params_path) or overwrite:
                    os.makedirs(f"{params_path}/{ligand_name}", exist_ok=True)
                    shutil.copyfile(f"{struct_path}/{ligand_name}.pdb", f"{params_path}/{ligand_name}/{ligand_name}.pdb")
                    os.chdir(f"{params_path}/{ligand_name}")

                    command = f"acpype -i {ligand_name}.pdb"
                    if ligand_name in charge:
                        command += f" -n {charge[ligand_name]}"
                    subprocess.run(command, shell=True)

                    with open(f"{ligand_name}.acpype/{ligand_name}_GMX.itp") as f:
                        lines = f.readlines()

                    atomtypes_lines, new_lines = [], []
                    atomtypes, atoms = False, False
                    for i, line in enumerate(lines):
                        if atomtypes:
                            if line.startswith("[ moleculetype ]"):
                                new_lines.append(line)
                                atomtypes = False
                            else:
                                spl = line.split()
                                if spl:
                                    spl[0] = ligand_name + spl[0]
                                    spl[1] = ligand_name + spl[1]
                                    atomtypes_lines.append(" ".join(spl))
                        elif atoms:
                            if line.startswith("[ bonds ]"):
                                new_lines.append(line)
                                atoms = False
                            else:
                                spl = line.split()
                                if spl:
                                    spl[1] = ligand_name + spl[1]
                                    new_lines.append(" ".join(spl) + "\n")
                        else:
                            new_lines.append(line)

                        if line.startswith(";name"):
                            if lines[i - 1].startswith("[ atomtypes ]"):
                                atomtypes = True

                        elif line.startswith(";"):
                            if lines[i - 1].startswith("[ atoms ]"):
                                atoms = True

                    print(lig_counter)
                    write_type = "w" if lig_counter == 1 else "a"
                    with open("../atomtypes.itp", write_type) as f:
                        if lig_counter == 1:
                            f.write("[ atomtypes ]\n")
                        for line in atomtypes_lines:
                            f.write(line + "\n")

                    with open(f"{ligand_name}.acpype/{ligand_name}_GMX.itp", "w") as f:
                        for line in new_lines:
                            if not line.startswith("[ atomtypes ]"):
                                f.write(line)

                    os.chdir("../../..")

                parser = PDB.PDBParser()
                ligand_structure = parser.get_structure("ligand", f"{params_path}/{ligand_name}/{ligand_name}.acpype/{ligand_name}_NEW.pdb")
                for i, atom in enumerate(ligand_structure.get_atoms()):
                    atom.coord = ligand_coords[lig_chain][i]
                io.set_structure(ligand_structure)
                io.save(f"{struct_path}/{ligand_name}.pdb")

            return ligand_res

        def _setupModelStructure(structure, ligand_chains, ion_chains):
            gmx_codes, ion_residues = [], []
            for mdl in structure:
                for chain in mdl:
                    for residue in chain:
                        if ion_chains and chain.get_id() in ion_chains:
                            ion_residues.append(residue.id[1])
                        HD1, HE2 = False, False
                        if residue.resname == "HIS":
                            for atom in residue:
                                if atom.name == "HD1":
                                    HD1 = True
                                if atom.name == "HE2":
                                    HE2 = True
                        if HD1 or HE2:
                            number = 0 if HD1 and not HE2 else 1 if HE2 and not HD1 else 2
                            gmx_codes.append(number)
            return str(gmx_codes)[1:-1].replace(",", ""), ion_residues

        def _createCAConstraintFile(structure, cst_file, sd=1.0):
            with open(cst_file, "w") as f:
                ref_res, ref_chain = None, None
                for r in structure.get_residues():
                    if r.id[0] != " ":
                        continue
                    res, chain = r.id[1], r.get_parent().id
                    if not ref_res:
                        ref_res, ref_chain = res, chain
                    if ref_chain != chain:
                        ref_res, ref_chain = res, chain

                    ca_atom = None
                    for atom in r.get_atoms():
                        if atom.name == "CA":
                            ca_atom = atom

                    if ca_atom != None:
                        ca_coordinate = list(ca_atom.coord)
                        cst_line = f"CoordinateConstraint CA {res}{chain} CA {ref_res}{ref_chain} "
                        cst_line += " ".join([f"{c:.4f}" for c in ca_coordinate]) + f" HARMONIC 0 {sd}\n"
                        f.write(cst_line)

        def _generateLocalCommand(command_name, model, i, ligand_chains, ligand_res, ion_residues, ion_chains, md_folder, ff, his_pro):
            command_local = f"cd {md_folder}\nexport GMXLIB=$(pwd)/FF\nmkdir -p output_models/{model}/{i}/topol\n"
            command_local += f"cp input_models/{model}/protein.pdb output_models/{model}/{i}/topol/protein.pdb\n"

            if ligand_chains:
                command_local += f"cp ligand_params/atomtypes.itp output_models/{model}/{i}/topol/atomtypes.itp\n"
                for ligand_name in ligand_res.values():
                    command_local += f"cp -r ligand_params/{ligand_name}/{ligand_name}.acpype output_models/{model}/{i}/topol/\n"

            command_local += f"cd output_models/{model}/{i}/topol\n"
            command_local += f"echo {his_pro} | {command_name} pdb2gmx -f protein.pdb -o prot.pdb -p topol.top -his -ignh -ff {ff} -water tip3p -vsite hydrogens\n"

            if ligand_chains:
                lig_files = " ".join([f" ../../../../input_models/{model}/{ligand_name}.pdb " for ligand_name in ligand_res.values()])
                command_local += f"grep -h ATOM prot.pdb {lig_files} >| complex.pdb\n"
                command_local += f"{command_name} editconf -f complex.pdb -o complex.gro\n"
                line = '#include "atomtypes.itp"\\n'
                included_ligands = []
                for ligand_name in ligand_res.values():
                    if ligand_name not in included_ligands:
                        included_ligands.append(ligand_name)
                        line += f'#include "{ligand_name}.acpype/{ligand_name}_GMX.itp"\\n'
                    line += "#ifdef POSRES\\n"
                    line += f'#include "{ligand_name}.acpype/posre_{ligand_name}.itp"\\n'
                    line += "#endif\\n"
                line += "'"
                local_path = (os.getcwd() + "/" + md_folder + "/FF").replace("/", "\/")
                command_local += f"sed -i '/#include \"{local_path}\/{ff}.ff\/forcefield.itp\"/a {line} topol.top\n"
                for ligand_name in ligand_res.values():
                    command_local += f"sed -i -e '$a{ligand_name.ljust(20)}1' topol.top\n"
            else:
                command_local += f"{command_name} editconf -f prot.pdb -o complex.gro\n"

            command_local += f"{command_name} editconf -f complex.gro -o prot_box.gro -c -d 1.0 -bt octahedron\n"
            command_local += f"{command_name} solvate -cp prot_box.gro -cs spc216.gro -o prot_solv.gro -p topol.top\n"
            command_local += f'echo q | {command_name} make_ndx -f prot_solv.gro -o index.ndx\n'

            return command_local

        def _removeBackupFiles(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.startswith("#") and file.endswith("#"):
                        os.remove(os.path.join(root, file))

        remote_command_name = "${GMXBIN}"
        if isinstance(models, str):
            models = [models]
        if not os.path.exists(md_folder):
            os.mkdir(md_folder)
        if not os.path.exists(f"{md_folder}/scripts"):
            os.mkdir(f"{md_folder}/scripts")
        if not os.path.exists(f"{md_folder}/FF"):
            os.mkdir(f"{md_folder}/FF")
        if not os.path.exists(f"{md_folder}/FF/{ff}.ff"):
            os.mkdir(f"{md_folder}/FF/{ff}.ff")
        if not os.path.exists(f"{md_folder}/input_models"):
            os.mkdir(f"{md_folder}/input_models")
        if not os.path.exists(f"{md_folder}/output_models"):
            os.mkdir(f"{md_folder}/output_models")

        if local_command_name is None:
            possible_command_names = ["gmx", "gmx_mpi"]
            command_name = None
            for command in possible_command_names:
                if shutil.which(command) is not None:
                    command_name = command
                    break
            if command_name is None:
                raise ValueError(f"Gromacs executable is required for the setup and was not found. The following executable names were tested: {','.join(possible_command_names)}")
        else:
            command_name = local_command_name

        if ligand_chains is not None:
            if isinstance(ligand_chains, str):
                ligand_chains = [ligand_chains]
            if not os.path.exists(f"{md_folder}/ligand_params"):
                os.mkdir(f"{md_folder}/ligand_params")

        self.saveModels(f"{md_folder}/input_models")

        for file in resource_listdir(Requirement.parse("prepare_proteins"), "prepare_proteins/scripts/md/gromacs/mdp"):
            if not file.startswith("__"):
                _copyScriptFile(f"{md_folder}/scripts", file, subfolder="md/gromacs/mdp")

        for file in resource_listdir(Requirement.parse("prepare_proteins"), f"prepare_proteins/scripts/md/gromacs/ff/{ff}"):
            if not file.startswith("__"):
                _copyScriptFile(f"{md_folder}/FF/{ff}.ff", file, subfolder=f"md/gromacs/ff/{ff}")

        for line in fileinput.input(f"{md_folder}/scripts/em.mdp", inplace=True):
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        for line in fileinput.input(f"{md_folder}/scripts/md.mdp", inplace=True):
            if "TIME_INTEGRATOR" in line:
                line = line.replace("TIME_INTEGRATOR", str(production_dt / 1000))
                if equilibration_dt > 2:
                    print("WARNING: you have selected a time integrator higher than 2 femtoseconds. Constraints have been automatically changed to all bonds. This may affect the accuracy of your simulation.")
                    cst = "all-bonds"
                else:
                    cst = "h-bonds"
            if "BOND_CONSTRAINTS" in line:
                line = line.replace("BOND_CONSTRAINTS", cst)
            if "NUMBER_OF_STEPS" in line:
                line = line.replace("NUMBER_OF_STEPS", str(int((sim_time * (1e6 / production_dt)) / frags)))
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        for line in fileinput.input(f"{md_folder}/scripts/nvt.mdp", inplace=True):
            if "TIME_INTEGRATOR" in line:
                line = line.replace("TIME_INTEGRATOR", str(equilibration_dt / 1000))
                if equilibration_dt > 2:
                    print("WARNING: you have selected a time integrator higher than 2 femtoseconds. Constraints have been automatically changed to all bonds. This may affect the accuracy of your simulation.")
                    cst = "all-bonds"
                else:
                    cst = "h-bonds"
            if "BOND_CONSTRAINTS" in line:
                line = line.replace("BOND_CONSTRAINTS", cst)
            if "NUMBER_OF_STEPS" in line:
                line = line.replace("NUMBER_OF_STEPS", str(int(nvt_time * (1e6 / equilibration_dt))))
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        for line in fileinput.input(f"{md_folder}/scripts/npt.mdp", inplace=True):
            if "TIME_INTEGRATOR" in line:
                line = line.replace("TIME_INTEGRATOR", str(equilibration_dt / 1000))
                if equilibration_dt > 2:
                    print("WARNING: you have selected a time integrator higher than 2 femtoseconds. Constraints have been automatically changed to all bonds. This may affect the accuracy of your simulation.")
                    cst = "all-bonds"
                else:
                    cst = "h-bonds"
            if "BOND_CONSTRAINTS" in line:
                line = line.replace("BOND_CONSTRAINTS", cst)
            if "NUMBER_OF_STEPS" in line:
                line = line.replace("NUMBER_OF_STEPS", str(int(npt_time * (1e6 / equilibration_dt))))
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "SYSTEM_OUTPUT" in line:
                line = line.replace("SYSTEM_OUTPUT", system_output)
            sys.stdout.write(line)

        jobs = []
        for model in self.models_names:
            if models and model not in models:
                continue

            if not os.path.exists(f"{md_folder}/input_models/{model}"):
                os.mkdir(f"{md_folder}/input_models/{model}")
            if not os.path.exists(f"{md_folder}/output_models/{model}"):
                os.mkdir(f"{md_folder}/output_models/{model}")

            for i in range(replicas):
                if not os.path.exists(f"{md_folder}/output_models/{model}/{i}"):
                    os.mkdir(f"{md_folder}/output_models/{model}/{i}")

            parser = PDB.PDBParser()
            structure = parser.get_structure("protein", f"{md_folder}/input_models/{model}.pdb")

            his_pro, ion_residues = _setupModelStructure(structure, ligand_chains, ion_chains)
            ligand_res = _getLigandParameters(structure, ligand_chains, f"{md_folder}/input_models/{model}", f"{md_folder}/ligand_params", charge=charge) if ligand_chains else shutil.copyfile(f"{md_folder}/input_models/{model}.pdb",f"{md_folder}/input_models/{model}/protein.pdb")

            for i in range(replicas):

                skip_local = os.path.exists(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx") and not overwrite
                if not skip_local:
                    command_local = _generateLocalCommand(command_name, model, i, ligand_chains, ligand_res, ion_residues, ion_chains, md_folder, ff, his_pro)
                    print(command_local)
                    #adfasdf
                    with open("tmp.sh", "w") as f:
                        f.write(command_local)
                    subprocess.run("bash tmp.sh", shell=True)
                    os.remove("tmp.sh")

                group_dics = {}
                group_dics["complex"] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx")

                if not skip_local:
                    os.system(f'echo "q"| {command_name} make_ndx -f {md_folder}/output_models/{model}/{i}/topol/complex.gro -o {md_folder}/output_models/{model}/{i}/topol/tmp_index.ndx')
                    group_dics['tmp_index'] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/tmp_index.ndx")

                    if 'Water' in group_dics['tmp_index']:
                        reading, crystal_waters_ndx_lines = False, '[ CrystalWaters ]\n'
                        for line in open(f"{md_folder}/output_models/{model}/{i}/topol/tmp_index.ndx"):
                            if '[' in line and reading:
                                reading = False
                            elif '[ Water ]' in line:
                                reading = True
                            elif reading:
                                crystal_waters_ndx_lines += line

                        with open(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx", 'a') as f:
                            f.write(crystal_waters_ndx_lines)
                        os.system(f'echo \"{group_dics["complex"]["Water"]} & !{len(group_dics["complex"])}\nq\" | {command_name} make_ndx -f {md_folder}/output_models/{model}/{i}/topol/prot_solv.gro -o {md_folder}/output_models/{model}/{i}/topol/index.ndx -n {md_folder}/output_models/{model}/{i}/topol/index.ndx')
                        os.system(f'echo \"del {group_dics["complex"]["SOL"]}\n name {len(group_dics["complex"])} SOL\nq\" | {command_name} make_ndx -f {md_folder}/output_models/{model}/{i}/topol/prot_solv.gro -o {md_folder}/output_models/{model}/{i}/topol/index.ndx -n {md_folder}/output_models/{model}/{i}/topol/index.ndx')

                        group_dics['complex'] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx")

                sol_group = 'SOL'
                skip_ions = os.path.exists(f"{md_folder}/output_models/{model}/{i}/topol/prot_ions.gro") and not overwrite
                if not skip_ions:
                    command_local = f"cd {md_folder}/output_models/{model}/{i}/topol\n"
                    command_local += f"{command_name} grompp -f ../../../../scripts/ions.mdp -c prot_solv.gro -p topol.top -o prot_ions.tpr -maxwarn 1\n"
                    command_local += f"echo {group_dics['complex'][sol_group]} | {command_name} genion -s prot_ions.tpr -o prot_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.1 -n index.ndx\n"
                    # if constatph add buffer
                    if constantph:
                        command_local += f"{command_name} grompp -f ../../../../scripts/ions.mdp -c prot_ions.gro -p topol.top -o prot_buf.tpr -maxwarn 1\n"
                        command_local += f"echo {group_dics['complex'][sol_group]} | {command_name} genion -s prot_buf.tpr -p topol.top -o prot_buf.gro -np 1 -rmin 1.0 -pname BUF -n index.ndx\n"
                        file_name = 'prot_buf'
                    else:
                        file_name = 'prot_ions'

                    command_local += f'echo "q"| {command_name} make_ndx -f {file_name}.gro -o index.ndx\n'

                    with open("tmp.sh", "w") as f:
                        f.write(command_local)
                    subprocess.run("bash tmp.sh", shell=True)
                    os.remove("tmp.sh")

                    group_dics["complex"] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx")

                if ligand_chains or ion_residues:
                    skip_ndx = os.path.exists(f"{md_folder}/output_models/{model}/{i}/topol/posre.itp") and not overwrite
                    if not skip_ndx:
                        command_local = f"cd {md_folder}/output_models/{model}/{i}/topol\n"
                        lig_selector = ""
                        if ligand_chains:
                            for ligand_name in ligand_res.values():
                                command_local += f'echo -e "0 & ! a H*\nq"| {command_name} make_ndx -f {ligand_name}.acpype/{ligand_name}_GMX.gro -o {ligand_name}_index.ndx\n'
                                lig_selector += f"{group_dics['complex'][ligand_name]}|"

                        ion_selector, water_and_solventions_selector = "", ""
                        if ion_residues:
                            for r in ion_residues:
                                ion_selector += f"r {r}|"
                                water_and_solventions_selector += f" ! r {r} &"

                        selector_line = ""
                        if lig_selector and ion_selector:
                            selector_line += f"{group_dics['complex']['Protein']}|{ion_selector[:-1]}|{lig_selector[:-1]}\n"
                            selector_line += f"{group_dics['complex']['Protein']}|{ion_selector[:-1]}\n"
                            selector_line += f"{group_dics['complex'][sol_group]} | {group_dics['complex']['Ion']} & {water_and_solventions_selector[:-1]}\n"
                        elif ion_selector:
                            selector_line += f"{group_dics['complex']['Protein']}|{ion_selector[:-1]}\n"
                            selector_line += f"{group_dics['complex'][sol_group]} | {group_dics['complex']['Ion']} & {water_and_solventions_selector[:-1]}\n"
                        elif lig_selector:
                            selector_line += f"{group_dics['complex']['Protein']}|{lig_selector[:-1]}\n"

                        command_local += f'echo -e "{selector_line}q"| {command_name} make_ndx -f {file_name}.gro -o index.ndx\n'


                        with open("tmp.sh", "w") as f:
                            f.write(command_local)
                        subprocess.run("bash tmp.sh", shell=True)
                        os.remove("tmp.sh")

                        group_dics["complex"] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/index.ndx")

                        if ligand_chains:
                            for ligand_name in ligand_res.values():
                                group_dics[ligand_name] = _readGromacsIndexFile(f"{md_folder}/output_models/{model}/{i}/topol/{ligand_name}_index.ndx")

                command = f"export GMXLIB=$(pwd)/{md_folder}/FF" + "\n"
                command += f"cd {md_folder}/output_models/{model}/{i}\n"
                local_path = os.getcwd() + f"/{md_folder}/FF"
                command += f'sed -i  "s#{local_path}#$GMXLIB#g" topol/topol.top\n'

                skip_em = os.path.exists(f"{md_folder}/output_models/{model}/{i}/em/prot_em.tpr") and not overwrite
                if not skip_em:
                    command += "mkdir -p em\n"
                    command += "cd em\n"
                    command += f"{remote_command_name} grompp -f ../../../../scripts/em.mdp -c ../topol/{file_name}.gro -p ../topol/topol.top -o prot_em.tpr\n"
                    command += f"{remote_command_name} mdrun -v -deffnm prot_em\n"
                    command += "cd ..\n"

                skip_nvt = os.path.exists(f"{md_folder}/output_models/{model}/{i}/nvt/prot_nvt.tpr") and not overwrite
                if not skip_nvt:
                    command += "mkdir -p nvt\n"
                    command += "cd nvt\n"
                    command += "cp -r ../../../../scripts/nvt.mdp .\n"
                    tc_grps1, tc_grps2 = ["Protein"], "SOL_Ion" if ion_residues else "Water_and_ions"
                    if ion_residues:
                        for r in ion_residues:
                            tc_grps1.append(f"r_{r}")
                            tc_grps2 += f"_&_!r_{r}"
                    if ligand_chains:
                        tc_grps1.extend(ligand_res.values())
                    command += f"sed -i  '/tc-grps/c\\tc-grps = {'_'.join(tc_grps1)} {tc_grps2}' nvt.mdp\n"
                    if ligand_chains:
                        for ligand_name in ligand_res.values():
                            command += f"echo {group_dics[ligand_name]['System_&_!H*']} | {remote_command_name} genrestr -f ../topol/{ligand_name}.acpype/{ligand_name}_GMX.gro -n ../topol/{ligand_name}_index.ndx -o ../topol/{ligand_name}.acpype/posre_{ligand_name}.itp -fc 1000 1000 1000\n"
                    sel = group_dics['complex']["Protein"] if not ion_residues else f"Protein_{'_'.join([f'r_{r}' for r in ion_residues])}"
                    command += f"echo {sel} | {remote_command_name} genrestr -f ../topol/{file_name}.gro -o ../topol/posre.itp -fc 1000 1000 1000 -n ../topol/index.ndx\n"
                    command += f"{remote_command_name} grompp -f nvt.mdp -c ../em/prot_em.gro -p ../topol/topol.top -o prot_nvt.tpr -r ../em/prot_em.gro -n ../topol/index.ndx\n"
                    command += f"{remote_command_name} mdrun -v -deffnm prot_nvt\n"
                    command += "cd ..\n"

                FClist = ("550", "300", "170", "90", "50", "30", "15", "10", "5")
                skip_npt = os.path.exists(f"{md_folder}/output_models/{model}/{i}/npt/prot_npt_{len(FClist)}.tpr") and not overwrite
                if not skip_npt:
                    command += "mkdir -p npt\n"
                    command += "cd npt\n"
                    tc_grps1, tc_grps2 = ["Protein"], "SOL_Ion" if ion_residues else "Water_and_ions"
                    if ion_residues:
                        for r in ion_residues:
                            tc_grps1.append(f"r_{r}")
                            tc_grps2 += f"_&_!r_{r}"
                    if ligand_chains:
                        tc_grps1.extend(ligand_res.values())
                    command += "cp -r ../../../../scripts/npt.mdp .\n"
                    command += f"sed -i  '/tc-grps/c\\tc-grps = {'_'.join(tc_grps1)} {tc_grps2}' npt.mdp\n"
                    sel = group_dics['complex']["Protein"] if not ion_residues else f"Protein_{'_'.join([f'r_{r}' for r in ion_residues])}"
                    for j in range(len(FClist) + 1):
                        if not os.path.exists(f"{md_folder}/output_models/{model}/{i}/npt/prot_npt_{j + 1}.tpr") or overwrite:
                            if j == 0:
                                command += f"{remote_command_name} grompp -f npt.mdp -c ../nvt/prot_nvt.gro -t ../nvt/prot_nvt.cpt -p ../topol/topol.top -o prot_npt_1.tpr -r ../nvt/prot_nvt.gro -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} mdrun -v -deffnm prot_npt_{j + 1}\n"
                            else:
                                if ligand_chains:
                                    for ligand_name in ligand_res.values():
                                        command += f"echo {group_dics[ligand_name]['System_&_!H*']} | {remote_command_name} genrestr -f ../topol/{ligand_name}.acpype/{ligand_name}_GMX.gro -n ../topol/{ligand_name}_index.ndx -o ../topol/{ligand_name}.acpype/posre_{ligand_name}.itp -fc {FClist[j - 1]} {FClist[j - 1]} {FClist[j - 1]}\n"
                                command += f"echo {sel} | {remote_command_name} genrestr -f ../topol/{file_name}.gro -o ../topol/posre.itp -fc {FClist[j - 1]} {FClist[j - 1]} {FClist[j - 1]} -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} grompp -f npt.mdp -c prot_npt_{j}.gro -t prot_npt_{j}.cpt -p ../topol/topol.top -o prot_npt_{j + 1}.tpr -r prot_npt_{j}.gro -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} mdrun -v -deffnm prot_npt_{j + 1}\n"
                    command += "cd ..\n"

                skip_md = os.path.exists(f"{md_folder}/output_models/{model}/{i}/md/prot_md_{frags}.xtc") and not overwrite
                if not skip_md:
                    command += "mkdir -p md\n"
                    command += "cd md\n"
                    tc_grps1, tc_grps2 = ["Protein"], "SOL_Ion" if ion_residues else "Water_and_ions"
                    if ion_residues:
                        for r in ion_residues:
                            tc_grps1.append(f"r_{r}")
                            tc_grps2 += f"_&_!r_{r}"
                    if ligand_chains:
                        tc_grps1.extend(ligand_res.values())
                    command += "cp -r ../../../../scripts/md.mdp .\n"
                    command += f"sed -i  '/tc-grps/c\\tc-grps = {'_'.join(tc_grps1)} {tc_grps2}' md.mdp\n"
                    for j in range(1, frags + 1):
                        if not os.path.exists(f"{md_folder}/output_models/{model}/{i}/md/prot_md_{j}.xtc") or overwrite:
                            if j == 1:
                                command += f"{remote_command_name} grompp -f md.mdp -c ../npt/prot_npt_{len(FClist) + 1}.gro  -t ../npt/prot_npt_{len(FClist) + 1}.cpt -p ../topol/topol.top -o prot_md_{j}.tpr -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} mdrun -v -deffnm prot_md_{j}\n"
                            else:
                                command += f"{remote_command_name} grompp -f md.mdp -c prot_md_{j - 1}.gro -t prot_md_{j - 1}.cpt -p ../topol/topol.top -o prot_md_{j}.tpr -n ../topol/index.ndx\n"
                                command += f"{remote_command_name} mdrun -v -deffnm prot_md_{j}\n"
                    command += "cd ../../../..\n"
                else:
                    command = ''

                if command.strip():
                    jobs.append(command)

        if remove_backups:
            _removeBackupFiles(md_folder)

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

    def setUpOpenMMSimulations(self, job_folder, replicas, simulation_time, ligand_charges=None, residue_names=None, ff='amber14',
                               add_bonds=None, skip_ligands=None, metal_ligand=None, metal_parameters=None, skip_replicas=None,
                               extra_frcmod=None, extra_mol2=None, dcd_report_time=100.0, data_report_time=100.0,
                               non_standard_residues=None, add_hydrogens=True, extra_force_field=None,
                               nvt_time=0.1, npt_time=0.2, nvt_temp_scaling_steps=50, npt_restraint_scaling_steps=50,
                               restraint_constant=100.0, chunk_size=100.0, equilibration_report_time=1.0, temperature=300.0,
                               collision_rate=1.0, time_step=0.002, cuda=False, fixed_seed=None, script_file=None, extra_script_options=None,
                               add_counterionsRand=False,charge_model='bcc'):
        """
        Set up OpenMM simulations for multiple models with customizable ligand charges, residue names, and force field options.
        Includes support for multiple replicas.

        Parameters:
        - job_folder (str): Path to the folder where the simulations will be set up.
        - replicas (int): The number of replicas to generate for each model.
        - simulation_time (float): Total simulation time in ns (mandatory).
        - ligand_charges (dict, optional): Dictionary of ligand charges to apply during parameterization.
        - residue_names (dict, optional): Dictionary of residue names to rename for each model. Format: {'model': {(chain, resnum): 'new_resname'}}.
        - ff (str): Force field to use for the simulations (default is 'amber14').
        - add_bonds (dict, optional): Dictionary of bonds to be added between atoms for each model.
        - skip_ligands (list, optional): List of ligand residue names to skip during parameterization.
        - metal_ligand (optional): Specify metal-ligand interactions if applicable.
        - metal_parameters (optional): Additional parameters for metals.
        - extra_frcmod (optional): Extra force field modifications.
        - extra_mol2 (optional): Extra mol2 files for ligand parameterization.
        - dcd_report_time (float): The DCD report interval in ps.
        - data_report_time (float): The data report interval in ps.
        - nvt_time (float): The NVT equilibration time in ns.
        - npt_time (float): The NPT equilibration time in ns.
        - nvt_temp_scaling_steps (int): The number of iterations for NVT temperature scaling.
        - npt_restraint_scaling_steps (int): The number of iterations for NPT restraint scaling.
        - restraint_constant (float): Force constant for positional restraints (kcal/mol/Å²).
        - chunk_size (float): The chunk size for output report files (data and dcd) in ns.
        - equilibration_report_time (float): The report interval for equilibration steps in ps.
        - temperature (float): The temperature for the simulation (K).
        - collision_rate (float): The collision rate for the Langevin integrator (1/ps).
        - time_step (float): The simulation time step (ps).
        - cuda (bool): Whether to use CUDA for GPU acceleration.
        - fixed_seed (int, optional): A fixed seed for the simulation, if provided.
        - script_file (str, optional): Path to the OpenMM simulation script.
        - add_counterionsRand(bool, optional): use the tleap function add_counterionsRand instead of addions2 to place the ions in the simulation box. It places
        the ions randomly in the simulation box without computing charges, so it is much faster than addions2. The default is False. Use when you have a large number of systems to prepare.

        """

        def setUpJobs(job_folder, openmm_md, script_file, simulation_time=simulation_time, dcd_report_time=dcd_report_time,
                      data_report_time=data_report_time, nvt_time=nvt_time, npt_time=npt_time,
                      nvt_temp_scaling_steps=nvt_temp_scaling_steps, npt_restraint_scaling_steps=npt_restraint_scaling_steps,
                      restraint_constant=restraint_constant, chunk_size=chunk_size,
                      equilibration_report_time=equilibration_report_time, temperature=temperature,
                      collision_rate=collision_rate, time_step=time_step, cuda=cuda, fixed_seed=fixed_seed, add_counterionsRand=add_counterionsRand):
            """
            Subfunction to set up individual OpenMM simulation jobs with inherited parameters.
            """
            if not os.path.exists(os.path.join(job_folder, 'input_files')):
                os.makedirs(os.path.join(job_folder, 'input_files'))

            prmtop_name = os.path.basename(openmm_md.prmtop_file)
            inpcrd_name = os.path.basename(openmm_md.inpcrd_file)

            shutil.copyfile(openmm_md.prmtop_file, os.path.join(job_folder, 'input_files', prmtop_name))
            shutil.copyfile(openmm_md.inpcrd_file, os.path.join(job_folder, 'input_files', inpcrd_name))
            script = os.path.basename(script_file)
            shutil.copyfile(script_file, os.path.join(job_folder, script))

            jobs = []
            command = ''
            if not fixed_seed:
                command += f'SEED=$(($SLURM_JOB_ID + $RANDOM % 100000))\n'
            else:
                command += f'SEED=' + str(fixed_seed) + '\n'
            command += 'echo employed seed: $SEED\n'
            command += f'cd {job_folder}\n'
            command += f'python {script} '
            command += f'input_files/{prmtop_name} '
            command += f'input_files/{inpcrd_name} '
            command += f'{simulation_time} '
            command += f'--dcd_report_time {dcd_report_time} '
            command += f'--data_report_time {data_report_time} '
            command += f'--nvt_time {nvt_time} '
            command += f'--npt_time {npt_time} '
            command += f'--nvt_temp_scaling_steps {nvt_temp_scaling_steps} '
            command += f'--npt_restraint_scaling_steps {npt_restraint_scaling_steps} '
            command += f'--restraint_constant {restraint_constant} '
            command += f'--chunk_size {chunk_size} '
            command += f'--equilibration_report_time {equilibration_report_time} '
            command += f'--temperature {temperature} '
            command += f'--collision_rate {collision_rate} '
            command += f'--time_step {time_step} '
            command += f'--seed $SEED '
            if extra_script_options:
                for option in extra_script_options:
                    command += f'--{option[0]} {str(option[1])} '
            command += '\n'
            command += f'cd {"../" * len(job_folder.split(os.sep))}\n'
            jobs.append(command)

            return jobs

        # Create the base job folder
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        self.openmm_md = {}  # Dictionary to hold openmm_md instances for each model

        ligand_parameters_folder = os.path.join(job_folder, 'parameters')
        if not os.path.exists(ligand_parameters_folder):
            os.mkdir(ligand_parameters_folder)

        script_folder = os.path.join(job_folder, 'scripts')
        if not os.path.exists(script_folder):
            os.mkdir(script_folder)

        if not script_file:
            _copyScriptFile(script_folder, "openmm_simulation.py", subfolder='md/openmm', hidden=False)
            script_file = script_folder + '/openmm_simulation.py'

        # Iterate over all models
        simulation_jobs = []
        for model in self:
            model_folder = os.path.join(job_folder, model)

            # Create a subdirectory for the model if it doesn't exist
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)

            # Generate OpenMM class for setting up calculations
            self.openmm_md[model] = prepare_proteins.MD.openmm_md(self.models_paths[model])
            self.openmm_md[model].setUpFF(ff)  # Define the force field

            if add_hydrogens:
                # Get and set protonation states
                variants = self.openmm_md[model].getProtonationStates()
                self.openmm_md[model].removeHydrogens()
                self.openmm_md[model].addHydrogens(variants=variants)

            # Parameterize ligands and build amber topology
            qm_jobs = self.openmm_md[model].parameterizePDBLigands(
                ligand_parameters_folder, charges=ligand_charges, metal_ligand=metal_ligand,
                add_bonds=add_bonds.get(model) if add_bonds else None,  # Use model-specific bond additions
                skip_ligands=skip_ligands, overwrite=False, metal_parameters=metal_parameters,
                extra_frcmod=extra_frcmod, extra_mol2=extra_mol2, cpus=20, return_qm_jobs=True,
                extra_force_field=extra_force_field,charge_model=charge_model,
                force_field='ff14SB', residue_names=residue_names.get(model) if residue_names else None,  # Use model-specific residue renaming
                add_counterions=True, add_counterionsRand=add_counterionsRand,save_amber_pdb=True, solvate=True, regenerate_amber_files=False,
                non_standard_residues=non_standard_residues
            )

            # Create folders for replicas
            zfill = max(len(str(replicas)), 2)
            for replica in range(1, replicas+1):

                if not isinstance(skip_replicas, type(None)) and replica in skip_replicas:
                    continue

                replica_str = str(replica).zfill(zfill)
                replica_folder = os.path.join(model_folder, f'replica_{replica_str}')

                if not os.path.exists(replica_folder):
                    os.mkdir(replica_folder)

                # Call the subfunction to set up the individual simulation for each replica
                simulation_jobs += setUpJobs(replica_folder, self.openmm_md[model], script_file)

        return simulation_jobs

    def setUpPLACERcalculation(self, PLACERfolder, output_folder="output_folder", PLACER_PATH="/gpfs/projects/bsc72/conda_envs/PLACER/", suffix=None, num_samples=50,
                           ligand=None, apo=False,rerank="prmsd", mutate=None, mutate_chain="A",
                           mutate_to=None, residue_json=None):
        """
        Set up PLACER calculations for evaluating catalytic centers, with or without ligand.
        Special amino acids can be added.

        Visit https://github.com/baker-laboratory/PLACER/tree/main for more options.

        Parameters
        ----------
        PLACERfolder : str
            Directory where all job-related files will be stored.
        output_folder : str, default="output_folder"
            Folder containing PDB files to run.
        suffix : str, optional
            Suffix added to output PDB file.
        num_samples : int, default=50
            Number of samples to generate. 50-100 is a good number in most cases.
        ligand : str, optional
            Ligand <name3>, <name3-resno>, or <chain-name3-resno> (e.g., "L-LIG-1") to be predicted.
            All other ligands will be fixed.
            If not specified, PLACER will detect the ligand automatically.
        apo : bool, default=False
            run PLACER in apo mode:
        rerank : str, optional
            Rank models using one of the input metrics: "prmsd", "plddt", or "plddt_pde".
            "prmsd" is sorted in ascending order, "plddt" and "plddt_pde" in descending order.
        mutate : dict, optional
            Dictionary with model names as keys and residue numbers to mutate as values.
            Requires `mutate_to` to be specified.
        mutate_chain : str, default="A"
            Chain where the residues to mutate are located.
        mutate_to : str, optional
            Residue name (3-letter code) to mutate into.
        residue_json : str, optional
            JSON file specifying custom residues used in the PDB or with `mutate`.

        Returns
        -------
        list
            List of command-line commands to be executed.
        """
        # validate PLACER_PATH option
        valid_PLACER_PATH = {"/gpfs/projects/bsc72/conda_envs/PLACER/","/gpfs/home/bsc/bsc072871/repos/PLACER/","/shared/work/NBD_Utilities/PLACER/"}
        if PLACER_PATH not in valid_PLACER_PATH:
            raise ValueError(f"Invalid path! option. Choose from {valid_PLACER_PATH}")

        # Validate rerank option
        rerank_options = {"prmsd", "plddt_pde", "plddt"}
        if rerank not in rerank_options:
            raise ValueError(f"Invalid rerank option '{rerank}'. Choose from {rerank_options}")

        # Validate mutation options
        if mutate:
            if not isinstance(mutate, dict):
                raise TypeError("Expected 'mutate' to be a dictionary mapping model names to residue numbers.")
            if not mutate_to or not isinstance(mutate_to, str) or len(mutate_to) != 3:
                raise ValueError("Expected 'mutate_to' to be a 3-letter residue code string.")

        # Validate ligand options
        if ligand:
            if apo:
                raise ValueError("Cannot specify both ligand and apo at the same time!.")

        # Prepare output directories
        os.makedirs(PLACERfolder, exist_ok=True)
        input_pdbs_folder = os.path.join(PLACERfolder, "input_pdbs")
        os.makedirs(input_pdbs_folder, exist_ok=True)

        # Save input models
        self.saveModels(input_pdbs_folder)

        # Generate PLACER commands
        jobs = []
        for model in self:
            pdb_name = f"{model}.pdb"
            pdb_path = os.path.join(input_pdbs_folder, pdb_name)

            command = f"python {PLACER_PATH}run_PLACER.py "
            command +=  f"-f {pdb_path} "
            command +=  f"-o {PLACERfolder}/{output_folder} "
            command +=  f"--nsamples {num_samples} "

            if suffix:
                command += f"--suffix {suffix} "
            if rerank:
                command += f"--rerank {rerank} "
            if ligand:
                command += f"--predict_ligand {ligand} "
            if apo:
                command += f"--no-use_sm "
            if mutate:
                command += f"--mutate {mutate[model]}{mutate_chain}:{mutate_to} "
                command += "--no-use_sm "
            if residue_json:
                command += f"--residue_json {residue_json} "
            command += "\n"
            jobs.append(command)

        return jobs

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
        overwrite=False,
        only_models=None,
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

        # Create scores data folder
        if not os.path.exists(docking_folder + '/'+output_folder+"/scores"):
            os.mkdir(docking_folder + '/'+output_folder+"/scores")

        # Create distance data folder
        if not os.path.exists(docking_folder + '/'+output_folder+"/atom_pairs"):
            os.mkdir(docking_folder + '/'+output_folder+"/atom_pairs")

        # Create angle data folder
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

        if isinstance(only_models, str):
            only_models = [only_models]

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
        if only_models:
            command += " --only_models " + ",".join(only_models)
        else:
            command += " --only_models " + ",".join(self.models_names)
        if overwrite:
            command += " --overwrite "
        command += ' --analysis_folder '+output_folder

        os.system(command)

        # # Read the CSV file into pandas
        # if not os.path.exists(docking_folder + '/'+output_folder+"/docking_data.csv"):
        #     raise ValueError(
        #         "Docking analysis failed. Check the ouput of the analyse_docking.py script."
        #     )

        # Read scores data
        scores_directory = docking_folder + '/'+output_folder+"/scores"
        self.docking_data = []
        for f in os.listdir(scores_directory):
            model = f.split(separator)[0]
            ligand = f.split(separator)[1].split(".")[0]

            # Read the CSV file into pandas
            self.docking_data.append(pd.read_csv(
                scores_directory+'/'+f
            ))

        # Concatenate the list of DataFrames into a single DataFrame
        self.docking_data = pd.concat(self.docking_data)
        self.docking_data.set_index(["Protein", "Ligand", "Pose"], inplace=True)

        distances_directory = docking_folder + '/'+output_folder+"/atom_pairs"
        for f in os.listdir(distances_directory):
            model = f.split(separator)[0]
            ligand = f.split(separator)[1].split(".")[0]

            # Read the CSV file into pandas
            self.docking_distances.setdefault(model, {})
            self.docking_distances[model][ligand] = pd.read_csv(
                distances_directory+'/'+f
            )
            self.docking_distances[model][ligand].set_index(
                ["Protein", "Ligand", "Pose"], inplace=True
            )

            self.docking_ligands.setdefault(model, [])
            if ligand not in self.docking_ligands[model]:
                self.docking_ligands[model].append(ligand)

        angles_directory = docking_folder + '/'+output_folder+"/angles"
        if os.path.exists(angles_directory):
            for f in os.listdir(angles_directory):
                model = f.split(separator)[0]
                ligand = f.split(separator)[1].split(".")[0]

                # Read the CSV file into pandas
                self.docking_angles.setdefault(model, {})
                self.docking_angles[model][ligand] = pd.read_csv(
                    angles_directory +'/'+ f
                )
                self.docking_angles[model][ligand].set_index(
                    ["Protein", "Ligand", "Pose"], inplace=True
                )

        if return_failed:
            with open(docking_folder + '/'+output_folder+"/._failed_dockings.json") as jifd:
                failed_dockings = json.load(jifd)
            return failed_dockings

    def analyseDockingParallel(self,
        docking_folder,
        protein_atoms=None,
        angles=None,
        atom_pairs=None,
        skip_chains=False,
        return_failed=False,
        ignore_hydrogens=False,
        separator="-",
        overwrite=False,
        only_models=None,
        compute_sasa=False,
        output_folder='.analysis'):
        """
        Set up jobs for analysing individual docking and creating CSV files. The files should be
        read by the analyseDocking function (i.e., the non-parallel version).
        """

        # Create analysis folder
        if not os.path.exists(docking_folder + '/'+output_folder):
            os.mkdir(docking_folder + '/'+output_folder)

        # Create scores data folder
        if not os.path.exists(docking_folder + '/'+output_folder+"/scores"):
            os.mkdir(docking_folder + '/'+output_folder+"/scores")

        # Create distance data folder
        if not os.path.exists(docking_folder + '/'+output_folder+"/atom_pairs"):
            os.mkdir(docking_folder + '/'+output_folder+"/atom_pairs")

        # Create angle data folder
        if angles:
            if not os.path.exists(docking_folder + '/'+output_folder+"/angles"):
                os.mkdir(docking_folder + '/'+output_folder+"/angles")

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        prepare_proteins._copyScriptFile(
            docking_folder + '/'+output_folder, "analyse_individual_docking.py"
        )
        script_path = docking_folder + '/'+output_folder+"/._analyse_individual_docking.py"

        # Write protein_atoms dictionary to json file
        if protein_atoms:
            with open(docking_folder + '/'+output_folder+"/._protein_atoms.json", "w") as jf:
                json.dump(protein_atoms, jf)

        if isinstance(only_models, str):
            only_models = [only_models]

        # Write atom_pairs dictionary to json file
        if atom_pairs:
            with open(docking_folder + '/'+output_folder+"/._atom_pairs.json", "w") as jf:
                json.dump(atom_pairs, jf)

        # Write angles dictionary to json file
        if angles:
            with open(docking_folder + '/'+output_folder+"/._angles.json", "w") as jf:
                json.dump(angles, jf)

        jobs = []
        for model in os.listdir(docking_folder+'/output_models'):

            # Skip models not given in only_models
            if only_models != None  and model not in only_models:
                continue

            # Check separator in model name
            if separator in model:
                raise ValueError('The separator %s was found in model name %s. Please use a different one!' % (separator, model))

            for f in os.listdir(docking_folder+'/output_models/'+model):

                subjobs = None
                mae_output = None

                if f.endswith('.maegz'):
                    ligand = f.replace(model+'_','').replace('_pv.maegz','')

                    # Check separator in ligand name
                    if separator in ligand:
                        raise ValueError('The separator %s was found in ligand name %s. Please use a different one!' % (separator, ligand))

                    mae_output = docking_folder+'/output_models/'+model+'/'+f

                    csv_name = model+separator+ligand+'.csv'
                    scores_csv = docking_folder+'/'+output_folder+'/scores/'+csv_name
                    distance_csv = docking_folder+'/'+output_folder+'/atom_pairs/'+csv_name
                    angles_csv = docking_folder+'/'+output_folder+'/angles/'+csv_name

                    skip_scores = True
                    skip_distances = True
                    skip_angles = True

                    if not os.path.exists(scores_csv) or overwrite:
                        skip_scores = False

                    if atom_pairs and not os.path.exists(distance_csv) or overwrite:
                        skip_distances = False

                    if angles and not os.path.exists(angles_csv) or overwrite:
                        skip_angles = False

                    if skip_scores and skip_distances and skip_angles:
                        continue

                    command  = 'run '
                    command += docking_folder+'/'+output_folder+"/._analyse_individual_docking.py "
                    command += docking_folder+' '
                    command += mae_output+' '
                    command += model+' '
                    command += ligand+' '
                    if atom_pairs:
                        command += "--atom_pairs " + docking_folder + '/'+output_folder+"/._atom_pairs.json "
                    elif protein_atoms:
                        command += "--protein_atoms " + docking_folder + '/'+output_folder+"/._protein_atoms.json "
                    if angles:
                        command += " --angles " + docking_folder + '/'+output_folder+"/._angles.json"
                    if skip_chains:
                        command += " --skip_chains"
                    if return_failed:
                        command += " --return_failed"
                    if ignore_hydrogens:
                        command += " --ignore_hydrogens"
                    if compute_sasa:
                        command += " --compute_sasa"
                    command += " --separator " + separator
                    command += '\n'
                    jobs.append(command)

        return jobs

    def analyseRosettaDocking(self, docking_folder, separator='_', only_models=None):

        # Initialize an empty DataFrame for all scores
        self.rosetta_docking = pd.DataFrame()

        # Initialize an empty dictionary for distances
        self.rosetta_docking_distances = {}

        output_models_path = os.path.join(docking_folder, 'output_models')
        model_ligands = os.listdir(output_models_path)

        # Filter models to process if only_models is provided
        if only_models is not None:
            if isinstance(only_models, str):
                only_models = [only_models]
            model_ligands = [ml for ml in model_ligands if ml.split(separator)[0] in only_models]

        total_files = len(model_ligands)
        processed_files = 0

        # Loop through models
        for model_ligand in model_ligands:
            if model_ligand.count(separator) != 1:
                raise ValueError(f"The separator '{separator}' was not found or found more than once in '{model_ligand}'")

            model, ligand = model_ligand.split(separator)

            try:
                # Check for .out file first
                scorefile_out = os.path.join(output_models_path, f'{model}{separator}{ligand}/{model}{separator}{ligand}.out')
                scorefile_sc = os.path.join(output_models_path, f'{model}{separator}{ligand}/{model}{separator}{ligand}.sc')

                if os.path.exists(scorefile_out):
                    scorefile = scorefile_out
                elif os.path.exists(scorefile_sc):
                    scorefile = scorefile_sc
                else:
                    raise FileNotFoundError(f"Neither '{model}{separator}{ligand}.out' nor '{model}{separator}{ligand}.sc' found for '{model_ligand}'")

                scores = _readRosettaScoreFile(scorefile)
                scores['Ligand'] = ligand  # Add ligand column to scores
                scores = scores.set_index(['Model', 'Ligand', 'Pose'])

                # Extract distance columns
                distance_columns = [col for col in scores.columns if col.startswith('distance_')]
                distance_df = scores[distance_columns]

                # Add to the distances dictionary
                if model not in self.rosetta_docking_distances:
                    self.rosetta_docking_distances[model] = {}
                self.rosetta_docking_distances[model][ligand] = distance_df

                # Remove distance columns from scores
                scores = scores.drop(columns=distance_columns)

                # Reorder columns to put 'interface_delta_*' after 'total_score'
                interface_delta_col = next((col for col in scores.columns if col.startswith('interface_delta_')), None)
                if interface_delta_col and 'total_score' in scores.columns:
                    cols = list(scores.columns)
                    cols.insert(cols.index('total_score') + 1, cols.pop(cols.index(interface_delta_col)))
                    scores = scores[cols]

                # Append the remaining scores to the global DataFrame
                self.rosetta_docking = pd.concat([self.rosetta_docking, scores])

            except FileNotFoundError as e:
                print(f"\nSkipping {model_ligand} due to missing file: {e}")
                continue

            # Update and print progress
            processed_files += 1
            progress = f"Processing: {processed_files}/{total_files} files"
            sys.stdout.write('\r' + progress)
            sys.stdout.flush()

        # Print a final newline character to move to the next line after the loop is done
        print()

    def combineRosettaDockingDistancesIntoMetrics(self, catalytic_labels, overwrite=False):
        """
        Combine different equivalent distances into specific named metrics. The function
        takes as input a dictionary (catalytic_labels) composed of inner dictionaries as follows:

            catalytic_labels = {
                metric_name = {
                    model = {
                        ligand = distances_list}}}

        The innermost distances_list object contains all equivalent distance names for
        a specific model and ligand pair to be combined under the same metric_name column.

        The combination is done by taking the minimum value of all equivalent distances.

        Parameters
        ==========
        catalytic_labels : dict
            Dictionary defining which distances will be combined under a common name.
            (for details see above).
        """

        # Initialize the metric type dictionary if it doesn't exist
        if not hasattr(self, 'rosetta_docking_metric_type'):
            self.rosetta_docking_metric_type = {}

        for name in catalytic_labels:
            if "metric_" + name in self.rosetta_docking.columns and not overwrite:
                print(
                    f"Combined metric {name} already added. Give overwrite=True to recombine"
                )
            else:
                values = []
                for model in self.rosetta_docking.index.get_level_values('Model').unique():

                    # Check whether model is found in docking distances
                    if model not in self.rosetta_docking_distances:
                        continue

                    model_series = self.rosetta_docking[
                        self.rosetta_docking.index.get_level_values("Model") == model
                    ]

                    for ligand in self.rosetta_docking.index.get_level_values('Ligand').unique():

                        # Check whether ligand is found in model's docking distances
                        if ligand not in self.rosetta_docking_distances[model]:
                            continue

                        ligand_series = model_series[
                            model_series.index.get_level_values("Ligand") == ligand
                        ]

                        # Check input metric
                        distance_metric = False
                        angle_metric = False
                        for x in catalytic_labels[name][model][ligand]:
                            if len(x.split("-")) == 2:
                                distance_metric = True
                            elif len(x.split("-")) == 3:
                                angle_metric = True

                        if distance_metric and angle_metric:
                            raise ValueError(
                                f"Metric {name} combines distances and angles which is not supported."
                            )

                        if distance_metric:
                            distances = catalytic_labels[name][model][ligand]
                            distance_values = (
                                self.rosetta_docking_distances[model][ligand][distances]
                                .min(axis=1)
                                .tolist()
                            )
                            assert ligand_series.shape[0] == len(distance_values)
                            values += distance_values
                            self.rosetta_docking_metric_type["metric_" + name] = "distance"
                        elif angle_metric:
                            angles = catalytic_labels[name][model][ligand]
                            if len(angles) > 1:
                                raise ValueError(
                                    "Combining more than one angle into a metric is not currently supported."
                                )
                            angle_values = (
                                self.rosetta_docking_angles[model][ligand][angles]
                                .min(axis=1)
                                .tolist()
                            )
                            assert ligand_series.shape[0] == len(angle_values)
                            values += angle_values
                            self.rosetta_docking_metric_type["metric_" + name] = "angle"

                self.rosetta_docking["metric_" + name] = values

    def rosettaDockingBindingEnergyLandscape(self, initial_threshold=3.5, vertical_line=None, xlim=None, ylim=None, clim=None, color=None,
                                             size=1.0, alpha=0.05, vertical_line_width=0.5, vertical_line_color='k', dataframe=None,
                                             title=None, no_xticks=False, no_yticks=False, no_xlabel=False, no_ylabel=False,
                                             no_cbar=False, xlabel=None, ylabel=None, clabel=None, relative_total_energy=False):
        """
        Plot binding energy as an interactive plot.

        Parameters
        ==========
        initial_threshold : float, optional
            Initial threshold value for metrics sliders. Default is 3.5.
        vertical_line : float, optional
            Position to plot a vertical line.
        xlim : tuple, optional
            The limits for the x-axis range.
        ylim : tuple, optional
            The limits for the y-axis range.
        clim : tuple, optional
            The limits for the color range.
        color : str, optional
            Column name to use for coloring the plot. Can also be a fixed color.
        size : float, optional
            Scale factor for the plot size. Default is 1.0.
        alpha : float, optional
            Alpha value for the scatter plot markers. Default is 0.05.
        vertical_line_width : float, optional
            Width of the vertical line. Default is 0.5.
        vertical_line_color : str, optional
            Color of the vertical line. Default is 'k' (black).
        dataframe : pandas.DataFrame, optional
            Dataframe containing the data. If not provided, self.rosetta_docking is used.
        title : str, optional
            Title of the plot.
        no_xticks : bool, optional
            If True, x-axis ticks are not shown. Default is False.
        no_yticks : bool, optional
            If True, y-axis ticks are not shown. Default is False.
        no_xlabel : bool, optional
            If True, the x-axis label is not shown. Default is False.
        no_ylabel : bool, optional
            If True, the y-axis label is not shown. Default is False.
        no_cbar : bool, optional
            If True, the color bar is not shown. Default is False.
        xlabel : str, optional
            Label for the x-axis. If not provided, defaults to the x parameter.
        ylabel : str, optional
            Label for the y-axis. If not provided, defaults to the y parameter.
        clabel : str, optional
            Label for the color bar. If not provided, defaults to color_column.
        relative_total_energy : bool, optional
            If True, color values are shown relative to their minimum value. Default is False.
        """

        if not self.rosetta_docking_distances:
            raise ValueError('There are no distances in the docking data. Use calculateDistances to show plot.')

        def getLigands(model, dataframe=None):
            if dataframe is not None:
                model_series = dataframe[dataframe.index.get_level_values('Model') == model]
            else:
                model_series = self.rosetta_docking[self.rosetta_docking.index.get_level_values('Model') == model]

            ligands = list(set(model_series.index.get_level_values('Ligand').tolist()))
            ligands_ddm = Dropdown(options=ligands, description='Ligand', style={'description_width': 'initial'})

            interact(getDistance, model_series=fixed(model_series), model=fixed(model), ligand=ligands_ddm)

        def getDistance(model_series, model, ligand, by_metric=False):
            ligand_series = model_series[model_series.index.get_level_values('Ligand') == ligand]

            distances = []
            distance_label = 'Distance'
            if by_metric:
                distances = [d for d in ligand_series if d.startswith('metric_') and not ligand_series[d].dropna().empty]
                distance_label = 'Metric'

            if not distances:
                if model in self.rosetta_docking_distances and ligand in self.rosetta_docking_distances[model]:
                    distances = [d for d in self.rosetta_docking_distances[model][ligand] if 'distance' in d or '_coordinate' in d]
                if model in self.rosetta_docking_angles and ligand in self.rosetta_docking_angles[model]:
                    distances += [d for d in self.rosetta_docking_angles[model][ligand] if 'angle' in d]
                if 'Ligand RMSD' in self.rosetta_docking:
                    distances.append('Ligand RMSD')

            if not distances:
                raise ValueError('No distances or metrics found for this ligand. Consider calculating some distances.')

            distances_ddm = Dropdown(options=distances, description=distance_label, style={'description_width': 'initial'})

            interact(getMetrics, distances=fixed(distances_ddm), ligand_series=fixed(ligand_series),
                     model=fixed(model), ligand=fixed(ligand))

        def getMetrics(ligand_series, distances, model, ligand, filter_by_metric=False, filter_by_label=False,
                       color_by_metric=False, color_by_labels=False):

            if color_by_metric or filter_by_metric:
                metrics = [k for k in ligand_series.keys() if 'metric_' in k]
                metrics_sliders = {}
                for m in metrics:
                    if self.rosetta_docking_metric_type[m] == 'distance':
                        m_slider = FloatSlider(value=initial_threshold, min=0, max=max(30, max(ligand_series[m])), step=0.1,
                                               description=f"{m}:", disabled=False, continuous_update=False,
                                               orientation='horizontal', readout=True, readout_format='.2f')
                    elif self.rosetta_docking_metric_type[m] in ['angle', 'torsion']:
                        m_slider = FloatRangeSlider(value=[110, 130], min=-180, max=180, step=0.1,
                                                    description=f"{m}:", disabled=False, continuous_update=False,
                                                    orientation='horizontal', readout=True, readout_format='.2f')

                    metrics_sliders[m] = m_slider
            else:
                metrics_sliders = {}

            if filter_by_label:
                labels_ddms = {}
                labels = [l for l in ligand_series.keys() if 'label_' in l]
                for l in labels:
                    label_options = [None] + sorted(list(set(ligand_series[l])))
                    labels_ddms[l] = Dropdown(options=label_options, description=l, style={'description_width': 'initial'})
            else:
                labels_ddms = {}

            interact(getColor, distance=distances, model=fixed(model), ligand=fixed(ligand),
                     metrics=fixed(metrics_sliders), ligand_series=fixed(ligand_series),
                     color_by_metric=fixed(color_by_metric), color_by_labels=fixed(color_by_labels), **labels_ddms)

        def getColor(distance, ligand_series, metrics, model, ligand, color_by_metric=False,
                     color_by_labels=False, **labels):

            if color is None:
                color_columns = [k for k in ligand_series.keys() if ':' not in k and 'distance' not in k and not k.startswith('metric_') and not k.startswith('label_')]
                color_columns = [None, 'Epoch'] + color_columns

                if 'interface_delta_B' in ligand_series:
                    be_column = 'interface_delta_B'
                else:
                    raise ValueError('No binding energy column (interface_delta_B) found in the data.')

                color_columns.remove(be_column)

                color_ddm = Dropdown(options=color_columns, description='Color', style={'description_width': 'initial'})
                if color_by_metric:
                    color_ddm.options = ['Color by metrics']
                    alpha_value = 0.10
                elif color_by_labels:
                    color_ddm.options = ['Color by labels']
                    alpha_value = 1.00
                else:
                    alpha_value = fixed(0.10)

                color_object = color_ddm
            else:
                color_object = fixed(color)

            interact(_bindingEnergyLandscape, color=color_object, ligand_series=fixed(ligand_series),
                     distance=fixed(distance), color_by_metric=fixed(color_by_metric), color_by_labels=fixed(color_by_labels),
                     Alpha=alpha_value, labels=fixed(labels), model=fixed(model), ligand=fixed(ligand), title=fixed(title),
                     no_xticks=fixed(no_xticks), no_yticks=fixed(no_yticks), no_cbar=fixed(no_cbar), clabel=fixed(clabel),
                     no_xlabel=fixed(no_xlabel), no_ylabel=fixed(no_ylabel), xlabel=fixed(xlabel), ylabel=fixed(ylabel),
                     relative_total_energy=fixed(relative_total_energy), clim=fixed(clim), **metrics)

        def _bindingEnergyLandscape(color, ligand_series, distance, model, ligand,
                                    color_by_metric=False, color_by_labels=False,
                                    Alpha=0.10, labels=None, title=None, no_xticks=False,
                                    no_yticks=False, no_cbar=False, no_xlabel=True, no_ylabel=False,
                                    xlabel=None, ylabel=None, clabel=None, relative_total_energy=False,
                                    clim=None, **metrics):

            skip_fp = False
            show_plot = True

            return_axis = False
            if color_by_metric:
                color = 'k'
                color_metrics = metrics
                metrics = {}
                return_axis = True
                show_plot = False

            elif color_by_labels:
                skip_fp = True
                return_axis = True
                show_plot = False

            if color == 'Total Energy' and relative_total_energy:
                relative_color_values = True
                if clim is None:
                    clim = (0, 27.631021116)
            else:
                relative_color_values = None

            if 'interface_delta_B' in ligand_series:
                be_column = 'interface_delta_B'
            else:
                raise ValueError('No binding energy column (interface_delta_B) found in the data.')

            if not skip_fp:
                axis = self.scatterPlotIndividualSimulation(model, ligand, distance, be_column, xlim=xlim, ylim=ylim,
                                                            vertical_line=vertical_line, color_column=color, clim=clim, size=size,
                                                            vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                            metrics=metrics, labels=labels, return_axis=return_axis, show=show_plot,
                                                            title=title, no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar,
                                                            no_xlabel=no_xlabel, no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel,
                                                            clabel=clabel, relative_color_values=relative_color_values, dataframe=ligand_series)

                # Set reasonable ticks
                if axis is not None:
                    if not no_xticks:
                        axis.set_xticks(axis.get_xticks()[::max(1, len(axis.get_xticks()) // 10 + 1)])
                    if not no_yticks:
                        axis.set_yticks(axis.get_yticks()[::max(1, len(axis.get_yticks()) // 10 + 1)])

            if color_by_metric:
                self.scatterPlotIndividualSimulation(model, ligand, distance, be_column, xlim=xlim, ylim=ylim,
                                                     vertical_line=vertical_line, color_column='r', clim=clim, size=size,
                                                     vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                     metrics=color_metrics, labels=labels, axis=axis, show=True, alpha=Alpha,
                                                     no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar, no_xlabel=no_xlabel,
                                                     no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel, clabel=clabel, dataframe=ligand_series)
            elif color_by_labels:
                all_labels = {l: sorted(list(set(ligand_series[l].to_list()))) for l in ligand_series.keys() if 'label_' in l}

                for l in all_labels:
                    colors = iter([plt.cm.Set2(i) for i in range(len(all_labels[l]))])
                    for i, v in enumerate(all_labels[l]):
                        if i == 0:
                            axis = self.scatterPlotIndividualSimulation(model, ligand, distance, be_column, xlim=xlim, ylim=ylim, plot_label=v,
                                                                        vertical_line=vertical_line, color_column=[next(colors)], clim=clim, size=size,
                                                                        vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                                        metrics=metrics, labels=labels, return_axis=return_axis, alpha=Alpha, show=show_plot,
                                                                        no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar, no_xlabel=no_xlabel,
                                                                        no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel, clabel=clabel, dataframe=ligand_series)
                            continue
                        elif i == len(all_labels[l]) - 1:
                            show_plot = True
                        axis = self.scatterPlotIndividualSimulation(model, ligand, distance, be_column, xlim=xlim, ylim=ylim, plot_label=v,
                                                                    vertical_line=vertical_line, color_column=[next(colors)], clim=clim, size=size,
                                                                    vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                                    metrics=metrics, labels={l: v}, return_axis=return_axis, axis=axis, alpha=Alpha, show=show_plot,
                                                                    show_legend=True, title=title, no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar,
                                                                    no_xlabel=no_xlabel, no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel, clabel=clabel,
                                                                    dataframe=ligand_series)

        models = self.rosetta_docking.index.get_level_values('Model').unique()
        models_ddm = Dropdown(options=models, description='Model', style={'description_width': 'initial'})

        interact(getLigands, model=models_ddm, dataframe=fixed(dataframe))

    def scatterPlotIndividualSimulation(self, model, ligand, x, y, vertical_line=None, color_column=None, size=1.0, labels_size=10.0, plot_label=None,
                                        xlim=None, ylim=None, metrics=None, labels=None, title=None, title_size=14.0, return_axis=False, dpi=300, show_legend=False,
                                        axis=None, xlabel=None, ylabel=None, vertical_line_color='k', vertical_line_width=0.5, marker_size=0.8, clim=None, show=False,
                                        clabel=None, legend_font_size=6, no_xticks=False, no_yticks=False, no_cbar=False, no_xlabel=False, no_ylabel=False,
                                        relative_color_values=False, dataframe=None, separator='_', **kwargs):
        """
        Creates a scatter plot for the selected model and ligand using the x and y
        columns. Data series can be filtered by specific metrics.

        Parameters
        ==========
        model : str
            The target model.
        ligand : str
            The target ligand.
        x : str
            The column name of the data to plot on the x-axis.
        y : str
            The column name of the data to plot on the y-axis.
        vertical_line : float, optional
            Position to plot a vertical line.
        color_column : str, optional
            The column name to use for coloring the plot. Also, a color can be given
            to use uniformly for the points.
        size : float, optional
            Scale factor for the plot size. Default is 1.0.
        labels_size : float, optional
            Font size for the labels. Default is 10.0.
        plot_label : str, optional
            Label for the plot. If not provided, it defaults to 'model_separator_ligand'.
        xlim : tuple, optional
            The limits for the x-axis range.
        ylim : tuple, optional
            The limits for the y-axis range.
        clim : tuple, optional
            The limits for the color range.
        metrics : dict, optional
            A set of metrics for filtering the data points.
        labels : dict, optional
            Use the label column values to filter the data.
        title : str, optional
            The title of the plot.
        title_size : float, optional
            Font size for the title. Default is 14.0.
        return_axis : bool, optional
            Whether to return the axis of this plot. Default is False.
        dpi : int, optional
            Dots per inch for the figure. Default is 300.
        show_legend : bool, optional
            Whether to show the legend. Default is False.
        axis : matplotlib.pyplot.axis, optional
            The axis to use for plotting the data. If None, a new axis is created.
        xlabel : str, optional
            Label for the x-axis. If not provided, it defaults to the x parameter.
        ylabel : str, optional
            Label for the y-axis. If not provided, it defaults to the y parameter.
        vertical_line_color : str, optional
            Color of the vertical line. Default is 'k' (black).
        vertical_line_width : float, optional
            Width of the vertical line. Default is 0.5.
        marker_size : float, optional
            Size of the markers. Default is 0.8.
        clabel : str, optional
            Label for the color bar. If not provided, it defaults to color_column.
        legend_font_size : float, optional
            Font size for the legend. Default is 6.
        no_xticks : bool, optional
            If True, x-axis ticks are not shown. Default is False.
        no_yticks : bool, optional
            If True, y-axis ticks are not shown. Default is False.
        no_cbar : bool, optional
            If True, the color bar is not shown. Default is False.
        no_xlabel : bool, optional
            If True, the x-axis label is not shown. Default is False.
        no_ylabel : bool, optional
            If True, the y-axis label is not shown. Default is False.
        relative_color_values : bool, optional
            If True, color values are shown relative to their minimum value. Default is False.
        dataframe : pandas.DataFrame, optional
            Dataframe containing the data. If not provided, self.rosetta_docking is used.
        separator : str, optional
            Separator used in the plot label. Default is '_'.
        **kwargs : additional keyword arguments
            Additional arguments to pass to the scatter function.

        Returns
        =======
        axis : matplotlib.pyplot.axis
            The axis object of the plot, if return_axis is True.

        Raises
        ======
        ValueError
            If the specified model or ligand is not found in the data.
        """

        def _addDistanceAndAngleData(ligand_series, model, ligand, dataframe):
            if model in self.rosetta_docking_distances:
                if ligand in self.rosetta_docking_distances[model]:
                    if self.rosetta_docking_distances[model][ligand] is not None:
                        for distance in self.rosetta_docking_distances[model][ligand]:
                            if dataframe is not None:
                                index_columns = ['Model', 'Ligand', 'Pose']
                                indexes = dataframe.reset_index().set_index(index_columns).index
                                ligand_series[distance] = self.rosetta_docking_distances[model][ligand][self.rosetta_docking_distances[model][ligand].index.isin(indexes)][distance].tolist()
                            else:
                                ligand_series[distance] = self.rosetta_docking_distances[model][ligand][distance].tolist()

            if model in self.rosetta_docking_angles:
                if ligand in self.rosetta_docking_angles[model]:
                    if self.rosetta_docking_angles[model][ligand] is not None:
                        for angle in self.rosetta_docking_angles[model][ligand]:
                            if dataframe is not None:
                                index_columns = ['Model', 'Ligand', 'Pose']
                                indexes = dataframe.reset_index().set_index(index_columns).index
                                ligand_series[angle] = self.rosetta_docking_angles[model][ligand][self.rosetta_docking_angles[model][ligand].index.isin(indexes)][angle].tolist()
                            else:
                                ligand_series[angle] = self.rosetta_docking_angles[model][ligand][angle].tolist()

            return ligand_series

        def _filterByMetrics(ligand_series, metrics):
            for metric, value in metrics.items():
                if isinstance(value, float):
                    mask = ligand_series[metric] <= value
                elif isinstance(value, tuple):
                    mask = (ligand_series[metric] >= value[0]) & (ligand_series[metric] <= value[1])
                ligand_series = ligand_series[mask]
            return ligand_series

        def _filterByLabels(ligand_series, labels):
            for label, value in labels.items():
                if value is not None:
                    mask = ligand_series[label] == value
                    ligand_series = ligand_series[mask]
            return ligand_series

        def _defineColorColumns(ligand_series):
            color_columns = [col for col in ligand_series.columns if ':' not in col and 'distance' not in col and 'angle' not in col and not col.startswith('metric_')]
            return color_columns

        def _plotScatter(axis, ligand_series, x, y, color_column, color_columns, plot_label, clim, marker_size, size, **kwargs):
            if color_column is not None:
                if clim is not None:
                    vmin, vmax = clim
                else:
                    vmin, vmax = None, None

                ascending = False
                colormap = 'Blues_r'

                if color_column == 'Step':
                    ascending = True
                    colormap = 'Blues'

                elif color_column in ['Epoch', 'Cluster']:
                    ascending = True
                    color_values = ligand_series.reset_index()[color_column]
                    cmap = plt.cm.jet
                    cmaplist = [cmap(i) for i in range(cmap.N)]
                    cmaplist[0] = (.5, .5, .5, 1.0)
                    max_value = max(color_values.tolist())
                    bounds = np.linspace(0, max_value + 1, max_value + 2)
                    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                    colormap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
                    color_values = color_values + 0.01
                    sc = axis.scatter(ligand_series[x], ligand_series[y], c=color_values, cmap=colormap, norm=norm, vmin=vmin, vmax=vmax, label=plot_label, s=marker_size*size, **kwargs)
                    if not no_cbar:
                        cbar = plt.colorbar(sc, ax=axis)
                        cbar.set_label(label=color_column, size=labels_size * size)
                        cbar.ax.tick_params(labelsize=labels_size * size)

                elif color_column in color_columns:
                    ligand_series = ligand_series.sort_values(color_column, ascending=ascending)
                    color_values = ligand_series[color_column]

                    if relative_color_values:
                        color_values = color_values - np.min(color_values)

                    sc = axis.scatter(ligand_series[x], ligand_series[y], c=color_values, cmap=colormap, vmin=vmin, vmax=vmax, label=plot_label, s=marker_size * size, **kwargs)
                    if not no_cbar:
                        cbar = plt.colorbar(sc, ax=axis)
                        if clabel is None:
                            clabel = color_column
                        cbar.set_label(label=clabel, size=labels_size * size)
                        cbar.ax.tick_params(labelsize=labels_size * size)
                else:
                    sc = axis.scatter(ligand_series[x], ligand_series[y], c=color_column, vmin=vmin, vmax=vmax, label=plot_label, s=marker_size * size, **kwargs)
            else:
                sc = axis.scatter(ligand_series[x], ligand_series[y], label=plot_label, s=marker_size * size, **kwargs)
            return sc

        # Extract model series from dataframe or self.rosetta_docking
        if dataframe is not None:
            model_series = dataframe[dataframe.index.get_level_values('Model') == model]
        else:
            model_series = self.rosetta_docking[self.rosetta_docking.index.get_level_values('Model') == model]

        if model_series.empty:
            raise ValueError(f'Model name {model} not found in data!')

        ligand_series = model_series[model_series.index.get_level_values('Ligand') == ligand]
        if ligand_series.empty:
            raise ValueError(f"Ligand name {ligand} not found in model's {model} data!")

        # Add distance and angle data to ligand_series
        if len(ligand_series) != 0:
            ligand_series = _addDistanceAndAngleData(ligand_series, model, ligand, dataframe)

        # Filter points by metrics
        if metrics is not None:
            ligand_series = _filterByMetrics(ligand_series, metrics)

        # Filter points by labels
        if labels is not None:
            ligand_series = _filterByLabels(ligand_series, labels)

        # Check if an axis has been given
        new_axis = False
        if axis is None:
            plt.figure(figsize=(4*size, 3.3*size), dpi=dpi)
            axis = plt.gca()
            new_axis = True

        # Define plot label
        if plot_label is None:
            plot_label = f"{model}{separator}{ligand}"

        # Define color columns
        color_columns = _defineColorColumns(ligand_series)

        # Plot scatter
        sc = _plotScatter(axis, ligand_series, x, y, color_column, color_columns, plot_label, clim, marker_size, size, **kwargs)

        # Plot vertical line if specified
        if vertical_line is not None:
            axis.axvline(vertical_line, c=vertical_line_color, lw=vertical_line_width, ls='--')

        # Set labels and title
        if xlabel is None and not no_xlabel:
            xlabel = x
        if ylabel is None and not no_ylabel:
            ylabel = y

        axis.set_xlabel(xlabel, fontsize=labels_size*size)
        axis.set_ylabel(ylabel, fontsize=labels_size*size)
        axis.tick_params(axis='both', labelsize=labels_size*size)

        # Set ticks visibility
        if no_xticks:
            axis.set_xticks([])
        if no_yticks:
            axis.set_yticks([])

        if title is not None:
            axis.set_title(title, fontsize=title_size*size)
        if xlim is not None:
            axis.set_xlim(xlim)
        if ylim is not None:
            axis.set_ylim(ylim)

        if show_legend:
            axis.legend(prop={'size': legend_font_size*size})

        if show:
            plt.show()

        if return_axis:
            return axis

    def rosettaDockingCatalyticBindingFreeEnergyMatrix(self, initial_threshold=3.5, initial_threshold_filter=3.5, measured_metrics=None,
                                                       store_values=False, lig_label_rot=90, observable='interface_delta_B',
                                                       matrix_file='catalytic_matrix.npy', models_file='catalytic_models.json',
                                                       max_metric_threshold=30, pele_data=None, KT=5.93, to_csv=None,
                                                       only_proteins=None, only_ligands=None, average_binding_energy=False,
                                                       nan_to_zero=False):

        def _bindingFreeEnergyMatrix(KT=KT, sort_by_ligand=None, models_file='catalytic_models.json',
                                     lig_label_rot=90, pele_data=None, only_proteins=None, only_ligands=None,
                                     abc=False, avg_ebc=False, n_poses=10, **metrics):

            metrics_filter = {m: metrics[m] for m in metrics if m.startswith('metric_')}
            labels_filter = {l: metrics[l] for l in metrics if l.startswith('label_')}

            if pele_data is None:
                pele_data = self.rosetta_docking

            if only_proteins is not None:
                proteins = [p for p in pele_data.index.get_level_values('Model').unique() if p in only_proteins]
            else:
                proteins = pele_data.index.get_level_values('Model').unique()

            if only_ligands is not None:
                ligands = [l for l in pele_data.index.get_level_values('Ligand').unique() if l in only_ligands]
            else:
                ligands = pele_data.index.get_level_values('Ligand').unique()

            if len(proteins) == 0:
                raise ValueError('No proteins were found!')
            if len(ligands) == 0:
                raise ValueError('No ligands were found!')

            # Create a matrix of length proteins times ligands
            M = np.zeros((len(proteins), len(ligands)))

            # Calculate the probability of each state
            for i, protein in enumerate(proteins):
                protein_series = pele_data[pele_data.index.get_level_values('Model') == protein]

                for j, ligand in enumerate(ligands):
                    ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]

                    if not ligand_series.empty:

                        if abc:
                            # Calculate partition function
                            total_energy = ligand_series['total_score']
                            energy_minimum = total_energy.min()
                            relative_energy = total_energy - energy_minimum
                            Z = np.sum(np.exp(-relative_energy / KT))

                        # Calculate catalytic binding energy
                        catalytic_series = ligand_series

                        for metric in metrics_filter:
                            if isinstance(metrics_filter[metric], float):
                                mask = catalytic_series[metric] <= metrics_filter[metric]
                            elif isinstance(metrics_filter[metric], tuple):
                                mask = (catalytic_series[metric] >= metrics_filter[metric][0]).to_numpy()
                                mask = mask & ((catalytic_series[metric] <= metrics_filter[metric][1]).to_numpy())
                            catalytic_series = catalytic_series[mask]

                        for l in labels_filter:
                            # Filter by labels
                            if labels_filter[l] is not None:
                                catalytic_series = catalytic_series[catalytic_series[l] == labels_filter[l]]

                        if abc:
                            total_energy = catalytic_series['total_score']
                            relative_energy = total_energy - energy_minimum
                            probability = np.exp(-relative_energy / KT) / Z
                            M[i][j] = np.sum(probability * catalytic_series[observable])
                        elif avg_ebc:
                            M[i][j] = catalytic_series.nsmallest(n_poses, observable)[observable].mean()
                    else:
                        M[i][j] = np.nan

            if nan_to_zero:
                M[np.isnan(M)] = 0.0

            if abc:
                binding_metric_label = '$A_{B}^{C}$'
            elif avg_ebc:
                binding_metric_label = '$\overline{E}_{B}^{C}$'
            else:
                raise ValueError('You should mark at least one option: $A_{B}^{C}$ or $\overline{E}_{B}^{C}$!')

            if store_values:
                np.save(matrix_file, M)
                if not models_file.endswith('.json'):
                    models_file = models_file + '.json'
                with open(models_file, 'w') as of:
                    json.dump(list(proteins), of)

            if to_csv is not None:
                catalytic_values = {
                    'Model': [],
                    'Ligand': [],
                    binding_metric_label: []
                }

                for i, m in zip(M, proteins):
                    for v, l in zip(i, ligands):
                        catalytic_values['Model'].append(m)
                        catalytic_values['Ligand'].append(l)
                        catalytic_values[binding_metric_label].append(v)
                catalytic_values = pd.DataFrame(catalytic_values)
                catalytic_values.set_index(['Model', 'Ligand'])
                catalytic_values.to_csv(to_csv)

            # Sort matrix by ligand or protein
            if sort_by_ligand == 'by_protein':
                protein_labels = proteins
            else:
                ligand_index = list(ligands).index(sort_by_ligand)
                sort_indexes = M[:, ligand_index].argsort()
                M = M[sort_indexes]
                protein_labels = [proteins[x] for x in sort_indexes]

            plt.figure(dpi=100, figsize=(0.28 * len(ligands), 0.2 * len(proteins)))
            plt.imshow(M, cmap='autumn')
            plt.colorbar(label=binding_metric_label)

            plt.xlabel('Ligands', fontsize=12)
            ax = plt.gca()
            ax.set_xticks(np.arange(len(ligands)))  # Set tick positions
            ax.set_xticklabels(ligands, rotation=lig_label_rot)
            plt.xticks(np.arange(len(ligands)), ligands, rotation=lig_label_rot)
            plt.ylabel('Proteins', fontsize=12)
            ax.set_yticks(np.arange(len(proteins)))  # Set tick positions
            plt.yticks(np.arange(len(proteins)), protein_labels)

            display(plt.show())

        # Check to_csv input
        if to_csv is not None and not isinstance(to_csv, str):
            raise ValueError('to_csv must be a path to the output csv file.')
        if to_csv is not None and not to_csv.endswith('.csv'):
            to_csv = to_csv + '.csv'

        # Define if PELE data is given
        if pele_data is None:
            pele_data = self.rosetta_docking

        # Add checks for the given pele data pandas df
        metrics = [k for k in pele_data.keys() if 'metric_' in k]
        labels = {}
        for m in metrics:
            for l in pele_data.keys():
                if 'label_' in l and l.replace('label_', '') == m.replace('metric_', ''):
                    labels[m] = sorted(list(set(pele_data[l])))

        metrics_sliders = {}
        labels_ddms = {}
        for m in metrics:
            if measured_metrics is not None:
                if m in measured_metrics:
                    threshold = initial_threshold
                else:
                    threshold = initial_threshold_filter
            else:
                threshold = initial_threshold_filter  # Ensure threshold is always defined

            if self.rosetta_docking_metric_type[m] == 'distance':
                m_slider = FloatSlider(
                    value=threshold,
                    min=0,
                    max=max_metric_threshold,
                    step=0.1,
                    description=m + ':',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.2f',
                    style={'description_width': 'initial'})

            elif self.rosetta_docking_metric_type[m] == 'angle':
                m_slider = FloatRangeSlider(
                    value=[110, 130],
                    min=-180,
                    max=180,
                    step=0.1,
                    description=m + ':',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.2f',
                )

            metrics_sliders[m] = m_slider

            if m in labels and labels[m] != []:
                label_options = [None] + labels[m]
                label_ddm = Dropdown(options=label_options, description=m.replace('metric_', 'label_'), style={'description_width': 'initial'})
                metrics_sliders[m.replace('metric_', 'label_')] = label_ddm

        if only_proteins is not None:
            if isinstance(only_proteins, str):
                only_proteins = [only_proteins]

        # Get only ligands if given
        if only_ligands is not None:
            if isinstance(only_ligands, str):
                only_ligands = [only_ligands]

            ligands = [l for l in self.rosetta_docking.index.get_level_values('Ligand').unique() if l in only_ligands]
        else:
            ligands = self.rosetta_docking.index.get_level_values('Ligand').unique()

        VB = []
        ligand_ddm = Dropdown(options=list(ligands) + ['by_protein'], description='Sort by ligand',
                              style={'description_width': 'initial'})
        VB.append(ligand_ddm)

        abc = Checkbox(value=True, description='$A_{B}^{C}$')
        VB.append(abc)

        if average_binding_energy:
            avg_ebc = Checkbox(value=False, description='$\overline{E}_{B}^{C}$')
            VB.append(avg_ebc)

            Ebc_slider = IntSlider(
                value=10,
                min=1,
                max=1000,
                step=1,
                description='N poses (only $\overline{E}_{B}^{C}$):',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True)
            VB.append(Ebc_slider)

        KT_slider = FloatSlider(
            value=KT,
            min=0.593,
            max=1000.0,
            step=0.1,
            description='KT:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f')

        for m in metrics_sliders:
            VB.append(metrics_sliders[m])
        for m in labels_ddms:
            VB.append(labels_ddms[m])
        VB.append(KT_slider)

        if average_binding_energy:
            plot = interactive_output(_bindingFreeEnergyMatrix, {'KT': KT_slider, 'sort_by_ligand': ligand_ddm,
                                      'pele_data': fixed(pele_data), 'models_file': fixed(models_file),
                                      'lig_label_rot': fixed(lig_label_rot), 'only_proteins': fixed(only_proteins),
                                      'only_ligands': fixed(only_ligands), 'abc': abc, 'avg_ebc': avg_ebc,
                                      'n_poses': Ebc_slider, **metrics_sliders})
        else:
            plot = interactive_output(_bindingFreeEnergyMatrix, {'KT': KT_slider, 'sort_by_ligand': ligand_ddm,
                                      'pele_data': fixed(pele_data), 'models_file': fixed(models_file),
                                      'lig_label_rot': fixed(lig_label_rot), 'only_proteins': fixed(only_proteins),
                                      'only_ligands': fixed(only_ligands), 'abc': abc, **metrics_sliders})

        VB.append(plot)
        VB = VBox(VB)

        display(VB)

    def getBestRosettaDockingPoses(
            self,
            filter_values,
            return_failed=False,
            exclude_models=None,
            exclude_ligands=None,
            exclude_pairs=None,
            score_column='interface_delta_B'
        ):
        """
        Get best docking poses based on the best SCORE and a set of metrics with specified thresholds.
        The filter thresholds must be provided with a dictionary using the metric names as keys
        and the thresholds as the values.

        Parameters
        ==========
        filter_values : dict
            Thresholds for the filter.
        return_failed : bool
            Whether to return a list of the dockings without any models fulfilling
            the selection criteria. It is returned as a tuple (index 0) alongside
            the filtered data frame (index 1).
        exclude_models : list, optional
            List of models to be excluded from the selection.
        exclude_ligands : list, optional
            List of ligands to be excluded from the selection.
        exclude_pairs : list, optional
            List of pair tuples (model, ligand) to be excluded from the selection.
        score_column : str, optional
            Column name to use for scoring. Default is 'interface_delta_B'.

        Returns
        =======
        pandas.DataFrame
            Dataframe containing the best poses based on the given criteria.
        """

        if exclude_models is None:
            exclude_models = []
        if exclude_ligands is None:
            exclude_ligands = []
        if exclude_pairs is None:
            exclude_pairs = []

        best_poses = []
        failed = []

        for model in self.rosetta_docking.index.get_level_values('Model').unique():

            if model in exclude_models:
                continue

            model_series = self.rosetta_docking[
                self.rosetta_docking.index.get_level_values("Model") == model
            ]

            for ligand in model_series.index.get_level_values('Ligand').unique():

                if ligand in exclude_ligands:
                    continue

                if (model, ligand) in exclude_pairs:
                    continue

                ligand_data = model_series[
                    model_series.index.get_level_values("Ligand") == ligand
                ]

                for metric, threshold in filter_values.items():

                    if metric not in [score_column, "RMSD"]:
                        if not metric.startswith("metric_"):
                            metric_label = "metric_" + metric
                        else:
                            metric_label = metric

                        if isinstance(threshold, (float, int)):
                            ligand_data = ligand_data[ligand_data[metric_label] <= threshold]
                        elif isinstance(threshold, (tuple, list)):
                            ligand_data = ligand_data[
                                (ligand_data[metric_label] >= threshold[0]) &
                                (ligand_data[metric_label] <= threshold[1])
                            ]
                    else:
                        metric_label = metric
                        ligand_data = ligand_data[ligand_data[metric_label] < threshold]

                if ligand_data.empty:
                    failed.append((model, ligand))
                    continue

                best_pose_idx = ligand_data[score_column].idxmin()
                best_poses.append(best_pose_idx)

        best_poses_df = self.rosetta_docking.loc[best_poses]

        if return_failed:
            return failed, best_poses_df

        return best_poses_df

    def extractRosettaDockingModels(self, docking_folder, input_df, output_folder, separator='_'):
        """
        Extract models based on an input DataFrame with index ['Model', 'Ligand', 'Pose'].

        Parameters
        ==========
        docking_folder : str
            Path to folder where the Rosetta docking files are contained.
        input_df : pd.DataFrame
            DataFrame containing the models to be extracted with index ['Model', 'Ligand', 'Pose'].
        separator : str
            Separator character used in file names. Default is '-'.

        Returns
        =======
        list
            List of models extracted.
        """

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        executable = "extract_pdbs.linuxgccrelease"
        models = []
        missing_models = []

        # Check if params were given
        params = {}
        for ligand in self.rosetta_docking.index.levels[1]:
            ligand_folder = os.path.join(docking_folder, "ligand_params", ligand)
            if os.path.exists(ligand_folder):
                params[ligand] = os.path.join(ligand_folder, ligand+'.params')

        for index, row in input_df.iterrows():
            model, ligand, pose = index

            output_model_dir = os.path.join(docking_folder, "output_models", f"{model}{separator}{ligand}")
            if not os.path.exists(output_model_dir):
                missing_models.append(model)
                continue

            silent_file = os.path.join(output_model_dir, f"{model}{separator}{ligand}.out")
            if not os.path.exists(silent_file):
                missing_models.append(model)
                continue

            best_model_tag =  row['description']

            command = f"{executable} -silent {silent_file} -tags {best_model_tag}"
            if params is not None:
                command += f" -extra_res_fa {params[ligand]} "
            os.system(command)

            pdb_filename = f"{best_model_tag}.pdb"

            shutil.move(pdb_filename, output_folder+'/'+pdb_filename)

        self.getModelsSequences()

        if missing_models:
            print("Missing models in Rosetta Docking folder:")
            print("\t" + ", ".join(missing_models))

        return models

    def convertLigandPDBtoMae(self, ligands_folder, change_ligand_name=True, keep_pdbs=False):
        """
        Convert ligand PDBs into MAE files.

        Parameters
        ==========
        ligands_folder : str
            Path to the folder where ligands are in PDB format
        """

        _copyScriptFile(ligands_folder, "PDBtoMAE.py")
        script_name = "._PDBtoMAE.py"

        cwd = os.getcwd()
        os.chdir(ligands_folder)
        command = "run ._PDBtoMAE.py"
        if change_ligand_name:
            command += " --change_ligand_name"
        if keep_pdbs:
            command += ' --keep_pdbs'
        os.system(command)
        os.chdir(cwd)

    def convertLigandMAEtoPDB(self, ligands_folder, change_ligand_name=True, modify_maes=False,
                              assign_pdb_names=False):
        """
        Convert ligand MAEs into PDB files.

        Parameters
        ==========
        ligands_folder : str
            Path to the folder where ligands are in MAE format
        """

        if modify_maes:
            keep_maes = True

        if isinstance(change_ligand_name, dict):
            with open(ligands_folder+'/ligand_names.json', 'w') as jf:
                json.dump(change_ligand_name, jf)
        elif isinstance(change_ligand_name, str):
            if len(change_ligand_name) != 3:
                raise ValueError('The ligand name should be three-letters long')

        # Copy analyse docking script (it depends on Schrodinger Python API so we leave it out to minimise dependencies)
        _copyScriptFile(ligands_folder, "MAEtoPDB.py")
        script_name = "._MAEtoPDB.py"

        cwd = os.getcwd()
        os.chdir(ligands_folder)
        command = "run ._MAEtoPDB.py"
        if isinstance(change_ligand_name, dict):
            command += " --residue_names ligand_names.json"
        elif isinstance(change_ligand_name, str):
            command += " --residue_names "+change_ligand_name
        if change_ligand_name:
            command += " --change_ligand_name"
        if modify_maes:
            command += ' --modify_maes'
        if assign_pdb_names:
            command += ' --assign_pdb_names'
        os.system(command)
        os.chdir(cwd)

        if isinstance(change_ligand_name, dict):
            os.remove(ligands_folder+'/ligand_names.json')

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
                        "Atom name %s was not found in residue %s of chain %s for model %s"
                        % (atom1[2], atom1[1], atom1[0], model)
                    )
                if atom2[2] not in coordinates[atom2[0]][atom2[1]]:
                    raise ValueError(
                        "Atom name %s was not found in residue %s of chain %s for model %s"
                        % (atom2[2], atom2[1], atom2[0], model)
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

        if isinstance(self.models_data, dict):
            self.models_data = pd.DataFrame(self.models_data)
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
        self.protonation_states["Model"] = []
        self.protonation_states["Chain"] = []
        self.protonation_states["Residue"] = []
        self.protonation_states["Name"] = []
        self.protonation_states["State"] = []

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

                    # Check for hydrogens
                    hydrogens = [a for a in atoms if a.startswith('H')]

                    if not hydrogens:
                        print(f'The model {model} have not been protonated.')
                        continue

                    if "HE2" in atoms and "HD1" in atoms:
                        self.protonation_states["State"].append("HIP")
                    elif "HD1" in atoms:
                        self.protonation_states["State"].append("HID")
                    elif "HE2" in atoms:
                        self.protonation_states["State"].append("HIE")
                    else:
                        print(f'HIS {r.id[1]} could not be assigned for model {model}')
                        continue
                    self.protonation_states["Model"].append(model)
                    self.protonation_states["Chain"].append(r.get_parent().id)
                    self.protonation_states["Residue"].append(r.id[1])
                    self.protonation_states["Name"].append(r.resname)

        if self.protonation_states['Model'] == []:
            raise ValueError("No protonation states were found. Did you run prepwizard?")

        # Convert dictionary to Pandas
        self.protonation_states = pd.DataFrame(self.protonation_states)
        self.protonation_states.set_index(["Model", "Chain", "Residue"], inplace=True)

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
                # Initialize a Series with NaN values, indexed the same as self.docking_data
                metric_series = pd.Series(np.nan, index=self.docking_data.index)

                for model in self.docking_data.index.get_level_values("Protein").unique():
                    # Check whether model is found in docking distances
                    if model not in self.docking_distances:
                        continue

                    model_series = self.docking_data.xs(model, level="Protein")

                    for ligand in model_series.index.get_level_values("Ligand").unique():
                        # Check whether ligand is found in model's docking distances
                        if ligand not in self.docking_distances[model]:
                            continue

                        ligand_series = model_series.xs(ligand, level="Ligand")

                        # Check input metric
                        distance_metric = False
                        angle_metric = False
                        for x in catalytic_labels[name][model][ligand]:
                            if len(x.split("-")) == 2:
                                distance_metric = True
                            elif len(x.split("-")) == 3:
                                angle_metric = True

                        if distance_metric and angle_metric:
                            raise ValueError(
                                f"Metric {name} combines distances and angles which is not supported."
                            )

                        if distance_metric:
                            distances = catalytic_labels[name][model][ligand]
                            distance_values = self.docking_distances[model][ligand][distances].min(axis=1)
                            # Align the indices
                            indices = ligand_series.index
                            metric_series.loc[(model, ligand, indices)] = distance_values.values
                            self.docking_metric_type[name] = "distance"
                        elif angle_metric:
                            angles = catalytic_labels[name][model][ligand]
                            if len(angles) > 1:
                                raise ValueError(
                                    "Combining more than one angle into a metric is not currently supported."
                                )
                            angle_values = self.docking_angles[model][ligand][angles].min(axis=1)
                            indices = ligand_series.index
                            metric_series.loc[(model, ligand, indices)] = angle_values.values
                            self.docking_metric_type[name] = "angle"

                # Assign the Series to the DataFrame
                self.docking_data["metric_" + name] = metric_series

    def combineDockingMetricsWithExclusions(self, combinations, exclusions, drop=True):
        """
        Combine mutually exclusive metrics into new metrics while handling exclusions.

        Parameters
        ----------
        combinations : dict
            Dictionary defining which metrics to combine under a new common name.
            Structure:
                combinations = {
                    new_metric_name: (metric1, metric2, ...),
                    ...
                }

        exclusions : list of tuples or dict
            List of tuples (for simple exclusions) or dictionary by metrics for by-metric exclusions.

        drop : bool, optional
            If True, drop the original metric columns after combining. Default is True.

        """

        # Determine exclusion type
        simple_exclusions = False
        by_metric_exclusions = False
        if isinstance(exclusions, list):
            simple_exclusions = True
        elif isinstance(exclusions, dict):
            by_metric_exclusions = True
        else:
            raise ValueError('exclusions should be a list of tuples or a dictionary by metrics.')

        # Collect all unique metrics from combinations
        unique_metrics = set()
        for new_metric, metrics in combinations.items():
            metric_types = [self.docking_metric_type[m] for m in metrics]
            if len(set(metric_types)) != 1:
                raise ValueError('Attempting to combine different metric types (e.g., distances and angles) is not allowed.')
            self.docking_metric_type[new_metric] = metric_types[0]
            unique_metrics.update(metrics)

        # Build a mapping from metric names to column indices
        metrics_list = list(unique_metrics)
        metrics_indexes = {m: idx for idx, m in enumerate(metrics_list)}

        # Add metric prefix if not given
        add_metric_prefix = True
        for m in metrics_list:
            if 'metric_' in m:
                raise ValueError('"metric_" prefix found in given metrics. Please, leave it out.')
        all_metrics_columns = ['metric_' + m for m in metrics_list]

        # Ensure all required metric columns exist in the data
        missing_columns = set(all_metrics_columns) - set(self.docking_data.columns)
        if missing_columns:
            raise ValueError(f"Missing metric columns in data: {missing_columns}")

        # Extract metric data
        data = self.docking_data[all_metrics_columns]

        # Positions of values to be excluded (row index, column index)
        excluded_positions = set()

        # Get labels of the shortest distance for each row
        min_metric_labels = data.idxmin(axis=1)  # Series of column names

        if simple_exclusions:
            for row_idx, metric_col_label in enumerate(min_metric_labels):
                m = metric_col_label.replace('metric_', '')

                # Exclude metrics specified in exclusions
                for exclusion_group in exclusions:
                    if m in exclusion_group:
                        others = set(exclusion_group) - {m}
                        for x in others:
                            if x in metrics_indexes:
                                col_idx = metrics_indexes[x]
                                excluded_positions.add((row_idx, col_idx))

                # Exclude other metrics in the same combination group
                for metrics_group in combinations.values():
                    if m in metrics_group:
                        others = set(metrics_group) - {m}
                        for y in others:
                            if y in metrics_indexes:
                                col_idx = metrics_indexes[y]
                                excluded_positions.add((row_idx, col_idx))

        if by_metric_exclusions:
            # Convert data to a NumPy array for efficient processing
            data_array = data.to_numpy()

            # Iterate over each row to handle exclusions iteratively
            for row_idx in range(data_array.shape[0]):

                considered_metrics = set()  # Track metrics already considered as minimums in this row

                while True:
                    # Find the minimum among metrics that haven't been excluded or considered as minimums
                    min_value = np.inf
                    min_col_idx = -1

                    # Identify the next lowest metric that hasn't been excluded or already considered
                    for col_idx, metric_value in enumerate(data_array[row_idx]):
                        if col_idx not in considered_metrics and (row_idx, col_idx) not in excluded_positions:
                            if metric_value < min_value:
                                min_value = metric_value
                                min_col_idx = col_idx
                    # if row_idx == 3:
                        # print(min_value, min_col_idx, data.columns[min_col_idx])

                    # Break the loop if no valid minimum metric is found
                    if min_col_idx == -1:
                        break

                    # Mark this metric as considered so it's not reused as minimum in future iterations
                    considered_metrics.add(min_col_idx)

                    # Get the name of the metric and retrieve exclusions based on this metric
                    min_metric_label = data.columns[min_col_idx]
                    min_metric_name = min_metric_label.replace('metric_', '')
                    excluded_metrics = exclusions.get(min_metric_name, [])

                    # Apply exclusions for this metric
                    for excluded_metric in excluded_metrics:
                        if excluded_metric in metrics_indexes:
                            excluded_col_idx = metrics_indexes[excluded_metric]
                            if (row_idx, excluded_col_idx) not in excluded_positions:
                                excluded_positions.add((row_idx, excluded_col_idx))
                                data_array[row_idx, excluded_col_idx] = np.inf  # Set excluded metric to infinity
                # if row_idx == 3:
                #     print()
                #     for x, m in zip(data_array[row_idx], metrics_indexes.items()):
                #         print(x, m)

        # Combine metrics and add new columns to the DataFrame
        for new_metric_name, metrics_to_combine in combinations.items():
            c_indexes = [metrics_indexes[m] for m in metrics_to_combine if m in metrics_indexes]

            if c_indexes:
                # Calculate the minimum value among the combined metrics, excluding inf-only combinations
                combined_min = np.min(data_array[:, c_indexes], axis=1)

                # Check if combined_min is all inf and handle accordingly
                if np.all(np.isinf(combined_min)):
                    print(f"Skipping combination for '{new_metric_name}' due to incompatible exclusions.")
                    continue
                self.docking_data['metric_' + new_metric_name] = combined_min
            else:
                raise ValueError(f"No valid metrics to combine for '{new_metric_name}'.")

        # Drop original metric columns if specified
        if drop:
            self.docking_data.drop(columns=all_metrics_columns, inplace=True)

        # Ensure compatibility of combinations with exclusions
        for new_metric_name, metrics_to_combine in combinations.items():
            non_excluded_found = False

            for metric in metrics_to_combine:
                # Use standardized names for consistent indexing
                metric_column_name = 'metric_' + metric if 'metric_' not in metric else metric
                col_idx = metrics_indexes.get(metric_column_name)

                if col_idx is not None:
                    # Check directly in data_array for non-excluded values
                    column_values = data_array[:, col_idx]
                    if not np.all(np.isinf(column_values)):
                        non_excluded_found = True
                        break

            # Print warning if all values for a combination are excluded
            if not non_excluded_found:
                print(f"Warning: No non-excluded metrics available to combine for '{new_metric_name}'.")

    def plotDockingData(self):
        """
        Generates an interactive scatter plot for docking data, allowing users to select
        the protein, ligand, and columns for the X and Y axes.

        The method assumes the docking data is a Pandas DataFrame stored in `self.docking_data`
        with a MultiIndex (Protein, Ligand, Pose) and numeric columns (Score, RMSD, Closest distance).

        The function creates interactive widgets to select a specific protein, ligand, and which
        numeric columns to plot on the X and Y axes.

        Returns:
            An interactive scatter plot that updates based on widget selections.
        """

        # Subfunction to handle filtering and plotting
        def scatter_plot(protein, ligand, x_axis, y_axis):
            """
            Subfunction to plot the scatter plot for the selected protein and ligand.

            Args:
                protein (str): Selected protein sequence.
                ligand (str): Selected ligand.
                x_axis (str): The column name for the X-axis.
                y_axis (str): The column name for the Y-axis.
            """
            # Filter the data based on selected Protein and Ligand
            filtered_df = df.loc[(protein, ligand)]

            # Plotting the scatter plot for the selected X and Y axes
            plt.figure(figsize=(8, 6))
            plt.scatter(filtered_df[x_axis], filtered_df[y_axis], color='blue')
            plt.title(f'Scatter Plot for {protein} - {ligand}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.grid(True)
            plt.show()

        # Get the docking data from the object's attribute
        df = self.docking_data

        # Create dropdown widgets for selecting the Protein, Ligand, X-axis, and Y-axis columns
        protein_dropdown = widgets.Dropdown(
            options=df.index.levels[0],  # Options for Protein selection
            description='Protein'        # Label for the dropdown
        )

        ligand_dropdown = widgets.Dropdown(
            options=df.index.levels[1],  # Options for Ligand selection
            description='Ligand'         # Label for the dropdown
        )

        x_axis_dropdown = widgets.Dropdown(
            options=df.columns,          # Options for selecting the X-axis (numeric columns)
            description='X-axis'         # Label for the dropdown
        )

        y_axis_dropdown = widgets.Dropdown(
            options=df.columns,          # Options for selecting the Y-axis (numeric columns)
            description='Y-axis'         # Label for the dropdown
        )

        # Create an interactive widget that dynamically updates the plot based on selections
        interact(
            scatter_plot,
            protein=protein_dropdown,
            ligand=ligand_dropdown,
            x_axis=x_axis_dropdown,
            y_axis=y_axis_dropdown
        )

    def getBestDockingPoses(
        self,
        filter_values,
        n_models=1,
        return_failed=False,
        exclude_models=None,
        exclude_ligands=None,
        exclude_pairs=None,
    ):
        if exclude_models is None:
            exclude_models = []
        if exclude_ligands is None:
            exclude_ligands = []
        if exclude_pairs is None:
            exclude_pairs = []

        if not isinstance(n_models, int):
            n_models = int(n_models)

        # Create exclusion masks
        docking_data = self.docking_data
        index = docking_data.index

        exclude_models_mask = ~index.get_level_values('Protein').isin(exclude_models)
        exclude_ligands_mask = ~index.get_level_values('Ligand').isin(exclude_ligands)

        pairs_to_exclude = set(exclude_pairs)
        if pairs_to_exclude:
            exclude_pairs_mask = ~index.map(lambda idx: (idx[0], idx[1]) in pairs_to_exclude)
        else:
            exclude_pairs_mask = np.ones(len(index), dtype=bool)  # Include all

        mask = exclude_models_mask & exclude_ligands_mask & exclude_pairs_mask

        filtered_data = docking_data[mask]

        # Apply filters
        for metric in filter_values:
            filter_value = filter_values[metric]
            if metric not in ["Score", "RMSD"]:
                if not metric.startswith("metric_") and metric != 'Closest distance':
                    metric_label = "metric_" + metric
                else:
                    metric_label = metric
            else:
                metric_label = metric

            if isinstance(filter_value, (float, int)):
                filtered_data = filtered_data[filtered_data[metric_label] <= filter_value]
            elif isinstance(filter_value, (tuple, list)):
                filtered_data = filtered_data[
                    (filtered_data[metric_label] >= filter_value[0]) &
                    (filtered_data[metric_label] <= filter_value[1])
                ]
            else:
                filtered_data = filtered_data[filtered_data[metric_label] < filter_value]

        # Ensure index levels are named
        if filtered_data.index.nlevels == 3:
            filtered_data.index.set_names(['Protein', 'Ligand', 'Pose'], inplace=True)
        else:
            # If index levels are not named, we can set default names
            filtered_data.index.set_names(['Protein', 'Ligand'], inplace=True)

        # Get all available pairs after exclusions
        available_pairs = docking_data[mask].index.to_frame(index=False)[['Protein', 'Ligand']].drop_duplicates()

        # Get pairs present in filtered_data
        filtered_pairs = filtered_data.index.to_frame(index=False)[['Protein', 'Ligand']].drop_duplicates()

        # Find failed pairs
        failed_pairs = pd.merge(
            available_pairs,
            filtered_pairs,
            on=['Protein', 'Ligand'],
            how='left',
            indicator=True
        )
        failed_pairs = failed_pairs[failed_pairs['_merge'] == 'left_only'][['Protein', 'Ligand']]
        failed = list(failed_pairs.itertuples(index=False, name=None))

        # Sort and group
        filtered_data = filtered_data.sort_values(by=['Protein', 'Ligand', 'Score'])

        # Use level indices if names are not consistent
        if filtered_data.index.nlevels >= 2:
            grouped = filtered_data.groupby(level=[0, 1], sort=False)
        else:
            grouped = filtered_data.groupby(['Protein', 'Ligand'], sort=False)

        # Select top n_models per group
        top_n = grouped.head(n_models)

        # Warning for groups with less than n_models
        group_sizes = grouped.size()
        # print("Group Sizes:")
        # print(group_sizes)
        # print("Data Types of Group Sizes:")
        # print(group_sizes.dtypes)
        if not group_sizes.empty:
            insufficient_groups = group_sizes[group_sizes < n_models]
            if not insufficient_groups.empty:
                for (protein, ligand), size in insufficient_groups.iteritems():
                    print(
                        "WARNING: less than %s models available for docking %s + %s"
                        % (n_models, protein, ligand)
                    )
        else:
            insufficient_groups = pd.Series(dtype=int)

        if return_failed:
            return failed, top_n
        else:
            return top_n

    def getBestDockingPosesIteratively(
        self,
        metrics,
        ligands=None,
        distance_step=0.1,
        angle_step=1.0,
        fixed=None,
        max_distance=None,
        max_distance_step_shift=None,
        verbose=False,
    ):
        """
        Iteratively select the best docking poses for protein-ligand pairs based on given metric thresholds.
        If not all protein-ligand pairs have acceptable models under the initial thresholds, the function
        iteratively relaxes the thresholds of the non-fixed metrics, starting with the ones that accept the
        fewest models, until at least one model is selected for each protein-ligand pair or until
        max_distance is reached.

        Parameters:
        - metrics (dict): Dictionary of metric thresholds. Keys are metric names, values are thresholds.
                          Thresholds can be a scalar (upper limit) or a tuple/list (lower and upper limits).
        - ligands (list, optional): List of ligands to consider. If None, all ligands are considered.
        - distance_step (float, optional): Step size to adjust distance metrics.
        - angle_step (float, optional): Step size to adjust angle metrics.
        - fixed (list, optional): List of metric names that should not be adjusted.
        - max_distance (float, optional): Maximum allowed value for distance metrics.
        - max_distance_step_shift (float, optional): New step size for distance metrics after reaching max_distance.

        Returns:
        - pandas.DataFrame: DataFrame containing the selected docking poses.
        """

        # Ensure fixed is a list
        if fixed is None:
            fixed = []
        elif isinstance(fixed, str):
            fixed = [fixed]

        # Ensure there is at least one non-fixed metric
        non_fixed_metrics = set(metrics.keys()) - set(fixed)
        if not non_fixed_metrics:
            raise ValueError("You must leave at least one metric not fixed")

        metrics = metrics.copy()

        # Filter data by ligands if provided
        if ligands is not None:
            # Assuming that the ligand identifier is at index level 1
            data = self.docking_data[self.docking_data.index.get_level_values(1).isin(ligands)]
        else:
            data = self.docking_data

        # Get all unique protein-ligand pairs
        protein_ligand_pairs = set(zip(
            data.index.get_level_values(0),  # Assuming protein identifier is at index level 0
            data.index.get_level_values(1)   # Ligand identifier at index level 1
        ))

        extracted_pairs = set()
        selected_indexes = []
        current_distance_step = distance_step
        step_shift_applied = False  # Flag to indicate if step shift has been applied

        while len(extracted_pairs) < len(protein_ligand_pairs):
            if verbose:
                ti = time.time()

            # Get best poses with current thresholds
            best_poses = self.getBestDockingPoses(metrics, n_models=1)  # Assuming self has this method

            # Select new models
            new_selected_pairs = set()
            for idx in best_poses.index:
                pair = (idx[0], idx[1])  # Adjust index levels if needed
                if pair not in extracted_pairs:
                    selected_indexes.append(idx)
                    new_selected_pairs.add(pair)

            extracted_pairs.update(new_selected_pairs)

            # If we've selected models for all pairs, break the loop
            if len(extracted_pairs) >= len(protein_ligand_pairs):
                break

            # Prepare remaining data
            remaining_pairs = protein_ligand_pairs - extracted_pairs
            mask = [((idx[0], idx[1]) in remaining_pairs) for idx in data.index]
            remaining_data = data[mask]

            if remaining_data.empty:
                break  # No more data to process

            # Compute acceptance counts for each metric
            metric_acceptance = {}
            for metric in metrics:
                if metric in fixed:
                    continue
                metric_label = metric if metric.startswith('metric_') else 'metric_' + metric
                metric_type = self.docking_metric_type.get(metric_label.replace('metric_', ''), None)
                if metric_type is None:
                    raise ValueError(f"Metric type for {metric_label} not defined.")

                metric_values = remaining_data[metric_label]

                if isinstance(metrics[metric], (int, float)):
                    if metric_type in ['distance', 'angle']:
                        acceptance = metric_values <= metrics[metric]
                    else:
                        acceptance = metric_values >= metrics[metric]
                elif isinstance(metrics[metric], (tuple, list)):
                    lower, upper = metrics[metric]
                    acceptance = (metric_values >= lower) & (metric_values <= upper)
                else:
                    raise ValueError(f"Invalid threshold type for metric {metric}")

                metric_acceptance[metric] = acceptance.sum()

            # Order metrics by acceptance count (ascending)
            ordered_metrics = sorted(
                [(m, a) for m, a in metric_acceptance.items() if m not in fixed],
                key=lambda x: x[1]
            )

            # Adjust thresholds for the metric with lowest acceptance
            updated = False
            for metric, _ in ordered_metrics:
                metric_label = metric if metric.startswith('metric_') else 'metric_' + metric
                metric_type = self.docking_metric_type.get(metric_label.replace('metric_', ''), None)
                if metric_type == 'distance':
                    step = current_distance_step
                elif metric_type == 'angle':
                    step = angle_step
                else:
                    raise ValueError(f"Unknown metric type for {metric_label}")

                if isinstance(metrics[metric], (int, float)):
                    # For upper limit thresholds (assuming distance and angle are upper limits)
                    new_value = metrics[metric] + step

                    if metric_type == 'distance' and max_distance is not None:
                        if not step_shift_applied and new_value >= max_distance:
                            if max_distance_step_shift is not None:
                                # Apply step shift
                                current_distance_step = max_distance_step_shift
                                step_shift_applied = True
                                print(f"Max distance {max_distance} reached for metric {metric}. Applying step shift to {current_distance_step}.")
                                # Do not cap the value; allow it to exceed max_distance
                            else:
                                # If no step shift, cap at max_distance and terminate
                                new_value = max_distance
                                metrics[metric] = new_value
                                print(f"Max distance {max_distance} reached for metric {metric}. Terminating iteration.")
                                updated = True
                                break  # Exit the for-loop to terminate the while-loop
                    # Update the metric
                    metrics[metric] = new_value
                    updated = True
                    break  # Adjusted one metric, exit the loop

                elif isinstance(metrics[metric], (tuple, list)):
                    # For range thresholds
                    lower, upper = metrics[metric]
                    new_lower = lower - step
                    new_upper = upper + step
                    metrics[metric] = (new_lower, new_upper)
                    updated = True
                    break  # Adjusted one metric, exit the loop
                else:
                    raise ValueError(f"Invalid threshold type for metric {metric}")

            # Check if step shift was applied and allow further adjustments
            if not updated:
                # Could not adjust any metric, exit the loop
                print("No metrics were updated. Terminating iteration.")
                break

            # If step shift was applied and already applied before, continue adjusting with new step size
            # No additional action needed as current_distance_step has been updated

            # Optional: Print progress for debugging
            if verbose:
                elapsed_time = time.time() - ti
                print(f"Max distance reached: {step_shift_applied}, Current step: {current_distance_step}, Metrics: {metrics}, Time elapsed: {elapsed_time:.2f}s", end='\r')

        # Collect selected models
        if selected_indexes:
            best_poses = data.loc[selected_indexes]
        else:
            best_poses = pd.DataFrame()  # Return empty DataFrame if no poses selected

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
                    if len(extracted_models):
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
        only_hetatoms_conect=False,
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

                                    if models and model not in models:
                                        continue

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
                            only_hetatoms=only_hetatoms_conect
                        )
                        load_count += 1

        self.getModelsSequences()

        # Gather missing models
        # Remove
        if models:
            missing_models = set(models) - set(all_models)
        else:
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
        skip_finished=False,
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

                if not os.path.exists(f'{rosetta_folder}/output_models/{m}/{m}_relax.out'):
                    print(f'Silent file for model {m} was not found!')
                    continue

                if skip_finished and os.path.exists(f'{rosetta_folder}/.analysis/scores/{m}.csv'):
                    continue

                commands.append(command.replace("MODEL", m))

            print("Returning jobs for running the analysis in parallel.")
            print(
                "After jobs have finished, rerun this function removing return_jobs=True!"
            )
            return commands

        else:
            count = 0
            for m in self:
                if not os.path.exists(f'{rosetta_folder}/output_models/{m}/{m}_relax.out'):
                    print(f'Silent file for model {m} was not found!')
                    continue
                if not os.path.exists(f'{rosetta_folder}/.analysis/scores/{m}.csv'):
                    count += 1

            if count:
                required = {}
                installed = {pkg.key for pkg in pkg_resources.working_set}
                if 'pyrosetta' not in installed:
                    raise ValueError('PyRosetta was not found in your Python environment.\
                    Consider using return_jobs=True or activating an environment the does have it.')
                else:
                    os.system(command)

        # Compile dataframes into rosetta_data attributes
        self.rosetta_data = []
        self.rosetta_distances = {}
        self.rosetta_ebr = []
        self.rosetta_neighbours = []
        self.rosetta_protonation = []
        binding_energy_df = []

        output_folder = '.analysis'
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

        return self.rosetta_data

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
                    if isinstance(model, int):
                        model = str(model)
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
        cst_files=None,
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

    # def loadModelsFromRosettaOptimization(
    #     self,
    #     optimization_folder,
    #     filter_score_term="score",
    #     min_value=True,
    #     tags=None,
    #     wat_to_hoh=True,
    #     return_missing=False,
    #     sugars=False,
    #     conect_update=False,
    # ):
    #     """
    #     Load the best energy models from a set of silent files inside a specfic folder.
    #     Useful to get the best models from a relaxation run.
    #
    #     Parameters
    #     ==========
    #     optimization_folder : str
    #         Path to folder where the Rosetta optimization files are contained
    #     filter_score_term : str
    #         Score term used to filter models
    #     relax_run : bool
    #         Is this a relax run?
    #     min_value : bool
    #         Grab the minimum score value. Set false to grab the maximum scored value.
    #     tags : dict
    #         The tag of a specific pose to be loaded for the given model. Each model
    #         must have a single tag in the tags dictionary. If a model is not found
    #         in the tags dictionary, normal processing will follow to select
    #         the loaded pose.
    #     wat_to_hoh : bool
    #         Change water names from WAT to HOH when loading.
    #     return_missing : bool
    #         Return missing models from the optimization_folder.
    #     """
    #
    #     def getConectLines(pdb_file, format_for_prepwizard=True):
    #
    #         ace_names = ['CO', 'OP1', 'CP2', '1HP2', '2HP2', '3HP2']
    #
    #         # Read PDB file
    #         atom_tuples = {}
    #         add_one = False
    #         previous_chain = None
    #         with open(pdb_file, "r") as f:
    #             for l in f:
    #                 if l.startswith("ATOM") or l.startswith("HETATM"):
    #                     index, name, resname, chain, resid = (
    #                         int(l[6:11]),        # Atom index
    #                         l[12:16].strip(),    # Atom name
    #                         l[17:20].strip(),    # Residue name
    #                         l[21],               # Chain identifier
    #                         int(l[22:26]),       # Residue index
    #                     )
    #
    #                     if not previous_chain:
    #                         previous_chain = chain
    #
    #                     if name in ace_names:
    #                         resid -= 1
    #
    #                         if format_for_prepwizard:
    #                             if name == 'CP2':
    #                                 name = 'CH3'
    #                             elif name == 'CO':
    #                                 name = 'C'
    #                             elif name == 'OP1':
    #                                 name = 'O'
    #                             elif name == '1HP2':
    #                                 name = '1H'
    #                             elif name == '2HP2':
    #                                 name = '2H'
    #                             elif name == '3HP2':
    #                                 name = '3H'
    #
    #                     if resname == 'NMA':
    #                         add_one = True
    #
    #                         if format_for_prepwizard:
    #                             if name == 'HN2':
    #                                 name = 'H'
    #                             elif name == 'C':
    #                                 name = 'CA'
    #                             elif name == 'H1':
    #                                 name = '1HA'
    #                             elif name == 'H2':
    #                                 name = '2HA'
    #                             elif name == 'H3':
    #                                 name = '3HA'
    #
    #                     if previous_chain != chain:
    #                         add_one = False
    #
    #                     if add_one:
    #                         resid += 1
    #
    #                     atom_tuples[index] = (chain, resid, name)
    #                     previous_chain = chain
    #
    #         conects = []
    #         with open(pdb_file) as pdbf:
    #             for l in pdbf:
    #                 if l.startswith("CONECT"):
    #                     l = l.replace("CONECT", "")
    #                     l = l.strip("\n").rstrip()
    #                     num = len(l) / 5
    #                     new_l = [int(l[i * 5 : (i * 5) + 5]) for i in range(int(num))]
    #                     conects.append([atom_tuples[int(x)] for x in new_l])
    #
    #         return conects
    #
    #     def writeConectLines(conects, pdb_file):
    #
    #         atom_indexes = {}
    #         with open(pdb_file, "r") as f:
    #             for l in f:
    #                 if l.startswith("ATOM") or l.startswith("HETATM"):
    #                     index, name, resname, chain, resid = (
    #                         int(l[6:11]),        # Atom index
    #                         l[12:16].strip(),    # Atom name
    #                         l[17:20].strip(),    # Residue name
    #                         l[21],               # Chain identifier
    #                         int(l[22:26]),       # Residue index
    #                     )
    #                     atom_indexes[(chain, resid, name)] = index
    #
    #         # Check atoms not found in conects
    #         with open(pdb_file + ".tmp", "w") as tmp:
    #             with open(pdb_file) as pdb:
    #                 # write all lines but skip END line
    #                 for line in pdb:
    #                     if not line.startswith("END"):
    #                         tmp.write(line)
    #
    #                 # Write new conect line mapping
    #                 for entry in conects:
    #                     line = "CONECT"
    #                     for x in entry:
    #                         line += "%5s" % atom_indexes[x]
    #                     line += "\n"
    #                     tmp.write(line)
    #             tmp.write("END\n")
    #         shutil.move(pdb_file + ".tmp", pdb_file)
    #
    #     def checkCappingGroups(pdb_file, format_for_prepwizard=True, keep_conects=True):
    #
    #         ace_names = ['CO', 'OP1', 'CP2', '1HP2', '2HP2', '3HP2']
    #
    #         if keep_conects:
    #             conect_lines = getConectLines(pdb_file)
    #
    #         # Detect capping groups
    #         structure = _readPDB(pdb_file, best_model_tag+".pdb")
    #         model = structure[0]
    #
    #         for chain in model:
    #
    #             add_one = False
    #             residues = [r for r in chain]
    #
    #             # Check for ACE atoms
    #             ace_atoms = []
    #             for a in residues[0]:
    #                 if a.name in ace_names:
    #                     ace_atoms.append(a)
    #
    #             # Check for NMA residue
    #             nma_residue = None
    #             for r in residues:
    #                 if r.resname == 'NMA':
    #                     nma_residue = r
    #
    #             # Build a separate residue for ACE
    #             new_chain = PDB.Chain.Chain(chain.id)
    #
    #             if ace_atoms:
    #
    #                 for a in ace_atoms:
    #                     residues[0].detach_child(a.name)
    #
    #                 ace_residue = PDB.Residue.Residue((' ', residues[0].id[1]-1, ' '), 'ACE', '')
    #
    #                 for i, a in enumerate(ace_atoms):
    #                     new_name = a.get_name()
    #
    #                     # Define the new name based on the old one
    #                     if format_for_prepwizard:
    #                         if new_name == 'CP2':
    #                             new_name = 'CH3'
    #                         elif new_name == 'CO':
    #                             new_name = 'C'
    #                         elif new_name == 'OP1':
    #                             new_name = 'O'
    #                         elif new_name == '1HP2':
    #                             new_name = '1H'
    #                         elif new_name == '2HP2':
    #                             new_name = '2H'
    #                         elif new_name == '3HP2':
    #                             new_name = '3H'
    #
    #                     # Create a new atom
    #                     new_atom = PDB.Atom.Atom(
    #                         new_name,                  # Atom name
    #                         a.get_coord(),             # Coordinates
    #                         a.get_bfactor(),           # B-factor
    #                         a.get_occupancy(),         # Occupancy
    #                         a.get_altloc(),            # AltLoc
    #                         "%-4s" % new_name,         # Full atom name (formatted)
    #                         a.get_serial_number(),     # Serial number
    #                         a.element                  # Element symbol
    #                     )
    #
    #                     ace_residue.add(new_atom)
    #
    #                 new_chain.add(ace_residue)
    #
    #             # Renumber residues and rename atoms
    #             for i, r in enumerate(residues):
    #
    #                 # Handle NMA residue atom renaming
    #                 if r == nma_residue and format_for_prepwizard:
    #                     renamed_atoms = []
    #                     for a in nma_residue:
    #
    #                         new_name = a.get_name()  # Original atom name
    #
    #                         # Rename the atom based on the rules
    #                         if new_name == 'HN2':
    #                             new_name = 'H'
    #                         elif new_name == 'C':
    #                             new_name = 'CA'
    #                         elif new_name == 'H1':
    #                             new_name = '1HA'
    #                         elif new_name == 'H2':
    #                             new_name = '2HA'
    #                         elif new_name == 'H3':
    #                             new_name = '3HA'
    #
    #                         # Create a new atom with the updated name
    #                         new_atom = PDB.Atom.Atom(
    #                             new_name,                  # New name
    #                             a.get_coord(),             # Same coordinates
    #                             a.get_bfactor(),           # Same B-factor
    #                             a.get_occupancy(),         # Same occupancy
    #                             a.get_altloc(),            # Same altloc
    #                             "%-4s" % new_name,         # Full atom name (formatted)
    #                             a.get_serial_number(),     # Same serial number
    #                             a.element                  # Same element
    #                         )
    #                         renamed_atoms.append(new_atom)
    #
    #                     # Create a new residue with renamed atoms
    #                     nma_residue = PDB.Residue.Residue(r.id, r.resname, r.segid)
    #                     for atom in renamed_atoms:
    #                         nma_residue.add(atom)
    #
    #                     r = nma_residue
    #                     add_one = True
    #
    #                 if add_one:
    #                     chain.detach_child(r.id)  # Deatach residue from old chain
    #                     new_id = (r.id[0], r.id[1]+1, r.id[2])  # New ID with updated residue number
    #                     r.id = new_id  # Update residue ID with renumbered value
    #
    #                 # Add residue to the new chain
    #                 new_chain.add(r)
    #
    #             model.detach_child(chain.id)
    #             model.add(new_chain)
    #
    #         _saveStructureToPDB(structure, pdb_file)
    #
    #         if keep_conects:
    #             writeConectLines(conect_lines, pdb_file)
    #
    #     executable = "extract_pdbs.linuxgccrelease"
    #     models = []
    #
    #     # Check if params were given
    #     params = None
    #     if os.path.exists(optimization_folder + "/params"):
    #         params = optimization_folder + "/params"
    #         patch_line = ""
    #         for p in os.listdir(params):
    #             if not p.endswith(".params"):
    #                 patch_line += params + "/" + p + " "
    #
    #     for d in os.listdir(optimization_folder + "/output_models"):
    #         if os.path.isdir(optimization_folder + "/output_models/" + d):
    #             for f in os.listdir(optimization_folder + "/output_models/" + d):
    #                 if f.endswith("_relax.out"):
    #                     model = d
    #
    #                     # skip models not loaded into the library
    #                     if model not in self.models_names:
    #                         continue
    #
    #                     scores = readSilentScores(
    #                         optimization_folder + "/output_models/" + d + "/" + f
    #                     )
    #                     if tags != None and model in tags:
    #                         print(
    #                             "Reading model %s from the given tag %s"
    #                             % (model, tags[model])
    #                         )
    #                         best_model_tag = tags[model]
    #                     elif min_value:
    #                         best_model_tag = scores.idxmin()[filter_score_term]
    #                     else:
    #                         best_model_tag = scores.idxmxn()[filter_score_term]
    #                     command = executable
    #                     command += (
    #                         " -silent "
    #                         + optimization_folder
    #                         + "/output_models/"
    #                         + d
    #                         + "/"
    #                         + f
    #                     )
    #                     if params != None:
    #                         command += " -extra_res_path " + params
    #                         if patch_line != "":
    #                             command += " -extra_patch_fa " + patch_line
    #                     command += " -tags " + best_model_tag
    #                     if sugars:
    #                         command += " -include_sugars"
    #                         command += " -alternate_3_letter_codes pdb_sugar"
    #                         command += " -write_glycan_pdb_codes"
    #                         command += " -auto_detect_glycan_connections"
    #                         command += " -maintain_links"
    #                     os.system(command)
    #
    #                     checkCappingGroups(best_model_tag+".pdb")
    #
    #                     self.readModelFromPDB(
    #                         model,
    #                         best_model_tag + ".pdb",
    #                         wat_to_hoh=wat_to_hoh,
    #                         conect_update=conect_update,
    #                     )
    #                     os.remove(best_model_tag + ".pdb")
    #                     models.append(model)
    #
    #     self.getModelsSequences()
    #
    #     missing_models = set(self.models_names) - set(models)
    #     if missing_models != set():
    #         print("Missing models in relaxation folder:")
    #         print("\t" + ", ".join(missing_models))
    #         if return_missing:
    #             return missing_models

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
        output_folder=None,
    ):
        """
        Load the best energy models from a set of silent files inside a specific folder.
        Useful to get the best models from a relaxation run.

        If output_folder is provided, the best model is extracted and saved there
        (with the pose index removed from the filename) rather than loaded into the class.

        Parameters
        ==========
        optimization_folder : str
            Path to folder where the Rosetta optimization files are contained
        filter_score_term : str
            Score term used to filter models
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
        sugars : bool
            Additional flag for sugar handling.
        conect_update : bool
            Flag to update CONECT lines.
        output_folder : str or None
            If provided, extracted PDB files (with pose index removed from filename) are written
            into this folder instead of being loaded into the class.
        """

        def getConectLines(pdb_file, format_for_prepwizard=True):

            ace_names = ['CO', 'OP1', 'CP2', '1HP2', '2HP2', '3HP2']

            # Read PDB file
            atom_tuples = {}
            add_one = False
            previous_chain = None
            with open(pdb_file, "r") as f:
                for l in f:
                    if l.startswith("ATOM") or l.startswith("HETATM"):
                        index, name, resname, chain, resid = (
                            int(l[6:11]),        # Atom index
                            l[12:16].strip(),    # Atom name
                            l[17:20].strip(),    # Residue name
                            l[21],               # Chain identifier
                            int(l[22:26]),       # Residue index
                        )

                        if not previous_chain:
                            previous_chain = chain

                        if name in ace_names:
                            resid -= 1

                            if format_for_prepwizard:
                                if name == 'CP2':
                                    name = 'CH3'
                                elif name == 'CO':
                                    name = 'C'
                                elif name == 'OP1':
                                    name = 'O'
                                elif name == '1HP2':
                                    name = '1H'
                                elif name == '2HP2':
                                    name = '2H'
                                elif name == '3HP2':
                                    name = '3H'

                        if resname == 'NMA':
                            add_one = True

                            if format_for_prepwizard:
                                if name == 'HN2':
                                    name = 'H'
                                elif name == 'C':
                                    name = 'CA'
                                elif name == 'H1':
                                    name = '1HA'
                                elif name == 'H2':
                                    name = '2HA'
                                elif name == 'H3':
                                    name = '3HA'

                        if previous_chain != chain:
                            add_one = False

                        if add_one:
                            resid += 1

                        atom_tuples[index] = (chain, resid, name)
                        previous_chain = chain

            conects = []
            with open(pdb_file) as pdbf:
                for l in pdbf:
                    if l.startswith("CONECT"):
                        l = l.replace("CONECT", "")
                        l = l.strip("\n").rstrip()
                        num = len(l) / 5
                        new_l = [int(l[i * 5 : (i * 5) + 5]) for i in range(int(num))]
                        conects.append([atom_tuples[int(x)] for x in new_l])

            return conects

        def writeConectLines(conects, pdb_file):

            atom_indexes = {}
            with open(pdb_file, "r") as f:
                for l in f:
                    if l.startswith("ATOM") or l.startswith("HETATM"):
                        index, name, resname, chain, resid = (
                            int(l[6:11]),        # Atom index
                            l[12:16].strip(),    # Atom name
                            l[17:20].strip(),    # Residue name
                            l[21],               # Chain identifier
                            int(l[22:26]),       # Residue index
                        )
                        atom_indexes[(chain, resid, name)] = index

            # Check atoms not found in conects
            with open(pdb_file + ".tmp", "w") as tmp:
                with open(pdb_file) as pdb:
                    # write all lines but skip END line
                    for line in pdb:
                        if not line.startswith("END"):
                            tmp.write(line)

                    # Write new conect line mapping
                    for entry in conects:
                        line = "CONECT"
                        for x in entry:
                            line += "%5s" % atom_indexes[x]
                        line += "\n"
                        tmp.write(line)
                tmp.write("END\n")
            shutil.move(pdb_file + ".tmp", pdb_file)

        def checkCappingGroups(pdb_file, format_for_prepwizard=True, keep_conects=True):

            ace_names = ['CO', 'OP1', 'CP2', '1HP2', '2HP2', '3HP2']

            if keep_conects:
                conect_lines = getConectLines(pdb_file)

            # Detect capping groups
            structure = _readPDB(pdb_file, best_model_tag+".pdb")
            model = structure[0]

            for chain in model:

                add_one = False
                residues = [r for r in chain]

                # Check for ACE atoms
                ace_atoms = []
                for a in residues[0]:
                    if a.name in ace_names:
                        ace_atoms.append(a)

                # Check for NMA residue
                nma_residue = None
                for r in residues:
                    if r.resname == 'NMA':
                        nma_residue = r

                # Build a separate residue for ACE
                new_chain = PDB.Chain.Chain(chain.id)

                if ace_atoms:

                    for a in ace_atoms:
                        residues[0].detach_child(a.name)

                    ace_residue = PDB.Residue.Residue((' ', residues[0].id[1]-1, ' '), 'ACE', '')

                    for i, a in enumerate(ace_atoms):
                        new_name = a.get_name()

                        # Define the new name based on the old one
                        if format_for_prepwizard:
                            if new_name == 'CP2':
                                new_name = 'CH3'
                            elif new_name == 'CO':
                                new_name = 'C'
                            elif new_name == 'OP1':
                                new_name = 'O'
                            elif new_name == '1HP2':
                                new_name = '1H'
                            elif new_name == '2HP2':
                                new_name = '2H'
                            elif new_name == '3HP2':
                                new_name = '3H'

                        # Create a new atom
                        new_atom = PDB.Atom.Atom(
                            new_name,                  # Atom name
                            a.get_coord(),             # Coordinates
                            a.get_bfactor(),           # B-factor
                            a.get_occupancy(),         # Occupancy
                            a.get_altloc(),            # AltLoc
                            "%-4s" % new_name,         # Full atom name (formatted)
                            a.get_serial_number(),     # Serial number
                            a.element                  # Element symbol
                        )

                        ace_residue.add(new_atom)

                    new_chain.add(ace_residue)

                # Renumber residues and rename atoms
                for i, r in enumerate(residues):

                    # Handle NMA residue atom renaming
                    if r == nma_residue and format_for_prepwizard:
                        renamed_atoms = []
                        for a in nma_residue:

                            new_name = a.get_name()  # Original atom name

                            # Rename the atom based on the rules
                            if new_name == 'HN2':
                                new_name = 'H'
                            elif new_name == 'C':
                                new_name = 'CA'
                            elif new_name == 'H1':
                                new_name = '1HA'
                            elif new_name == 'H2':
                                new_name = '2HA'
                            elif new_name == 'H3':
                                new_name = '3HA'

                            # Create a new atom with the updated name
                            new_atom = PDB.Atom.Atom(
                                new_name,                  # New name
                                a.get_coord(),             # Same coordinates
                                a.get_bfactor(),           # Same B-factor
                                a.get_occupancy(),         # Same occupancy
                                a.get_altloc(),            # Same altloc
                                "%-4s" % new_name,         # Full atom name (formatted)
                                a.get_serial_number(),     # Same serial number
                                a.element                  # Same element
                            )
                            renamed_atoms.append(new_atom)

                        # Create a new residue with renamed atoms
                        nma_residue = PDB.Residue.Residue(r.id, r.resname, r.segid)
                        for atom in renamed_atoms:
                            nma_residue.add(atom)

                        r = nma_residue
                        add_one = True

                    if add_one:
                        chain.detach_child(r.id)  # Deatach residue from old chain
                        new_id = (r.id[0], r.id[1]+1, r.id[2])  # New ID with updated residue number
                        r.id = new_id  # Update residue ID with renumbered value

                    # Add residue to the new chain
                    new_chain.add(r)

                model.detach_child(chain.id)
                model.add(new_chain)

            _saveStructureToPDB(structure, pdb_file)

            if keep_conects:
                writeConectLines(conect_lines, pdb_file)

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

        if output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        for d in os.listdir(optimization_folder + "/output_models"):
            subfolder = os.path.join(optimization_folder, "output_models", d)
            if os.path.isdir(subfolder):
                for f in os.listdir(subfolder):
                    if f.endswith("_relax.out"):
                        model = d

                        # Skip models not loaded into the library
                        if model not in self.models_names:
                            continue

                        scores = readSilentScores(os.path.join(subfolder, f))
                        if tags is not None and model in tags:
                            print("Reading model %s from the given tag %s" % (model, tags[model]))
                            best_model_tag = tags[model]
                        elif min_value:
                            best_model_tag = scores.idxmin()[filter_score_term]
                        else:
                            best_model_tag = scores.idxmxn()[filter_score_term]
                        command = executable
                        command += " -silent " + os.path.join(subfolder, f)
                        if params is not None:
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

                        checkCappingGroups(best_model_tag + ".pdb")

                        # Remove the pose index from the name.
                        base_name = '_'.join(best_model_tag.split("_")[:-1])
                        new_filename = base_name + ".pdb"
                        if output_folder:
                            os.rename(best_model_tag + ".pdb", os.path.join(output_folder, new_filename))
                        else:
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
        if missing_models:
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
                            if '_' in l.strip().split()[i]:
                                scores[t].append(l.strip().split()[i])
                            else:
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
    output_folder, script_name, no_py=False, subfolder=None, hidden=True, path="prepare_proteins/scripts",
):
    """
    Copy a script file from the prepare_proteins package.

    Parameters
    ==========

    """
    # Get script

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

def _readRosettaScoreFile(score_file, indexing=False, skip_empty=False):
    """
    Reads a Rosetta score file and returns a DataFrame of the scores.

    Arguments:
    ==========
    score_file : str
        Path to the input score file.
    indexing : bool, optional
        If True, sets the DataFrame index to ['Model', 'Pose'].

    Returns:
    ========
    DataFrame
        A DataFrame containing the scores from the score file.
    """
    with open(score_file) as sf:
        lines = [x.strip() for x in sf if x.startswith("SCORE:")]

    if len(lines) < 2:
        if not skip_empty:
            raise ValueError("The score file does not contain enough data.")
        else:
            return None

    score_terms = lines[0].split()[1:]  # Get the terms excluding the initial "SCORE:"
    scores = {term: [] for term in score_terms}
    models = []
    poses = []
    descriptions = []

    for line in lines[1:]:
        parts = line.split()[1:]  # Get the parts excluding the initial "SCORE:"
        if len(parts) != len(score_terms):
            continue  # Skip lines that are headers or do not match the number of score terms
        if parts[0] == score_terms[0]:  # Check if this is a repeated header
            continue

        for i, score in enumerate(score_terms):
            try:
                scores[score].append(float(parts[i]))
            except ValueError:
                scores[score].append(parts[i])

        # Extract model and pose from the 'description' field
        description_index = score_terms.index("description")
        description = parts[description_index]
        model, pose = "_".join(description.split("_")[:-1]), description.split("_")[-1]
        models.append(model)
        poses.append(int(pose))
        descriptions.append(description)

    scores.pop("description")
    scores["Model"] = np.array(models)
    scores["Pose"] = np.array(poses)
    scores["description"] = np.array(descriptions)

    scores_df = pd.DataFrame(scores)

    if indexing:
        scores_df = scores_df.set_index(["Model", "Pose"])

    return scores_df

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
