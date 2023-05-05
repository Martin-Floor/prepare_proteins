from prepare_proteins.alignment import mafft
import mdtraj as md

def alignTrajectoryBySequenceAlignment(trajectory, reference, reference_frame=0,
                                       chain_indexes=None, trajectory_chain_indexes=None,
                                       reference_chain_indexes=None, aligment_mode='aligned',
                                       reference_residues=None):
    """
    Align two trajectories based on a sequence alignment of specific chains. The
    chains are specified using their indexes. When the trajectories have corresponding
    chains use the option chain_indexes to specify the list of chains to align.
    Otherwise, specify the chains with trajectory_chain_indexes and reference_chain_indexes
    options. Note that the list of chain indexes must be corresponding.

    Example:

    trajectory_chain_indexes = [0,1]
    reference_chain_indexes = [2,3]

    This input will align the sequence of chain 2 from the reference topology against
    the sequence of chain 0 from the target topology to find common alpha carbon
    atoms. Then the alignment is carried out for the sequence of chain 2 of the
    reference trajectory against the sequence of chain 1 in the target trajectory
    to find common alpha carbon atoms. All matched atoms will be used to align the
    target trajectory against the specified frame of the reference trajectory.

    The input chain_indexes = [0,1] is equivalent to inputting:

    trajectory_chain_indexes = [0,1]
    reference_chain_indexes = [0,1]

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Trajectory to align.
    reference : mdtraj.Trajectory
        Reference trajectory.
    reference_frame : int (0)
        Index of the frame in the reference trajectory to use as reference for the
        alignment.
    chain_indexes : int or list
        Chain indexes to use for the alignment. Use this option when the trajectories
        have corresponding chains in their topologies.
    trajectory_chain_indexes : int
        Chain indexes of the target trajectory to use in the alignment.
    reference_chain_indexes : int
        Chain indexes of the reference trajectory to use in the alignment.
    aligment_mode : str
        The mode defines how sequences are aligned. 'exact' for structurally
        aligning positions with exactly the same aminoacids after the sequence
        alignemnt or 'aligned' for structurally aligining sequences using all
        positions aligned in the sequence alignment.
    """

    # Check for correct input
    if chain_indexes == None:
        if trajectory_chain_indexes == None or reference_chain_indexes == None:
            raise ValueError('You must enter the chain index(es) to align against.')
        elif trajectory_chain_indexes != None and reference_chain_indexes == None:
            raise ValueError('If you give the chain index(es) for the target topology\
            with the option trajectory_chain_indexes you also need to specify the\
            chain index(es) for the reference topology with the option reference_chain_indexes')
        elif trajectory_chain_indexes == None and reference_chain_indexes != None:
            raise ValueError('If you give the chain index(es) for the reference\
            topology with the option reference_chain_indexes you also need to specify the\
            chain index(es) for the target topology with the option trajectory_chain_index')
        else:
            if isinstance(trajectory_chain_indexes, int):
                 trajectory_chain_indexes = [trajectory_chain_indexes]
            elif not isinstance(trajectory_chain_indexes, list):
                raise ValueError('Trajectory chain indexes must be given as a integer or as a list of\
                integers.')
            if isinstance(reference_chain_indexes, int):
                 reference_chain_indexes = [reference_chain_indexes]
            elif not isinstance(reference_chain_indexes, list):
                raise ValueError('Reference chain indexes must be given as a integer or as a list of\
                integers.')
    elif chain_indexes != None and (trajectory_chain_indexes != None or reference_chain_indexes != None):
        raise ValueError('If you use the chain_indexes option. Please do not specify\
        with the options trajectory_chain_indexes or reference_chain_indexes.')
    else:
        if isinstance(chain_indexes, int):
             chain_indexes = [chain_indexes]
        elif not isinstance(chain_indexes, list):
            raise ValueError('Chain indexes must be given as a integer or as a list of\
            integers.')
        trajectory_chain_indexes = chain_indexes
        reference_chain_indexes = chain_indexes

    if len(trajectory_chain_indexes) != len(reference_chain_indexes):
        raise ValueError('You must input the same number of chains to align the\
        target trajectory with the reference trajectory.')

    # Get dictionaries linking sequence positions to residue chain_indexes
    trajectory_indexes = getChainIndexesToResidueIndexes(trajectory.topology)
    reference_indexes = getChainIndexesToResidueIndexes(reference.topology)

    # Align sequences and store common residues
    trajectory_residues = []
    reference_residues = []
    for i in range(len(trajectory_chain_indexes)):
        sequences = {}
        # Get corresponding chain ids
        tci = trajectory_chain_indexes[i]
        rci = reference_chain_indexes[i]

        # Store sequences into a dictionary
        sequences['target'] = getTopologySequence(trajectory.topology, tci)
        sequences['reference'] = getTopologySequence(reference.topology, rci)

        # Align sequences
        alignment = mafft.multipleSequenceAlignment(sequences, stdout=False, stderr=False)

        # Get coincident positions in the alignment
        positions = getCommonPositions(alignment[0].seq, alignment[1].seq,
                                       mode=aligment_mode)

        # Store common residues
        for p in positions:
            trajectory_residues.append(trajectory_indexes[tci][p[0]])
            reference_residues.append(reference_indexes[rci][p[1]])

    # Store common alpha-carbon atoms
    for p in zip(trajectory_residues, reference_residues):
        trj_ca = trajectory.topology.residue(p[0])
        ref_ca = trajectory.topology.residue(p[0])

    trajectory_atoms = [ a.index for a in trajectory.topology.atoms if a.name == 'CA'\
                         and a.residue.index in trajectory_residues ]
    reference_atoms = [ a.index for a in reference.topology.atoms if a.name == 'CA' \
                        and a.residue.index in reference_residues ]

    assert len(trajectory_atoms) == len(reference_atoms)

    # Align trajectory
    trajectory.superpose(reference, frame=reference_frame, atom_indices=trajectory_atoms,
                         ref_atom_indices=reference_atoms)

    rmsd = md.rmsd(trajectory, reference, atom_indices=trajectory_atoms, ref_atom_indices=reference_atoms)[0]*10.0

    return rmsd, len(trajectory_atoms)

def getCommonPositions(sequence1, sequence2, mode='exact', sequence1_residues=None):
    """
    Get the common positions of two aligned sequences. Two modes are possible:
    mode = 'exact', in which positions returned are equal in sequence.
    mode = 'aligned', in which positions returned are aligned together.

    Parameters
    ----------
    sequence1 : Bio.Seq.Seq
        Input sequence 1
    sequence2 : Bio.Seq.Seq
        Input sequence 2
    mode : str ('exact')
        Mode to analyze common positions

    Returns
    -------
    positions : list
        List of tuples containing the common positions between the two sequences
    """

    # Check input variables
    if mode not in ['exact', 'aligned']:
        raise ValueError('The mode option of getCommonPositions() can be either:\
        "exact" or "aligned".')
    if len(sequence1) != len(sequence2):
        raise ValueError('Input sequences are not the same length. Input sequences\
        must have been previously aligned.')

    if len(sequence1) != len(sequence2):
        raise ValuError('The sequences given must be previously aligned. Chech your input')

    positions = []
    # Initialize idependent counters for sequence positions
    s1p = 0
    s2p = 0
    # Iterate through the aligned positions
    for i in range(len(sequence1)):

        # Skip residues not found in given list
        if sequence1_residues != None:
            if s1p+1 not in sequence1_residues:
                continue

        # Compare sequences according to selected mode
        if sequence1[i] != '-' and sequence2[i] != '-':
            if mode == 'exact':
                if sequence1[i] == sequence2[i]:
                    positions.append((s1p, s2p))
            elif mode == 'aligned':
                positions.append((s1p, s2p))

        # Add to position counters
        if sequence1[i] != '-':
            s1p += 1
        if sequence2[i] != '-':
            s2p += 1

    return positions

def getTopologySequence(topology, chain_index, non_protein='X',
                        only_protein=True):
    """
    Get the sequence of a specfic chain in a topology object.

    Parameters
    ----------
    topology : mdtraj.Topology
        Input topology
    chain_indexes : int
        Index of the chain to extract its sequence

    Returns
    -------
    sequence : str
        Sequence of the topology object
    """
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

    sequence = ''
    for r in topology.chain(chain_index).residues:
        if only_protein:
            if r.name in three_to_one:
                sequence += three_to_one[r.name]
            else:
                if not only_protein:
                    sequence += non_protein

    return sequence

def getChainIndexesToResidueIndexes(topology):
    """
    Get a dictionary linking chain sequence positions to residues positions in the
    topology.

    Parameters
    ----------
    topology : mdtraj.Topology
        Input topology

    Returns
    -------
    residue_indexes : dict
        Dictionary containing chain indexes as keys and subdictionaries as values.
        Each subdictionary contains chain sequence indexes as keys and residue index in the
        topology as values.
    """

    residue_indexes  = {}
    for chain in topology.chains:
        residue_indexes[chain.index] = {}
        for i,residue in enumerate(chain.residues):
            residue_indexes[chain.index][i] = residue.index

    return residue_indexes
