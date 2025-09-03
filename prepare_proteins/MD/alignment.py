from prepare_proteins.alignment import mafft
import mdtraj as md

def alignTrajectoryBySequenceAlignment(trajectory, reference, reference_frame=0,
                                       chain_indexes=None, trajectory_chain_indexes=None,
                                       reference_chain_indexes=None, alignment_mode='aligned',
                                       reference_residues=None):
    """
    Align a trajectory to a reference trajectory using sequence alignment of specific chains.
    Chains can be specified using either `chain_indexes` (if chains correspond) or
    separately using `trajectory_chain_indexes` and `reference_chain_indexes`.

    Example:
    --------
    trajectory_chain_indexes = [0, 1]
    reference_chain_indexes = [2, 3]

    This aligns chain 0 of the trajectory to chain 2 of the reference, and chain 1
    to chain 3, using common alpha-carbon atoms identified via sequence alignment.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Trajectory to align.
    reference : mdtraj.Trajectory
        Reference trajectory.
    reference_frame : int, default=0
        Frame of the reference trajectory to align to.
    chain_indexes : int or list of int, optional
        If provided, these are used for both trajectory and reference chains.
    trajectory_chain_indexes : int or list of int, optional
        Chain indexes in the trajectory to align.
    reference_chain_indexes : int or list of int, optional
        Chain indexes in the reference trajectory to align.
    alignment_mode : str, default='aligned'
        'exact' to align only residues with identical amino acids.
        'aligned' to align all residues matched in the sequence alignment.

    Returns
    -------
    rmsd : float
        RMSD (in Ångströms) after alignment.
    n_atoms : int
        Number of atoms used in the alignment.
    """

    # Validate chain input
    if chain_indexes is None:
        if trajectory_chain_indexes is None or reference_chain_indexes is None:
            raise ValueError("Specify either `chain_indexes` or both `trajectory_chain_indexes` and `reference_chain_indexes`.")

        if isinstance(trajectory_chain_indexes, int):
            trajectory_chain_indexes = [trajectory_chain_indexes]
        elif not isinstance(trajectory_chain_indexes, list):
            raise ValueError("`trajectory_chain_indexes` must be an int or a list of ints.")

        if isinstance(reference_chain_indexes, int):
            reference_chain_indexes = [reference_chain_indexes]
        elif not isinstance(reference_chain_indexes, list):
            raise ValueError("`reference_chain_indexes` must be an int or a list of ints.")
    else:
        if trajectory_chain_indexes is not None or reference_chain_indexes is not None:
            raise ValueError("Use `chain_indexes` OR (`trajectory_chain_indexes` and `reference_chain_indexes`), not both.")

        if isinstance(chain_indexes, int):
            chain_indexes = [chain_indexes]
        elif not isinstance(chain_indexes, list):
            raise ValueError("`chain_indexes` must be an int or a list of ints.")

        trajectory_chain_indexes = chain_indexes
        reference_chain_indexes = chain_indexes

    if len(trajectory_chain_indexes) != len(reference_chain_indexes):
        raise ValueError("Trajectory and reference must have the same number of chain indexes.")

    # Get residue indexes per chain
    trajectory_indexes = getChainIndexesToResidueIndexes(trajectory.topology)
    reference_indexes = getChainIndexesToResidueIndexes(reference.topology)

    # Align sequences and get matched residues
    trajectory_residues = []
    reference_residues = []

    for tci, rci in zip(trajectory_chain_indexes, reference_chain_indexes):
        sequences = {
            'target': getTopologySequence(trajectory.topology, tci),
            'reference': getTopologySequence(reference.topology, rci)
        }

        alignment = mafft.multipleSequenceAlignment(sequences, stdout=False, stderr=False)

        positions = getCommonPositions(alignment[0].seq, alignment[1].seq, mode=alignment_mode)

        for p in positions:
            trajectory_residues.append(trajectory_indexes[tci][p[0]])
            reference_residues.append(reference_indexes[rci][p[1]])

    # Get CA atom indices from matched residues
    trajectory_atoms = [a.index for a in trajectory.topology.atoms
                        if a.name == 'CA' and a.residue.index in trajectory_residues]

    reference_atoms = [a.index for a in reference.topology.atoms
                       if a.name == 'CA' and a.residue.index in reference_residues]

    if len(trajectory_atoms) != len(reference_atoms):
        raise ValueError("Mismatch in number of aligned alpha-carbon atoms.")

    # Superpose and compute RMSD
    trajectory.superpose(reference, frame=reference_frame,
                         atom_indices=trajectory_atoms, ref_atom_indices=reference_atoms)

    rmsd = md.rmsd(trajectory, reference,
                   atom_indices=trajectory_atoms,
                   ref_atom_indices=reference_atoms)[0] * 10.0  # Convert nm to Å

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
