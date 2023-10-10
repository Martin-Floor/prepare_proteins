from Bio import SeqIO, AlignIO

def readFastaFile(fasta_file, replace_slash=False):
    """
    Read a fasta file and get the sequences as a dictionary

    Parameters
    ----------
    fasta_file : str
        Path to the input fasta file

    Returns
    -------
    sequences : dict
        Dictionary containing the IDs and squences in the fasta file.
    replace_slash : bool
        If a slash symbol is found in name, raplaces it for a middle dash.
    """

    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        if replace_slash:
            sequences[record.id.replace('/','-')] = str(record.seq)
        else:
            sequences[record.id] = str(record.seq)


    return sequences

def writeFastaFile(sequences, output_file):
    """
    Write sequences to a fasta file.

    Parameters
    ----------
    sequences : dict
        Dictionary containing as values the strings representing the sequences
        of the proteins to align and their identifiers as keys.

    output_file : str
        Path to the output fasta file
    """

    # Write fasta file containing the sequences
    with open(output_file, 'w') as of:
        for name in sequences:
            of.write('>'+name+'\n')
            of.write(sequences[name]+'\n')

def writeMsaToFastaFile(msa, output_file):
    """
    Write sequences inside an MSA to a fasta file.

    Parameters
    ----------
    msa : Bio.AlignIO
        Multiple sequence aligment in Biopython format.

    output_file : str
        Path to the output fasta file
    """

    # Write fasta file containing the sequences
    with open(output_file, 'w') as of:
        for s in msa:
            of.write('>'+s.id+'\n')
            of.write(str(s.seq)+'\n')


def ReadMsaFromFastaFile(alignment_file):
    """
    Read an MSA from a fasta file.

    Parameters
    ----------
    alignment_file : str
        Path to the alignment fasta file

    msa : Bio.AlignIO
        Multiple sequence aligment in Biopython format.
    """

    msa = AlignIO.read(alignment_file, 'fasta')

    return msa

def msaIndexesFromSequencePositions(msa, sequence_id, sequence_positions):
    """
    Get the multiple sequence alignment position indexes matching those positions (zero-based) of a specific target sequence.

    Parameters
    ==========
    msa : Bio.AlignIO
        Multiple sequence aligment in Biopython format.
    sequence_id : str
        ID of the target sequence
    sequence_positions : list
        Target sequence positions to match (one-based indexes)

    Returns
    =======
    msa_indexes : list
        MSA indexes matching the target sequence positions (zero-based indexes)
    """

    # Check whether the given ID is presetnin the MSA.
    if sequence_id not in [x.id for x in msa]:
        raise ValueError('Entry %s not found in MSA' % sequence_id)

    # Gather MSA index positions mathing the target sequence positions.
    msa_indexes = []
    p = 0
    for a in msa:
        if a.id == sequence_id:

            # Check that all given positions are lower than the sequence length
            for x in sequence_positions:
                sl = len([x for x in a.seq if x != '-'])
                if x > sl:
                    raise ValueError('The given position %s is larger than the length %s of the target sequence in the MSA.' % (x, sl))

            for i in range(msa.get_alignment_length()):
                if a.seq[i] != '-':
                    p += 1
                    if p in sequence_positions:
                        msa_indexes.append(i)

    return msa_indexes
