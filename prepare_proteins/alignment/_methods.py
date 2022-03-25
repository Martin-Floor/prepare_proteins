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

def msaPositionsFromSequencePositions(msa, sequence_id, sequence_indexes):
    """
    Get the multiple sequence alignment position indexes matching those positions of a specific sequence target.

    Parameters
    ==========
    msa : Bio.AlignIO
        Multiple sequence aligment in Biopython format.
    sequence_id : str
        ID of the target sequence
    sequence_indexes : list
        Target sequence positions to match.

    Returns
    =======
    msa_indexes : list
        MSA indexes matching the target sequence positions
    """

    msa_indexes = []
    p = 0
    for i in range(msa.get_alignment_length()):
        if p in sequence_indexes:
            msa_indexes.append(i)
        for a in msa:
            if a.id == sequence_id:
                if a.seq[i] != '-':
                    p += 1

    return msa_indexes
