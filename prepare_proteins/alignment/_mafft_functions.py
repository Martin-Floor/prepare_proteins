import os
from Bio import AlignIO
import shutil
from ._methods import *

class mafft:
    """
    Class to hold methods to work with mafft executable.

    Methods
    -------
    multipleSequenceAlignment()
        Execute a multiple sequence alignment of the input sequences
    """

    def multipleSequenceAlignment(sequences, output=None, anysymbol=False):
        """
        Use the mafft executable to perform a multiple sequence alignment.

        Parameters
        ----------
        sequences : dict
            Dictionary containing as values the strings representing the sequences
            of the proteins to align and their identifiers as keys.
        output : str
            File name to write the fasta formatted alignment output.
        anysymbol : bool
            Use unusual symbols in the alignment

        Returns
        -------
        alignment : Bio.AlignIO
            Multiple sequence alignment in Biopython format.
        """

        # Write input file containing the sequences
        writeFastaFile(sequences, 'sequences.fasta.tmp')

        # Calculate alignment
        command = 'mafft --auto'
        if anysymbol:
            command += ' --anysymbol'
        command += ' sequences.fasta.tmp > sequences.aligned.fasta.tmp'
        os.system(command)

        # Read aligned file
        alignment = AlignIO.read("sequences.aligned.fasta.tmp", "fasta")

        # Remove temporary file
        os.remove('sequences.fasta.tmp')
        if output != None:
            shutil.copyfile('sequences.aligned.fasta.tmp', output)
        os.remove('sequences.aligned.fasta.tmp')

        return alignment

    def addSequenceToMSA(sequences, msa, output=None, anysymbol=False, keeplength=False,
                         reorder=True):
        """
        Use the mafft executable to add sequences to an already existing multiple
        sequence alignment.

        Parameters
        ----------
        sequences : dict
            Dictionary containing as values the strings representing the sequences
            of the proteins to add and their identifiers as keys.
        msa : Bio.AlignIO
            Multiple sequence aligment in Biopython format.
        output : str
            File name to write the fasta formatted alignment output.
        anysymbol : bool
            Use unusual symbols in the alignment
        reorder : bool
            Gaps in existing_alignment are preserved, but the alignment length may
            be changed. All sequences are conserved.
        keeplength : bool
            If keeplength option is True, then the alignment length is unchanged.
            Insertions at the new sequences are deleted.

        Returns
        -------
        alignment : Bio.AlignIO
            Multiple sequence alignment in Biopython format.
        """

        if keeplength:
            reorder = False

        # Write input file containing the sequences
        with open('sequences.fasta.tmp', 'w') as iff:
            for name in sequences:
                iff.write('>'+name+'\n')
                iff.write(sequences[name]+'\n')

        # Write input file containing the msa sequences as fasta file
        with open('sequences.msa.fasta.tmp', 'w') as iff:
            for name in msa:
                iff.write('>'+name.id+'\n')
                iff.write(str(name.seq)+'\n')

        # Calculate alignment
        command = 'mafft'
        if anysymbol:
            command += ' --anysymbol'
        command += ' --add sequences.fasta.tmp'
        if reorder:
            command += ' --reorder sequences.msa.fasta.tmp'
        if keeplength:
            command += ' --keeplength sequences.msa.fasta.tmp'
        command += '  > sequences.aligned.fasta.tmp'
        os.system(command)

        # Read aligned file
        alignment = AlignIO.read("sequences.aligned.fasta.tmp", "fasta")

        # Remove temporary file
        os.remove('sequences.fasta.tmp')
        os.remove('sequences.msa.fasta.tmp')
        if output != None:
            shutil.copyfile('sequences.aligned.fasta.tmp', output)
        os.remove('sequences.aligned.fasta.tmp')

        return alignment

    def readSequenceFastaFile(fasta_file):
        """
        Function to read the sequences in a fasta file into a dictionary.

        Parameters
        ----------
        fasta_file : str
            Path to the input fasta file

        Returns
        -------

        sequences : dict
            Dictionary containing as values the sequences and their identifiers
            as keys.
        """

        sequences = {}
        sequence = ''
        with open(fasta_file) as ff:
            for l in ff:
                if l.startswith('>'):
                    if sequence != '':
                        sequences[identifier] = sequence
                    identifier = l.strip().replace('>','')
                    sequence = ''
                else:
                    sequence += l.strip()
            if sequence != '':
                sequences[identifier] = sequence

        return sequences
