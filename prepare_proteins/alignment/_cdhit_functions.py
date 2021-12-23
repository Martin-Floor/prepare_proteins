import os
import subprocess
import shutil
import sys
from ._methods import *

class cdhit:

    def clusterSequences(sequences, pid_threshold=0.9, return_identities=False,
                         keep_sequence_file=None):
        """
        Cluster sequences by PID using CD-HIT program.

        Parameters
        ----------
        sequences : dict or str
            Dictionary containing as values the strings representing the sequences
            of the proteins to align and their identifiers as keys. It also can be the
            path to the fasta file containing the sequences.
        pid_threshold : float
            PID value to cluster the sequences
        return_identities : bool
            Whether to return the identities of cluster member to the representative
            sequence.
        keep_sequence_file : str
            keep CD-HIT fasta sequence file
        Returns
        -------
        clusters : dict
            Clustered sequences. The dictionary contains the cluster representatives
            as keys and the cluster memebers as values. If return identities is True,
            then the subkey 'members' access the cluster members and the subkey 'identities'
            access their identities to the representative sequence.
        """
        temp_fasta_file = 'sequences.fasta.tmp'
        delete_temp = True

        # Check input format
        if isinstance(sequences, dict):
            # Write input file containing the sequences
            writeFastaFile(sequences, 'sequences.fasta.tmp')
        elif isinstance(sequences, str):
            temp_fasta_file = sequences
            delete_temp = False

        # Set command to calculate clusters
        command = 'cd-hit -i '+temp_fasta_file
        command += ' -c '+str(pid_threshold)
        command += ' -o cdhit_clusters.tmp'

        # Run CD-HIT
        # subprocess.check_call([command])
        # cdhit_output = os.popen(command).read()
        try:
            output = subprocess.check_output(
                        command, stderr=subprocess.STDOUT, shell=True,
                        universal_newlines=True)

        except subprocess.CalledProcessError as exc:
            print(exc.output, exc.returncode)
            raise Exception("Problem with CDHIT input")

        clusters = {}

        if return_identities:
            clusters['members'] = {}
            clusters['identities'] = {}

        # Parse CDHIT output file
        """ a “>” starts a new cluster
            a “*” at the end means that this sequence is the representative of this cluster
            a “%” is the identity between this sequence and the representative"""

        with open('cdhit_clusters.tmp.clstr') as cdo:
            cond = False
            cluster_members = []
            identities = []
            cluster_center = None
            for l in cdo:
                if l.startswith('>') and cluster_members != []:
                    cond = True
                    if not return_identities:
                        clusters[cluster_center] = cluster_members
                    else:
                        clusters['members'][cluster_center] = cluster_members
                        for z in zip(cluster_members, identities):
                            clusters['identities'][z[0]] = z[1]
                        cluster_members = []
                else:
                    if not l.startswith('>'):
                        name = l.split('>')[-1].split('...')[0]
                        if 'at' in l:
                            identities.append(float(l.strip().split('at')[-1][:-1]))
                        else:
                            cluster_center = name
                            identities.append(100.0)
                        cluster_members.append(name)

            # Process the last cluster in the file
            if not return_identities:
                clusters[cluster_center] = cluster_members
            else:
                clusters['members'][cluster_center] = cluster_members
                for z in zip(cluster_members, identities):
                    clusters['identities'][z[0]] = z[1]

        if delete_temp:
            os.remove(temp_fasta_file)

        if keep_sequence_file:
            shutil.copyfile('cdhit_clusters.tmp', keep_sequence_file)

        os.remove('cdhit_clusters.tmp.clstr')
        os.remove('cdhit_clusters.tmp')

        return clusters
