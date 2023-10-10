import pandas as pd
import mdtraj as md
import shutil
import os

class rosettaAnalysis:
    """
    Container for managing data from Rosetta silent files.
    """

    def __init__(self, silent_file, params_path=None):
        """
        Initializes the class reading content from the silent file and generating
        a topology (.pdb) and a trajectory file for reading non-virtual coordinates.
        Calculated data is stored into an csv file with pandas for easy retrieving.

        Parameters
        ==========
        silent_file : str
            Path to the silent file to be read
        """

        # Define paths and file names
        self.silent_file = silent_file
        self.silent_name = silent_file.split('/')[-1]
        self.source_dir = '/'.join(silent_file.split('/')[:-1])
        self.dcd_name = self.silent_name.replace('.out', '.dcd')
        self.dcd_file = self.source_dir+'/'+self.dcd_name
        self.params_path = params_path

        # Read score term data
        self.data = self.readScoreTerms(self)

    def readScoreTerms(self):
        """
        Process the SCORE lines in the associated silent file
        """
        scores = {}
        with open(self.silent_file) as sf:
            count = 0
            for l in sf:
                if l.startswith('SCORE'):
                    if count == 0:
                        score_terms = l.split()[1:]
                        for t in score_terms:
                            scores[t] = []
                    else:
                        for t,v in zip(score_terms, l.split()[1:]):
                            scores[t].append(v)
                    count += 1
        scores = pd.DataFrame(scores)
        scores.set_index('description', inplace=True)
        scores.sort_index(inplace=True)
        return scores
