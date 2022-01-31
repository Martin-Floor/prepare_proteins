import __main__
__main__.pymol_argv = ['pymol','-qc']
import pymol
from pymol import cmd, stored
import sys
stdout = sys.stdout
stderr = sys.stderr
pymol.finish_launching(['pymol', '-qc'])
sys.stdout = stdout
sys.stderr = stderr

import os
import string

def alignPDBs(folder_name, pdb_list=None, alignment_index=0, align=True, verbose=True):
    """
    Performs alpha-carbon based alignment of PDB models inside a specific folder.
    The PDBs to be aligned can be narrowed down by giving a list of PDB file names.
    An index can be used to use a specific PDB in the list as the reference for
    the full alignment. The method returns a dictionary containing the RMSD values
    and the number of atoms used into a specific alignment.

    The method can be used only to calculate the RMSD values without saving the
    aligned models by giving the option align=False.

    Parameters
    ----------
    folder_name : str
        Path to the folder where PDB models are found.
    pdb_list : list
        Optional list with the PDBs to consider in the alignment.
    alignment_index : int
        Index of the file in the list to be used as the reference in the alignment.
    align : bool
        Whether to align or not the PDBs in the folder (set to False to only calculate
        RMSDs).
    verbose : bool
        Verbose mode

    Return
    ------
    alignment_details : dict
        Dictionary containing the RMSD values and the number of alpha carbon atoms
        used in the alignment.
    """

    cwd = os.getcwd()
    #Align all structures inside this folder
    os.chdir(folder_name)

    #Read pdbs in folder or read given pdbs
    if pdb_list == None:
        pdbs = {}
        count = 0
        for f in sorted(os.listdir()):
            if f.endswith('.pdb'):
                pdbs[count] = f
                count += 1
        if pdbs == {}:
            os.chdir(cwd)
            raise ValueError('There is no files with .pdb extension in the input folder. Please check that your input folder is correct.')
    else:
        pdbs = { i:pdb for i,pdb in enumerate(pdb_list) }

    # Load pdbs and align them
    for i in sorted(pdbs):
        if i == alignment_index:
            cmd.load(pdbs[i], 'reference')
        else:
            cmd.load(pdbs[i], pdbs[i])

    if verbose:
        print('Aligning to model: '+pdbs[alignment_index])
    sys.stdout = open('pymolout.tmp', 'w')
    cmd.extra_fit( 'name CA', 'reference', 'super', object='aln_super')
    sys.stdout = stdout

    alignment_details = {}
    with open('pymolout.tmp') as pof:
        for l in pof:
            f = l.split()[0]
            rms = float(l.split()[3])
            atoms = int(l.split()[4].replace('(',''))
            alignment_details[f] = {}
            alignment_details[f]['RMS'] = rms
            alignment_details[f]['atoms'] = atoms
    if align:
        for i in sorted(pdbs):
            if i != alignment_index:
                cmd.save(pdbs[i], pdbs[i])

    cmd.delete('all')
    os.remove('pymolout.tmp')
    os.chdir(cwd)

    return alignment_details

def createSymmetryMolecules(pdb_file, chain_id, distance, output_pdb, letter='Z'):

    # Load input PDB
    cmd.load(pdb_file, 'reference')

    # Select chain
    cmd.select(chain_id, 'chain '+chain_id)

    # Get all chain letters in use
    # Define additional chain letters
    letters = set([l for l in string.ascii_uppercase])
    model = cmd.get_model('reference')
    used_letters = []
    for a in model.atom:
        used_letters.append(a.chain)
    unused_letters = sorted(list(letters - set(used_letters)))

    # Create symmetry relacted molecules inside the threshold distance
    cmd.symexp('symmetry', 'reference', chain_id, distance)

    # Get all symmetry reconstructed objects
    symmetry_objects = []
    for o in cmd.get_object_list('all'):
        if 'symmetry' in o:
            symmetry_objects.append(o)

    # Rename and renumber each reconstructed object
    for symmetry in symmetry_objects:

        letter = unused_letters[-1]
        if symmetry == None:
            raise ValueError('No symmetry molecules were build.')

        # Change symmetry related molecules chain id
        cmd.alter(symmetry, "chain='"+letter+"'")

        # Renumber symmetry related residues
        model = cmd.get_model(symmetry)
        for i,a in enumerate(model.atom):
            a.resi = i+1

        cmd.alter(symmetry, 'resi = next(atom_it).resi',
                space={'atom_it': iter(model.atom), 'next': next})

        cmd.group('reconstructed', 'reference '+symmetry, action='add')

        unused_letters = unused_letters[:-1]

    cmd.save(output_pdb, 'reconstructed')
