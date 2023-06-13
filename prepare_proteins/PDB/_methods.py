from Bio import PDB
import os
import shutil

def retrievePDBs(pdb_codes, names=None, pdb_directory='PDB'):
    """
    Download a set of pdb structures from the PDB database.

    Parameters
    ----------
    pdb_codes : list or str
        A list with the PDB codes to retrieve their files. Optionally a string
        with a unique code can be given.
    names : dict
        Dictionary mapping the name of each PDB structure
    pdb_directory : str
        The name of the directory to store the retrieved files.

    Returns
    -------
    pdb_paths : list
        A list containing the paths to the retrieved PDB files.
    """

    pdb_paths = []

    pdbl = PDB.PDBList()

    # Create directory to store files
    if not os.path.exists(pdb_directory):
        os.mkdir(pdb_directory)

    # Convert to list if only a string is given
    if isinstance(pdb_codes, str):
        pdb_codes = [pdb_codes]

    if isinstance(pdb_codes, list):
        # Iterate pdb codes
        for code in pdb_codes:
            # Download PDB file
            if names == None:
                output_file = pdb_directory+'/'+code.upper()+'.pdb'
            else:
                output_file = pdb_directory+'/'+names[code.upper()]+'.pdb'

            if not os.path.exists(output_file):
                pdbl.retrieve_pdb_file(code, file_format='pdb', pdir=pdb_directory)
            else: # If file already exists
                print('Structure exists: '+code.upper()+'.pdb')
                if names != None:
                    print('It was named as: '+names[code.upper()]+'.pdb')
                pdb_paths.append(output_file)

    for f in os.listdir(pdb_directory):
        if f.endswith('.ent'):
            # Rename file
            if names == None:
                reanamed_file = pdb_directory+'/'+f.replace('pdb','').upper().replace('.ENT','.pdb')
            else:
                reanamed_file = pdb_directory+'/'+names[f.replace('pdb','').upper().replace('.ENT','')]+'.pdb'
            os.rename(pdb_directory+'/'+f, reanamed_file)
            # Append path
            pdb_paths.append(reanamed_file)

    # Remove unnecesary folders created by Bio.PDB method
    shutil.rmtree('obsolete')

    return pdb_paths

def renumberResidues(structure, by_chain=False):
    """
    Renumber residues in a structure object starting from one. Two methods are possible:
    if by_chain is set to True the renumbering is restarted at the begining of every
    chain.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        Input structure object
    by_chain : bool
        Whether each chain should be renumerated from one.

    Returns
    -------

    structure_copy : Bio.PDB.Structure
        Renumbered copy of the input structure
    """

    count = 0
    structure_copy = PDB.Structure.Structure(0)
    model = PDB.Model.Model(0)
    auxiliar = PDB.Chain.Chain(0)

    for chain in structure.get_chains():
        new_chain = PDB.Chain.Chain(chain.id)
        if by_chain:
            count = 0
        for residue in chain.get_residues():
            count += 1
            residue.set_parent(auxiliar)
            residue.id = (residue.id[0], count, residue.id[2])
            new_chain.add(residue)
        model.add(new_chain)
    structure_copy.add(model)

    return structure_copy

def chainsAsStructure(chains):
    """
    This method creates a new Structure object containing only the given chains.

    Parameters
    ----------
    chains : list or Bio.PDB.Chain
        Chain or chains to be added to the new structure object.

    Returns
    -------
    structure : Bio.PDB.Structure
    """

    if not isinstance(chains, list):
        chains = [chains]

    structure = PDB.Structure.Structure(0)
    model = PDB.Model.Model(0)
    for chain in chains:
        model.add(chain)
    structure.add(model)

    return structure
