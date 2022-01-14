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
