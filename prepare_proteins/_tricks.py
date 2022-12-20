import shutil
import os

class tricks:
    """
    Collection of useful functions to fix PDB formats that, ideally, should not be
    useful.
    """

    def getProteinLigandInputFiles(pele_folder, protein, ligand, separator='-'):
        """
        Returns the paths of the input PDB files of a PELE simulation folder
        for a specific protein and ligand.
        """

        pdb_files = []
        for d in os.listdir(pele_folder):

            if d == 'templates':
                continue

            if separator not in d:
                raise ValueError('Separator %s not found in PELE folder.' % separator)
            if d.count(separator) > 1:
                raise ValueError('Separator %s appears more than one time in the PELE folder!' % separator)

            protein_name = d.split(separator)[0]
            ligand_name = d.split(separator)[1]

            if protein == protein_name and ligand == ligand_name:
                for f in os.listdir(pele_folder+'/'+d):
                    if f.endswith('.pdb'):
                        pdb_files.append(pele_folder+'/'+d+'/'+f)

        return pdb_files

    def changeResidueAtomNames(input_pdb, residue, atom_names):
        """
        Change the atom names of a specific residue in a pdb file.

        Parameters
        ==========
        input_pdb : str
            Path to the target PDB file
        residue : tuple
            Residue to change as (chain_id, resname)
        atom_names : dict
            Mapping the old atom names to the new atom names
        """

        if not isinstance(residue, tuple):
            raise ValueError('The residue must be a two element tuple (chain, resid)')

        found = {}
        for atom in atom_names:
            found[atom] = False

        with open(input_pdb+'.tmp', 'w') as tmp:
            with open(input_pdb) as pdb:
                for l in pdb:
                    if l.startswith('ATOM') or l.startswith('HETATM'):
                        index, name, resname, chain, resid = (int(l[7:12]), l[12:16], l[17:20], l[21], int(l[22:27]))
                        if (chain, resid) == residue:
                            old_atom_name = name
                            if old_atom_name in atom_names:
                                new_atom_name = atom_names[old_atom_name]
                                if len(new_atom_name) != len(old_atom_name):
                                    raise ValueError('The new and old atom names should be the same length.')
                                found[old_atom_name] = True
                                l = l.replace(old_atom_name, new_atom_name)

                    tmp.write(l)
        shutil.move(input_pdb+'.tmp', input_pdb)
        for atom in found:
            if not found[atom]:
                print('Given atom %s was not found in residue %s' % (atom, residue))

    def displaceLigandAtomNames(input_pdb, atom, alignment='right', verbose=False):
        """
        Displace the name of the ligand atom name in the PDB.

        Parameters
        ==========
        input_pdb : str
            Path to the target PDB file
        atom : tuple
            Residue to change as (resname, atom_name)
        """
        if alignment not in ['right', 'left']:
            raise ValueError('Alignment must be either "left" or "right"')

        with open(input_pdb+'.tmp', 'w') as tmp:
            with open(input_pdb) as pdb:
                for l in pdb:
                    if l.startswith('ATOM') or l.startswith('HETATM'):
                        atom_name = l.split()[2]
                        resname = l.split()[3]
                        if (resname, atom_name) == atom:
                            if alignment == 'right':
                                if verbose:
                                    print('Changing atom name %s-%s to the right' % atom)
                                l = l.replace(atom_name+' ', ' '+atom_name)
                            elif alignment == 'left':
                                if verbose:
                                    print('Changing atom name %s-%s to the left' % atom)
                                l = l.replace(' '+atom_name, atom_name+' ')
                    tmp.write(l)
        shutil.move(input_pdb+'.tmp', input_pdb)

    def displaceResidueAtomNames(input_pdb, atom, alignment='right'):
        """
        Displace the name of the atom name of a specific residue in the PDB.

        Parameters
        ==========
        input_pdb : str
            Path to the target PDB file
        atom : tuple
            Residue to change as (resname, atom_name)
        """
        if alignment not in ['right', 'left']:
            raise ValueError('Alignment must be either "left" or "right"')

        with open(input_pdb+'.tmp', 'w') as tmp:
            with open(input_pdb) as pdb:
                for l in pdb:
                    if l.startswith('ATOM') or l.startswith('HETATM'):
                        atom_name = l.split()[2]
                        resname = l.split()[3]
                        if (resname, atom_name) == atom:
                            if alignment == 'right':
                                l = l.replace(atom_name+' ', ' '+atom_name)
                            elif alignment == 'left':
                                l = l.replace(' '+atom_name, atom_name+' ')
                    tmp.write(l)
        shutil.move(input_pdb+'.tmp', input_pdb)
