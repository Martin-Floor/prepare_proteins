from Bio import PDB

class notHydrogen(PDB.Select):
    def accept_atom(self, atom):
        """
        Verify if atom is not Hydrogen.
        """
        _hydrogen = re.compile("[123 ]*H.*")
        name = atom.get_id()
        if _hydrogen.match(name):
            return 0
        else:
            return 1

class notWater(PDB.Select):
    def accept_residue(self, residue):
        """
        Verify if residue is water.
        """
        _restype = residue.id[0]
        if _restype == 'W':
            return 0
        else:
            return 1

class onlyProtein(PDB.Select):

    def accept_residue(self, residue):
        """
        Verify if residues are protein.
        """
        _restype = residue.id[0]
        if _restype != ' ':
            return 0
        else:
            return 1
