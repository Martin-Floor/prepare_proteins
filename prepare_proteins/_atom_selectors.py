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

    def __init__(self, keep_residues, *args, **kwargs):
        """
        Pass a list of non-protein residues to keep.

        Parameters
        ==========
        keep_residues : list
            List of non proteins residues to keep: the list should contain tuples
            of the form (chain_id, residue_id), e.g., ('A', 203).
        """
        super(PDB.Select, self).__init__(*args, **kwargs)
        self.keep_residues = keep_residues

    def accept_residue(self, residue):
        """
        Verify if residues are protein.
        """
        _restype = residue.id[0]
        _residue_chain = residue.get_parent().id
        _residue_id = residue.id[1]
        _match_residue = (_residue_chain, _residue_id)

        if _restype != ' ':
            print(_restype, _residue_chain, _residue_id, _match_residue)
            print(self.keep_residues)
            if _match_residue in self.keep_residues:
                print('keeping residue', residue)
                return 1
            else:
                return 0
        else:
            return 1
