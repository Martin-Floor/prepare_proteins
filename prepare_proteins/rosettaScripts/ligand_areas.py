import xml.etree.ElementTree as ElementTree

class ligandArea:

    def __init__(self, name, chain="X", cutoff=6.0, add_nbr_radius=True,
                 all_atom_mode=True, minimize_ligand=None, calpha_restraints=None):

        self.type = 'ligand_area'
        self.name = name
        self.chain = chain
        self.cutoff = cutoff
        self.add_nbr_radius = add_nbr_radius
        self.all_atom_mode = all_atom_mode
        self.minimize_ligand = minimize_ligand
        self.calpha_restraints = calpha_restraints

    def generateXml(self):

        self.xml = ElementTree
        self.root = self.xml.Element('LigandArea')
        self.root.set('name', self.name)
        self.root.set('chain', self.chain)
        self.root.set('cutoff', str(self.cutoff))
        self.root.set('add_nbr_radius', str(self.add_nbr_radius).lower())
        self.root.set('all_atom_mode', str(self.all_atom_mode).lower())
        if self.minimize_ligand:
            self.root.set('minimize_ligand', str(self.minimize_ligand).lower())
        if self.calpha_restraints:
            self.root.set('Calpha_restraints', str(self.calpha_restraints))
