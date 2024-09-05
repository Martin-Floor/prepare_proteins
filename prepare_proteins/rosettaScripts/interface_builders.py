import xml.etree.ElementTree as ElementTree

class interfaceBuilder:

    def __init__(self, name, ligand_areas, extension_window=None):
        self.name = name
        if hasattr(ligand_areas, 'name'):
            ligand_areas = [ligand_areas]
        self.ligand_areas = ligand_areas
        self.extension_window = extension_window

    def generateXml(self):
        self.xml = ElementTree
        self.root = self.xml.Element('InterfaceBuilder')
        self.root.set('name', self.name)
        self.root.set('ligand_areas', ','.join([a.name for a in self.ligand_areas]))
        if self.extension_window:
            self.root.set('extension_window', str(self.extension_window))
