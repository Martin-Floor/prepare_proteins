import xml.etree.ElementTree as ElementTree

class movemapBuilder:

    def __init__(self, name, sc_interface=None, bb_interface=None, minimize_water=False):
        self.name = name
        self.sc_interface = sc_interface
        self.bb_interface = bb_interface
        self.minimize_water = minimize_water

    def generateXml(self):
        self.xml = ElementTree
        self.root = self.xml.Element('MoveMapBuilder')
        self.root.set('name', self.name)
        if self.sc_interface:
            self.root.set('sc_interface', self.sc_interface.name)
        if self.bb_interface:
            self.root.set('bb_interface', self.bb_interface.name)
        if self.minimize_water:
            self.root.set('minimize_water', str(self.minimize_water).lower())
