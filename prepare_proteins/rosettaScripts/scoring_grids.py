import xml.etree.ElementTree as ElementTree

class scoringGrid:

    class classicGrid:

        def __init__(self, grid_name='classicGrid', weight=1.0):
            self.grid_name = grid_name
            self.weight = weight

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('ClassicGrid')
            self.root.set('grid_name', self.grid_name)
            self.root.set('weight', str(self.weight))
