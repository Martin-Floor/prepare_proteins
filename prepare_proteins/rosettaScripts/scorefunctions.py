import xml.etree.ElementTree as ElementTree

class scorefunctions:

    class new_scorefunction:

        def __init__(self, name, weights_file=None):
            self.name = name
            self.weights_file = weights_file
            self.reweights = {}

        def addReweight(self, scoretype, weight):

            self.reweights[scoretype] = weight

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('ScoreFunction')
            self.root.set('name', self.name)
            if self.weights_file != None:
                self.root.set('weights', str(self.weights_file))

            if self.reweights != {}:
                self.xml.reweights = {}
                for scoretype in self.reweights:
                    self.xml.reweights[scoretype] = self.xml.SubElement(self.root, 'Reweight')
                    self.xml.reweights[scoretype].set('scoretype', scoretype)
                    self.xml.reweights[scoretype].set('weight', str(self.reweights[scoretype]))
