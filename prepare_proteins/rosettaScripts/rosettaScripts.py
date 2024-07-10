import xml.etree.ElementTree as ElementTree
from xml.dom import minidom
from collections import OrderedDict
from .selectors import residueSelectors

import xml.etree.ElementTree as ElementTree
from xml.dom import minidom
from collections import OrderedDict
import pickle

class xmlScript:

    def __init__(self):
        self.root = ElementTree.Element('ROSETTASCRIPTS')
        self.scorefxns = {}
        self.scorefxns['root'] = ElementTree.SubElement(self.root, 'SCOREFXNS')
        self.residueSelectors = {}
        self.residueSelectors['root'] = ElementTree.SubElement(self.root, 'RESIDUE_SELECTORS')
        self.jumpSelectors = {}
        self.jumpSelectors['root'] = ElementTree.SubElement(self.root, 'JUMP_SELECTORS')
        self.taskOperations = {}
        self.taskOperations['root'] = ElementTree.SubElement(self.root, 'TASKOPERATIONS')
        self.moveMapFactories = {}
        self.moveMapFactories['root'] = ElementTree.SubElement(self.root, 'MOVE_MAP_FACTORIES')
        self.ligandAreas = {}
        self.ligandAreas['root'] = ElementTree.SubElement(self.root, 'LIGAND_AREAS')
        self.interfaceBuilders = {}
        self.interfaceBuilders['root'] = ElementTree.SubElement(self.root, 'INTERFACE_BUILDERS')
        self.movemapBuilders = {}
        self.movemapBuilders['root'] = ElementTree.SubElement(self.root, 'MOVEMAP_BUILDERS')
        self.scoringgrids = {}
        self.scoringgrids['root'] = ElementTree.SubElement(self.root, 'SCORINGGRIDS')
        self.simpleMetrics = {}
        self.simpleMetrics['root'] = ElementTree.SubElement(self.root, 'SIMPLE_METRICS')
        self.filters = {}
        self.filters['root'] = ElementTree.SubElement(self.root, 'FILTERS')
        self.movers = OrderedDict()
        self.movers['root'] = ElementTree.SubElement(self.root, 'MOVERS')
        self.protocols = OrderedDict()
        self.protocols['root'] = ElementTree.SubElement(self.root, 'PROTOCOLS')
        self.output = ElementTree.SubElement(self.root, 'OUTPUT')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['root'] = ElementTree.tostring(self.root).decode('utf-8')  # Convert root to string for pickling
        return state

    def __setstate__(self, state):
        root_str = state.pop('root')
        self.root = ElementTree.fromstring(root_str.encode('utf-8'))  # Convert string back to Element
        self.__dict__.update(state)
        # Reinitialize the root-dependent dictionaries
        self.scorefxns['root'] = self.root.find('SCOREFXNS')
        self.residueSelectors['root'] = self.root.find('RESIDUE_SELECTORS')
        self.jumpSelectors['root'] = self.root.find('JUMP_SELECTORS')
        self.taskOperations['root'] = self.root.find('TASKOPERATIONS')
        self.moveMapFactories['root'] = self.root.find('MOVE_MAP_FACTORIES')
        self.ligandAreas['root'] = self.root.find('LIGAND_AREAS')
        self.interfaceBuilders['root'] = self.root.find('INTERFACE_BUILDERS')
        self.movemapBuilders['root'] = self.root.find('MOVEMAP_BUILDERS')
        self.scoringgrids['root'] = self.root.find('SCORINGGRIDS')
        self.simpleMetrics['root'] = self.root.find('SIMPLE_METRICS')
        self.filters['root'] = self.root.find('FILTERS')
        self.movers['root'] = self.root.find('MOVERS')
        self.protocols['root'] = self.root.find('PROTOCOLS')
        self.output = self.root.find('OUTPUT')

    def addResidueSelector(self, residue_selector):
        self.residueSelectors[residue_selector.name] = residue_selector
        self.residueSelectors[residue_selector.name].generateXml()
        self.residueSelectors['root'].append(self.residueSelectors[residue_selector.name].root)

    def addJumpSelector(self, jump_selector):
        self.jumpSelectors[jump_selector.name] = jump_selector
        self.jumpSelectors[jump_selector.name].generateXml()
        self.jumpSelectors['root'].append(self.jumpSelectors[jump_selector.name].root)

    def addTaskOperation(self, task_operation):
        self.taskOperations[task_operation.name] = task_operation
        self.taskOperations[task_operation.name].generateXml()
        self.taskOperations['root'].append(self.taskOperations[task_operation.name].root)

    def addMoveMapFactory(self, movemap_factory):
        self.moveMapFactories[movemap_factory.name] = movemap_factory
        self.moveMapFactories[movemap_factory.name].generateXml()
        self.moveMapFactories['root'].append(self.moveMapFactories[movemap_factory.name].root)

    def addFilter(self, rs_filter):
        self.filters[rs_filter.name] = rs_filter
        self.filters[rs_filter.name].generateXml()
        self.filters['root'].append(self.filters[rs_filter.name].root)

    def addScorefunction(self, scorefunction):
        self.scorefxns[scorefunction.name] = scorefunction
        self.scorefxns[scorefunction.name].generateXml()
        self.scorefxns['root'].append(self.scorefxns[scorefunction.name].root)

    def addLigandArea(self, ligand_area):
        self.ligandAreas[ligand_area.name] = ligand_area
        self.ligandAreas[ligand_area.name].generateXml()
        self.ligandAreas['root'].append(self.ligandAreas[ligand_area.name].root)

    def addInterfaceBuilder(self, interface_builder):
        self.interfaceBuilders[interface_builder.name] = interface_builder
        self.interfaceBuilders[interface_builder.name].generateXml()
        self.interfaceBuilders['root'].append(self.interfaceBuilders[interface_builder.name].root)

    def addMovemapBuilder(self, movemap_builder):
        self.movemapBuilders[movemap_builder.name] = movemap_builder
        self.movemapBuilders[movemap_builder.name].generateXml()
        self.movemapBuilders['root'].append(self.movemapBuilders[movemap_builder.name].root)

    def addScoringGrid(self, scoring_grid, ligand_chain=None, width=None):
        if ligand_chain:
            self.scoringgrids['root'].set('ligand_chain', ligand_chain)
        if width:
            self.scoringgrids['root'].set('width', str(width))

        self.scoringgrids[scoring_grid.grid_name] = scoring_grid
        self.scoringgrids[scoring_grid.grid_name].generateXml()
        self.scoringgrids['root'].append(self.scoringgrids[scoring_grid.grid_name].root)

    def addSimpleMetric(self, simple_metric):
        self.simpleMetrics[simple_metric.name] = simple_metric
        self.simpleMetrics[simple_metric.name].generateXml()
        self.simpleMetrics['root'].append(self.simpleMetrics[simple_metric.name].root)

    def addMover(self, mover):
        self.movers[mover.name] = mover
        self.movers[mover.name].generateXml()
        self.movers['root'].append(self.movers[mover.name].root)

    def setProtocol(self, protocol_list=None):
        if protocol_list is None:
            raise ValueError('A list of movers and filters with the execution order is necessary')

        for mover in protocol_list:
            self.protocols[mover.name] = ElementTree.SubElement(self.protocols['root'], 'Add')
            if mover.type == 'mover':
                self.protocols[mover.name].set('mover', mover.name)
            elif mover.type == 'filter':
                self.protocols[mover.name].set('filter', mover.name)
            else:
                print('Object is not registered as a mover or filter')
                print(mover)

    def addOutputScorefunction(self, scorefunction):
        self.output.set('scorefxn', scorefunction.name)

    def write_xml(self, file_name):
        xmlstr = minidom.parseString(ElementTree.tostring(self.root)).toprettyxml(indent="  ")
        with open(file_name, "w") as f:
            f.write(xmlstr)

class taskOperations:
    class operateOnResidueSubset:
        def __init__(self, name, selector, operation=None, aas=None):
            if operation == None or operation not in ['RestrictToRepackingRLT', 'PreventRepackingRLT', 'RestrictAbsentCanonicalAASRLT']:
                raise ValueError('Must define one of these operations: RestrictToRepackingRLT, PreventRepackingRLT, RestrictAbsentCanonicalAASRLT')
            self.name = name
            self.selector = selector
            self.operation = operation
            self.aas = aas
            if aas == None and operation == 'RestrictAbsentCanonicalAASRLT':
                raise ValueError('Designable amino acids must be defined for RestrictAbsentCanonicalAASRLT operation.')

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('OperateOnResidueSubset')
            self.root.set('name', self.name)
            self.root.set('selector', self.selector)
            if self.aas != None:
                self.root.set('aas', self.aas)
            self.xml.SubElement(self.root, self.operation)

    class restrictToRepacking:
        def __init__(self, name):
            self.name = name

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('RestrictToRepacking')
            self.root.set('name', self.name)

    class preventResiduesFromRepacking:
        def __init__(self, name, residues):
            self.name = name
            self.residues = residues

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('PreventResiduesFromRepacking')
            self.root.set('name', self.name)
            self.root.set('residues', ','.join(self.residues))

    class DetectProteinLigandInterface:
        def __init__(self, name, cut1=6.0, cut2=8.0, cut3=10.0, cut4=12.0, design=True, catres_interface=True):
            self.name = name
            self.cut1 = cut1
            self.cut2 = cut2
            self.cut3 = cut3
            self.cut4 = cut4
            self.design = design
            self.catres_interface = catres_interface

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('DetectProteinLigandInterface')
            self.root.set('name', self.name)
            self.root.set('cut1', str(self.cut1))
            self.root.set('cut2', str(self.cut2))
            self.root.set('cut3', str(self.cut3))
            self.root.set('cut4', str(self.cut4))
            self.root.set('design', str(int(self.design)))
            self.root.set('catres_interface', str(int(self.catres_interface)))

    class RestrictAbsentCanonicalAAS:
        def __init__(self, name, resnum=0, keep_aas=None):
            self.name = name
            self.resnum = resnum
            self.keep_aas = keep_aas

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('RestrictAbsentCanonicalAAS')
            self.root.set('name', self.name)
            self.root.set('resnum', str(self.resnum))
            self.root.set('keep_aas', self.keep_aas)

    class extraRotamersGeneric:
        def __init__(self, name, ex1=1, ex2=1, ex1_sample_level=1, ex2_sample_level=1):
            self.name = name
            self.ex1 = ex1
            self.ex2 = ex2
            self.ex1_sample_level = ex1_sample_level
            self.ex2_sample_level = ex2_sample_level

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('ExtraRotamersGeneric')
            self.root.set('name', self.name)
            self.root.set('ex1', str(self.ex1))
            self.root.set('ex2', str(self.ex2))
            self.root.set('ex1_sample_level', str(self.ex1_sample_level))
            self.root.set('ex2_sample_level', str(self.ex2_sample_level))

    class ReadResfile:
        def __init__(self, name='ReadResfile', filename=None, selector=None):
            self.name = name
            if filename == None:
                raise ValueError('File must be given to ReadResfile task operation.')
            self.filename = filename
            self.selector = selector

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('ReadResfile')
            self.root.set('name', self.name)
            self.root.set('filename', self.filename)
            if self.selector != None:
                self.root.set('selector', self.selector)

class taskOperations:

    class operateOnResidueSubset:

        def __init__(self, name, selector, operation=None, aas=None):


            if operation == None or operation not in ['RestrictToRepackingRLT', 'PreventRepackingRLT', 'RestrictAbsentCanonicalAASRLT']:
                raise ValueError('Must define one of these operations: RestrictToRepackingRLT, PreventRepackingRLT, RestrictAbsentCanonicalAASRLT')

            self.name = name
            self.selector = selector
            self.operation = operation
            self.aas = aas
            if aas == None and operation == 'RestrictAbsentCanonicalAASRLT':
                raise ValueError('Designable amino acids must be defined for RestrictAbsentCanonicalAASRLT operation.')

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('OperateOnResidueSubset')
            self.root.set('name', self.name)
            self.root.set('selector', self.selector)
            if self.aas != None:
                self.root.set('aas', self.aas)
            self.xml.SubElement(self.root, self.operation)

    class restrictToRepacking:

        def __init__(self, name):

            self.name = name

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('RestrictToRepacking')
            self.root.set('name', self.name)

    class preventResiduesFromRepacking:

        def __init__(self, name, residues):

            self.name = name
            self.residues = residues

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('PreventResiduesFromRepacking')
            self.root.set('name', self.name)
            self.root.set('residues', ','.join(self.residues))

    class DetectProteinLigandInterface:

        def __init__(self, name, cut1=6.0, cut2=8.0, cut3=10.0, cut4=12.0, design=True, catres_interface=True):

            self.name = name
            self.cut1 = cut1
            self.cut2 = cut2
            self.cut3 = cut3
            self.cut4 = cut4
            self.design = design
            self.catres_interface = catres_interface

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('DetectProteinLigandInterface')
            self.root.set('name', self.name)
            self.root.set('cut1', str(self.cut1))
            self.root.set('cut2', str(self.cut2))
            self.root.set('cut3', str(self.cut3))
            self.root.set('cut4', str(self.cut4))
            self.root.set('design', str(int(self.design)))
            self.root.set('catres_interface', str(int(self.catres_interface)))

    class RestrictAbsentCanonicalAAS:

        def __init__(self, name, resnum=0, keep_aas=None):
            self.name = name
            self.resnum = resnum
            self.keep_aas = keep_aas

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('RestrictAbsentCanonicalAAS')
            self.root.set('name', self.name)
            self.root.set('resnum', str(self.resnum))
            self.root.set('keep_aas', self.keep_aas)


    class extraRotamersGeneric:

        def __init__(self, name, ex1=1, ex2=1, ex1_sample_level=1, ex2_sample_level=1):

            self.name = name
            self.ex1 = ex1
            self.ex2 = ex2
            self.ex1_sample_level = ex1_sample_level
            self.ex2_sample_level = ex2_sample_level

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('ExtraRotamersGeneric')
            self.root.set('name', self.name)
            self.root.set('ex1', str(self.ex1))
            self.root.set('ex2', str(self.ex2))
            self.root.set('ex1_sample_level', str(self.ex1_sample_level))
            self.root.set('ex2_sample_level', str(self.ex2_sample_level))


            #<ReadResfile name="(&string)" filename="(&string)" selector="(&string)" />

    class ReadResfile:

        def __init__(self, name='ReadResfile', filename=None, selector=None):

            self.name = name
            if filename == None:
                raise ValuError('File must be given to ReadResfile task operation.')
            self.filename = filename
            self.selector = selector

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('ReadResfile')
            self.root.set('name', self.name)
            self.root.set('filename', self.filename)
            if self.selector != None:
                self.root.set('selector', self.selector)
