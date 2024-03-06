import xml.etree.ElementTree as ElementTree
from .selectors import residueSelectors
from .selectors import jumpSelectors

class moveMapFactory:

    def __init__(self, name, bb=False, chi=False, nu=False, branches=False, jumps=False, cartesian=False):

        self.name = name
        self.bb = bb
        self.chi = chi
        self.nu = nu
        self.branches = branches
        self.jumps = jumps
        self.cartesian = cartesian
        self.operations = {}
        self.operations['bb'] = []
        self.operations['chi'] = []
        self.operations['nu'] = []
        self.operations['branches'] = []
        self.operations['jumps'] = []

    def addBackboneOperation(self, enable=True, residue_selector=None, bb_tor_index=None):

        if residue_selector == None:
            raise ValueError('residue_selector = None, you must provide the name of a Residue Selector in order to add a backbone operation.')

        bb_operation = {}
        bb_operation['enable'] = enable
        if isinstance(residue_selector, str):
            bb_operation['residue_selector'] = residue_selector
        else:
            bb_operation['residue_selector'] = residue_selector.name
        bb_operation['bb_tor_index'] = bb_tor_index
        self.operations['bb'].append(bb_operation)

    def addChiOperation(self, enable=True, residue_selector=None):

        if residue_selector == None:
            raise ValueError('residue_selector = None, you must provide the name of a Residue Selector in order to add a chi operation.')

        chi_operation = {}
        chi_operation['enable'] = enable
        if isinstance(residue_selector, str):
            chi_operation['residue_selector'] = residue_selector
        else:
            chi_operation['residue_selector'] = residue_selector.name
        self.operations['chi'].append(chi_operation)

    def addNuOperation(self, enable=True, residue_selector=None):

        if residue_selector == None:
            raise ValueError('residue_selector = None, you must provide the name of a Residue Selector in order to add a nu operation.')

        nu_operation = {}
        nu_operation['enable'] = enable
        if isinstance(residue_selector, str):
            nu_operation['residue_selector'] = residue_selector
        else:
            nu_operation['residue_selector'] = residue_selector.name
        self.operations['nu'].append(nu_operation)

    def addBranchesOperation(self, enable=True, residue_selector=None):

        if residue_selector == None:
            raise ValueError('residue_selector = None, you must provide the name of a Residue Selector in order to add a branches operation.')

        br_operation = {}
        br_operation['enable'] = enable
        if isinstance(residue_selector, str):
            br_operation['residue_selector'] = residue_selector
        else:
            br_operation['residue_selector'] = residue_selector.anme
        self.operations['branches'].append(br_operation)

    def addJumpsOperation(self, enable=True, jump_selector=None):

        if jump_selector == None:
            raise ValueError('jump_selector = None, you must provide the name of a Jump Selector in order to add a jumps operation.')

        jmp_operation = {}
        jmp_operation['enable'] = enable
        if isinstance(jump_selector, str):
            jmp_operation['jump_selector'] = jump_selector
        else:
            jmp_operation['jump_selector'] = jump_selector.name
        self.operations['jumps'].append(jmp_operation)

    def generateXml(self):

        self.xml = ElementTree
        self.root = self.xml.Element('MoveMapFactory')
        self.root.set('name', str(self.name))
        if self.bb:
            self.root.set('bb', str(self.bb))
        if self.chi:
            self.root.set('chi', str(self.chi))
        if self.nu:
            self.root.set('nu', str(self.nu))
        if self.branches:
            self.root.set('branches', str(self.branches))
        if self.jumps:
            self.root.set('jumps', str(self.jumps))
        if self.cartesian:
            self.root.set('cartesians', str(self.cartesian))

        if self.operations['bb'] != []:
            for bbo in self.operations['bb']:
                operation = self.xml.SubElement(self.root, 'Backbone')
                operation.set('enable', str(bbo['enable']))
                operation.set('residue_selector', str(bbo['residue_selector']))
                if bbo['bb_tor_index'] != None:
                    operation.set('bb_tor_index', str(bbo['bb_tor_index']))

        if self.operations['chi'] != []:
            for chio in self.operations['chi']:
                operation = self.xml.SubElement(self.root, 'Chi')
                operation.set('enable', str(chio['enable']))
                operation.set('residue_selector', str(chio['residue_selector']))

        if self.operations['nu'] != []:
            for nuo in self.operations['nu']:
                operation = self.xml.SubElement(self.root, 'Nu')
                operation.set('enable', str(nuo['enable']))
                operation.set('residue_selector', str(nuo['residue_selector']))

        if self.operations['branches'] != []:
            for bro in self.operations['branches']:
                operation = self.xml.SubElement(self.root, 'Branches')
                operation.set('enable', str(bro['enable']))
                operation.set('residue_selector', str(bro['residue_selector']))

        if self.operations['jumps'] != []:
            for jmpo in self.operations['jumps']:
                operation = self.xml.SubElement(self.root, 'Jumps')
                operation.set('enable', str(jmpo['enable']))
                operation.set('jump_selector', str(jmpo['jump_selector']))
