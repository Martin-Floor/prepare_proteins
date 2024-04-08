import xml.etree.ElementTree as ElementTree

class simpleMetrics:

    class totalEnergyMetric:

        def __init__(self, name='totalEnergyMetric', reference_name=None, residue_selector=None, use_native=False,
                     scoretype='total_score', scorefxn=None, custom_type=None):

            self.name = name
            self.reference_name = reference_name
            self.residue_selector = residue_selector
            self.use_native = use_native
            self.scoretype = scoretype
            self.scorefxn = scorefxn
            self.custom_type = custom_type

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('TotalEnergyMetric')
            self.root.set('name', self.name)
            self.root.set('scoretype', self.scoretype)
            if self.reference_name != None:
                self.root.set('reference_name', self.reference_name)
            if self.residue_selector != None:
                self.root.set('residue_selector', self.residue_selector)
            if self.custom_type != None:
                self.root.set('custom_type', self.custom_type)
            if self.scorefxn != None:
                if isinstance(self.scorefxn, str):
                    self.root.set('scorefxn', self.scorefxn)
                else:
                    self.root.set('scorefxn', self.scorefxn.name)
            if self.use_native:
                self.root.set('use_native', str(int(use_native)))

    class RMSDMetric:

        def __init__(self, name='RMSDMetric', custom_type=None, reference_name=None,
                    residue_selector=None, residue_selector_ref=None, residue_selector_super=None,
                    residue_selector_super_ref=None, robust=True, use_native=False, super=False,
                    rmsd_type=None):

            self.name = name
            self.custom_type = custom_type
            self.reference_name = reference_name
            self.residue_selector = residue_selector
            self.residue_selector_ref = residue_selector_ref
            self.residue_selector_super = residue_selector_super
            self.residue_selector_super_ref = residue_selector_super_ref
            self.robust = robust
            self.use_native = use_native
            self.super = super
            self.rmsd_type = rmsd_type

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('RMSDMetric')
            self.root.set('name', self.name)
            if self.custom_type != None:
                self.root.set('custom_type', self.custom_type)
            if self.residue_selector != None:
                self.root.set('residue_selector', self.residue_selector)
            if self.residue_selector_ref != None:
                self.root.set('residue_selector_ref', self.residue_selector_ref)
            if self.residue_selector_super != None:
                self.root.set('residue_selector_super', self.residue_selector_super)
            if self.residue_selector_super_ref != None:
                self.root.set('residue_selector_super_ref', self.residue_selector_super_ref)
            if self.rmsd_type != None:
                self.root.set('rmsd_type', self.rmsd_type)
            if self.robust:
                self.root.set('robust', str(int(self.robust)))
            if self.use_native:
                self.root.set('use_native', str(int(self.use_native)))
            if self.super:
                self.root.set('super', str(int(self.super)))

    
