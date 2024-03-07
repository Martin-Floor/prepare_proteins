import xml.etree.ElementTree as ElementTree

class filters:

    class ddg:

        def __init__(self, name="ddg", scorefxn=None, threshold=None, jump=1, confidence=1.0,
                     chain_num=None, repeats=None, repack=None, relax_mover=None,
                     repack_bound=None, repack_unbound=None, relax_bound=None,
                     relax_unbound=None, filter=None, extreme_value_removal=None):

            self.type = 'filter'
            self.name = name
            self.scorefxn = scorefxn
            self.threshold = threshold
            self.jump = jump
            self.confidence = confidence
            self.chain_num = chain_num
            self.repeats = repeats
            self.repack = repack
            self.relax_mover = relax_mover
            self.repack_bound = repack_bound
            self.repack_unbound = repack_unbound
            self.relax_bound = relax_bound
            self.relax_unbound = relax_unbound
            self.filter = filter
            self.extreme_value_removal = extreme_value_removal

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('Ddg')
            self.root.set('name', str(self.name))
            self.root.set('jump', str(self.jump))
            if self.confidence != 1.0:
                self.root.set('confidence', str(self.confidence))
    
    class DisulfideEntropy:

        def __init__(self, name="entropy", tightness=0,lower_bound=0):

            self.type = 'filter'
            self.name = name
            self.tightness = tightness
            self.lower_bound = lower_bound
     
            self.filter = filter


        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('DisulfideEntropy')
            self.root.set('name', str(self.name))
            self.root.set('tightness', str(self.tightness))
            self.root.set('lower_bound', str(self.lower_bound))


    class sasa:

        def __init__(self, name="sasa_filter", threshold=800, upper_threshold=1000000,
                     hydrophobic=False, polar=False, jump=1, sym_dof_names="", confidence=1.0):

            self.type = 'filter'
            self.name = name
            self.confidence = confidence
            self.jump = jump
            self.threshold = threshold
            self.upper_threshold = upper_threshold
            self.hydrophobic = hydrophobic
            self.polar = polar
            self.jump = jump
            self.sym_dof_names =sym_dof_names

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('Sasa')
            self.root.set('name', str(self.name))
            self.root.set('jump', str(self.jump))
            if self.confidence != 1.0:
                self.root.set('confidence', str(self.confidence))
            if self.hydrophobic:
                self.root.set('hydrophobic', str(1.0))
            if self.polar:
                self.root.set('polar', str(1.0))

    class rmsd:

        def __init__(self, name='rmsd', reference_name=None, symmetry=False,
                     chains=None, superimpose_on_all=False, superimpose=True,
                     threshold=5, by_aln=False, aln_files=None, template_names=None,
                     query_names=None, rms_residues_from_pose_cache=None, confidence=1.0):

            self.type = 'filter'
            self.name = name
            self.reference_name = reference_name
            self.symmetry = symmetry
            self.chains = chains
            self.superimpose_on_all = superimpose_on_all
            self.superimpose = superimpose
            self.threshold = threshold
            self.by_aln = by_aln
            self.aln_files = aln_files
            self.template_names = template_names
            self.query_names = query_names
            self.rms_residues_from_pose_cache = rms_residues_from_pose_cache
            self.confidence = confidence

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('Rmsd')
            self.root.set('name', str(self.name))
            self.root.set('threshold', str(self.threshold))
            self.root.set('superimpose', str(int(self.superimpose)))
            if self.confidence != 1.0:
                self.root.set('confidence', str(self.confidence))
            if self.symmetry:
                self.root.set('symmetry', str(int(self.symmetry)))
            if self.superimpose_on_all:
                self.root.set('superimpose_on_all', str(int(self.superimpose_on_all)))
            if self.by_aln:
                self.root.set('by_aln', str(int(self.by_aln)))
            if self.aln_files != None:
                self.root.set('aln_files', self.aln_files)
            if self.reference_name != None:
                self.root.set('reference_name', self.reference_name)
            if self.chains != None:
                self.root.set('chains', self.chains)
            if self.aln_files != None:
                self.root.set('aln_files', self.aln_files)
            if self.template_names != None:
                self.root.set('template_names', self.template_names)
            if self.query_names != None:
                self.root.set('query_names', self.query_names)
            if self.rms_residues_from_pose_cache != None:
                self.root.set('rms_residues_from_pose_cache', self.rms_residues_from_pose_cache)

    class time:

        def __init__(self, name='Time'):

            self.type = 'filter'
            self.name = name

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('Time')
            self.root.set('name', self.name)

    class scoreType:

        def __init__(self, name='scoreType', score_type='total_score', threshold=None,
                     scorefxn=None, confidence=1.0):

            self.type = 'filter'
            self.name = name
            self.score_type = score_type
            self.threshold = threshold
            self.scorefxn = scorefxn
            self.confidence = confidence

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('ScoreType')
            self.root.set('name', self.name)
            self.root.set('score_type', self.score_type)
            self.root.set('confidence', str(self.confidence))
            if self.threshold != None:
                self.root.set('threshold', str(self.threshold))
            if self.scorefxn != None:
                self.root.set('scorefxn', self.scorefxn.name)

    class torsion:

        def __init__(self, name='torsion', torsion=None, lower=0, upper=0, resnum=None,
                     task_operations=None, confidence=1.0):

            self.type = 'filter'
            self.name = name
            self.lower = lower
            self.upper = upper
            self.resnum = resnum
            self.torsion = torsion
            self.task_operations = task_operations
            self.confidence = confidence

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('Torsion')
            self.root.set('name', self.name)
            self.root.set('lower', str(self.lower))
            self.root.set('upper', str(self.upper))
            self.root.set('confidence', str(self.confidence))
            if self.resnum != None:
                self.root.set('resnum', str(self.resnum))
            if self.torsion != None:
                self.root.set('torsion', self.torsion)
            if self.task_operations != None:
                self.root.set('task_operations', self.task_operations)
