import xml.etree.ElementTree as ElementTree

class movers:

    class fastRelax:

        def __init__(self, name="fastRelax", scorefxn=None, disable_design=False, movemap_factory=None,
                     task_operations=None, optimize_jumps=None, task_factory=False, packer_palette=False,
                     repeats=5, relaxscript=False, cst_file=False, batch=False,
                     cartesian=False, dualspace=False, ramp_down_constraints=False,
                     bondangle=False, bondlength=False, min_type=False, movemaps=None,
                     delete_virtual_residues_after_FastRelax=False):

            self.type = 'mover'
            self.name = name
            self.repeats = repeats
            self.scorefxn = scorefxn
            self.movemap_dict = movemaps
            self.optimize_jumps = optimize_jumps
            self.task_operations = task_operations
            self.movemap_factory = movemap_factory

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('FastRelax')
            self.root.set('name', self.name)
            self.root.set('repeats', str(self.repeats))

            if self.scorefxn != None:
                if isinstance(self.scorefxn, str):
                    self.root.set('scorefxn', self.scorefxn)
                else:
                    self.root.set('scorefxn', self.scorefxn.name)

            if self.task_operations != None:
                self.root.set('task_operations', ','.join([t.name for t in self.task_operations]))

            if self.movemap_factory != None:
                self.root.set('movemap_factory', self.movemap_factory.name)

            if self.movemap_dict != None:

                if not isinstance(self.movemap_dict, dict):
                    raise ValueError('movemaps must be given as a special dictionary')

                self.movemap = self.xml.SubElement(self.root, 'MoveMap')

                self.spans = {}
                for i in sorted(self.movemap_dict):
                    self.spans[i] = self.xml.SubElement(self.movemap, 'Span')
                    self.spans[i].set('begin', self.movemap_dict[i]['begin'])
                    self.spans[i].set('end', self.movemap_dict[i]['end'])
                    self.spans[i].set('chi', self.movemap_dict[i]['chi'])
                    self.spans[i].set('bb', self.movemap_dict[i]['bb'])

                if self.optimize_jumps != None:
                    if isinstance(self.optimize_jumps, int):
                        self.jump = self.xml.SubElement(self.movemap, 'Jump')
                        self.jump.set('number', str(self.optimize_jumps))
                        self.jump.set('setting', 'true')
                    elif isinstance(self.optimize_jump, list):
                        self.jump = {}
                        for i,j in enumerate(self.optimize_jump):
                            self.jump[i] = self.xml.SubElement(self.movemap, 'Jump')
                            self.jump[i].set('number', str(j))
                            self.jump[i].set('setting', 'true')

    class packRotamersMover:

        def __init__(self, name='packRotamersMover', nloop=1, scorefxn=None, task_operations=None, packer_palette=None):

            self.type = 'mover'
            self.name = name
            self.nloop = nloop
            self.scorefxn = scorefxn
            self.task_operations = task_operations
            self.packer_palette = packer_palette

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('PackRotamersMover')
            self.root.set('name', self.name)
            if self.scorefxn != None:
                self.root.set('scorefxn', self.scorefxn.name)
            if self.task_operations != None:
                self.root.set('task_operations', ','.join([t.name for t in self.task_operations]))

    class minMover:

        def __init__(self, name="minMover", jump=None, abs_score_convergence_threshold=None,
                     max_iter=200, min_type='lbfgs_armijo_nonmonotone', tolerance=0.01,
                     cartesian=False, bondangle=False, bondlength=False, chi=False,
                     bb=True, omega=True, bb_task_operations=None, chi_task_operations=None,
                     bondangle_task_operations=None, bondlength_task_operations=None,
                     movemap_factory=None, scorefxn=None):

            types = ['linmin', 'dfpmin', 'dfpmin_armijo', 'lbfgs_armijo_nonmonotone']
            if min_type not in types:
                raise ValuError('Incorrect minimizer. Accepted types: '+' '.join((types)))

            self.type = 'mover'
            self.name = name
            self.jump = jump
            self.abs_score_convergence_threshold = abs_score_convergence_threshold
            self.max_iter = max_iter
            self.min_type = min_type
            self.tolerance = tolerance
            self.cartesian = cartesian
            self.bondangle = bondangle
            self.bondlength = bondlength
            self.chi = chi
            self.bb = bb
            self.omega = omega
            self.bb_task_operations = bb_task_operations
            self.chi_task_operations = chi_task_operations
            self.bondangle_task_operations = bondangle_task_operations
            self.bondlength_task_operations = bondlength_task_operations
            self.movemap_factory = movemap_factory
            self.scorefxn = scorefxn

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('MinMover')
            self.root.set('name', self.name)
            self.root.set('type', self.min_type)
            self.root.set('max_iter', str(self.max_iter))
            self.root.set('tolerance', str(self.tolerance))
            self.root.set('bb', str(int(self.bb)))
            self.root.set('omega', str(int(self.omega)))
            if self.cartesian:
                self.root.set('cartesian', str(int(self.cartesian)))
            if self.bondangle:
                self.root.set('bondangle', str(int(self.bondangle)))
            if self.bondlength:
                self.root.set('bondlength', str(int(bself.ondlength)))
            if self.chi:
                self.root.set('chi', str(int(self.chi)))
            if self.jump != None:
                self.root.set('jump', str(self.jump))
            if self.abs_score_convergence_threshold != None:
                self.root.set('abs_score_convergence_threshold', self.abs_score_convergence_threshold)
            if self.bb_task_operations != None:
                self.root.set('bb_task_operations', ','.join([t.name for t in self.bb_task_operations]))
            if self.chi_task_operations != None:
                self.root.set('chi_task_operations', ','.join([t.name for t in self.chi_task_operations]))
            if self.bondangle_task_operations != None:
                self.root.set('bondangle_task_operations', ','.join([t.name for t in self.bondangle_task_operations]))
            if self.bondlength_task_operations != None:
                self.root.set('bondlength_task_operations', ','.join([t.name for t in self.bondlength_task_operations]))

    class parsedProtocol:

        def __init__(self, name, movers, mode='sequence', report_filter_at_end=False):

            self.type = 'mover'
            self.name = name
            self.movers = movers
            self.report_filter_at_end = report_filter_at_end

            modes = ['sequence', 'random_order', 'single_random']
            if mode not in modes:
                raise ValueError('mode not found in permited modes: '+str(modes))
            self.mode = mode

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('ParsedProtocol')
            self.root.set('name', self.name)
            self.xml.movers = {}
            self.root.set('mode', self.mode)
            for i in range(len(self.movers)):

                # Given as pair (mover, filter)
                if isinstance(self.movers[i], tuple) or isinstance(self.movers[i], list):
                    if len(self.movers[i]) == 2:
                        if not isinstance(self.movers[i][0], type(None)) and self.movers[i][0].type == 'mover':
                            self.xml.movers[i] = self.xml.SubElement(self.root, 'Add')
                            self.xml.movers[i].set('mover', self.movers[i][0].name)
                        elif isinstance(self.movers[i][0], type(None)):
                            self.xml.movers[i] = self.xml.SubElement(self.root, 'Add')
                        else:
                            raise ValuError('When given as pairs the order is (mover, filter)!')
                        if self.movers[i][1].type == 'filter':
                            self.xml.movers[i].set('filter_name', self.movers[i][1].name)
                            if not self.report_filter_at_end:
                                self.xml.movers[i].set('report_at_end', str(int(self.report_filter_at_end)))
                        else:
                            raise ValuError('When given as pairs the order is (mover, filter)!')
                    else:
                        raise ValuError('More than two elements were given!')
                else:
                    self.xml.movers[i] = self.xml.SubElement(self.root, 'Add')
                    self.xml.movers[i].set('mover', self.movers[i].name)

    class genericMonteCarlo:

        def __init__(self, name, mover=None, trials=100, temperature=1.0, preapply=False, scorefxn=None, recover_low=None):

            self.type = 'mover'
            self.name = name
            self.mover = mover
            self.trials = trials
            self.temperature = temperature
            self.preapply = preapply
            self.scorefxn = scorefxn
            self.recover_low = recover_low

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('GenericMonteCarlo')
            self.root.set('name', self.name)
            self.root.set('mover_name', self.mover.name)
            self.root.set('trials', str(self.trials))
            self.root.set('temperature', str(self.temperature))
            if self.scorefxn != None:
                self.root.set('scorefxn_name', self.scorefxn.name)
            if self.recover_low != None:
                self.root.set('recover_low', str(self.recover_low))

    class loopOver:
        def __init__(self, name='loopOver', mover=None ,filter_name=None, iterations=1, drift=True, ms_whenfail=None):
            self.type = 'mover'
            self.name = name
            self.mover = mover
            self.filter_name = filter_name
            self.iterations = str(iterations)
            self.drift = str(drift).lower()
            self.ms_whenfail = ms_whenfail

            if self.mover == None:
                raise ValueError('You need to specify a mover to loop over. Set the option mover=mover_object')

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('LoopOver')
            self.root.set('name', self.name)
            self.root.set('drift', self.drift)
            self.root.set('iterations', self.iterations)
            self.root.set('mover_name', self.mover.name)
            if self.filter_name != None:
                self.root.set('filter_name', self.filter_name)
            if self.ms_whenfail != None:
                self.root.set('ms_whenfail', self.ms_whenfail)

    class peptideStubMover:

        def __init__(self, name='peptideStubMover', repeat=1):

            self.type = 'mover'
            self.name = name
            self.residues = []
            self.repeat = repeat

        def setResidues(self, residue_list, action=None):

            if action == None or action not in  ['prepend', 'append', 'insert']:
                raise ValueError('No action given for adding residues. Actions can be: \'prepend\', \'append\' or \'insert\'')

            if isinstance(residue_list, list):
                for i in residue_list:
                    if len(i) != 2:
                        raise ValueError('The list of residues must contain tuples of the form (residue_number, residue_name)')
                    self.residues.append((i,action))
            else:
                raise ValueError('The list of residues must contain tuples of the form (residue_number, residue_name)')

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('PeptideStubMover')
            self.root.set('name', self.name)

            if self.residues != {}:
                self.addResidue = {}
                for r in self.residues:
                    if r[1] == 'prepend':
                        self.addResidue[r] = self.xml.SubElement(self.root, 'Prepend')
                    elif r[1] == 'append':
                        self.addResidue[r] = self.xml.SubElement(self.root, 'Append')
                    elif r[1] == 'insert':
                        self.addResidue[r] = self.xml.SubElement(self.root, 'Insert')
                    self.addResidue[r].set('anchor_rsd', str(r[0][0]))
                    self.addResidue[r].set('resname', r[0][1])
                    self.addResidue[r].set('repeat', str(self.repeat))

    class declareBond:

        def __init__(self, name='addBond', atom1=None, atom2=None, res1=None, res2=None):

            if res1 == None:
                raise ValueError('res1 == None, you need to give the residue index of the first residue to bond!')
            if res2 == None:
                raise ValueError('res2 == None, you need to give the residue index of the second residue to bond!')
            if atom1 == None:
                raise ValueError('atom1 == None, you need to specify the atom name of residue 1!')
            if atom2 == None:
                raise ValueError('atom2 == None, you need to specify the atom name of residue 2!')

            self.type = 'mover'
            self.name = name
            self.atom1 = str(atom1)
            self.atom2 = str(atom2)
            self.res1 = str(res1)
            self.res2 = str(res2)

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('DeclareBond')
            self.root.set('name', self.name)
            self.root.set('atom1', self.atom1)
            self.root.set('atom2', self.atom2)
            self.root.set('res1', self.res1)
            self.root.set('res2', self.res2)

    class generalizedKic:

        def __init__(self, name, selector=None, selector_scorefunction=None,
                     perturber=None, filter_type=None, perturbation_angle=None,
                     closure_attempts=100, stop_when_n_solutions_found=100):

            self.type = 'mover'
            self.name = name
            self.selector = selector
            self.selector_scorefunction = selector_scorefunction
            self.closure_attempts = closure_attempts
            self.stop_when_n_solutions_found = stop_when_n_solutions_found
            self.added_perturbers = {}
            self.filters = {}
            self.close_bonds = {}

        def addKICResidues(self, residues_list, pivot_residues=None, pivot_atoms=None):

            self.kic_residues = residues_list
            if pivot_residues == None:
                self.pivot_residues = [residues_list[0], residues_list[int(len(residues_list)/2)], residues_list[-1]]
            else:
                self.pivot_residues = pivot_residues
            if pivot_atoms == None:
                self.pivot_atoms = ['CA','CA','CA']
            else:
                self.pivot_atoms = pivot_atoms

        def addPerturber(self, perturber):

            if self.added_perturbers == {}:
                n = 0
            else:
                n = max(list(self.added_perturbers.keys()))+1
            self.added_perturbers[n] = perturber

        def addCloseBond(self, close_bond):

            if self.close_bonds == {}:
                n = 0
            else:
                n = max(list(self.close_bonds.keys()))+1

            self.close_bonds[n] = close_bond

        def addFilter(self, filter_type):

            if self.filters == {}:
                n = 0
            else:
                n = max(list(self.filters.keys()))+1

            self.filters[n] = filter_type

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('GeneralizedKIC')
            self.root.set('name', self.name)
            self.root.set('closure_attempts', str(self.closure_attempts))
            self.root.set('stop_when_n_solutions_found', str(self.stop_when_n_solutions_found))
            self.root.set('selector', self.selector)
            self.root.set('selector_scorefunction', self.selector_scorefunction)

            self.residues = {}
            for i in range(len(self.kic_residues)):
                self.residues[i] = self.xml.SubElement(self.root, 'AddResidue')
                self.residues[i].set('res_index', str(self.kic_residues[i]))

            self.pivots = self.xml.SubElement(self.root, 'SetPivots')
            self.pivots.set('res1', str(self.pivot_residues[0]))
            self.pivots.set('res2', str(self.pivot_residues[1]))
            self.pivots.set('res3', str(self.pivot_residues[2]))
            self.pivots.set('atom1', str(self.pivot_atoms[0]))
            self.pivots.set('atom2', str(self.pivot_atoms[1]))
            self.pivots.set('atom3', str(self.pivot_atoms[2]))

            if self.added_perturbers != {}:
                for i in self.added_perturbers:
                    self.added_perturbers[i].generateXml()
                    self.root.append(self.added_perturbers[i].root)

            if self.close_bonds != {}:
                for i in self.close_bonds:
                    self.close_bonds[i].generateXml()
                    self.root.append(self.close_bonds[i].root)

            if self.filters != {}:
                for i in self.filters:
                    f = self.xml.SubElement(self.root,'AddFilter')
                    f.set('type', self.filters[i])

        class pertubers:

            class setDihedrals:

                def __init__(self, residues1, atoms1, residues2, atoms2, value):

                    self.effect = 'set_dihedral'
                    self.residues1 = residues1
                    self.atoms1 = atoms1
                    self.residues2 = residues2
                    self.atoms2 = atoms2
                    self.value =value

                def generateXml(self):

                    self.xml = ElementTree
                    self.root = self.xml.Element('AddPerturber')
                    self.root.set('effect', self.effect)
                    self.xml.dihedrals = {}
                    for i in range(len(self.residues1)):
                        self.xml.dihedrals[i] = self.xml.SubElement(self.root, 'AddAtoms')
                        self.xml.dihedrals[i].set('res1', str(self.residues1[i]))
                        self.xml.dihedrals[i].set('atom1', str(self.atoms1[i]))
                        self.xml.dihedrals[i].set('res2', str(self.residues2[i]))
                        self.xml.dihedrals[i].set('atom2', str(self.atoms2[i]))
                    self.xml.value = self.xml.SubElement(self.root, 'AddValue')
                    self.xml.value.set('value', str(self.value))

            class perturbDihedrals:

                def __init__(self, residues1, atoms1, residues2, atoms2, value):

                    self.effect = 'perturb_dihedral'
                    self.residues1 = residues1
                    self.atoms1 = atoms1
                    self.residues2 = residues2
                    self.atoms2 = atoms2
                    self.value =value

                def generateXml(self):

                    self.xml = ElementTree
                    self.root = self.xml.Element('AddPerturber')
                    self.root.set('effect', self.effect)
                    self.xml.dihedrals = {}
                    for i in range(len(self.residues1)):
                        self.xml.dihedrals[i] = self.xml.SubElement(self.root, 'AddAtoms')
                        self.xml.dihedrals[i].set('res1', str(self.residues1[i]))
                        self.xml.dihedrals[i].set('atom1', str(self.atoms1[i]))
                        self.xml.dihedrals[i].set('res2', str(self.residues2[i]))
                        self.xml.dihedrals[i].set('atom2', str(self.atoms2[i]))
                    self.xml.value = self.xml.SubElement(self.root, 'AddValue')
                    self.xml.value.set('value', str(self.value))

            class randomizeBackboneByRamaPrepro:

                def __init__(self, residues):

                    self.effect = 'randomize_backbone_by_rama_prepro'
                    self.residues = residues

                def generateXml(self):

                    self.xml = ElementTree
                    self.root = self.xml.Element('AddPerturber')
                    self.root.set('effect', self.effect)
                    self.xml.residues = {}
                    for i in range(len(self.residues)):
                        self.xml.residues[i] = self.xml.SubElement(self.root, 'AddResidue')
                        self.xml.residues[i].set('index', str(self.residues[i]))

        class closeBond:

            def __init__(self, res1=None, res2=None, atom1=None, atom2=None, bondlength=1.328685, angle1=121.699997, angle2=116.199993, torsion=180.0):

                if res1 == None:
                    raise ValueError('res1 == None, you need to give the residue index of the first residue to close bond!')
                if res2 == None:
                    raise ValueError('res2 == None, you need to give the residue index of the second residue to close bond!')
                if atom1 == None:
                    raise ValueError('atom1 == None, you need to specify the atom name of residue 1!')
                if atom2 == None:
                    raise ValueError('atom2 == None, you need to specify the atom name of residue 2!')

                self.res1 = str(res1)
                self.res2 = str(res2)
                self.atom1 = str(atom1)
                self.atom2 = str(atom2)
                self.bondlength = str(bondlength)
                self.angle1 = str(angle1)
                self.angle2 = str(angle2)
                self.torsion = str(torsion)

            def generateXml(self):

                self.xml = ElementTree
                self.root = self.xml.Element('CloseBond')
                self.root.set('res1', self.res1)
                self.root.set('res2', self.res2)
                self.root.set('atom1', self.atom1)
                self.root.set('atom2', self.atom2)
                self.root.set('bondlength', self.bondlength)
                self.root.set('angle1', self.angle1)
                self.root.set('angle2', self.angle2)
                self.root.set('torsion', self.torsion)

    class deleteRegionMover:

        def __init__(self, name='DeleteRegionMover', residue_selector=None, start=None, end=None, rechain=None):

            if residue_selector == None and (start == None or end == None):
                raise ValueError('No information for deleting residues. "start" and "end" tags, or a residue selector must be given')
            elif residue_selector != None and (start != None or end != None):
                raise ValueError('Only "start" and "end", or a residue selector must be given, not both')
            elif (start == None and end != None) or (start != None and end == None):
                raise ValueError('"start" and "end" tags must be provided together')

            self.type = 'mover'
            self.name = name
            self.residue_selector = residue_selector
            self.start = start
            self.end = end
            self.rechain = rechain

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('DeleteRegionMover')
            self.root.set('name', str(self.name))
            if self.residue_selector != None:
                self.root.set('residue_selector', str(self.residue_selector.name))
            if self.start != None:
                self.root.set('start', str(self.start))
                self.root.set('end', str(self.end))

            if self.rechain != None:
                self.root.set('rechain', str(self.rechain))

    class savePoseMover:

        def __init__(self, name="savePoseMover", restore_pose=True, reference_name='reference_pose', pdb_file=None):

            self.type = 'mover'
            self.name = name
            self.restore_pose = restore_pose
            self.reference_name = reference_name
            self.pdb_file = pdb_file

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('SavePoseMover')
            self.root.set('name', str(self.name))
            self.root.set('restore_pose', str(int(self.restore_pose)))
            self.root.set('reference_name', str(self.reference_name))
            if self.pdb_file != None:
                self.root.set('pdb_file', str(self.pdb_file))

    class keepRegionMover:

        def __init__(self, name="keepRegionMover", start=None, end=None):

            if start == None or end == None:
                raise ValueError('You must give a starting and ending pdb number')

            self.type = 'mover'
            self.name = name
            self.start = start
            self.end = end

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('KeepRegionMover')
            self.root.set('name', str(self.name))
            self.root.set('start', str(self.start))
            self.root.set('end', str(self.end))

    class insertPoseIntoPoseMover:

        def __init__(self, name="insertPoseIntoPoseMover", start_res_num=None, end_res_num=None, copy_pdbinfo=False, spm_reference_name=None):

            if spm_reference_name == None:
                raise ValueError('spm_reference_name = None. You must give a pose to insert!')
            if start_res_num == None or end_res_num == None:
                raise ValueError('You must give a starting and ending residue number')

            self.type = 'mover'
            self.name = name
            self.start = start
            self.end = end
            self.copy_pdbinfo = copy_pdbinfo
            self.spm_reference_name = spm_reference_name

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('InsertPoseIntoPoseMover')
            self.root.set('name', str(self.name))
            self.root.set('start_res_num', str(self.start))
            self.root.set('end_res_num', str(self.end))
            if self.copy_pdbinfo == True:
                self.root.set('copy_pdbinfo', str(self.copy_pdbinfo).lower())
            self.root.set('spm_reference_name', str(self.spm_reference_name))

    class replaceRegionMover:

        def __init__(self, name="replaceRegionMover", src_pose_start=None, target_pose_start=None, span=None, spm_reference_name=None):

            if spm_reference_name == None:
                raise ValueError('spm_reference_name = None. You must give a pose to insert!')
            if src_pose_start == None:
                raise ValueError('You must indicate the starting residue of the reference pose.')
            if target_pose_start == None:
                raise ValueError('You must indicate the starting residue of the target pose.')
            if span == None:
                raise ValueError('You must indicate the span of the inserted region.')

            self.type = 'mover'
            self.name = name
            self.src_pose_start = src_pose_start
            self.target_pose_start = target_pose_start
            self.span = span
            self.spm_reference_name = spm_reference_name

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('ReplaceRegionMover')
            self.root.set('name', str(self.name))
            self.root.set('src_pose_start', str(self.src_pose_start))
            self.root.set('target_pose_start', str(self.target_pose_start))
            self.root.set('span', str(self.span))
            self.root.set('spm_reference_name', str(self.spm_reference_name))

    class addChain:

        def __init__(self, name="addChain", file_name=None, new_chain=True, scorefxn=None,
                    random_access=False, swap_chain_number=False, spm_reference_name=None,
                    update_PDBInfo=True):

            if file_name == None and spm_reference_name == None:
                raise ValueError('You must give a pose or pdb file to use this mover.')

            if file_name != None and spm_reference_name != None:
                raise ValueError('You must give a pose or pdb file to use this mover. Not Both.')

            self.type = 'mover'
            self.name = name
            self.file_name = file_name
            self.new_chain = new_chain
            self.scorefxn = scorefxn
            self.random_access = random_access
            self.swap_chain_number = swap_chain_number
            self.spm_reference_name = spm_reference_name
            self.update_PDBInfo = update_PDBInfo

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('AddChain')
            self.root.set('name', str(self.name))
            if self.scorefxn != None:
                self.root.set('scorefxn', str(self.scorefxn))
            if self.file_name != None:
                self.root.set('file_name', str(self.file_name))
            if self.spm_reference_name != None:
                self.root.set('spm_reference_name', str(self.spm_reference_name))
            self.root.set('new_chain', str(int(self.new_chain)))
            self.root.set('random_access', str(int(self.random_access)))
            self.root.set('swap_chain_number', str(int(self.swap_chain_number)))
            self.root.set('update_PDBInfo', str(int(self.update_PDBInfo)))

    class dumpPdb:

        def __init__(self, name='dumpPdb', file_name=None, scorefxn=None, tag_time=False):

            if file_name == None:
                raise ValueError('file_name = None. You need to give a pdb file name to write the current pose')

            self.type = 'mover'
            self.name = name
            self.file_name = file_name
            self.scorefxn = scorefxn
            self.tag_time = tag_time

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('DumpPdb')
            self.root.set('name', self.name)
            self.root.set('fname', self.file_name)
            if self.scorefxn != None:
                self.root.set('scorefxn', self.scorefxn)
            if self.tag_time:
                self.root.set('tag_time', str(int(self.tag_time)))

    class switchResidueTypeSetMover:

        def __init__(self, name='switch_residue_type', atom_type_set=None):
            if atom_type_set == None:
                raise ValueError('atom_type_set = None. You need to give an atom type set: centroid, centroid_rot, fa_standard...')

            self.type = 'mover'
            self.name = name
            self.atom_type_set = atom_type_set

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('SwitchResidueTypeSetMover')
            self.root.set('name', self.name)
            self.root.set('set', self.atom_type_set)

    class prepareForCentroid:
        def __init__(self, name='PrepareForCentroid'):
            self.type = 'mover'
            self.name = name
        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('PrepareForCentroid')
            self.root.set('name', self.name)

    class scoreMover:
        def __init__(self, name='scoreMover', scorefxn=None, verbose=True):

            if scorefxn == None:
                raise ValueError('You must give a scorefunction touse this mover')
            self.type = 'mover'
            self.name = name
            self.scorefxn = scorefxn
            self.verbose = verbose

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('ScoreMover')
            self.root.set('name', self.name)
            if self.scorefxn != None:
                if isinstance(self.scorefxn, str):
                    self.root.set('scorefxn', self.scorefxn)
                else:
                    self.root.set('scorefxn', self.scorefxn.name)
            self.root.set('verbose', str(int(self.verbose)))

    class mutate:
        def __init__(self, name='MutateResidue', target_residue=None, new_residue=None):

            if target_residue == None:
                raise ValueError('target_residue = None. You need to give a target residue to mutate.')
            if new_residue == None:
                raise ValueError('new_residue = None. You need to give a new residue identity to mutate to.')
            self.type = 'mover'
            self.name = name
            self.target_residue = str(target_residue)
            self.new_residue = new_residue

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('MutateResidue')
            self.root.set('name', self.name)
            self.root.set('target', self.target_residue)
            self.root.set('new_res', self.new_residue)


    class ForceDisulfides:
        def __init__(self, name='ForceDisulfides', disulfides=None, remove_existing=False, repack=True,scorefxn="ref2015"):
        

            if disulfides == None:
                raise ValueError('disulfides = None. You need to give a colon separator respair cs list')
            
            self.type = 'mover'
            self.name = name
            self.disulfides = str(disulfides)
            self.remove_existing = str(remove_existing).lower()
            self.repack = str(repack).lower()
            self.scorefxn = scorefxn
            

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('ForceDisulfides')
            self.root.set('name', self.name)
            self.root.set('disulfides', self.disulfides)
            self.root.set('repack', self.repack)
            self.root.set('remove_existing', self.remove_existing)
            self.root.set('scorefxn', self.scorefxn)
            
            
    class atomTree:

        def __init__(self, name='AtomTree', fold_tree_file=None):

            if fold_tree_file == None:
                raise ValueError('You need to give a path to the fold tree file')
            self.type = 'mover'
            self.name = name
            self.fold_tree_file = fold_tree_file

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('AtomTree')
            self.root.set('name', self.name)
            self.root.set('fold_tree_file', self.fold_tree_file)

    class ddG:

        def __init__(self, name='ddG', jump=1, per_residue_ddg=0, repack_bound=None, repack_unbound=None,
                     relax_bound=None, relax_unbound=None, relax_mover=None, scorefxn=None, chain_num=None,
                     chain_name=None, filter=None):
            self.type = 'mover'
            self.name = name
            self.jump = jump
            self.per_residue_ddg = per_residue_ddg

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('ddG')
            self.root.set('name', self.name)
            if self.jump != 1:
                self.root.set('jump', str(self.jump))
            if self.per_residue_ddg != 0:
                self.root.set('per_residue_ddg', str(self.per_residue_ddg))

    class idealize:

        def __init__(self, name='idealize'):
            self.type = 'mover'
            self.name = name

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('Idealize')
            self.root.set('name', self.name)

    class flexPepDock:

        def __init__(self, name="flexPepDock", min_only=False, pep_refine=False, lowres_abinitio=False,
                     peptide_chain=None, receptor_chain=None, ppk_only=False, scorefxn=None,
                     extra_scoring=False):

            self.type = 'mover'
            self.name = name
            self.min_only = min_only
            self.pep_refine = pep_refine
            self.lowres_abinitio = lowres_abinitio
            self.peptide_chain = peptide_chain
            self.receptor_chain = receptor_chain
            self.ppk_only = ppk_only
            self.scorefxn = scorefxn
            self.extra_scoring = extra_scoring

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('FlexPepDock')
            self.root.set('name', self.name)
            if self.peptide_chain != None:
                self.root.set('peptide_chain', self.peptide_chain)
            if self.receptor_chain != None:
                self.root.set('receptor_chain', self.receptor_chain)
            if self.scorefxn != None:
                if isinstance(self.scorefxn, str):
                    self.root.set('scorefxn', self.scorefxn)
                else:
                    self.root.set('scorefxn', self.scorefxn.name)
            if self.min_only:
                self.root.set('min_only', 'true')
            if self.pep_refine:
                self.root.set('pep_refine', 'true')
            if self.lowres_abinitio:
                self.root.set('lowres_abinitio', 'true')
            if self.ppk_only:
                self.root.set('ppk_only', 'true')
            if self.extra_scoring:
                self.root.set('extra_scoring', 'true')

    class null:

        def __init__(self, name='null'):
            self.type = 'mover'
            self.name = name

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('Null')
            self.root.set('name', self.name)


    class dockingProtocol:

        def __init__(self, name="dockingProtocol", docking_score_low=None, docking_score_high=None,
                     ignore_default_docking_task=False, task_operations=None, partners=None,
                     low_res_protocol_only=False, docking_local_refine=False, dock_min=False):

            self.type = 'mover'
            self.name = name
            self.docking_score_low = docking_score_low
            self.docking_score_high = docking_score_high
            self.ignore_default_docking_task = ignore_default_docking_task
            self.task_operations = task_operations
            self.partners = partners
            self.low_res_protocol_only = low_res_protocol_only
            self.docking_local_refine = docking_local_refine
            self.dock_min = dock_min

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('DockingProtocol')
            self.root.set('name', self.name)
            if self.docking_score_low != None:
                self.root.set('docking_score_low', self.docking_score_low)
            if self.docking_score_high != None:
                self.root.set('docking_score_high', self.docking_score_high)
            if self.task_operations != None:
                self.root.set('task_operations', self.task_operations)
            if self.partners != None:
                self.root.set('partners', self.partners)
            if self.ignore_default_docking_task:
                self.root.set('ignore_default_docking_task', str(self.ignore_default_docking_task))
            if self.low_res_protocol_only:
                self.root.set('low_res_protocol_only', str(self.low_res_protocol_only))
            if self.docking_local_refine:
                self.root.set('docking_local_refine', str(self.docking_local_refine))
            if self.dock_min:
                self.root.set('dock_min', str(self.dock_min))

    class singleFragmentMover:

        def __init__(self, name="singleFragmentMover", fragments=None, policy="uniform"):
            self.type = 'mover'
            self.name = name
            self.fragments = fragments
            self.policy = policy
            self.movemap = None

            if self.fragments == None:
                raise ValueError('You need to specify the fragment containg file path with the fragments="path_to_file" option.')

        def addMoveMapSpan(self, begin=None, end=None, chi=True, bb=True):

            # Create movemap dictionary to store spans if does not exists.
            if self.movemap == None:
                self.movemap = {}

            # Check for previous span objects indexes
            if 0 not in self.movemap:
                i = 0
            else:
                i = max(list(self.movemap.keys()))+1

            # Create move map span entry in dictionary
            self.movemap[i] = {}
            self.movemap[i]['begin'] = str(begin)
            self.movemap[i]['end'] = str(end)
            self.movemap[i]['chi'] = str(chi).lower()
            self.movemap[i]['bb'] = str(bb).lower()

            if self.movemap[i]['begin'] == None:
                raise ValueError('Specify begin option for movemap!')
            if self.movemap[i]['end'] == None:
                raise ValueError('Specify end option for movemap!')

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('SingleFragmentMover')
            self.root.set('name', self.name)
            self.root.set('fragments', self.fragments)
            self.root.set('policy', self.policy)
            if self.movemap != None:
                spans = {}
                mm = self.xml.SubElement(self.root, 'MoveMap')
                for i in self.movemap:
                    spans[i] = self.xml.SubElement(mm, 'Span')
                    spans[i].set('begin', self.movemap[i]['begin'])
                    spans[i].set('end', self.movemap[i]['end'])
                    spans[i].set('chi', self.movemap[i]['chi'])
                    spans[i].set('bb', self.movemap[i]['bb'])

    class interfaceAnalyzerMover:

        def __init__(self, name="interfaceAnalyzerMover", scorefxn=None, pack_separated=False,
                     pack_input=False, resfile=False, packstat=False, interface_sc=False,
                     tracer=False, use_jobname=False, fixedchains=None, interface=None,
                     ligandchain=None, jump=None, scorefile_reporting_prefix=None):

            self.type = 'mover'
            self.name = name
            self.scorefxn = scorefxn
            self.pack_separated = pack_separated
            self.pack_input = pack_input
            self.resfile = resfile
            self.packstat = packstat
            self.interface_sc = interface_sc
            self.tracer = tracer
            self.use_jobname = use_jobname
            self.fixedchains = fixedchains
            self.interface = interface
            self.ligandchain = ligandchain
            self.jump = jump
            self.scorefile_reporting_prefix = scorefile_reporting_prefix

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('InterfaceAnalyzerMover')
            self.root.set('name', self.name)

            if self.scorefxn != None:
                self.root.set('scorefxn', self.scorefxn)

    class rigidBodyPerturbNoCenterMover:

        def __init__(self, name='rigidBodyPerturbNoCenter', rot_mag=0.1, trans_mag=0.4):

            self.type = 'mover'
            self.name = name
            self.rot_mag = rot_mag
            self.trans_mag = trans_mag

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('RigidBodyPerturbNoCenter')
            self.root.set('name', self.name)
            self.root.set('rot_mag', str(self.rot_mag))
            self.root.set('trans_mag', str(self.trans_mag))

    class shear:

        def __init__(self, name='shear', residue_selector=None, scorefxn=None,
        temperature=0.5, nmoves=1, angle_max=6.0, preserve_detailed_balance=False):

            self.type = 'mover'
            self.name = name
            self.residue_selector = residue_selector
            self.scorefxn = scorefxn
            self.temperature = temperature
            self.nmoves = nmoves
            self.angle_max = angle_max
            self.preserve_detailed_balance = preserve_detailed_balance

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('Shear')
            self.root.set('name', self.name)
            if self.residue_selector != None:
                self.root.set('residue_selector', self.residue_selector.name)
            if self.scorefxn != None:
                self.root.set('scorefxn', self.scorefxn)
            self.root.set('temperature', str(self.temperature))
            self.root.set('nmoves', str(self.nmoves))
            self.root.set('angle_max', str(self.angle_max))
            if self.preserve_detailed_balance:
                self.root.set('preserve_detailed_balance', str(1.0))

    class small:

        def __init__(self, name='small', residue_selector=None, scorefxn=None,
        temperature=0.5, nmoves=1, angle_max=6.0, preserve_detailed_balance=False):

            self.type = 'mover'
            self.name = name
            self.residue_selector = residue_selector
            self.scorefxn = scorefxn
            self.temperature = temperature
            self.nmoves = nmoves
            self.angle_max = angle_max
            self.preserve_detailed_balance = preserve_detailed_balance

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('Small')
            self.root.set('name', self.name)
            if self.residue_selector != None:
                self.root.set('residue_selector', self.residue_selector.name)
            if self.scorefxn != None:
                self.root.set('scorefxn', self.scorefxn)
            self.root.set('temperature', str(self.temperature))
            self.root.set('nmoves', str(self.nmoves))
            self.root.set('angle_max', str(self.angle_max))
            if self.preserve_detailed_balance:
                self.root.set('preserve_detailed_balance', str(1.0))

    class constraintSetMover:

        def __init__(self, name="constraintSetMover", add_constraints=False, cst_file=None):

            self.type = 'mover'
            self.name = name
            self.add_constraints = add_constraints
            self.cst_file = cst_file

            if self.cst_file == None:
                raise ValueError('You must give the path to the constraint file')

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('ConstraintSetMover')
            self.root.set('name', self.name)
            self.root.set('cst_file', self.cst_file)
            if self.add_constraints:
                self.root.set('add_constraints', str(self.add_constraints))

    class atomCoordinateCstMover:

        def __init__(self, name="atomCoordinateCstMover", coord_dev=0.5, bounded=False,
                     bound_width=0.0, sidechain=False, native=False, task_operations=None,
                     func_groups=False):

            self.type = 'mover'
            self.name = name
            self.coord_dev = coord_dev
            self.bounded = bounded
            self.bound_width = bound_width
            self.sidechain = sidechain
            self.native = native
            self.task_operations = task_operations
            self.func_groups = func_groups

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('AtomCoordinateCstMover')
            self.root.set('name', self.name)
            self.root.set('coord_dev', str(self.coord_dev))
            self.root.set('bound_width', str(self.bound_width))
            if self.task_operations != None:
                if isinstance(self.task_operations, str):
                    self.task_operations = [self.task_operations]
                self.root.set('task_operations', ','.join(self.task_operations))
            if self.bounded:
                self.root.set('bounded', str(int(self.bounded)))
            if self.sidechain:
                self.root.set('sidechain', str(int(self.sidechain)))
            if self.native:
                self.root.set('native', str(int(self.native)))
            if self.func_groups:
                self.root.set('func_groups', str(int(self.func_groups)))

    class cstInfoMover:

        def __init__(self, name='cstInfoMover', cst_file=None, dump_cst_file=None,
                           prefix='CST', recursive=False):

            self.type = 'mover'
            self.name = name
            self.cst_file = cst_file
            self.dump_cst_file = dump_cst_file
            self.prefix = prefix
            self.recursive = recursive

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('CstInfoMover')
            self.root.set('name', self.name)
            self.root.set('cst_file', self.cst_file)
            self.root.set('prefix', self.prefix)
            if self.dump_cst_file != None:
                self.root.set('dump_cst_file', self.dump_cst_file)
            if self.recursive:
                self.root.set('recursive', str(int(self.recursive)))

    class clearConstraintsMover:

        def __init__(self, name="clearConstraints"):

            self.type = 'mover'
            self.name = name

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('ClearConstraintsMover')
            self.root.set('name', self.name)

    class virtualRoot:

        def __init__(self, name="virtualRoot", removable=False, remove=False):

            self.type = 'mover'
            self.name = name
            self.removable = removable
            self.remove = remove

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('VirtualRoot')
            self.root.set('name', self.name)
            if self.removable:
                self.root.set('removable', str(int(self.removable)))
            if self.remove:
                self.root.set('remove', str(int(self.remove)))

    class genericMonteCarlo:

        def __init__(self, name='genericMonteCarlo', mover=None, scorefxn_name=None,
                     scorefxn=None, trials=10, sample_type='low', temperature=0, drift=True,
                     recover_low=True, stopping_condition=None, preapply=True, saved_accept_file_name=None,
                     saved_trial_file_name=None):

            self.type = 'mover'
            self.name = name
            self.mover = mover
            self.scorefxn_name = scorefxn_name
            self.scorefxn = scorefxn
            self.trials = trials
            self.sample_type = sample_type
            self.temperature = temperature
            self.drift = drift
            self.recover_low = recover_low
            self.stopping_condition = stopping_condition
            self.preapply = preapply
            self.saved_accept_file_name = saved_accept_file_name
            self.saved_trial_file_name = saved_trial_file_name

            if self.mover == None:
                raise ValueError('You must give a mover to use the genericMonteCarlo mover.')

            if self.scorefxn_name == None:
                if self.scorefxn == None:
                    raise ValueError('You must give a scorefunction with scorefxn_name or scorefxn options.')
            else:
                if self.scorefxn == None:
                    if not isinstance(self.scorefxn_name, str):
                        raise ValueError('Wrong scorefxn_name input. It must be a string')
                if self.scorefxn != None:
                    raise ValueError('You must give only one scorefunction with scorefxn_name or scorefxn options.')

            sample_types = ['low', 'high']
            if self.sample_type not in sample_types:
                raise ValueError('sample_type must be "low" or "high"')

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('GenericMonteCarlo')
            self.root.set('name', self.name)
            self.root.set('mover_name', self.mover.name)
            if self.scorefxn_name != None:
                self.root.set('scorefxn_name', self.scorefxn_name)
            if self.scorefxn != None:
                self.root.set('scorefxn_name', self.scorefxn.name)
            self.root.set('trials', str(self.trials))
            self.root.set('sample_type', self.sample_type)
            self.root.set('temperature', str(self.temperature))
            self.root.set('drift', str(int(self.drift)))
            self.root.set('recover_low', str(int(self.recover_low)))
            self.root.set('preapply', str(int(self.preapply)))
            self.root.set('preapply', str(int(self.preapply)))
            if self.stopping_condition != None:
                self.root.set('stopping_condition', self.stopping_condition)
            if self.saved_trial_file_name != None:
                self.root.set('saved_trial_file_name', self.saved_trial_file_name)
            if self.saved_accept_file_name != None:
                self.root.set('saved_accept_file_name', self.saved_accept_file_name)

    class runSimpleMetrics:

        def __init__(self, name='runSimpleMetrics', metrics=[], prefix=None, suffix=None,
                     override=False):

            self.type = 'mover'
            self.name = name
            self.metrics = metrics
            self.prefix = prefix
            self.suffix = suffix
            self.override = override

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('RunSimpleMetrics')
            self.root.set('name', self.name)
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
            if isinstance(self.metrics, list):
                self.root.set('metrics', ','.join(self.metrics))
            if self.prefix != None:
                self.root.set('prefix', self.prefix)
            if self.suffix != None:
                self.root.set('suffix', self.suffix)
            if self.override:
                self.root.set('override', str(int(self.override)))

    class pyMOLMover:

        def __init__(self, name='pyMOLMover', keep_history=False):

            self.type = 'mover'
            self.name = name
            self.keep_history = keep_history

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('PyMOLMover')
            self.root.set('name', self.name)
            if self.keep_history:
                self.root.set('keep_history', str(int(self.keep_history)))

    class saveAndRetrieveSidechains:

        def __init__(self, name="saveAndRetrieveSidechains", allsc=False,
                     multi_use=False, two_step=False, jumpid=1, reference_name=None):

            self.type = 'mover'
            self.name = name
            self.allsc = allsc
            self.multi_use = multi_use
            self.two_step = two_step
            self.jumpid = jumpid
            self.reference_name = reference_name

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('SaveAndRetrieveSidechains')
            self.root.set('name', self.name)
            self.root.set('jumpid', str(self.jumpid))

            if self.allsc:
                self.root.set('allsc', str(int(self.allsc)))
            if self.multi_use:
                self.root.set('multi_use', str(int(self.multi_use)))
            if self.two_step:
                self.root.set('two_step', str(int(self.two_step)))
            if self.reference_name != None:
                self.root.set('reference_name', self.reference_name)

class rosetta_MP:

    class movers:

        class addMembraneMover:

            def __init__(self, name="addMembraneMover", spanfile=None):

                self.type = 'mover'
                self.name = name
                self.spanfile = spanfile

            def generateXml(self):
                self.xml = ElementTree
                self.root = self.xml.Element('AddMembraneMover')
                self.root.set('name', self.name)
                if self.spanfile != None:
                    self.root.set('spanfile', self.spanfile)

        class membranePositionFromTopologyMover:

            def __init__(self, name="init", spanfile=None, structure_based=False):

                self.type = 'mover'
                self.name = name
                self.structure_based = structure_based
                self.spanfile = spanfile

            def generateXml(self):
                self.xml = ElementTree
                self.root = self.xml.Element('MembranePositionFromTopologyMover')
                self.root.set('name', self.name)
                if self.structure_based:
                    self.root.set('structure_based', str(int(self.structure_based)))
                if self.spanfile != None:
                    self.root.set('spanfile', self.spanfile)

class topologyBroker:

    class environment:

        def __init__(self, name='environment', auto_cut=1):
            self.type = 'mover'
            self.name = name
            self.auto_cut = auto_cut
            self.registered = []
            self.apply = []

        def registerMover(self, mover):

            if isinstance(mover, str):
                self.registered.append(mover)
            elif mover.type == 'mover':
                self.registered.append(mover.name)

        def applyMover(self, mover):
            if isinstance(mover, str):
                self.apply.append(mover)
            elif mover.type == 'mover':
                self.apply.append(mover.name)

        def generateXml(self):
            self.xml = ElementTree
            self.root = self.xml.Element('Environment')
            self.root.set('name', self.name)
            self.root.set('auto_cut', str(self.auto_cut))
            if self.registered != []:
                for mover in self.registered:
                    register = self.xml.SubElement(self.root, 'Register')
                    register.set('mover', mover)
            if self.apply != []:
                for mover in self.apply:
                    apply = self.xml.SubElement(self.root, 'Apply')
                    apply.set('mover', mover)

    class movers:

        class fragmentCM:

            def __init__(self, name='fragment', frag_type="classic", fragments=None,
                        selector=None,  initialize=True):

                if fragments == None:
                    raise ValueError('You must sepecify which fragments to use.')
                if selector == None:
                    raise ValueError('You need to specify a residue selector to \
apply the fragment insertions.')


                self.type = 'mover'
                self.name = name
                self.frag_type = frag_type
                self.fragments = fragments
                self.selector = selector
                self.initialize = initialize

            def generateXml(self):
                self.xml = ElementTree
                self.root = self.xml.Element('FragmentCM')
                self.root.set('name', self.name)
                self.root.set('frag_type', self.frag_type)
                self.root.set('fragments', self.fragments)
                if self.selector != None:
                    self.root.set('selector', self.selector)
                if self.initialize != None:
                    self.root.set('initialize', str(self.initialize).lower())

        class rigidChunkCM:

            def __init__(self, name="rigidChunk", template="INPUT", region_selector=None,
                         selector=None, apply_to_template=None):

                self.type = 'mover'
                self.name = name
                self.template = template
                self.region_selector = region_selector
                self.selector = selector
                self.apply_to_template = apply_to_template

            def generateXml(self):
                self.xml = ElementTree
                self.root = self.xml.Element('RigidChunkCM')
                self.root.set('name', self.name)
                self.root.set('template', self.template)
                if self.region_selector != None:
                    self.root.set('region_selector', self.region_selector)
                if self.selector != None:
                    self.root.set('selector', self.selector)
                if self.apply_to_template != None:
                    self.root.set('apply_to_template', self.apply_to_template)

        class abscriptMover:

            def __init__(self, name="abscript", cycles=2):

                self.type = 'mover'
                self.name = name
                self.cycles = cycles
                self.frags = False
                self.large_frags = None
                self.small_frags = None
                self.stages = []
                self.movers = []

            def addFragments(self, large_frags=None, small_frags=None):
                if large_frags ==None or small_frags == None:
                    raise ValueError('You must provide fragment files with the options\
                    large_frags and small_frags.')

                self.frags = True
                self.large_frags = large_frags
                self.small_frags = small_frags

            def addStage(self, stage_ids=None, mover_name=None):
                if stage_ids == None:
                    raise ValueError('You must provide an abinitio stage id(s) (e.g. "I-IVb").\n\
                    Possible values: I, II, IIIa, IIIb, IVa, IVb')
                elif mover_name == None:
                    raise ValueError('You must provide a mover name to apply to the given stage(s) (e.g. Jump)')

                self.stages.append(stage_ids)
                self.movers.append(mover_name)

            def generateXml(self):
                self.xml = ElementTree
                self.root = self.xml.Element('AbscriptMover')
                self.root.set('name', self.name)
                self.root.set('cycles', str(self.cycles))
                if self.frags:
                    frags = self.xml.SubElement(self.root, 'Fragments')
                if self.large_frags != None:
                    frags.set('large_frags', self.large_frags)
                if self.small_frags != None:
                    frags.set('small_frags', self.small_frags)
                if self.stages != []:
                    for i, stage in enumerate(self.stages):
                        stg_i = self.xml.SubElement(self.root, 'Stage')
                        stg_i.set('ids', stage)
                        mover_i = self.xml.SubElement(stg_i, 'Mover')
                        mover_i.set('name', self.movers[i])

        class abscriptLoopCloserCM():

            def __init__(self, name='fragment', fragments=None, selector=None):

                if fragments == None:
                    raise ValueError('You must sepecify which fragments to use.')
                if selector == None:
                    raise ValueError('You need to specify a residue selector to \
apply the fragment insertions.')

                self.type = 'mover'
                self.name = name
                self.fragments = fragments
                self.selector = selector

            def generateXml(self):
                self.xml = ElementTree
                self.root = self.xml.Element('AbscriptLoopCloserCM')
                self.root.set('name', self.name)
                self.root.set('fragments', self.fragments)
                if self.selector != None:
                    self.root.set('selector', self.selector)
