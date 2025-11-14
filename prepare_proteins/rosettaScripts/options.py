import xml.etree.ElementTree as ElementTree

class flags:

    def __init__(self, rosetta_script_xml_file, s=None, l=None, input_silent=None,
                 input_fasta=None, input_native=None, nstruct=1, out_prefix=None,
                 output_silent_file=None, output_score_file=None, no_output=False,
                 output_path=None, overwrite=False, ignore_zero_occupancy=True, centroid=False,
                 score_weights=None, score_only=False):

        self.s = s
        self.l = l
        self.inputSilentFile = input_silent
        self.input_fasta = input_fasta
        self.input_native = input_native
        self.protocol = rosetta_script_xml_file
        self.nstruct = str(nstruct)
        self.out_prefix = out_prefix
        self.output_silent_file = output_silent_file
        self.output_score_file = output_score_file
        self.no_output = no_output
        self.output_path = output_path
        self.overwrite = overwrite
        self.ignore_zero_occupancy = ignore_zero_occupancy
        self.centroid = centroid
        self.score_weights = score_weights
        self.score_only = score_only
        self.flags_files = []
        self.others = {}
        self.rosetta_db = None

        #Protocol specific
        self.relax_cst = False
        self.relax = False
        self.ligand_docking = False

    def addOption(self, option, value=None, append=False):
        if append:
            existing = self.others.get(option)
            if existing is None:
                self.others[option] = [value]
            else:
                if isinstance(existing, list):
                    existing.append(value)
                else:
                    self.others[option] = [existing, value]
        else:
            self.others[option] = value

    def add_relax_options(self):
        self.relax = True

    def add_ligand_docking_options(self):
        self.ligand_docking = True

    def add_relax_cst_options(self):
        self.relax = True
        self.relax_cst = True

    def includeFile(self, flags_file):
        self.flags_files.append(flags_file)

    def setRosettaDatabase(self, database_path):
        self.rosetta_db = database_path

    def write_flags(self, file_name):

        with open(file_name, 'w') as ff:
            ff.write('-nstruct '+self.nstruct+'\n')
            ff.write('-parser:protocol '+self.protocol+'\n')
            if self.s != None:
                ff.write('-s '+ self.s+'\n')
            elif self.l != None:
                ff.write('-l '+ self.l+'\n')
            elif self.inputSilentFile != None:
                ff.write('-in:file:silent '+ self.inputSilentFile+'\n')
            if self.input_fasta != None:
                ff.write('-in:file:fasta '+ self.input_fasta+'\n')
            if self.input_native != None:
                ff.write('-in:file:native '+ self.input_native+'\n')
            if self.rosetta_db != None:
                ff.write('-database '+ self.rosetta_db+'\n')
            if self.output_silent_file != None:
                if not self.output_silent_file.endswith('.out'):
                    self.output_silent_file = self.output_silent_file+'.out'
                ff.write('-out:file:silent '+self.output_silent_file+'\n')
                if self.output_score_file == None:
                    ff.write('-out:file:scorefile '+self.output_silent_file.replace('.out','.sc')+'\n')
                else:
                    ff.write('-out:file:scorefile '+self.output_score_file+'\n')
            elif self.output_score_file != None:
                ff.write('-out:file:scorefile '+self.output_score_file+'\n')
            if self.output_path != None:
                ff.write('-out:path:pdb '+self.output_path+'\n')
            if self.out_prefix != None:
                ff.write('-out:prefix '+self.out_prefix+'\n')
            if self.score_only:
                ff.write('-out:file:score_only\n')
            if self.no_output:
                ff.write('-out:nooutput\n')
            if self.overwrite:
                ff.write('-overwrite\n')

            #Add protocol specfic flags
            if self.relax_cst:
                ff.write('-relax:constrain_relax_to_start_coords\n')
                ff.write('-relax:coord_constrain_sidechains\n')
                ff.write('-relax:ramp_constraints false\n')

            if self.relax:
                ff.write('-ex1\n')
                ff.write('-ex2\n')
                ff.write('-use_input_sc\n')
                ff.write('-flip_HNQ\n')
                ff.write('-no_optH false\n')

            if self.ligand_docking:
                #the packing options allow Rosetta to sample additional rotamers for
                 #protein sidechain angles chi 1 (ex1) and chi 2 (ex2)
                ff.write('-ex1\n')
                ff.write('-ex2\n')
                ff.write('-ex1aro\n')
                ff.write('-ex2aro\n')
                ff.write('-no_optH false\n')
                #flip_HNQ tells Rosetta to consider HIS,ASN,GLN hydrogen flips
                ff.write('-flip_HNQ true\n')
                #ignore_ligand_chi prevents Roseta from adding additional ligand rotamer
                ff.write('-ignore_ligand_chi true\n')

            if self.ignore_zero_occupancy == False:
                ff.write('-ignore_zero_occupancy false\n')

            if self.centroid:
                ff.write('-in:file:centroid\n')

            if self.score_weights != None:
                ff.write('-score:weights '+self.score_weights+'\n')

            if self.others != {}:
                for option in self.others:
                    raw_value = self.others[option]
                    values = raw_value if isinstance(raw_value, list) else [raw_value]
                    for value in values:
                        if value == None:
                            ff.write('-'+option+'\n')
                        else:
                            ff.write('-'+option+' '+value+'\n')

            if self.flags_files != []:
                for flag_file in self.flags_files:
                    ff.write('@'+flag_file+'\n')
