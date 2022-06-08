import os

class mdp:
    """
    Container for functions to generate Gromacs Molecular Dynamics Protocol files.
    """
    def write_mdp_file(output_file,
                        # Run parameters
                        define = '',
                        integrator='steep',
                        nsteps=5000,
                        emtol=None,
                        emstep=None,
                        dt = None,
                        # Constraints
                        constraints='all-bonds',
                        constraint_algorithm=None,
                        lincs_order=None,
                        lincs_iter =None,
                        continuation='no',
                        # Neigbour search
                        cutoff_scheme='Verlet',
                        verlet_buffer_tolerance=None,
                        nstlist=40,
                        ns_type='grid',
                        pbc='xyz',
                        rlist = 1.0,
                        # Output files
                        nstxout=0,
                        nstvout=0,
                        nstfout=0,
                        nstlog=0,
                        nstenergy=0,
                        nstxout_compressed=0,
                        compressed_x_grps = "System",
                        # Groups
                        xtc_grps = None,
                        energygrps = None,
                        # Electrostatics and vdw
                        coulombtype = 'PME',
                        pme_order = None,
                        fourierspacing = None,
                        rcoulomb = 1.0,
                        rvdw = 1.0,
                        DispCorr = 'EnerPres',
                        # Temperature coupling
                        tcoupl = 'V-rescale',
                        tc_grps = None,
                        tau_t = None,
                        ref_t = None,
                        # Pressure coupling
                        pcoupl = "berendsen",
                        pcoupltype = "isotropic",
                        tau_p = 1.0,
                        ref_p = 1.0,
                        compressibility = 4.5e-5,
                        refcoord_scaling = 'com',
                        # Velocity generation
                        gen_vel = 'no',
                        gen_temp = 300,
                        gen_seed = -1,
                        # Simulated annealing
                        annealing = 'single single',
                        annealing_npoints = '2 2',
                        annealing_time = '0 100 0 100',
                        annealing_temp = '10 300 10 300'):
        """
        Create Gromacs Molecular Dynamics Protocol for GROMACS based on a set of options

        Parameters
        ==========
        output_file : str
            Path to write the MDP file.

        # Run parameters

        integrator : str
            Integration algorithm.
        nsteps : int
            maximum number of steps to integrate or minimize, -1 is no maximum.
        emtol : float
            Convergence criterion for energy minimization.
        emstep : float
            Intial step size for energy minimization.
        dt : float
            Time step for integration.

        # Constraints

        constraints : str
            Type of contraints to be applied.
        constraint_algorithm : str
            Constraint algorithm.
        lincs-order : str
            Order of the Lincs algorithm; normally 4, but 6 is needed for 4 fs timestep.
        lincs-iter : str
            Number of iterations to correct for rotational lengthening in LINCS.
        continuation : bool
            Is this a continuation from a previous simulation?

        # Neigbour search

        cutoff-scheme : str
            Cut off scheme for updating pair-list for non-bonded calculations.
        verlet_buffer_tolerance : float
            Tolerance for the pair-list buffer used by the Verlet algorithm. Sets the
            maximum allowed error for pair interactions per particle.
        nstlist : int
            Step frequency for updating neighbour list.
        ns_type : str
            Method for neighbour searching.
        pbc : bool
            Use periodic boundary conditions.

        # Output files

        nstxout : int
            Frequency for writing coords to output.
        nstvout : int
            Frequency for writing velocities to output.
        nstfout : int
            Frequency for writing forces to output.
        nstlog : int
            Frequency for writing energies to log file.
        nstenergy : int
            Frequency for writing energies to energy file.
        nstxout-compressed : int
            Frequency for writing coords to xtc traj.
        compressed_x_grps: str
            Group(s) to write to the compressed trajectory file.

        # Groups

        xtc_grps : str or list
             Group(s) whose coords are to be written in xtc traj.
        energygrps : str or list
            Group(s) whose energy is to be written in energy file.

        # Electrostatics and vdw

        coulombtype : str
            Type of Coulomb scheme.
        pme-order : int
            Interpolation order for the PME electrostatic method.
        fourierspacing : float
            Grid dimensions for the PME electrostatic method.
        rlist : float
            Cut-off distance for the short-range neighbor list.
        rcoulomb: float
            The distance for the Coulomb cut-off.
        rvdw : float
            Distance for the LJ or Buckingham cut-off.
        DispCorr: str
            Define dispersion corrections.

        # Temperature coupling

        tcoupl : str
            Type of temperature coupling to be used.
        tc_grps : str
            Groups to couple to separate temperature baths.
        tau_t : str
            Time constant for coupling for each group.
        ref_t : str
            Reference temperature for coupling for each group.

        # Pressure coupling

        pcoupl : str
            Type of pressure coupling to be used.
        pcoupltype : str
            Specifies the kind of isotropy of the pressure coupling used.
        tau_p : str
            Time constant for coupling.
        ref_p : str
            The reference pressure for coupling
        compressibility : float
        refcoord_scaling : str
            Scaling of the reference coordinates

        # Velocity generation

        gen_vel : str
            generate velocities?
        gen_temp : int
            temperature of Maxwell distribution
        gen_seed : int
            used to initialize random generator for random velocities

        # Simulated annealing

        annealing : str
            Type of annealing for each temperature group.
        annealing_npoints : str
            A list with the number of annealing reference/control points used for each temperature group.
        annealing_time : str
            List of times at the annealing reference/control points for each group.
        annealing_temp = str
            List of temperatures at the annealing reference/control points for each group.
        """

        keywords = locals()
        del keywords['output_file']

        if keywords['integrator'] == 'steep' and keywords['emtol'] == None:
            keywords['emtol'] = 1000
        elif keywords['integrator'] == 'steep' and keywords['emstep'] == None:
            keywords['emstep'] = 0.001
        elif keywords['integrator'] == 'md' and keywords['dt'] == None:
            keywords['dt'] = 0.004

        if keywords['constraints'] != 'none' and keywords['constraint_algorithm'] == None:
            keywords['constraint_algorithm'] = 'lincs'
        if keywords['constraint_algorithm'] == 'lincs' and keywords['lincs_iter'] == None:
            keywords['lincs_iter']  = 1
        if keywords['constraint_algorithm'] == 'lincs' and keywords['lincs_order'] == None:
            keywords['lincs_order'] = 6

        if keywords['cutoff_scheme'] == 'Verlet' and keywords['verlet_buffer_tolerance'] == None:
            keywords['verlet_buffer_tolerance'] = 0.005

        if keywords['coulombtype'] == 'PME' and keywords['pme_order'] == None:
            keywords['pme_order'] = 4
        if keywords['coulombtype'] == 'PME' and keywords['fourierspacing'] == None:
            keywords['fourierspacing'] = 0.12

        if keywords['tcoupl'] != 'no' and keywords['tc_grps'] == None:
            keywords['tc_grps'] = 'Protein Non-Protein'
        if keywords['tcoupl'] != 'no' and keywords['tau_t'] == None:
            keywords['tau_t'] = '0.1 0.1'
        if keywords['tcoupl'] != 'no' and keywords['ref_t'] == None:
            keywords['ref_t'] = ref_t = '300 300'


        # keywords = {
        # 'define' : '',
        # 'integrator' : integrator,
        # 'nsteps' :  nsteps,
        # 'emtol' : emtol,
        # 'constraints' : constraints,
        # 'constraint-algorithm' : constraint_algorithm,
        # 'lincs-order' : lincs_order,
        # 'continuation' : continuation,
        # 'cutoff-scheme' : cutoff_scheme,
        # 'verlet-buffer-tolerance' : verlet_buffer_tolerance,
        # 'nstlist' : nstlist,
        # 'ns_type' : ns_type,
        # 'pbc' : pbc,
        # 'nstxout' : nstxout,
        # 'nstvout' : nstvout,
        # 'nstfout' : nstfout,
        # 'nstlog' : nstlog,
        # 'nstenergy' : nstenergy,
        # 'nstxout-compressed' : nstxout_compressed,
        # 'xtc_grps' : xtc_grps,
        # 'energygrps' : energygrps,
        # 'coulombtype' : coulombtype,
        # 'pme-order' : pme_order,
        # 'fourierspacing' : fourierspacing,
        # 'rlist' : rlist,
        # 'rcoulomb' : rcoulomb,
        # 'rvdw' : rvdw,
        # 'DispCorr' : DispCorr}

        # Write MDP file with given options
        with open(output_file, 'w') as em_mdp:
            for k in keywords:
                if keywords[k] != None:
                    em_mdp.write(k+' = '+str(keywords[k])+'\n')
