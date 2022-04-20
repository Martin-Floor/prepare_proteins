import os

class mdp:
    """
    Container for functions to generate Gromacs Molecular Dynamics Protocol files
    """
    def write_mdp_file(output_file,
                        integrator='steep'
                        nsteps=2000,
                        emtol=1000,
                        emstep=0.001,
                        constraints='all-bonds',
                        constraint_algorithm='lincs',
                        lincs_order=6,
                        continuation=False,
                        cutoff_scheme='Verlet',
                        verlet_buffer_tolerance=0.005,
                        nstlist=40,
                        ns_type='grid',
                        pbc=True,
                        nstxout=25,
                        nstvout=25,
                        nstfout=25,
                        nstlog=25,
                        nstenergy=25,
                        nstxout_compressed=0,
                        xtc_grps = None,
                        energygrps = None,
                        coulombtype = 'PME',
                        pme_order = 4,
                        fourierspacing = 0.12,
                        rlist = 1.0,
                        rcoulomb = 1.0,
                        rvdw = 1.0,
                        DispCorr = 'EnerPres'):

        """
        Create Gromacs Molecular Dynamics Protocol for GROMACS based on a set of options

        Parameters
        ==========
        output_file : str
            Path to write the MDP file.
        integrator : str
            Integration algorithm.
        nsteps : int
            Number of MD steps.
        emtol : float
            Convergence criterion for energy minimization.
        emstep : float
            Intial step size for energy minimization.
        constraints : str
            Type of contraints to be applied.
        constraint_algorithm : str
            Constraint algorithm.
        lincs-order : str
            Order of the Lincs algorithm; normally 4, but 6 is needed for 4 fs timestep.
        continuation : bool
            Is this a continuation from a previous simulation?
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
        xtc_grps : str or list
             Group(s) whose coords are to be written in xtc traj.
        energygrps : str or list
            Group(s) whose energy is to be written in energy file.
        coulombtype : str
            Type of Coulomb scheme.
        pme-order : int
            Interpolation order for the PME electrostatic method.
        fourierspacing : float
            Grid dimensions for the PME electrostatic method.
        """

        keywords = {
        'define' : ''
        'integrator' : integrator,
        'nsteps' :  nsteps,
        'emtol' : emtol,
        'constraints' : constraints,
        'constraint-algorithm' : constraint_algorithm,
        'lincs-order' : lincs_order,
        'continuation' : continuation,
        'cutoff-scheme' : cutoff_scheme,
        'verlet-buffer-tolerance' : verlet_buffer_tolerance,
        'nstlist' : nstlist,
        'ns_type' : ns_type,
        'pbc' : pbc,
        'nstxout' : nstxout,
        'nstvout' : nstvout,
        'nstfout' : nstfout,
        'nstlog' : nstlog,
        'nstenergy' : nstenergy,
        'nstxout-compressed' : nstxout_compressed,
        'xtc_grps' : xtc_grps,
        'energygrps' : energygrps,
        'coulombtype' : coulombtype,
        'pme-order' : pme_order,
        'fourierspacing' : fourierspacing,
        'rlist' : rlist,
        'rcoulomb' : rcoulomb,
        'rvdw' : rvdw,
        'DispCorr' : DispCorr}

        # Write MDP file with given options
        wiht open(output_file, 'w') as em_mdp:
            for k in keywords:
                em_mdp.write(k+'\t'+keywords[k])
