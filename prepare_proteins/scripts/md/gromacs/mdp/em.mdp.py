; Run parameters
define					= 				;-DPOSRES ; -DPOSRES_IONS ;DFLEX_SPC; FLEXible SPC and POSition REStraints
integrator				= steep    		; steepest descent algorithm
nsteps              	= 2000     		; number of steps
emtol               	= 1000 			; convergence criterion
emstep              	= 0.001   		; intial step size
; Constraints
constraints             = all-bonds
constraint-algorithm    = lincs
lincs-order             = 6			; normally 4, but 6 is needed for 4 fs timestep
continuation            = no
; Neighbor search
cutoff-scheme           = Verlet		; pair list with buffering
verlet-buffer-tolerance = 0.005   		; Sets the maximum allowed error for pair interactions per particle
nstlist             	= 40		   	; step frequency for updating neighbour list
ns_type             	= grid 			; method for nighbour searching (?)
pbc                 	= xyz    		; use pbc
; Output files
nstxout             	= 25    		; frequency for writing coords to output
nstvout             	= 25    		; frequency for writing velocities to output
nstfout             	= 25    		; frequency for writing forces to output
nstlog              	= 25    		; frequency for writing energies to log file
nstenergy           	= 25		  	; frequency for writing energies to energy file
nstxout-compressed      =  0    		; frequency for writing coords to xtc traj
; Groups
xtc_grps            	= Protein Non-Protein ; group(s) whose coords are to be written in xtc traj
energygrps          	= Protein Non-Protein ; group(s) whose energy is to be written in energy file
; Electrostatics
coulombtype         	= PME		  	; truncation for minimisation, with large cutoff
pme-order		= 4
fourierspacing		= 0.12			; A lower bound for the number of Fourier-space grid points
rlist               	= 1.0   		; cutoff (nm)
rcoulomb            	= 1.0
rvdw                	= 1.0
DispCorr    	   	= EnerPres		; apply long range dispersion corrections for Energy and Pressure
