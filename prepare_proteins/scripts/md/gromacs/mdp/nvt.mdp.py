define				= -DPOSRES	; position restrain the protein
; Run parameters
integrator			= md		; leap-frog integrator
nsteps				= 5000		; 200 ps
dt				= 0.004		; 4 fs with virtual sites
; Output control
nstxout				= 0		; don't write coordinates to .trr file
nstvout				= 0		; don't write velocities
nstfout 	            	= 0         	; don't write forces
nstenergy			= 2500		; save energies every 10 ps
nstlog				= 2500		; update log file every 10 ps
compressed-x-grps            	= System	; write xtc for the whole system
nstxout-compressed           	= 2500		; write coordinates every 10 ps to .xtc file
; Bond parameters
continuation			= no		; first dynamics run
constraint_algorithm		= lincs		; holonomic constraints
constraints	        	= all-bonds	; all bonds (even heavy atom-H bonds) constrained
lincs_iter	        	= 1		; accuracy of LINCS
lincs_order	        	= 6		; normally 4, but 6 is needed for 4 fs timestep
; Neighborsearching
cutoff-scheme   		= Verlet
ns_type		    		= grid		; search neighboring grid cells
nstlist		    		= 20		; 20 fs, largely irrelevant with Verlet
rlist				= 1.0		; short-range neighborlist cutoff (in nm)
rcoulomb	    		= 1.0		; short-range electrostatic cutoff (in nm)
rvdw		    		= 1.0		; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype	    		= PME		; Particle Mesh Ewald for long-range electrostatics
pme_order	    		= 4		; cubic interpolation
fourierspacing			= 0.12		; grid spacing for FFT
; Temperature coupling is on
tcoupl				= V-rescale	; modified Berendsen thermostat
tc-grps				= Protein Non-Protein	; two coupling groups - more accurate
tau_t				= 0.1	  0.1   ; time constant, in ps
ref_t				= 298 	  298   ; reference temperature, one for each group, in K
; Pressure coupling is off
pcoupl				= no 		; no pressure coupling in NVT
; Periodic boundary conditions
pbc				= xyz  		; 3-D PBC
; Dispersion correction
DispCorr			= EnerPres	; account for cut-off vdW scheme
; Velocity generation
gen_vel				= yes		; assign velocities from Maxwell distribution
gen_temp			= 300		; temperature for Maxwell distribution
gen_seed			= -1		; generate a random seed
; Simulated annealing
annealing			= single single ; single sequence of points for each T-coupling group
annealing_npoints		= 2 	 2 		; two points - start and end temperatures
annealing_time 			= 0 100  0 100  ; time frame of heating - heat over period of 500 ps
annealing_temp			= 10 300 10 300 ; start and end temperatures
