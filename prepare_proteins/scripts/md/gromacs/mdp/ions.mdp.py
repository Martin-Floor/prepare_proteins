; Parameters describing what to do, when to stop and what to save
integrator			= steep		; Algorithm (steep = steepest descent minimization)
emtol				= 1000.0  	; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep				= 0.01  	; Energy step size
nsteps				= 50000		; Maximum number of (minimization) steps to perform

; Neighbor search
cutoff-scheme		= Verlet
verlet-buffer-tolerance	= -1 	; use rlist
;rlist				= 1.1		; Cut-off distance for the short-range neighbor list
pbc		        	= xyz 		; Periodic Boundary Conditions

; Electrostatics and vdW
coulombtype	    	= PME		; Treatment of long range electrostatic interactions
pme-order		= 4			; Fourth-order (cubic) interpolation
fourierspacing		= 0.12		; A lower bound for the number of Fourier-space grid points
rcoulomb	    	= 1.0		; Short-range electrostatic cut-off
rvdw		    	= 1.0		; Short-range Van der Waals cut-off
