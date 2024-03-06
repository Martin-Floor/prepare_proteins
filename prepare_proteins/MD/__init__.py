from .alignment import alignTrajectoryBySequenceAlignment
from .gromacs_files import mdp
try:
    import openmm
    from .openmm_setup import openmm_md
except:
    None
