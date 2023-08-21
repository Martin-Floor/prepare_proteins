from pyrosetta import *
import argparse
import numpy as np
import itertools

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('input_pdb', help='Path to the input PDB to calculate metal distances.')
parser.add_argument('cst_file', help='Path to the output constraint file containing the metal constraints.')
parser.add_argument('--sugars', action='store_true', help='Use carbohydrate aware Rosetta PDB reading.')
parser.add_argument('--params_folder', help='Folder containing parms files to use.')
args=parser.parse_args()

# Assign variable names
input_pdb = args.input_pdb
cst_file = args.cst_file
sugars = args.sugars
params_folder = args.params_folder

# Initilise PyRosetta
init_options = ''
if sugars:
    init_options =  '-include_sugars '
    init_options += '-write_pdb_link_records '
    init_options += '-load_PDB_components false '
    init_options += '-alternate_3_letter_codes pdb_sugar '
    init_options += '-write_glycan_pdb_codes '
    init_options += '-auto_detect_glycan_connections '
    init_options += '-maintain_links '

init(init_options)

# Get all params files if fiven
params_files = []
if params_folder != None:
    for f in os.listidir(params_folder):
        if f.endswith('.params'):
            params_files.append(params_folder+'/'+f)

# Create empty pose
pose = Pose()

if params_files != []:

    # Feed params files as a new residue type set
    params = Vector1(params_files)
    res_set = pose.conformation().modifiable_residue_type_set_for_conf()
    res_set.read_files_for_base_residue_types(params)
    pose.conformation().reset_residue_type_set_for_conf(res_set)

# Load PDB file as a pose
pose_from_file(pose, input_pdb)


def getMetalCoordinations(pose, max_dist_cutoff=2.9, step_dist_cutoff=0.01):
    """
    Get the atoms coordinating all metal residues in an iterative fashion. The distance cutoff
    is changed until the expected metal coordination is fulfilled. In case the max_dist_cutoff
    value is reached the iterative iteration is stopped and a warning is issued to check the
    input values.

    Parameters
    ==========
    max_dist_cutoff : float
        Maximum distance cutoff to select coordinating atoms.
    step_dist_cutoff : float
        Step size to take in the iterative search for coordinating atoms.

    Returns
    =======
    coord_atoms : dict
        Dictionary containing as keys the metal and as values the coordinating atoms.
        The atoms are given as PDB-numbered tuples, i.e., (chain, residue, atom).
    """

    pose2pdb = {}
    for r in range(1, pose.total_residue()+1):
        resid, chain = pose.pdb_info().pose2pdb(r).split()
        pose2pdb[r] = (int(resid), chain)

    expected_coordination = {
        'CA' : 6,
    }

    # Get metal residues
    metal_residues = []
    metal_coordinates = []
    for r in range(1, pose.total_residue()+1):
        residue = pose.residue(r)
        if residue.has_property('METAL'):
            metal_residues.append((r, residue.name()))
            metal_coordinates.append(np.array(residue.atom(1).xyz()))

    metal_coordinates = np.array(metal_coordinates)

    # Get acceptor atoms coordinates of metal binding or water residues
    coordinates = []
    atoms = []
    # Iterate metalbinding or water residues
    for r in range(1, pose.total_residue()+1):
        residue = pose.residue(r)

        if residue.is_metalbinding() or residue.is_water():
            atom_type_set = residue.atom_type_set()

            # Iterate over acceptor atoms
            for a in range(1, pose.residue_type(r).natoms()+1):
                at_type_index = pose.residue(r).atom_type_index(a)
                at_type = atom_type_set[at_type_index]

                # Get atoms and coordiantes
                if at_type.is_acceptor():
                    coordinates.append(np.array(residue.atom(a).xyz()))
                    atoms.append((r,a))

    coordinates = np.array(coordinates)
    atoms = np.array(atoms)

    # Get coordinating atoms
    dist_cutoff = 2.2
    coord_atoms = {}
    failed = False
    for (metal, name), mcoord in zip(metal_residues, metal_coordinates):

        if name not in expected_coordination:
            m = 'Your metal {name} has not an expected coordination defined!'
            m += 'Please add it manually to this function.'
            raise ValueError(m)

        resid, chain = pose2pdb[metal]

        coordination = np.array([])
        while coordination.shape[0] < expected_coordination[name]:

            # Calculate metal- acceptor distances
            dist = np.linalg.norm(mcoord-coordinates, axis=1)

            # Get atoms closer than the cutoff
            coordination = atoms[dist <= dist_cutoff]

            if dist_cutoff >= max_dist_cutoff:
                if coordination.shape[0] != expected_coordination[name]:
                    m = f'WARNING: Maximum coordination found for {name} {resid} {chain} is {len(coordination)}.'
                    m += f'Expected coordination is {expected_coordination[name]}.'
                    print(m)
                    failed = True
                break

            # Update the dist_cutoff value
            dist_cutoff += step_dist_cutoff

        metal_residue = (chain, resid, name)
        coord_atoms[metal_residue] = []
        for r, a in coordination:
            c_resid, c_chain = pose2pdb[r]
            c_atom_name = pose.residue(r).atom_name(a).strip()
            c_residue = (c_chain, c_resid, c_atom_name)
            coord_atoms[metal_residue].append(c_residue)

    if failed:
        print('For correctness, please check your input structure or try increasing the max_distcutoff_multiplier value.')

    return coord_atoms

def createMetalConstraintFile(pose, cst_file, max_dist_cutoff=2.9, bond_cst_std=0.1, angle_cst_std=0.1):
    """
    Create a constraint file containing all metal bond and angle constraints.

    Paramters
    =========
    cst_file : str
        Path to the output constraint file.
    max_dist_cutoff : float
        Maximum distance cutoff to select coordinating atoms (for details see getMetalCoordinations()
        function in this class).
    bond_cst_std : float
        std term in the bond harmonic constraint.
    angle_cst_std : float
        std term in the angle harmonic constraint.
    """

    def computeAngle(coord1, coord2, coord3):
        """
        Compute angle from three coordinates
        """
        v1 = coord1 - coord2
        v2 = coord3 - coord2
        cos_theta = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1, 1))

        return angle

    coordinations = getMetalCoordinations(pose, max_dist_cutoff=max_dist_cutoff)

    pdb2pose= {}
    for r in range(1, pose.total_residue()+1):
        resid, chain = pose.pdb_info().pose2pdb(r).split()
        pdb2pose[(int(resid), chain)] = r

    bonds = {}
    angles = {}

    for metal in coordinations:
        atoms_and_metal = [metal]

        # Get all constraint bonds
        bonds[metal] = []
        for atom in coordinations[metal]:

            # Calculate distance
            res1 = pdb2pose[(metal[1], metal[0])]
            res2 = pdb2pose[(atom[1], atom[0])]

            atom1 = pose.residue(res1).atom(metal[2])
            atom2 = pose.residue(res2).atom(atom[2])

            atom1_xyz = np.array(atom1.xyz())
            atom2_xyz = np.array(atom2.xyz())

            distance = np.linalg.norm(atom1_xyz-atom2_xyz)

            if distance > 3.0:
                print(distance)

            # Store values
            bond = ((metal, atom), distance)
            bonds[metal].append(bond)
            atoms_and_metal.append(atom)

        # Get all constraint angles
        angles[metal] = set()
        for c in itertools.combinations(atoms_and_metal, 3):
            if metal in c:

                # Put metal atom in the middle
                ang_atms = (c[1], c[0], c[2])

                # Calculate distance
                res1 = pdb2pose[(ang_atms[0][1], ang_atms[0][0])]
                res2 = pdb2pose[(ang_atms[1][1], ang_atms[1][0])]
                res3 = pdb2pose[(ang_atms[2][1], ang_atms[2][0])]

                atom1 = pose.residue(res1).atom(ang_atms[0][2])
                atom2 = pose.residue(res2).atom(ang_atms[1][2])
                atom3 = pose.residue(res3).atom(ang_atms[2][2])

                atom1_xyz = np.array(atom1.xyz())
                atom2_xyz = np.array(atom2.xyz())
                atom3_xyz = np.array(atom3.xyz())

                distance = computeAngle(atom1_xyz, atom2_xyz, atom3_xyz)
                angle = ((c[1], c[0], c[2]), distance)
                angles[metal].add(angle)


    # Write atom cst lines
    cf = open(cst_file, 'w')
    for metal in bonds:
        for (atom1, atom2), distance in bonds[metal]:
            bond_cst_line =  'AtomPair '
            bond_cst_line += atom1[2]+' '+str(atom1[1])+atom1[0]+' '
            bond_cst_line += atom2[2]+' '+str(atom2[1])+atom2[0]+' '
            bond_cst_line += 'HARMONIC '
            bond_cst_line += '%.4f' % (distance)+' '
            bond_cst_line += str(bond_cst_std)+'\n'
            cf.write(bond_cst_line)

    for metal in angles:
        for (atom1, atom2, atom3), angle in angles[metal]:
            angle_cst_line =  'Angle '
            angle_cst_line += atom1[2]+' '+str(atom1[1])+atom1[0]+' '
            angle_cst_line += atom2[2]+' '+str(atom2[1])+atom2[0]+' '
            angle_cst_line += atom3[2]+' '+str(atom3[1])+atom3[0]+' '
            angle_cst_line += 'HARMONIC '
            angle_cst_line += '%.4f' % (angle)+' '
            angle_cst_line += str(angle_cst_std)+'\n'
            cf.write(angle_cst_line)

    cf.close()

createMetalConstraintFile(pose, cst_file)
