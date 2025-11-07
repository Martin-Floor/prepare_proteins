import os
import mdtraj as md
import prepare_proteins
import numpy as np
from scipy.spatial import distance_matrix

def setUpPrepwizardOptimization(job_folder, trajectory_files, stride=10, topology=None,
                                pH=7.0, remove_hydrogens=False):
    """
    Set up prepwizard optimization for models inside a trajectory

    Paramters
    =========
    job_folder : str
        Job folder name
    trajectory_files : dict
        Contains the path to each trajectory file. Job folder names are refered
        by the keys in the dictionary.
    topology : str
        Path to the topology file.
    """

    # Create job folder
    if not os.path.exists(job_folder):
        os.mkdir(job_folder)

    # Create input models folders
    if not os.path.exists(job_folder+'/input_models'):
        os.mkdir(job_folder+'/input_models')

    # Create output folder
    if not os.path.exists(job_folder+'/output_models'):
        os.mkdir(job_folder+'/output_models')

    output_folder = job_folder+'/output_models/'

    # Copy control file to prepare folder
    prepare_proteins._copySchrodingerControlFile(job_folder)

    # Iterate over given trajectory files
    jobs = []
    for name in trajectory_files:

        input_folder = job_folder+'/input_models/'+name
        if not os.path.exists(input_folder):
            os.mkdir(input_folder)

        output_folder = job_folder+'/output_models/'+name
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Read trajectory
        traj = md.load(trajectory_files[name], top=topology, stride=stride)

        # Save frames into input models folder
        zf = len(str(traj.n_frames*stride))
        for i,t in enumerate(traj):
            i = (i+1)*stride
            frame_name = '.'.join(trajectory_files[name].split('/')[-1].split('.')[:-1])+'_'+str(i).zfill(zf)
            t.save(input_folder+'/'+frame_name+'.pdb')

            command = 'cd '+output_folder+'\n'
            command += '"${SCHRODINGER}/utilities/prepwizard" '
            command += '../../input_models/'+name+'/'+frame_name+'.pdb '
            command += frame_name+'.pdb '
            command += '-fillsidechains '
            command += '-disulfides '
            if remove_hydrogens:
                command += '-rehtreat '
            command += '-epik_pH '+str(pH)+' '
            command += '-epik_pHt 2.0 '
            command += '-propka_pH '+str(pH)+' '
            command += '-f OPLS_2005 '
            command += '-rmsd 0.3 '
            command += '-samplewater '
            command += '-delwater_hbond_cutoff 3 '
            command += '-JOBNAME prepare_'+frame_name+' '
            command += '-HOST localhost:1\n'

            # Add control script command
            command += 'python3 ../../._schrodinger_control.py '
            command += frame_name+'.log '
            command += '--job_type prepwizard\n'
            command += 'cd ../../..\n'
            jobs.append(command)

    return jobs

def setUpSiteMapCalculation(job_folder, prepwizard_folder, target_residue, site_box=10,
                            resolution='fine', reportsize=100, overwrite=False):
    """
    Set up SiteMap calculations from a previous prepwizard optimization folder

    Parameters
    ==========
    job_folder : str
        Name of the job folder
    prepwizard_folder : str
        Path to the prepwizard optimization folder
    target_residue : str
        Target residue given as "residue_id"
    """

    # Create site map job folders
    if not os.path.exists(job_folder):
        os.mkdir(job_folder)

    if not os.path.exists(job_folder+'/input_models'):
        os.mkdir(job_folder+'/input_models')

    if not os.path.exists(job_folder+'/output_models'):
        os.mkdir(job_folder+'/output_models')

    # Copy script to generate protein and ligand mae inputs, separately.
    prepare_proteins._copyScriptFile(job_folder, 'prepareForSiteMap.py')
    script_path = job_folder+'/._prepareForSiteMap.py'

    # Iterate over prepwizard output files
    jobs = []
    for d in os.listdir(prepwizard_folder+'/output_models'):
        if not os.path.exists(job_folder+'/input_models/'+d):
            os.mkdir(job_folder+'/input_models/'+d)

        if not os.path.exists(job_folder+'/output_models/'+d):
            os.mkdir(job_folder+'/output_models/'+d)

        for f in os.listdir(prepwizard_folder+'/output_models/'+d):
            if f.endswith('.pdb'):
                name = f.replace('.pdb', '')
                input_protein = job_folder+'/input_models/'+d+'/'+name+'_protein.mae'
                if not os.path.exists(input_protein) or overwrite:
                    command = 'run '+script_path+' '
                    command += prepwizard_folder+'/output_models/'+d+'/'+f+' '
                    command += job_folder+'/input_models/'+d+' '
                    command += '--protein_only '
                    os.system(command)

                # Add site map command
                command = 'cd '+job_folder+'/output_models/'+d+'\n'
                command += '"${SCHRODINGER}/sitemap" '
                command += '-prot ../../input_models/'+d+'/'+name+'_protein.mae'+' '
                command += '-sitebox '+str(site_box)+' '
                command += '-resolution '+str(resolution)+' '
                command += '-reportsize '+str(reportsize)+' '
                command += '-keepvolpts yes '
                command += '-keeplogs yes '
                command += '-siteasl \"res.num {'+target_residue+'}\" '
                command += '-HOST localhost:1 '
                command += '-TMPLAUNCHDIR '
                command += '-WAIT\n'

                jobs.append(command)

    return jobs

def readSiteMapResults(sitemap_folder, target_residue):
    """
    Read results of a SiteMap trajectory analysis.

    Parameters
    ==========
    sitemap_folder : str
        Path to the SiteMap trajectory calculation
    """

    sitemap_data = {}
    for d in os.listdir(sitemap_folder+'/output_models'):
        Ddir_out = sitemap_folder+'/output_models/'+d
        Ddir_in = sitemap_folder+'/input_models/'+d

        sitemap_data[d] = {}
        sitemap_data[d]['vpts'] = {}
        sitemap_data[d]['target_coordinates'] = {}
        sitemap_data[d]['step'] = []
        sitemap_data[d]['closest_distance'] = []
        sitemap_data[d]['volume'] = []

        for f in sorted(os.listdir(Ddir_out)):
            Fdir = Ddir_out+'/'+f
            # Get Volume points coordinates as numpy array
            if f.endswith('_volpts.pdb'):
                # Get data points
                step = int(f.split('_')[2])
                traj = md.load(Fdir)
                sitemap_data[d]['vpts'][step] = traj.xyz[0,:,:]*10

                # Get target residue coordinates from input mae
                coordinates = []
                input_mae = Ddir_in+'/'+'_'.join(f.split('_')[:4])+'.mae'
                c = False
                atom = False
                with open(input_mae) as mae:
                    columns = []
                    for l in mae:
                        if not l.strip().startswith('#'):
                            if 'm_atom[' in l:
                                c = True
                                continue
                            elif ':::' in l:
                                if c:
                                    atom = True
                                    c = False
                                    continue
                                if atom:
                                    atom = False
                                c = False
                                continue

                            if c:
                                columns.append(l.strip())
                            if atom:
                                resid = int(l.split()[columns.index('i_m_residue_number')+1])
                                if resid == target_residue:
                                    x = float(l.split()[columns.index('r_m_x_coord')+1])
                                    y = float(l.split()[columns.index('r_m_y_coord')+1])
                                    z = float(l.split()[columns.index('r_m_z_coord')+1])
                                    coordinates.append(np.array([x,y,z]))
                sitemap_data[d]['target_coordinates'][step] = np.array(coordinates)

                # Calculate closest distance between vpts and target residue coordinates
                M = distance_matrix(sitemap_data[d]['target_coordinates'][step],
                                    sitemap_data[d]['vpts'][step])

                sitemap_data[d]['step'].append(step)
                sitemap_data[d]['closest_distance'].append(np.amin(M))

            if f.endswith('_eval.log'):
                # Extract volume information
                volume = False
                with open(Fdir) as eval:
                    for l in eval:
                        if 'volume' in l:
                            columns = l.strip().split()
                            volume = True
                            continue
                        if volume:
                            ls = l.strip().split()
                            sitemap_data[d]['volume'].append(float(ls[columns.index('volume')]))
                            volume = False

        sitemap_data[d]['step'] = np.array(sitemap_data[d]['step'])
        sitemap_data[d]['closest_distance'] = np.array(sitemap_data[d]['closest_distance'])
        sitemap_data[d]['volume'] = np.array(sitemap_data[d]['volume'])

    return sitemap_data
