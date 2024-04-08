import os
import json
import numpy as np
import io
import shutil
import pandas as pd
import subprocess

import mdtraj as md

from pkg_resources import resource_stream, Requirement

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact,fixed

import prepare_proteins

class md_analysis:

    def __init__(self,md_folder, command='gmx', output_group='System', ligand=False, remove_redundant_files=False, overwrite=False):

        self.models = [folder for folder in os.listdir(md_folder+'/output_models/') if os.path.isdir(md_folder+'/output_models/'+folder)]

        self.traj_paths = {}
        self.traj_paths['nvt'] = {}
        self.traj_paths['npt'] = {}
        self.traj_paths['md'] = {}

        self.top_paths = {}

        self.distances = {}

        for model in self.models:
            self.traj_paths['nvt'][model] = {}
            self.traj_paths['npt'][model] = {}
            self.traj_paths['md'][model] = {}
            self.top_paths[model] = {}

            for replica in os.listdir(md_folder+'/output_models/'+model):
                # Define paths
                top_path = md_folder+'/output_models/'+model+'/'+replica+'/em/prot_em.gro'

                nvt_path = md_folder+'/output_models/'+model+'/'+replica+'/nvt/prot_nvt.xtc'
                npt_path = md_folder+'/output_models/'+model+'/'+replica+'/npt'
                md_path = md_folder+'/output_models/'+model+'/'+replica+'/md'

                # Print warnings for missing files
                if not os.path.exists(top_path):
                    self.top_paths[model][replica] = ''
                    print('WARNING: Topology for model '+model+' and replica '+replica+' could not be found. EM gro does not exist')

                if not os.path.exists(nvt_path):
                    self.traj_paths['nvt'][model][replica] = ''
                    print('WARNING: NVT trajectory for model '+model+' and replica '+replica+' could not be found. NVT xtc does not exist')

                if not os.path.exists(npt_path) or len(os.listdir(npt_path))==0:
                    self.traj_paths['npt'][model][replica] = ''
                    print('WARNING: NPT trajectory for model '+model+' and replica '+replica+' could not be found. Folder does not exist or is empty')

                if not os.path.exists(md_path) or len(os.listdir(md_path))==0:
                    self.traj_paths['md'][model][replica] = ''
                    print('WARNING: MD trajectory for model '+model+' and replica '+replica+' could not be found. Folder does not exist or is empty')

                # Get groups from topol index
                index_path = md_folder+'/'+'output_models/'+model+'/'+replica+'/topol/index.ndx'
                group_dics = _readGromacsIndexFile(index_path)
                if output_group not in group_dics:
                    raise ValueError('The selected output group is not available in the topol/index.ndx file. The following groups are available: '+','.join(group_dics.keys()))


                # Remove PBC
                # In case of ligand PBC center is between protein ligand interface
                if ligand:
                    centering_selector = group_dics['Protein']
                else:
                    centering_selector = group_dics['Protein']


                # TOP
                if not os.path.exists(top_path.replace('.gro','_noPBC.gro')) or overwrite:
                    os.system('echo -e '+centering_selector+' '+group_dics[output_group]+' | '+command+'  trjconv -s '+top_path.replace('.gro','.tpr')+' -f '+top_path+' -o '+top_path.replace('.gro','_noPBC.gro')+' -center -pbc res -ur compact'+' -n '+index_path)
                    if remove_redundant_files:
                        os.remove(top_path)

                self.top_paths[model][replica] = top_path.replace('.gro','_noPBC.gro')

                # NVT
                if not os.path.exists(nvt_path.replace('.xtc','_noPBC.xtc')) or overwrite:
                    os.system('echo '+centering_selector+' '+group_dics[output_group]+' | '+command+'  trjconv -s '+nvt_path.replace('.xtc','.tpr')+' -f '+nvt_path+' -o '+nvt_path.replace('.xtc','_noPBC.xtc')+' -center -pbc res -ur compact'+' -n '+index_path)
                    if remove_redundant_files:
                        os.remove(nvt_path)

                self.traj_paths['nvt'][model][replica] = nvt_path.replace('.xtc','_noPBC.xtc')

                # NPT
                if not os.path.exists(npt_path+'/prot_npt_cat_noPBC.xtc') or overwrite:
                    remove_paths = []
                    for file in os.listdir(npt_path):
                        if file.endswith('.xtc') and 'noPBC' not in file:
                            path = npt_path+'/'+file
                            os.system('echo '+centering_selector+' '+group_dics[output_group]+' | '+command+'  trjconv -s '+path.replace('.xtc','.tpr')+' -f '+path+' -o '+path.replace('.xtc','_noPBC.xtc')+' -center -pbc res -ur compact'+' -n '+index_path)
                            remove_paths.append(path)
                    os.system(command+' trjcat -f '+npt_path+'/*_noPBC.xtc -o '+npt_path+'/prot_npt_cat_noPBC.xtc -cat')
                    if remove_redundant_files:
                        [os.remove(path) for path in remove_paths]

                self.traj_paths['npt'][model][replica] = npt_path+'/prot_npt_cat_noPBC.xtc'

                # MD
                if not os.path.exists(md_path+'/prot_md_cat_noPBC.xtc') or overwrite:
                    remove_paths = []
                    for file in os.listdir(md_path):
                        if file.endswith('.xtc') and 'noPBC' not in file:
                            path = md_path+'/'+file
                            os.system('echo '+centering_selector+' '+group_dics[output_group]+' | '+command+'  trjconv -s '+path.replace('.xtc','.tpr')+' -f '+path+' -o '+path.replace('.xtc','_noPBC.xtc')+' -center -pbc res -ur compact'+' -n '+index_path)
                            remove_paths.append(path)
                    os.system(command+' trjcat -f '+md_path+'/*_noPBC.xtc -o '+md_path+'/prot_md_cat_noPBC.xtc -cat')
                    if remove_redundant_files:
                        [os.remove(path) for path in remove_paths]

                self.traj_paths['md'][model][replica] = md_path+'/prot_md_cat_noPBC.xtc'


    def calculateDistances(self,metrics,step='md',overwrite=False):

        def euclidean(XA, XB, *, out=None):
            return np.sqrt(np.add.reduce(np.square(XA - XB), 1), out=out)

        for model in self.traj_paths[step]:

            if model not in self.distances:
                self.distances[model] = {}

            for replica in self.traj_paths[step][model]:

                if replica not in self.distances[model] or overwrite:
                    # Check if trajectory and topology files were generated correctly
                    if not os.path.exists(self.traj_paths[step][model][replica]):
                        print('WARNING: trajectory file for model '+model+' and replica '+replica+' does not exist.')
                        continue
                    if not os.path.exists(self.top_paths[model][replica]):
                        print('WARNING: topology file for model '+model+' and replica '+replica+' does not exist.')
                        continue

                    traj = md.load(self.traj_paths[step][model][replica],top=self.top_paths[model][replica])
                    top = traj.topology

                    distance = {}
                    for m in metrics[model]:
                        joined_distances = []
                        for d in metrics[model][m]:
                            atom1 = top.select('resSeq '+str(d[0][0])+' and name '+d[0][1])
                            atom2 = top.select('resSeq '+str(d[1][0])+' and name '+d[1][1])

                            atom1_xyz = traj.xyz[:,atom1[0]]
                            atom2_xyz = traj.xyz[:,atom2[0]]

                            joined_distances.append(euclidean(atom1_xyz,atom2_xyz))

                        distance[m] = [min(values) for values in zip(*joined_distances)]

                    self.distances[model][replica] = distance


    def setupCalculateDistances(self,metrics,step='md',job_folder='MD_analysis_data',overwrite=False):


        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        # Get control script
        script_name = "calculateDistances.py"
        control_script = resource_stream(Requirement.parse("prepare_proteins"),
                                             "prepare_proteins/scripts/md/analysis/"+script_name)
        control_script = io.TextIOWrapper(control_script)

        # Write control script to job folder
        with open(job_folder+'/._'+script_name, 'w') as sof:
            for l in control_script:
                sof.write(l)

        jobs = []
        for model in self.traj_paths[step]:
            if not os.path.exists(job_folder+'/'+model):
                os.mkdir(job_folder+'/'+model)
            for replica in self.traj_paths[step][model]:

                # Check if trajectory and topology files were generated correctly
                if not os.path.exists(self.traj_paths[step][model][replica]):
                    print('WARNING: trajectory file for model '+model+' and replica '+replica+' does not exist.')
                    continue
                if not os.path.exists(self.top_paths[model][replica]):
                    print('WARNING: topology file for model '+model+' and replica '+replica+' does not exist.')
                    continue

                job_traj_path = model+'/'+replica+'/'+self.traj_paths[step][model][replica].split('/')[-1]
                job_top_path = model+'/'+replica+'/'+self.top_paths[model][replica].split('/')[-1]

                # Copy trajectory and topology files
                if not os.path.exists(job_folder+'/'+model+'/'+replica):
                    os.mkdir(job_folder+'/'+model+'/'+replica)
                if not os.path.exists(job_folder+'/'+job_traj_path):
                    shutil.copyfile(self.traj_paths[step][model][replica],job_folder+'/'+job_traj_path)
                if not os.path.exists(job_folder+'/'+job_top_path):
                    shutil.copyfile(self.top_paths[model][replica],job_folder+'/'+job_top_path)

                new_path = job_folder+'/'+model+'/'+replica

                if not os.path.exists(new_path+'/dist.json') or overwrite:
                    command = 'cd '+job_folder+'/'+model+'/'+replica+'/'+'\n'
                    command += 'python ../../._'+script_name+' '
                    command += '--trajectory '+self.traj_paths[step][model][replica].split('/')[-1]+' '
                    command += '--topology '+self.top_paths[model][replica].split('/')[-1]+' '
                    if metrics[model] != None:
                        with open(new_path+'/metrics.json','w') as f:
                            json.dump(metrics[model],f)
                        command += '--metrics metrics.json '
                    command += '\n'
                    command += 'cd ..\n'
                    jobs.append(command)
        return jobs


    def readDistances(self,folder='MD_analysis_data'):

        if not os.path.exists(folder):
            os.mkdir(folder)

        for model in self.top_paths:
            self.distances[model] = {}
            for replica in self.top_paths[model]:
                if os.path.exists(folder+'/'+model+'/'+replica+'/dist.json'):
                    with open(folder+'/'+model+'/'+replica+'/dist.json','r') as f:
                        self.distances[model][replica] = json.load(f)
                    print('Reading distance for '+model)
                    continue

    def plot_distances(self,threshold=0.45,lim=True):

        def options1(model):
            if self.distances[model] != {}:
                for replica in self.distances[model]:
                    for metric in self.distances[model][replica]:
                        y = np.array(self.distances[model][replica][metric])*10
                        x = [x*0.1 for x in range(len(y))]
                        plt.plot(x,y)

                    plt.axhline(y=threshold, color='r', linestyle='-')
                    if lim:
                        plt.ylim(1,10)

                    plt.title(replica)
                    plt.xlabel("time (ns)")
                    plt.ylabel("distance (Å)")
                    plt.legend(list(self.distances[model][replica].keys()))
                    plt.show()
            else:
                print('No distance data for model '+model)

        interact(options1,model = sorted(self.distances.keys()))

    def get_distance_prob(self,threshold=0.45,group_metrics=None):

        data = {}
        column_names = []
        for protein in self.distances:
            for replica in self.distances[protein]:
                masks = {}
                for metric in self.distances[protein][replica]:
                    if (protein,replica) not in data:
                        data[(protein,replica)] = []

                    n_frames = len(self.distances[protein][replica][metric])
                    metric_mask = [x < threshold for x in self.distances[protein][replica][metric]]

                    masks[metric] = (metric_mask)

                    #data[(protein,replica)].append(np.mean(self.distances[protein][replica][metric])
                    #data[(protein,replica)].append(np.std(self.distances[protein][replica][metric])
                    data[(protein,replica)].append(metric_mask.count(True)/n_frames)
                    if metric not in column_names:
                        column_names.append(metric)

                if group_metrics != None:
                    for group in group_metrics:
                        group_mask = [True]*n_frames
                        for metric in masks:
                            if metric in group_metrics[group]:
                                group_mask = [x and y for x, y in zip(masks[metric], group_mask)]

                        data[(protein,replica)].append(group_mask.count(True)/n_frames)
                        column_names.append(group)

        df = pd.DataFrame(data).transpose()

        df.columns = column_names

        return(df)

def _readGromacsIndexFile(file):

    f = open(file, 'r')
    groups = [x.replace('[','').replace(']','').replace('\n','').strip() for x in f.readlines() if x.startswith('[')]

    group_dict = {}
    for i,g in enumerate(groups):
        group_dict[g] = str(i)

    return group_dict
