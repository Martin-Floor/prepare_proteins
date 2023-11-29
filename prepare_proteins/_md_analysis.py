import os
import json
import numpy as np
import io
import shutil
import pandas as pd

import mdtraj as md

from pkg_resources import resource_stream, Requirement

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact,fixed

import prepare_proteins

class md_analysis:

    def __init__(self,path,triads=None,lig_atoms=None,command='gmx',step='md',traj_name='prot_md_cat_noPBC.xtc',topol_name='prot_md_1.gro',remove_water=True,peptide=False,ligand=False,remove_traj_files=False):

        self.trajectory_paths = {}
        self.topology_paths = {}
        self.distances = {}
        self.angles = {}
        self.triads = triads
        self.models = [folder for folder in os.listdir(path+'/output_models/') if os.path.isdir(path+'/output_models/'+folder)]
        self.lig_atoms = lig_atoms

        if remove_water == True:
            if peptide or ligand:
                option = '18'
            else:
                option = '1'
        else:
            option = '0'

        #Remove boundary conditions from gromacs simulation trajectory file and get paths
        for model in self.models:
            self.trajectory_paths[model] = {}
            self.topology_paths[model] = {}
            for replica in os.listdir(path+'/output_models/'+model):
                traj_path = path+'/output_models/'+model+'/'+replica+'/'+step
                top_em = path+'/output_models/'+model+'/'+replica+'/em/prot_em.gro'

                if not os.path.exists(traj_path) or len(os.listdir(traj_path))==0:
                    self.trajectory_paths[model][replica] = ''
                    self.topology_paths[model][replica] = ''
                    print('WARNING: model '+model+' has no trajectories')

                else:

                    self.trajectory_paths[model][replica] = (traj_path+'/'+traj_name)
                    if remove_water:
                        self.topology_paths[model][replica] =  (traj_path+'/'+topol_name.split('.')[0]+'_no_water.'+topol_name.split('.')[1])
                    else:
                        self.topology_paths[model][replica] =  (traj_path+'/'+topol_name)

                    for file in os.listdir(traj_path):
                        if not os.path.exists(traj_path+'/'+topol_name):
                            print('WARNING: No gro file')
                            continue

                        if file.endswith('.xtc') and not file.endswith('_noPBC.xtc') and not os.path.exists(traj_path+'/'+file.split(".")[0]+'_noPBC.xtc'):

                            if peptide:
                                for line in open(traj_path+'/'+topol_name).readlines():
                                    if 'ACE' in line:
                                        atom_num = int(line[16:21])-2
                                        break

                                line = '[ chainA ]\n'
                                counter = 0
                                for x in range(1,atom_num):
                                    if counter == 0:
                                        line += str(x).rjust(4)
                                    else:
                                        line += str(x).rjust(5)

                                    counter += 1
                                    if counter == 15:
                                        line += '\n'
                                        #lines.append(line)
                                        counter = 0
                                        #line = ''
                                line += '\n'

                                os.system('echo q | '+command+' make_ndx -f '+traj_path+'/'+topol_name+' -o '+'/'.join(traj_path.split('/')[:-1])+'/md/index.ndx')

                                with open('/'.join(traj_path.split('/')[:-1])+'/md/index.ndx','a') as f:
                                    f.write(line)

                                os.system('echo 20 '+option+' | '+command+' trjconv -s '+ traj_path+'/'+file.split(".")[0] +'.tpr -f '+traj_path+'/'+file+' -o '+traj_path+'/'+file.split(".")[0]+'_noPBC.xtc -center -pbc res -ur compact -n '+'/'.join(traj_path.split('/')[:-1])+'/md/index.ndx')
                                #os.system('echo 20 '+option+' | '+command+' trjconv -s '+ traj_path+'/'+file.split(".")[0] +'.tpr -f '+traj_path+'/'+file.split(".")[0]+'_noPBC_int.xtc -o '+traj_path+'/'+file.split(".")[0]+'_noPBC.xtc -center -pbc whole -ur compact')
                            elif ligand:
                                os.system('echo 1 '+option+' | '+command+' trjconv -s '+ traj_path+'/'+file.split(".")[0] +'.tpr -f '+traj_path+'/'+file+' -o '+traj_path+'/'+file.split(".")[0]+'_noPBC.xtc -center -pbc res -ur compact')
                            else:
                                os.system('echo '+option+' | '+command+' trjconv -s '+ traj_path+'/'+file.split(".")[0] +'.tpr -f '+traj_path+'/'+file+' -o '+traj_path+'/'+file.split(".")[0]+'_noPBC.xtc -pbc mol -ur compact')

                    if not os.path.exists(traj_path+'/'+traj_name) and os.path.exists(traj_path+'/'+topol_name):
                        os.system(command+' trjcat -f '+traj_path+'/*_noPBC.xtc -o '+traj_path+'/'+traj_name+' -cat')
                        if remove_traj_files:
                            os.system('rm '+traj_path+'/prot_md_?.xtc')
                            os.system('rm '+traj_path+'/prot_md_?_noPBC.xtc')

                    if not os.path.exists(traj_path+'/'+topol_name.split('.')[0]+'_no_water.'+topol_name.split('.')[1]) and os.path.exists(traj_path+'/'+topol_name) and remove_water == True:
                        os.system('echo '+option+' | '+command+' editconf -ndef -f '+traj_path+'/'+topol_name+' -o '+traj_path+'/'+topol_name.split('.')[0]+'_no_water.'+topol_name.split('.')[1])
                    if not os.path.exists(top_em.split('.')[0]+'_no_water.gro') and os.path.exists(top_em) and remove_water == True:
                        os.system('echo '+option+' | '+command+' editconf -ndef -f '+top_em+' -o '+top_em.split('.')[0]+'_no_water.gro')

    def setupCalculateDistances(self,job_folder='MD_analysis_data',overwrite=False):

        if self.triads == None:
            raise ValueError('Triads were not given to the object')

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
        for model in self.trajectory_paths:
            if not os.path.exists(job_folder+'/'+model):
                os.mkdir(job_folder+'/'+model)
            for replica in self.trajectory_paths[model]:
                if not os.path.exists(self.trajectory_paths[model][replica]):
                    print('WARNING: trajectory file for model '+model+' and replica '+replica+' does not exist.')
                    continue
                if not os.path.exists(self.topology_paths[model][replica]):
                    print('WARNING: topology file for model '+model+' and replica '+replica+' does not exist.')
                    continue

                if not os.path.exists(job_folder+'/'+model+'/'+replica):
                    os.mkdir(job_folder+'/'+model+'/'+replica)
                if not os.path.exists(job_folder+'/'+model+'/'+replica+'/'+self.trajectory_paths[model][replica].split('/')[-1]):
                    shutil.copyfile(self.trajectory_paths[model][replica],job_folder+'/'+model+'/'+replica+'/'+self.trajectory_paths[model][replica].split('/')[-1])
                if not os.path.exists(job_folder+'/'+model+'/'+replica+'/'+self.topology_paths[model][replica].split('/')[-1]):
                    shutil.copyfile(self.topology_paths[model][replica],job_folder+'/'+model+'/'+replica+'/'+self.topology_paths[model][replica].split('/')[-1])

                new_path = job_folder+'/'+model+'/'+replica

                if not os.path.exists(new_path+'/dist.json') or overwrite:
                    command = 'cd '+job_folder+'\n'
                    command += 'python ._'+script_name+' -p '+new_path +' '
                    command += '--triad '+','.join(self.triads[model])+' '
                    if self.lig_atoms != None:
                        with open(new_path+'/lig_atoms.json','w') as f:
                            print(self.lig_atoms[model])
                            json.dump(self.lig_atoms[model],f)
                        command += ' --lig_atoms lig_atoms.json '
                    command += '\n'
                    command += 'cd ..\n'
                    jobs.append(command)
        return jobs


    def calculateDistances(self,folder='MD_analysis_data',calculate=False,overwrite=False):

        if not os.path.exists(folder):
            os.mkdir(folder)

        for model in self.trajectory_paths:
            self.distances[model] = {}
            for replica,path in self.trajectory_paths[model].items():
                if (os.path.exists(folder+'/'+model+'/'+replica+'/dist.json') and not overwrite):
                    with open(folder+'/'+model+'/'+replica+'/dist.json','r') as f:
                        self.distances[model][replica] = json.load(f)
                    print('Reading distance for '+model)
                    continue

                ### WARNING: Option not implemented  with replicas ###
                if calculate:
                    if os.trajectory_paths.exists(path) and os.trajectory_paths.exists('/'.join(path.split('/')[:-2])+'/md/prot_md_1.gro'):
                        print('Calculating distance for '+model)
                        t = md.load(path, top='/'.join(path.split('/')[:-2])+'/md/prot_md_1.gro')
                        #t = t[0:100] #Test#
                        dist_dic = {}
                        dist_dic['ser_his'] = []
                        dist_dic['asp_his'] = []
                        dist_dic['pep'] = []

                        for frame in t:
                            top = frame.top

                            ser = self.triads[model][0][1:]
                            his = self.triads[model][1].split('-')[0][1:]
                            his_name = self.triads[model][1].split('-')[1]
                            asp = self.triads[model][2][1:]
                            asp_res_name = self.triads[model][2][0]

                            carbonyl = self.carbonyls[model]

                            pep_atom = top.select('resSeq '+carbonyl+' and name C')

                            ser_atom = top.select('resSeq '+ser+' and name HG')

                            if his_name == 'NE2':
                                his_asp_atom = top.select('resSeq '+his+' and name HD1')
                                heavy_his_asp_atom = top.select('resSeq '+his+' and name ND1')
                                his_ser_atom = top.select('resSeq '+his+' and name NE2')

                            elif his_name == 'ND1':
                                his_asp_atom = top.select('resSeq '+his+' and name HE2')
                                his_ser_atom = top.select('resSeq '+his+' and name ND1')

                            if asp_res_name == 'D':
                                asp_atom1 = top.select('resSeq '+asp+' and name OD1')
                                asp_atom2 = top.select('resSeq '+asp+' and name OD2')

                            elif asp_res_name == 'G':
                                asp_atom1 = top.select('resSeq '+asp+' and name OE1')
                                asp_atom2 = top.select('resSeq '+asp+' and name OE2')

                            if len(his_asp_atom) == 0:
                                print('WARNING: wrong protonation in model '+model)
                                break

                            dist1 = float(np.linalg.norm(frame.xyz[0][asp_atom1] - frame.xyz[0][his_asp_atom]))
                            dist2 = float(np.linalg.norm(frame.xyz[0][asp_atom2] - frame.xyz[0][his_asp_atom]))

                            if dist1 > dist2:
                                asp_atom = asp_atom2
                                asp_dist = dist2
                            else:
                                asp_atom = asp_atom1
                                asp_dist = dist1

                            dist_dic['ser_his'].append(float(np.linalg.norm(frame.xyz[0][ser_atom] - frame.xyz[0][his_ser_atom])))
                            dist_dic['asp_his'].append(asp_dist)
                            dist_dic['pep'].append(float(np.linalg.norm(frame.xyz[0][ser_atom] - frame.xyz[0][pep_atom])))

                        with open(folder+'/'+model+'_dist.json', 'w') as f:
                            json.dump(dist_dic,f)

                        self.distances[model] = dist_dic

    def plot_distances(self,threshold=0.45,lim=True):
        '''
        def options2(model,replica):

            y1 = self.distances[model][replica]['ser_his']
            x1 = [x*0.1 for x in range(len(y1))]

            y2 = self.distances[model][replica]['asp_his']
            x2 = [x*0.1 for x in range(len(y2))]

            y3 = self.distances[model][replica]['pep']
            x3 = [x*0.1 for x in range(len(y3))]

            plt.plot(x1,y1)
            plt.plot(x2,y2)
            plt.plot(x3,y3)
            plt.axhline(y=threshold, color='r', linestyle='-')
            if lim:
                plt.ylim(0.1,1)

            plt.xlabel("time (ns)")
            plt.ylabel("distance")
            plt.legend(['ser-his','his-asp','pep'])
            plt.show()

        def options1(model):
            interact(options2,model = fixed(model),replica=self.distances[model].keys())
        '''
        def options1(model):
            if self.distances[model] != {}:
                for replica in self.distances[model]:
                    y1 = np.array(self.distances[model][replica]['ser_his'])*10
                    x1 = [x*0.1 for x in range(len(y1))]

                    y2 = np.array(self.distances[model][replica]['asp_his'])*10
                    x2 = [x*0.1 for x in range(len(y2))]

                    y3 = np.array(self.distances[model][replica]['pep'])*10
                    x3 = [x*0.1 for x in range(len(y3))]

                    plt.plot(x1,y1)
                    plt.plot(x2,y2)
                    plt.plot(x3,y3)
                    plt.axhline(y=threshold, color='r', linestyle='-')
                    if lim:
                        plt.ylim(1,10)

                    plt.title(replica)
                    plt.xlabel("time (ns)")
                    plt.ylabel("distance (Å)")
                    plt.legend(['ser-his','his-acd','ser-pep'])
                    plt.show()
            else:
                print('No distance data for model '+model)

        interact(options1,model = sorted(self.distances.keys()))


    def get_distance_prob(self,threshold=0.45):
        data = {}
        for protein in self.distances:
            for replica in self.distances[protein]:

                n_frames = len(self.distances[protein][replica]['ser_his'])
                ser_mask = [x < threshold for x in self.distances[protein][replica]['ser_his']]
                asp_mask = [x < threshold for x in self.distances[protein][replica]['asp_his']]
                pep_mask = [x < threshold for x in self.distances[protein][replica]['pep']]
                ct_mask = [(x and y) for x,y in zip(ser_mask,asp_mask)]
                all_mask = [(x and y and z) for x,y,z in zip(ser_mask,asp_mask,pep_mask)]

                data[(protein,replica)] = [np.mean(self.distances[protein][replica]['ser_his']),
                                            np.std(self.distances[protein][replica]['ser_his']),
                                            np.mean(self.distances[protein][replica]['asp_his']),
                                            np.std(self.distances[protein][replica]['asp_his']),
                                            np.mean(self.distances[protein][replica]['pep']),
                                            np.std(self.distances[protein][replica]['pep']),
                                            ser_mask.count(True)/n_frames,
                                            asp_mask.count(True)/n_frames,
                                            pep_mask.count(True)/n_frames,
                                            ct_mask.count(True)/n_frames,
                                            all_mask.count(True)/n_frames]

                column_names = ['ser_his_mean','ser_his_std',
                                'acd_his_mean','acd_his_std',
                                'ser_pep_mean','ser_pep_std',
                                'ser_his_per','acd_his_per',
                                'ser_pep_per','ct_per','all_per']

        df = pd.DataFrame(data).transpose()

        df.columns = column_names

        return(df)
