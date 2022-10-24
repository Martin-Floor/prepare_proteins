import os
import json
import numpy as np
import io

import mdtraj as md

from pkg_resources import resource_stream, Requirement

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact,fixed

import prepare_proteins

class md_analysis:

    def __init__(self,path,triads,command='gmx',step='md',traj_name='prot_md_cat_noPBC.xtc',remove_water=True):

        self.paths = {}
        self.distances = {}
        self.angles = {}
        self.triads = triads
        self.models = os.listdir(path+'/output_models/')

        #Remove boundary conditions from gromacs simulation trajectory file and get paths

        for model in self.models:
            traj_path = path+'/output_models/'+model+'/'+step
            if len(os.listdir(traj_path))==0:
                print('WARNING: model '+model+' has no trajectories')
            else:
                for file in os.listdir(traj_path):
                    if file.endswith('.xtc') and not file.endswith('_noPBC.xtc') and not os.path.exists(traj_path+'/'+file.split(".")[0]+'_noPBC.xtc'):
                        if remove_water == True:
                            option = '14'
                        else:
                            option = '0'
                        os.system('echo '+option+' | '+command+' trjconv -s '+ traj_path+'/'+file.split(".")[0] +'.tpr -f '+traj_path+'/'+file+' -o '+traj_path+'/'+file.split(".")[0]+'_noPBC.xtc -pbc mol -ur compact')

                if not os.path.exists(traj_path+'/'+traj_name):
                    os.system(command+' trjcat -f '+traj_path+'/*_noPBC.xtc -o '+traj_path+'/'+traj_name+' -cat')

                if not os.path.exists('/'.join(traj_path.split('/')[:-1])+'/md/prot_md_1_no_water.gro') and remove_water == True:
                    os.system('echo 14 | gmx editconf -ndef -f '+'/'.join(traj_path.split('/')[:-1])+'/md/prot_md_1.gro -o '+'/'.join(traj_path.split('/')[:-1])+'/md/prot_md_1_no_water.gro')

        for model in self.models:
            traj_path = path+'/output_models/'+model+'/'+step
            self.paths[model] = (traj_path+'/'+traj_name)

    def calculateDistances(self,folder='MD_analysis_data',overwrite=False):


        if not os.path.exists(folder):
            os.mkdir(folder)

        for model,path in self.paths.items():
            self.distances[model] = {}
            self.distances[model]['ser_his'] = []
            self.distances[model]['asp_his'] = []

            if os.path.exists(folder+'/'+model+'_dist.json') and not overwrite:
                with open(folder+'/'+model+'_dist.json','r') as f:
                    self.distances[model] = json.load(f)
                print('Reading distance for '+model)
                continue

            if os.path.exists(path) and os.path.exists('/'.join(path.split('/')[:-2])+'/md/prot_md_1.gro'):
                print('Calculating distance for '+model)
                t = md.load(path, top='/'.join(path.split('/')[:-2])+'/md/prot_md_1.gro')
                #t = t[0:100] #Test#
                dist_dic = {}
                dist_dic['ser_his'] = []
                dist_dic['asp_his'] = []

                for frame in t:
                    top = frame.top

                    ser = self.triads[model][0][1:]
                    his = self.triads[model][1].split('-')[0][1:]
                    his_name = self.triads[model][1].split('-')[1]
                    asp = self.triads[model][2][1:]
                    asp_res_name = self.triads[model][2][0]

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

                with open(folder+'/'+model+'_dist.json', 'w') as f:
                    json.dump(dist_dic,f)

                self.distances[model] = dist_dic

    def plot_distances(self,threshold=0.45):
        def options1(protein):

            y1 = self.distances[protein]['ser_his']
            x1 = [x*0.1 for x in range(len(y1))]

            y2 = self.distances[protein]['asp_his']
            x2 = [x*0.1 for x in range(len(y1))]

            plt.plot(x1,y1)
            plt.plot(x2,y2)
            plt.axhline(y=threshold, color='r', linestyle='-')
            plt.ylim(0.1,1)

            plt.xlabel("time (ns)")
            plt.ylabel("distance")
            plt.legend(['ser-his','his-asp'])
            plt.show()

        interact(options1,protein = self.distances.keys())

    def get_distance_prob(self,threshold=0.45):
        dist_prob = {}
        for protein in self.distances:
            if self.distances[protein]['ser_his'] != [] and self.distances[protein]['asp_his'] != []:
                dist_prob[protein] = {}
                dist_prob[protein]['ser_his'] = [x < threshold for x in self.distances[protein]['ser_his']].count(True)/len(self.distances[protein]['ser_his'])
                dist_prob[protein]['asp_his'] = [x < threshold for x in self.distances[protein]['asp_his']].count(True)/len(self.distances[protein]['asp_his'])
                dist_prob[protein]['both'] = [(x < threshold and y < threshold) for x,y in zip(self.distances[protein]['ser_his'], self.distances[protein]['asp_his'])].count(True)/len(self.distances[protein]['ser_his'])

        return(dist_prob)
