import argparse
import json
import mdtraj as md
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path")
parser.add_argument("-t", "--triad")
parser.add_argument("-c", "--carbonyl")
args = parser.parse_args()
path = '../'+args.path
triad = args.triad.split(',')
if args.carbonyl != '0':
    carbonyl = int(args.carbonyl)
else:
    carbonyl = None

t = md.load(path+'/prot_md_cat_noPBC.xtc', top=path+'/prot_md_1_no_water.gro')
#t = t[0:100] #Test#
dist_dic = {}
dist_dic['ser_his'] = []
dist_dic['asp_his'] = []
dist_dic['pep'] = []

print('Calculating distances for model '+path.split('/')[-2]+' replica '+path.split('/')[-1])

for frame in t:
    top = frame.top

    ser = triad[0][1:]
    his = triad[1].split('-')[0][1:]
    his_name = triad[1].split('-')[1]
    asp = triad[2][1:]
    asp_res_name = triad[2][0]

    ser_atom = top.select('resSeq '+ser+' and name HG')
    heavy_ser_atom = top.select('resSeq '+ser+' and name OG')

    if carbonyl != None:
        first_pep_res = top.atom(top.select('resname ACE')[1])
        resnum = first_pep_res.residue.index
        pep_atom = top.select('resid '+str(int(carbonyl)+resnum)+' and name C')[0]

    '''
    if his_name == 'NE2':
        his_asp_atom = top.select('resSeq '+his+' and name HD1')
        heavy_his_asp_atom = top.select('resSeq '+his+' and name ND1')
        his_ser_atom = top.select('resSeq '+his+' and name NE2')

    elif his_name == 'ND1':
        his_asp_atom = top.select('resSeq '+his+' and name HE2')
        his_ser_atom = top.select('resSeq '+his+' and name ND1')
    '''

    if  len(top.select('resSeq '+his+' and (name HE2 or name HD1)'))>1:
            raise ValueError('Wrong Protonation')

    if len(top.select('resSeq '+his+' and name HE2'))>0:
        his_asp_atom = top.select('resSeq '+his+' and name HE2')
        heavy_his_asp_atom = top.select('resSeq '+his+' and name NE2')
        his_ser_atom = top.select('resSeq '+his+' and name ND1')
    else:
        his_asp_atom = top.select('resSeq '+his+' and name HD1')
        heavy_his_asp_atom = top.select('resSeq '+his+' and name ND1')
        his_ser_atom = top.select('resSeq '+his+' and name NE2')

    if asp_res_name == 'D':
        asp_atom1 = top.select('resSeq '+asp+' and name OD1')
        asp_atom2 = top.select('resSeq '+asp+' and name OD2')

    elif asp_res_name == 'E':
        asp_atom1 = top.select('resSeq '+asp+' and name OE1')
        asp_atom2 = top.select('resSeq '+asp+' and name OE2')

    #if len(his_asp_atom) == 0:
    #    print('WARNING: wrong protonation in model '+path.split('/')[-3])
    #    break

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

    if carbonyl != None:
        dist_dic['pep'].append(float(np.linalg.norm(frame.xyz[0][heavy_ser_atom] - frame.xyz[0][pep_atom])))
    else:
        dist_dic['pep'].append(0)

with open(path+'/dist.json', 'w') as f:
    json.dump(dist_dic,f)
    print('Saved distances for model '+path.split('/')[-2]+' replica '+path.split('/')[-1])
