import argparse
import json
import mdtraj as md
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--trajectory", help='Path to the trajectory.')
parser.add_argument("--topology", help='Path to the topology.')
parser.add_argument("-m", "--metrics",default=None)

args = parser.parse_args()

trajectory = args.trajectory
topology = args.topology
if args.metrics != None:
    with open(args.metrics,'r') as f:
        metrics = json.load(f)
else:
    metrics = None

t = md.load(trajectory, top=topology)

results = {}

for frame in t:
    top = frame.top
    for m in metrics:
        if m not in results:
            results[m] = []

        distances = []
        for d in metrics[m]:

            atom1 = top.select('resSeq '+str(d[0][0])+' and name '+d[0][1])
            atom2 = top.select('resSeq '+str(d[1][0])+' and name '+d[1][1])

            if len(atom1) < 1 or len(atom2) < 1:
                raise ValueError('Something wrong with atom definition in metric '+m+'. Atom selection is empty')

            if len(atom1) > 1 or len(atom2) > 1:
                raise ValueError('Something wrong with atom definition in metric '+m+'. Atom selection is more than 1 atom')

            dist = float(np.linalg.norm(frame.xyz[0][atom1] - frame.xyz[0][atom2]))

            distances.append(dist)

        results[m].append(min(distances))

with open('dist.json', 'w') as f:
    json.dump(results,f)
