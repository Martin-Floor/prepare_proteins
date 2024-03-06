import os
import json
import argparse

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', help='Path to the PELE output folder.')
parser.add_argument('point', help='Comma-separated xyz coordinate of the target point.')
parser.add_argument('--epsilon', default=0.50, help='Bias epsilon parameter')
parser.add_argument('--ligand_chain', default='L', help='Ligand chain')
parser.add_argument('--ligand_index', default=1, help='Ligand index')
args=parser.parse_args()

coords = args.point.replace("point_","")
pele_output = args.pele_output
point = [float(x) for x in coords.split(',')]
epsilon = float(args.epsilon)
ligand_chain = args.ligand_chain
ligand_index = int(args.ligand_index)

# Modify pele.conf file to remove unquoted words
pele_words = ['COMPLEXES', 'PELE_STEPS', 'SEED']
with open(pele_output+'/pele.conf.tmp', 'w') as tmp:
    with open(pele_output+'/pele.conf') as pc:
        for l in pc:
            for w in pele_words:
                if w in l:
                    l = l.replace('$'+w, '"$'+w+'"')
                    break
            tmp.write(l)

# Load modified pele.conf as json
with open(pele_output+'/pele.conf.tmp') as tmp:
    json_conf = json.load(tmp)

# Get PELE csts
pele_tasks = json_conf['commands'][0]['PeleTasks'][0]['metrics']
random_tasks = [1 for t in pele_tasks if t['type'] == 'random']

# Add to pint bias
task = {}
task['type'] = 'distanceToPoint'
task['point'] = point
task['atoms'] = {}
task['atoms']['links'] = {}
task['atoms']['links']['ids'] = [ligand_chain+':'+str(ligand_index)]
pele_tasks.append(task)
task_index = len(pele_tasks) - len(random_tasks) + 4

# Write pele.conf
with open(pele_output+'/pele.conf.tmp', 'w') as tmp:
    json.dump(json_conf, tmp, indent=4)

# Add quotes to pele reserved words
with open(pele_output+'/pele.conf', 'w') as pc:
    with open(pele_output+'/pele.conf.tmp') as tmp:
        for l in tmp:
            for w in pele_words:
                if w in l:
                    l = l.replace('"$'+w+'"','$'+w)
            pc.write(l)

os.remove(pele_output+'/pele.conf.tmp')

# Modify adaptive.conf
# Load modified pele.conf as json
with open(pele_output+'/adaptive.conf') as ac:
    adaptive_conf = json.load(ac)

spawning = {}
spawning['type'] = 'epsilon'
spawning['params'] = {}
spawning['params']['reportFilename'] = 'report'
spawning['params']['metricColumnInReport'] = task_index
spawning['params']['epsilon'] = epsilon
spawning['params']['condition'] = 'min'
spawning['density'] = {}
spawning['density']['type']= 'continuous'
adaptive_conf['spawning'] = spawning

# Write adaptive.conf
with open(pele_output+'/adaptive.conf', 'w') as ac:
    json.dump(adaptive_conf, ac, indent=4)
