import argparse
import json

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', default=None, help='Path to the PELE output folder.')
parser.add_argument('--iterations', help='New number of iterations.')
parser.add_argument('--steps', help='New number of steps.')

args=parser.parse_args()
pele_output = args.pele_output
iterations= args.iterations
steps= args.steps

if not isinstance(iterations, type(None)):
    iterations = int(iterations)
if not isinstance(iterations, type(None)):
    steps = int(steps)

# Load modified pele.conf as json
with open(pele_output+'/adaptive.conf') as ac:
    adaptive_conf = json.load(ac)

if iterations != None:
    # Update the number of iterations
    adaptive_conf['simulation']['params']['iterations'] = iterations
if steps != None:
    adaptive_conf['simulation']['params']['peleSteps'] = steps

# Write adaptive.conf
with open(pele_output+'/adaptive.conf', 'w') as ac:
    json.dump(adaptive_conf, ac, indent=4)
