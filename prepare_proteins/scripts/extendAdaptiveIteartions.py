import argparse
import json

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('pele_output', default=None, help='Path to the PELE output folder.')
parser.add_argument('iterations', help='New number of iterations.')

args=parser.parse_args()
pele_output = args.pele_output
iterations= int(args.iterations)

# Load modified pele.conf as json
with open(pele_output+'/adaptive.conf') as ac:
    adaptive_conf = json.load(ac)

# Update the number of iterations
adaptive_conf['simulation']['params']['iterations'] = iterations

# Write adaptive.conf
with open(pele_output+'/adaptive.conf', 'w') as ac:
    json.dump(adaptive_conf, ac, indent=4)
