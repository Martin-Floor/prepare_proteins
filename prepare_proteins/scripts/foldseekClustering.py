#!/usr/bin/env python3
import argparse
import os
import sys
import json
import subprocess
import shutil

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Foldseek clustering and save results.")
parser.add_argument("input_folder", help="Path to the folder containing input PDB models.")
parser.add_argument("output_folder", help="Path to store clustering results.")
parser.add_argument("tmp_folder", help="Temporary folder for Foldseek intermediate files.")
parser.add_argument("--cov_mode", type=int, default=0, help="Coverage mode for Foldseek (default: 0).")
parser.add_argument("--evalue", type=float, default=10.0, help="E-value threshold for clustering (default: 10.0).")
parser.add_argument("--c", type=float, default=0.9, help="Fraction of aligned residues for clustering (default: 0.9).")
parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary folder after execution.")
parser.add_argument("--cluster-reassign", action="store_true", help="Cascaded clustering can cluster sequence that do not fulfill the clustering criteria. \
Cluster reassignment corrects these errors.")
args = parser.parse_args()

# Normalize paths so they remain relative
input_folder = os.path.normpath(args.input_folder)
output_folder = os.path.normpath(args.output_folder)
tmp_folder = os.path.normpath(args.tmp_folder)

# Ensure output and temporary directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(tmp_folder, exist_ok=True)

# Run Foldseek clustering
foldseek_cmd  = f"foldseek easy-cluster {input_folder} {output_folder} {tmp_folder} "
foldseek_cmd += f"--cov-mode {args.cov_mode} -e {args.evalue} -c {args.c} "
if args.cluster_reassign:
    foldseek_cmd += f"--cluster-reassign "
subprocess.run(foldseek_cmd, shell=True, check=True)

# Parse the clustering results
cluster_output_file = "result_cluster.tsv"
output_json = os.path.join(output_folder, "clusters.json")
clusters = {}

if not os.path.exists(cluster_output_file):
    print(f"WARNING: Clustering output file {cluster_output_file} not found.", file=sys.stderr)
    sys.exit(1)

with open(cluster_output_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        centroid = parts[0].replace('.pdb', '')
        member = parts[1].replace('.pdb', '')
        clusters.setdefault(centroid, []).append(member)

renamed_clusters = {
    f"cluster_{i+1:02d}": {"centroid": k, "members": v}
    for i, (k, v) in enumerate(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))
}

with open(output_json, "w") as json_file:
    json.dump(renamed_clusters, json_file, indent=2)

print(f"Saved clustering results to {output_json}")

# Delete the temporary folder unless the --keep-tmp flag is used
if not args.keep_tmp:
    try:
        shutil.rmtree(tmp_folder)
        print(f"Deleted temporary folder: {tmp_folder}")
    except Exception as e:
        print(f"Error deleting temporary folder {tmp_folder}: {e}", file=sys.stderr)
else:
    print(f"Keeping temporary folder: {tmp_folder}")
