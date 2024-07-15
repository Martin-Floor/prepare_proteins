import argparse
import json
import mdtraj as md
import numpy as np

def compute_distances(trajectory, topology, metrics_file):
    """
    Computes distances between specified atom pairs over the trajectory.

    Parameters:
    trajectory (str): Path to the trajectory file.
    topology (str): Path to the topology file.
    metrics_file (str): Path to the JSON file containing distance metrics.

    Returns:
    dict: Dictionary with computed distances.
    """
    # Load trajectory and topology
    t = md.load(trajectory, top=topology)

    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    results = {m: [] for m in metrics}

    for m in metrics:
        distances = []
        for d in metrics[m]:
            atom1 = t.top.select(f'resSeq {d[0][0]} and name {d[0][1]}')
            atom2 = t.top.select(f'resSeq {d[1][0]} and name {d[1][1]}')

            if len(atom1) != 1 or len(atom2) != 1:
                raise ValueError(f'Something wrong with atom definition in metric {m}. Expected single atom selection, got {len(atom1)} and {len(atom2)}.')

            # Compute distances for all frames at once
            pair_distances = md.compute_distances(t, np.array([[atom1[0], atom2[0]]]))
            distances.append(pair_distances)

        # Aggregate minimum distances across all pairs for each frame
        min_distances = np.min(distances, axis=0)
        results[m] = min_distances.tolist()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", help='Path to the trajectory.', required=True)
    parser.add_argument("--topology", help='Path to the topology.', required=True)
    parser.add_argument("-m", "--metrics", help='Path to the metrics JSON file.', required=True)

    args = parser.parse_args()

    results = compute_distances(args.trajectory, args.topology, args.metrics)

    with open('dist.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
