#!/usr/bin/env python3
"""
bioemu_batch_filter.py

Apply BioEmu filtering in batches for a single model folder.
Supports:
  • skip-finished (based on existing samples_XXXX.xtc)
  • merge trajectories into samples.xtc
  • remove per‐batch samples after merge
"""

import os
import re
import shutil
import subprocess
import argparse
import mdtraj as md

def filter_model(model_folder,
                 batch_size,
                 sequence,
                 skip_finished=False,
                 merge_trajectories=False,
                 remove_batch_samples=False,
                 cache_subdir='cache',
                 tmp_subdir='tmp',
                 verbose=False):

    # 1) optional skip-finished logic
    if skip_finished:
        npz_files = sorted(f for f in os.listdir(model_folder) if f.endswith('.npz'))
        expected, load = 0, 0
        for fn in npz_files:
            try:
                s, e = fn.split('_')[1:3]
                load += int(e.split('.')[0]) - int(s)
            except:
                continue
            if load >= batch_size:
                expected += 1
                load = 0

        pat = re.compile(r'^samples_(\d{4})\.xtc$')
        existing = [int(m.group(1))
                    for f in os.listdir(model_folder)
                    if (m := pat.match(f))]
        if existing and max(existing) >= expected:
            if verbose:
                print(f"[SKIP] {model_folder}: already has {max(existing)} ≥ {expected} batches")
            return

    # 2) prepare cache
    cache_dir = os.path.join(model_folder, cache_subdir)
    os.makedirs(cache_dir, exist_ok=True)

    # 3) batch loop
    npz_files = sorted(f for f in os.listdir(model_folder) if f.endswith('.npz'))
    batch_load = 0
    batch_files = []
    batch_index = 0

    for fn in npz_files:
        try:
            parts = fn.split('_')
            length = int(parts[2].split('.')[0]) - int(parts[1])
        except:
            if verbose:
                print(f"[WARN] malformed {fn}, skipping")
            continue

        batch_load += length
        batch_files.append(fn)

        if batch_load < batch_size:
            continue

        batch_index += 1
        idx = str(batch_index).zfill(4)
        tmp_dir = os.path.join(model_folder, tmp_subdir)

        # move .npz files to tmp
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        for f in batch_files:
            shutil.move(os.path.join(model_folder, f), tmp_dir)

        # run bioemu.sample
        cmd = (
            f"cd {tmp_dir} && "
            f"python -m bioemu.sample "
            f"--sequence {sequence} "
            f"--num_samples 1 --batch_size_100 1 "
            f"--cache_embeds_dir {cache_dir} "
            f"--output_dir {tmp_dir} && "
            f"cd .."
        )

        if verbose:
            print(f"[{model_folder}] Running batch {idx} ({len(batch_files)} files)")

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] batch {idx} failed:\n{result.stderr}")
            return

        # copy sample output
        src_xtc = os.path.join(tmp_dir, 'samples.xtc')
        if not os.path.exists(src_xtc):
            raise RuntimeError(f"No samples.xtc in {tmp_dir}")
        dst_xtc = os.path.join(model_folder, f"samples_{idx}.xtc")
        shutil.copy(src_xtc, dst_xtc)
        shutil.copy(os.path.join(tmp_dir, 'topology.pdb'),
                    os.path.join(model_folder, 'topology.pdb'))

        # move .npz files back
        for f in batch_files:
            shutil.move(os.path.join(tmp_dir, f),
                        os.path.join(model_folder, f))

        # cleanup
        shutil.rmtree(tmp_dir)
        batch_load = 0
        batch_files = []

    # 4) optional merge
    sample_files = sorted(
        os.path.join(model_folder, f)
        for f in os.listdir(model_folder)
        if f.startswith('samples_') and f.endswith('.xtc')
    )
    if merge_trajectories and sample_files:
        topo = os.path.join(model_folder, 'topology.pdb')
        traj = md.load(sample_files, top=topo)
        traj.superpose(md.load(topo))
        traj.save(os.path.join(model_folder, 'samples.xtc'))
        if verbose:
            print(f"[{model_folder}] Merged into samples.xtc")

    # 5) optional cleanup
    if remove_batch_samples and sample_files:
        for sf in sample_files:
            os.remove(sf)
        if verbose:
            print(f"[{model_folder}] Removed per-batch samples")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run BioEmu filtering in batches on a single model folder"
    )
    p.add_argument("model_folder",
                   help="Folder with your .npz files")
    p.add_argument("--sequence", required=True,
                   help="FASTA sequence or path for --sequence")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Cumulative length per batch")
    p.add_argument("--skip-finished", action="store_true",
                   help="Skip if samples_XXXX.xtc already cover all batches")
    p.add_argument("--merge-trajectories", action="store_true",
                   help="After filtering, merge into samples.xtc")
    p.add_argument("--remove-batch-samples", action="store_true",
                   help="Delete per-batch samples_XXXX.xtc after merge")
    p.add_argument("--cache-subdir", default="cache",
                   help="Cache folder name in model_folder")
    p.add_argument("--tmp-subdir", default="tmp",
                   help="Temporary folder name in model_folder")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose output")

    args = p.parse_args()

    filter_model(
        model_folder=os.path.abspath(args.model_folder),
        batch_size=args.batch_size,
        sequence=args.sequence,
        skip_finished=args.skip_finished,
        merge_trajectories=args.merge_trajectories,
        remove_batch_samples=args.remove_batch_samples,
        cache_subdir=args.cache_subdir,
        tmp_subdir=args.tmp_subdir,
        verbose=args.verbose,
    )
