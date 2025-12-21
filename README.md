# prepare_proteins
A Python package developed for the BSC-EAPM group to set up protein models calculations.

## Installation and dependencies

For instructions, please check the README file located in the dependencies folder.

## Tutorials

A tutorial for basic usage is available in the docs folder:

 * [Basic protein set up for running PELE](https://github.com/Martin-Floor/prepare_proteins/blob/main/docs/tutorial/01-BasicProteinSetUpForPELE/01-BasicProteinSetUpForPELE.ipynb)

## MLCG simulations (experimental)

MLCG setup is available via `proteinModels.setUpMLCGSimulations`. It builds 5-bead CG inputs (N/CA/CB/C/O, GLY uses 4 beads), strips ligands/ions by default, and writes a `*.pt` structure file plus a CG PDB for analysis. You must provide a trained model checkpoint (`model_with_prior.pt` or the paper's `model_and_prior.pt`). If the file is not found locally, the helper can download it into `~/.cache/prepare_proteins/mlcg` from the Zenodo paper archive (zip) or a custom URL; using the default paper filenames will automatically pull from Zenodo.

Example (shared model folder is created next to `mlcg_jobs`):

```python
from prepare_proteins import proteinModels

models = proteinModels("path/to/pdbs")
jobs = models.setUpMLCGSimulations(
    "mlcg_jobs",
    model_file="model_and_prior.pt",
    model_url="https://zenodo.org/api/records/15465782/files/simulating_a_trained_cg_model.zip/content",
    model_shared_dir="_mlcg_models",
    protocols=("langevin", "pt"),
)
```

Convert MLCG `*_coords_*.npy` outputs into one DCD per trajectory for mdtraj:

```python
from prepare_proteins.MD import mlcg_setup

mlcg_setup.write_mlcg_dcds(
    "mlcg_jobs/1qys_A/langevin/sims",
    "mlcg_jobs/1qys_A/input_files/1qys_A_cg.pdb",
)
```

Analyze an entire `mlcg_jobs` folder and auto-convert missing/mismatched DCDs:

```python
from prepare_proteins.MD.mlcg_analysis import MLCGAnalysis

analysis = MLCGAnalysis("mlcg_jobs", auto_convert=True)
for run in analysis.runs:
    print(run.model_name, run.protocol, run.dcd_paths)
```

Compute RMSD and Q (fraction of native contacts) from the same runs:

```python
analysis = MLCGAnalysis("mlcg_jobs")
run = analysis.runs[0]

rmsd = analysis.compute_rmsd(run, native_pdb="path/to/native.pdb")
q = analysis.compute_q(run, native_pdb="path/to/native.pdb")
```
