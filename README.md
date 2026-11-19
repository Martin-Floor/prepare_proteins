# prepare_proteins
A Python package developed for the BSC-EAPM group to set up protein models calculations.

## Installation and dependencies

For instructions, please check the README file located in the dependencies folder.

## Tutorials

A tutorial for basic usage is available in the docs folder:

 * [Basic protein set up for running PELE](https://github.com/Martin-Floor/prepare_proteins/blob/main/docs/tutorial/01-BasicProteinSetUpForPELE/01-BasicProteinSetUpForPELE.ipynb)
 * [AF3 tutorial for UPOs with heme and Mg](https://github.com/Martin-Floor/prepare_proteins/blob/main/docs/tutorial/02-AF3UPOWithHemeAndMg/02-AF3UPOWithHemeAndMg.ipynb)
 * [AF3 tutorial continuation: Compound-I waters + Rosetta relax](https://github.com/Martin-Floor/prepare_proteins/blob/main/docs/tutorial/03-AF3UPOCompoundIWatersRosetta/03-AF3UPOCompoundIWatersRosetta.ipynb)
 * [CHARMM-GUI Quick Bilayer backend notes](https://github.com/Martin-Floor/prepare_proteins/blob/main/docs/charmm_gui_quick_bilayer.md)

## BioEmu 2

`sequenceModels.setUpBioEmu2` prepares protein systems for equilibrium sampling
or structure prediction. It writes a `system.yaml` and a `run.sh` per model and
returns shell commands for dispatch. Install [BioEmu 2](https://github.com/microsoft/bioemu2)
in a separate environment on the execution host.

```python
from prepare_proteins import sequenceModels

models = sequenceModels("proteins.fasta")
jobs = models.setUpBioEmu2(
    "bioemu2_jobs",
    num_samples=8,
    batch_size=2,
    seed=42,
    bioemu2_executable="/path/to/bioemu2-env/bin/bioemu2",
    checkpoint_path="/path/to/checkpoint.ckpt",
    msa_cache_dir="/path/to/colabfold_msa",
)
```

For offline execution, stage the checkpoint with its matching `config.yaml`
and populate the BioEmu 2 MSA cache for the requested sequences before running
the commands. Set `mode="structure"` for structure prediction.

Each model's `output/` contains `topology.cif`, `samples.xtc`, and
`physicality.json`. The report gives the status of each frame in the XTC.
The existing `setUpBioEmu` and its trajectory helpers use the BioEmu v1 layout.

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
