# CHARMM-GUI Quick Bilayer Backend

`prepare_proteins` now includes a `parameterization_method='charmm_gui'` backend for
`proteinModels.setUpOpenMMSimulations()`.

This backend is designed for membrane-protein systems that should be built remotely with
CHARMM-GUI Quick Bilayer and then run locally through the existing OpenMM replica/job
machinery in `prepare_proteins`.

## Current scope

- can reuse an existing CHARMM-GUI PDB Reader `jobid`
- can create a PDB Reader job automatically from a local model PDB or a remote `pdb_id`
- submits Quick Bilayer through the public CHARMM-GUI API
- polls job status and downloads the resulting archive
- extracts AMBER topology/coordinate inputs from the downloaded bundle
- caches request, status, archive, and selected input metadata under the job `parameters/`
  folder

## Required options

Pass the backend through `parameterization_method='charmm_gui'` and provide the backend
options inside `parameterization_options`.

Required keys:

- either `token` / `token_env` or `email` / `password`
- `quick_bilayer`

The `quick_bilayer` mapping must include:

- `margin`
- either `membtype` or both `upper` and `lower`

Useful optional keys:

- `ppm`
- `run_ffconverter`
- `wdist`
- `ion_conc`
- `ion_type`
- `clone_job`
- `prot_projection_upper`
- `prot_projection_lower`
- `workflow_mode`

Supported `workflow_mode` values:

- `full`: submit, wait, download, extract, and generate OpenMM-ready inputs
- `submit_only`: submit jobs and stop before download/extraction
- `collect`: reuse cached submission metadata, poll finished jobs, and download/extract

For PDB Reader creation you have two options:

- pass `pdb_reader_jobid` directly
- pass a `pdb_reader` mapping and let `prepare_proteins` create the PDB Reader job

Useful `pdb_reader` keys:

- `jobid`
- `pdb_id`
- `upload_path`
- `source`
- `pdb_format`
- `correct_pdb`
- `include_hetero`
- `include_water`
- `include_dna`
- `include_rna`
- `system_pH`
- `preserve_hydrogens`

If no `pdb_reader_jobid` or `pdb_reader.jobid` is provided, the backend defaults to:

- uploading the current model PDB (`input_pdb`) to PDB Reader
- keeping protein chains only
- excluding hetero chains and waters
- submitting the default PDB manipulation form

That default is intentionally conservative because it avoids the common ligand/cofactor
pitfalls that can stop unattended PDB Reader jobs.

For glycoprotein models where glycans are stored as separate HET chains, set
`include_hetero=True` for those specific models so the automatic PDB Reader path keeps
the glycan chain instead of silently dropping it.

## Example

```python
from prepare_proteins import proteinModels

models = proteinModels("path/to/pdbs")

jobs = models.setUpOpenMMSimulations(
    job_folder="md_jobs",
    replicas=3,
    simulation_time=100,
    parameterization_method="charmm_gui",
    parameterization_options={
        "token_env": "CHARMMGUI_TOKEN",
        "pdb_reader_jobid": {
            "modelA": "5804504324",
        },
        "quick_bilayer": {
            "modelA": {
                "membtype": "PMm",
                "margin": 20.0,
                "wdist": 22.5,
                "ion_conc": 0.15,
                "ion_type": "NaCl",
                "ppm": True,
                "run_ffconverter": True,
            },
        },
    },
)
```

Login-based authentication is also supported:

```python
jobs = models.setUpOpenMMSimulations(
    job_folder="md_jobs",
    replicas=3,
    simulation_time=100,
    parameterization_method="charmm_gui",
    parameterization_options={
        "email_env": "CHARMMGUI_EMAIL",
        "password_env": "CHARMMGUI_PASSWORD",
        "pdb_reader_jobid": "5804504324",
        "quick_bilayer": {
            "membtype": "PMm",
            "margin": 20.0,
            "ppm": True,
            "run_ffconverter": True,
        },
    },
)
```

In that mode, `prepare_proteins` first calls `POST /api/login`, reads the returned JWT,
and then uses the resulting bearer token for the remaining API calls.

Automatic PDB Reader creation uses the same email/password credentials for the
cookie-backed CHARMM-GUI website session. A token alone is not enough for that step.

## Two-Phase Queueing

If you want to queue several CHARMM-GUI jobs quickly and collect them later, use:

1. `workflow_mode='submit_only'` to submit every model without waiting for downloads
2. a later rerun with `workflow_mode='collect'` to poll/download completed jobs

Do not combine this with `skip_preparation=True`. For CHARMM-GUI, `skip_preparation`
means "use only already-downloaded local inputs" and disables remote collection.

## Automatic PDB Reader example

```python
jobs = models.setUpOpenMMSimulations(
    job_folder="md_jobs",
    replicas=3,
    simulation_time=100,
    parameterization_method="charmm_gui",
    parameterization_options={
        "email_env": "CHARMMGUI_EMAIL",
        "password_env": "CHARMMGUI_PASSWORD",
        "pdb_reader": {
            "system_pH": 7.0,
            "preserve_hydrogens": False,
            "include_hetero": False,
        },
        "quick_bilayer": {
            "membtype": "PMm",
            "margin": 20.0,
            "ppm": True,
            "run_ffconverter": True,
        },
    },
)
```

You can also start PDB Reader from a remote structure instead of the local model file:

```python
"pdb_reader": {
    "pdb_id": "1C3W",
    "source": "RCSB",
    "include_hetero": False,
}
```

## Important limitations

- `membrane_system` and the new `charmm_gui` backend are separate workflows. Do not use
  them together.
- Local ligand/MCPB parameterization options are not supported by this backend yet.
- `ligand_only=True` is not supported by this backend yet.
- Automatic PDB Reader creation currently targets the conservative protein-only path by
  default. If your system needs ligands, cofactors, glycans, or custom PDB Reader edits,
  you may still need to provide a pre-built `pdb_reader_jobid`.

## Cache layout

For model `modelA`, downloaded CHARMM-GUI artifacts are stored under:

```text
<job_folder>/parameters/charmm_gui/modelA/
```

Typical files:

- `pdb_reader_manifest.json`
- `pdb_reader_step1_chain_selection.html`
- `pdb_reader_step2_options.html`
- `pdb_reader_step3_reader.html`
- `request.json`
- `submit_response.json`
- `status_history.jsonl`
- `download.tgz`
- `selected_inputs.json`
- `extracted/`

Authentication credentials and tokens are not written to the cache directory.
