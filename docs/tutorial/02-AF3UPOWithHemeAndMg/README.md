# AF3 Tutorial for UPOs with Heme and Mg

This tutorial shows how to run AlphaFold 3 jobs for UPO sequences with one heme (`HEM`) and one magnesium ion (`MG`) using `prepare_proteins`.

The workflow uses:
- `sequenceModels.setUpAlphaFold3` to create AF3 JSON inputs and submission commands.
- `bsc_calculations.mn5.jobArrays(..., program="alphafold3")` to create a SLURM array script.
- `sequenceModels.collectAlphaFold3Results` to gather scores and export top structures.
- Optional `filter_strict=True` to apply built-in PROT-HEME-MG quality gates.

## 1. Prerequisites

- `prepare_proteins` installed in your Python environment.
- `bsc_calculations` installed in your Python environment.
- `bsc_alphafold` available in `PATH` (MN5/BSC wrapper expected by this method).
- `sbatch` available (SLURM environment).
- `WEIGHTS` environment variable pointing to AF3 weights:

```bash
export WEIGHTS=/path/to/alphafold3/weights
```

Input file requirements:
- FASTA headers are used as model names.
- Sequences must contain standard amino-acid one-letter codes.
- For secreted UPOs, use the mature sequence if that is your target state.

## 2. Prepare the UPO FASTA

Create `upo_sequences.fasta`:

```fasta
>UPO_001
MALVAVLLAASAFAAPVAAEGVTVVGPDATVKPASPVTVTTVGATPTVTAVDGYTVRV
>UPO_002
MKTLTALALALAGSAFAAPAQATVVGPKATVKPSAPVTVTTVGATPTVTVVDGYTVRV
```

Replace these toy sequences with your real full-length (or mature-domain) UPO sequences before production runs.

## 3. Set up AF3 jobs (protein + HEM + MG)

```python
from prepare_proteins import sequenceModels

seqs = sequenceModels("upo_sequences.fasta")

jobs = seqs.setUpAlphaFold3(
    job_folder="af3_upo_heme_mg",
    ligands=["HEM", "MG"],     # keeps PROT, HEM, MG order
    model_seeds=[1, 2, 3, 4, 5],
    skip_finished=True,
    benchmark=True,
)

print(f"Prepared {len(jobs)} AF3 jobs")
print(jobs[0])
```

What this writes:
- `af3_upo_heme_mg/<MODEL>/input/<MODEL>.json`
- `af3_upo_heme_mg/<MODEL>/output/`

Each generated command runs:
1. `bsc_alphafold input output $WEIGHTS`
2. `sbatch runner input`

If your runner script has a different name, pass `runner_name="your_runner_script"`.

## 4. Create and submit a SLURM array script (MN5)

```python
import bsc_calculations
import subprocess

slurm_script = "slurm_array_af3_upo.sh"
bsc_calculations.mn5.jobArrays(
    jobs,
    script_name=slurm_script,
    job_name="AF3_UPO_HEM_MG",
    output="AF3_UPO_HEM_MG",
    partition="acc_bscls",
    program="alphafold3",
    gpus=1,
)

# Submit:
subprocess.run(["sbatch", slurm_script], check=True)
```

To resume partially finished runs, keep `skip_finished=True` in `setUpAlphaFold3`. A model is considered finished when `ranking_scores.csv` exists under its `output` tree.

## 5. Collect AF3 results and export top models

```python
scores_df, selected_df, copied_paths = seqs.collectAlphaFold3Results(
    af_folder="af3_upo_heme_mg",
    output_folder="af3_upo_heme_mg_top_models",
    metric="ranking_score",
    top_models=5,
    append_model_index=True,   # required to keep >1 model per sequence
    return_selected=True,
)

print(scores_df.head())
print(selected_df.head())
print(copied_paths)
```

Important details:
- If `top_models > 1` and `append_model_index=False`, only one structure per model name is copied.
- Existing files are not overwritten unless `overwrite=True`.
- Export converts AF3 CIF files to PDB.
- During CIF->PDB export, multi-character AF3 ligand chain IDs (for example `MGA`) are remapped to one-character PDB chains.

## 6. Optional strict PROT-HEME-MG filtering

```python
scores_df, selected_df, copied_paths, filter_info = seqs.collectAlphaFold3Results(
    af_folder="af3_upo_heme_mg",
    output_folder="af3_upo_heme_mg_filtered_models",
    metric="ranking_score",
    top_models=5,
    append_model_index=True,
    return_selected=True,
    filter_strict=True,
    return_filter=True,
)

print(filter_info["counts"])
print(filter_info["reasons"])
```

`filter_strict=True` adds:
- `af3_strict_pass`
- `af3_strict_reason`

to the returned dataframe.

Chain-order requirement for strict filtering:
- The built-in filter assumes chain index order `0=protein`, `1=heme`, `2=mg`.
- Keep sequence + ligands in this order (as shown above).
- If you add extra cofactors/ligands or change ordering, strict defaults are no longer aligned with the intended chain mapping.

## 7. Per-model ligand customization (optional)

You can override ligands for specific UPOs:

```python
jobs = seqs.setUpAlphaFold3(
    job_folder="af3_upo_heme_mg",
    model_seeds=[1, 2, 3],
    ligands={
        "default": ["HEM", "MG"],
        "UPO_007": {"HEM": 1, "MG": 2},
    },
)
```

## 8. Troubleshooting checklist

- `No ranking scores were found`: the AF3 jobs likely did not finish or failed before writing `ranking_scores.csv`.
- `Metric ... not found`: check the `metric` argument against columns in `ranking_scores.csv`.
- No copied files but scores exist: check `output_folder`, `overwrite`, and `append_model_index`.
- `WEIGHTS` errors: confirm `echo $WEIGHTS` points to the AF3 parameter directory used by your cluster wrapper.
