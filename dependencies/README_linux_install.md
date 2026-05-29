# Installation on Linux (Modern Systems)

The original `prepare_proteins.yaml` uses package versions from 2022 that are no longer resolvable on current systems. This guide describes a verified installation method for modern Linux environments.

## Requirements

- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) or [conda](https://docs.conda.io/en/latest/miniconda.html)
- Python environment (Python >= 3.6)

## Step 1 — Install binary tools

Create a dedicated environment for the bioinformatics binaries (blast, dssp, mafft, cd-hit, entrez-direct). These tools require Python <= 3.13 so they go in a separate environment.

```bash
micromamba create -n tools python=3.12 blast dssp mafft cd-hit entrez-direct \
    -c bioconda -c conda-forge --channel-priority strict -y
```

## Step 2 — Add tools to PATH

Add the tools environment binaries to your PATH by appending to your `.bashrc`:

```bash
echo 'export PATH="$HOME/micromamba/envs/tools/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

> Adjust the path to match your micromamba/conda installation location.

## Step 3 — Install Python dependencies

Activate your working Python environment and install the required packages via pip:

```bash
pip install biopython mdtraj pandas numpy scipy matplotlib seaborn tqdm beautifulsoup4 requests
```

## Step 4 — Install prepare_proteins

From the root directory of this repository (where `setup.py` is located):

```bash
pip install -e .
```

## Step 5 — Verify installation

```python
import prepare_proteins
import Bio, mdtraj, pandas, numpy, scipy, matplotlib, seaborn, tqdm
print("All packages imported successfully")
```

Check binary tools are accessible:

```bash
blastn -version
mafft --version
cd-hit -h
mkdssp --version
efetch -help
```

> Note: DSSP is installed as `mkdssp` on modern systems. If any scripts call `dssp`, create a symlink:
> ```bash
> ln -s $(which mkdssp) ~/.local/bin/dssp
> ```

## Step 6 — Schrödinger (optional)

If you use scripts that require Schrödinger's API, add the following to your `.bashrc` (adjust the path):

```bash
export SCHRODINGER=/path/to/schrodinger2021-4
export PATH=$PATH:$SCHRODINGER
```
