# Automatic Lipid Bilayer Building Plan

## Goal

Add first-class membrane-system preparation to `prepare_proteins` for OpenMM workflows, so a membrane protein model can be converted into a solvated bilayer system without leaving the library.

The primary target is:

- protein or protein-complex input PDB/mmCIF
- automatic POPC bilayer build
- OpenMM-ready output through the existing `setUpOpenMMSimulations()` path

## Current State

### Already present

- OpenMM simulation setup:
  - [prepare_proteins/_protein_models.py](/home/martin/repos/prepare_proteins/prepare_proteins/_protein_models.py)
  - `proteinModels.setUpOpenMMSimulations()`
- OpenMM model wrapper:
  - [prepare_proteins/MD/openmm_setup.py](/home/martin/repos/prepare_proteins/prepare_proteins/MD/openmm_setup.py)
  - `openmm_md`
- Rosetta membrane positioning helper:
  - [prepare_proteins/_protein_models.py](/home/martin/repos/prepare_proteins/prepare_proteins/_protein_models.py)
  - `setUpMembranePositioning()`
- Local OpenMM exposes `Modeller.addMembrane()`
- Local OpenMM data contains:
  - `amber14/lipid17.xml`

### Missing

- no bilayer-building entry point in the OpenMM workflow
- no force-field selection logic for membrane runs
- no membrane-orientation handoff between the Rosetta membrane helper and OpenMM setup
- no standardized membrane metadata saved alongside generated jobs

## Design Choice

Use `OpenMM Modeller.addMembrane()` as the first implementation path.

Reason:

- it is already available in the local environment
- it keeps the feature inside `prepare_proteins`
- it avoids making CHARMM-GUI or Packmol-Memgen a hard dependency

This should be phase 1 only. More advanced builders can be added later as optional backends.

## Scope For Phase 1

Support:

- single membrane protein or membrane protein complex
- input already roughly oriented along the membrane normal
- lipid type `POPC`
- `amber14 + lipid17 + TIP3P`
- automatic solvent and ions through `Modeller.addMembrane()`

Do not support yet:

- glycosylated membrane systems as a turnkey mode
- mixed bilayers
- asymmetric membranes
- automatic topology inference from arbitrary structures without any orientation hint
- non-POPC lipids unless confirmed available in the local OpenMM force-field install

## Proposed Public API

Extend `proteinModels.setUpOpenMMSimulations()` with a new option:

```python
setUpOpenMMSimulations(
    ...,
    membrane_system=None,
)
```

Where `membrane_system` is either `None` or a dict like:

```python
membrane_system = {
    "lipid_type": "POPC",
    "minimum_padding_nm": 1.0,
    "membrane_center_z_nm": 0.0,
    "positive_ion": "Na+",
    "negative_ion": "Cl-",
    "ionic_strength_molar": 0.15,
    "neutralize": True,
    "orientation_mode": "as_is",
    "oriented_pdb": None,
    "protein_chain_ids": ["A"],
}
```

`orientation_mode` values for phase 1:

- `"as_is"`: assume the input is already membrane-oriented
- `"use_oriented_pdb"`: use `oriented_pdb` if provided

Planned later:

- `"rosetta_topology"`: call the existing Rosetta membrane positioning helper
- `"tm_residue_pca"`: infer orientation from user-supplied TM residues

## Internal Implementation Plan

### 1. Extend `openmm_md`

File:

- [prepare_proteins/MD/openmm_setup.py](/home/martin/repos/prepare_proteins/prepare_proteins/MD/openmm_setup.py)

Add:

- `openmm_md.addMembrane(...)`
- `openmm_md.buildMembraneSystem(...)`

Responsibilities:

- call `self.modeller.addMembrane(...)`
- update `self.positions`
- record build metadata on the object
- write an intermediate membrane-built PDB for inspection

### 2. Extend force-field setup

File:

- [prepare_proteins/MD/openmm_setup.py](/home/martin/repos/prepare_proteins/prepare_proteins/MD/openmm_setup.py)

Update `setUpFF()` so membrane runs can request:

- `amber14-all.xml`
- `amber14/tip3pfb.xml` or `amber14/tip3p.xml`
- `amber14/lipid17.xml`

For phase 1, hard-code the supported membrane combination and fail loudly on unsupported requests.

### 3. Thread membrane options through the public setup

File:

- [prepare_proteins/_protein_models.py](/home/martin/repos/prepare_proteins/prepare_proteins/_protein_models.py)

Update `setUpOpenMMSimulations()`:

- add `membrane_system=None`
- validate membrane options
- before backend parameterization, orient or replace the input model if requested
- call `openmm_md.buildMembraneSystem(...)` for membrane runs
- skip the normal `addSolvent` route when membrane mode is active

### 4. Parameterization backend behavior

Files:

- [prepare_proteins/MD/parameterization/openff.py](/home/martin/repos/prepare_proteins/prepare_proteins/MD/parameterization/openff.py)
- [prepare_proteins/MD/parameterization/ambertools.py](/home/martin/repos/prepare_proteins/prepare_proteins/MD/parameterization/ambertools.py)

Plan:

- phase 1 membrane support should be `ambertools` only
- reject `parameterization_method='openff'` for membrane mode until tested

Reason:

- the current membrane path is easiest to stabilize through Amber/OpenMM export
- `openff` currently centers on soluble protein + ligand handling

### 5. Save membrane metadata

Per model, write:

- `membrane_build.json`
- `oriented_input.pdb`
- `membrane_built.pdb`

Useful fields:

- lipid type
- padding
- ion conditions
- force-field files
- orientation mode
- source input file

## Orientation Strategy

This is the real risk area.

Phase 1 should not promise full automatic orientation. It should only support:

1. input already oriented correctly
2. user passes an oriented structure generated elsewhere

Phase 2 should connect the existing Rosetta helper:

- [prepare_proteins/_protein_models.py](/home/martin/repos/prepare_proteins/prepare_proteins/_protein_models.py)
  `setUpMembranePositioning()`

That would allow:

- generate span/topology estimate
- produce an oriented protein model
- feed that into the OpenMM bilayer builder

## Validation Plan

### Unit tests

Add focused tests that mock or lightly exercise:

- membrane option validation
- force-field selection for membrane mode
- metadata writing
- error on unsupported backend

### Integration tests

Use a small membrane protein test case to verify:

- bilayer is built
- system exports `prmtop` and `inpcrd`
- OpenMM runner starts

### Visual/structural checks

Add a helper notebook or script that reports:

- number of lipids
- box vectors
- protein center relative to membrane center
- number of atoms/residues after build

## Recommended Development Order

1. Add `membrane_system` validation to `setUpOpenMMSimulations()`
2. Add `openmm_md.addMembrane()` and membrane metadata writing
3. Support `amber14 + lipid17 + POPC` only
4. Add one integration test
5. Add one tutorial notebook using an Erv29p-like membrane protein
6. Add optional orientation handoff from `setUpMembranePositioning()`

## Explicit Non-Goals For The First Merge

- no automatic mixed bilayer composition
- no glycoprotein membrane setup without extra force-field validation
- no attempt to replace CHARMM-GUI feature-for-feature

## Practical First Milestone

The first milestone is successful when this works:

```python
models.setUpOpenMMSimulations(
    job_folder='md_jobs',
    replicas=3,
    simulation_time=100,
    ff='amber14',
    parameterization_method='ambertools',
    membrane_system={
        'lipid_type': 'POPC',
        'minimum_padding_nm': 1.0,
        'membrane_center_z_nm': 0.0,
        'ionic_strength_molar': 0.15,
        'orientation_mode': 'as_is',
    },
)
```

and produces a membrane-embedded OpenMM system with runnable `prmtop`/`inpcrd` inputs.
