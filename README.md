# prepare_proteins
A Python package developed for the BSC-EAPM group to set up protein models calculations.

## Installation and dependencies

For instructions, please check the README file located in the dependencies folder.

## Tutorials

A tutorial for basic usage is available in the docs folder:

 * [Basic protein set up for running PELE](https://github.com/Martin-Floor/prepare_proteins/blob/main/docs/tutorial/01-BasicProteinSetUpForPELE/01-BasicProteinSetUpForPELE.ipynb)

### Bugs / issues 

* When the systems are prepared, some HETATMS change its chain to a different one. For example, in a system with zincs as a cofactor, after the processing, some structures change the chain from "A" to "B". 
