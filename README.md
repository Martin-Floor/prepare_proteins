# prepare_proteins
A Python package to fix PDB models for further refinement


### Pending implementation

* For prepare protein optimization
	- Control script for log file monitoring.

### Bugs / issues 

* When the systems are prepared, some HETATMS change its chain to a different one. For example, in a system with zincs as a cofactor, after the processing, some structures change the chain from "A" to "B". 
