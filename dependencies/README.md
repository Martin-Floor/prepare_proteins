To install dependencies, create a new conda environment from the prepare_proteins.yaml file in this folder as follows:

	conda env create -f prepare_proteins.yaml

After this, you need to activate the prepare_proteins environment:

	conda activate prepare_proteins

Then, install the prepare_proteins.yaml library from the library's root directory (i.e., where the setup.py script is located) and run:

	python setup.py install

Finally to successfully run the scripts that use Schrödinger's API you need to add the following lines to your .bashrc file (adapting the path to your case):

	# Export SCHRODINGER
	SCHRODINGER=/path/to/schrodinger2021-4
	PATH=$PATH:$SCHRODINGER

