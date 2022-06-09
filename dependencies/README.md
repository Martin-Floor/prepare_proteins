To install dependencies, create a new conda environment from the prepare_proteins.yaml file in this folder as follows:

	conda env create -f prepare_proteins.yaml

After this, you need to activate the prepare_proteins environment:

	conda activate prepare_proteins

Finally, install the prepare_proteins.yaml library from the library's root directory (i.e., where the setup.py script is located) and run:

	python setup.py install
