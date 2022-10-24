import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prepare_proteins",
    version="0.0.1",
    author="Martin Floor",
    author_email="martinfloor@gmail.com",
    description="A Python package to fix PDB models for further refinement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    package_data = {'':['scripts/md/gromacs/ff/amber99sb-star-ildn/*','scripts/md/gromacs/mdp/*']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
    'matplotlib',
    ],
)
