import os
import shutil
import networkx as nx
import pandas as pd
import math
import prepare_proteins
from Bio import PDB

class pdb_formating:

    def __init__(self):
        pass

    
    def _getBondPele(self, pele_params, ligand_name):
        """
        Calculate the bonds and distance for pele params
        :param pele_params: path to the pele_params file
        :return: dataframe with the atomnames of the bounds
        Parameters
        ----------
        pele_params: str
            Path to the pele_params file
        ligand_name: str
            Ligand or chain name

        Returns
        -------
        df_pele: pd.dataframe
            Dataframe with the bound and distance

        """
        df_pele = pd.DataFrame(columns=['AnSource', 'AnTarget', 'Dist'])
        atom_names = {}
        with open(pele_params) as pele:
            bnd = False
            an = False
            count = 1
            for l in pele:
                if l.startswith(ligand_name):
                    total_an = int(l.split()[1])
                if l.split()[0] == '1':
                    an = True
                if l.startswith('NBON'):
                    an = False
                if l.startswith('BOND'):
                    bnd = True
                if l.startswith('THET'):
                    break
                if an and len(l.split()) == 9 and count <= total_an:
                    # Get atom_name
                    atom_names[int(l.split()[0])] = l.split()[4].replace('_', '')
                    count += 1
                if bnd and len(l.split()) == 4:
                    df_pele = df_pele.append(
                        {'AnSource': atom_names[int(l.split()[0])],
                         'AnTarget': atom_names[int(l.split()[1])],
                         'Dist': round(float(l.split()[3]), 1)},
                        ignore_index=True)
        return df_pele

    def _getBondRosetta(self, rosetta_params):
        """
        Get the bonds from rosetta params file.

        Parameters
        ----------
        rosetta_params: str
            Path to the rosetta params file

        Returns
        -------
        df_rosetta: pd.dataframe
            Dataframe with the bounds and distance for rosetta params

        """
        df_rosetta = pd.DataFrame(columns=['AnSource', 'AnTarget', 'Dist'])
        with open(rosetta_params) as ros:
            bnd, icor = [], {}
            for l in ros:
                if l.startswith('BOND'):
                    bnd.append(f"{l.split()[1]} {l.split()[2]}")
                if l.startswith('ICOOR_INTERNAL'):
                    icor[f"{l.split()[1]} {l.split()[5]}"] = round(float(l.split()[4]), 1)
                    icor[f"{l.split()[5]} {l.split()[1]}"] = round(float(l.split()[4]), 1)
            for bound in bnd:
                # if there is no distance for the bound
                if bound not in icor.keys():
                    icor[bound] = 0.0
                df_rosetta = df_rosetta.append({'AnSource': bound.split()[0],
                                                'AnTarget': bound.split()[1],
                                                'Dist': icor[bound]},
                                               ignore_index=True)
        return df_rosetta

    def _getBondTopology(self, df_pdb):
        """
        Calculate the bond topology for a certain dataframe

        Parameters
        ----------
        df_pdb: np.dataframe
            Dataframe with the atoms names with bounds

        Returns
        -------
        matrix_pdb_an:
            Matrix with the bound topology for each atom name
        """
        G = nx.from_pandas_edgelist(df_pdb, source='AnSource', target='AnTarget')
        dist_pdb = dict(nx.all_pairs_shortest_path_length(G))
        matrix_pdb_an = {}

        for node1 in dist_pdb:
            if node1[0] not in matrix_pdb_an.keys():
                matrix_pdb_an[node1[0]] = {}
            matrix_pdb_an[node1[0]][node1] = []

            for node2 in dist_pdb[node1]:
                matrix_pdb_an[node1[0]][node1].append(f"{dist_pdb[node1][node2]}{node2[0]}")
            sorted(matrix_pdb_an[node1[0]][node1])

        return matrix_pdb_an

    def _mapSymmetrycal(self, ros_an, pel_ans, df_pele, df_rosetta):
        """
        Maps the symmetrical atoms with the help of the individual connections or the distances
        Parameters
        ----------
        ros_an: str
            Rosetta atom name to map
        pel_ans: list
            List of strings with the pele equivalents for the rosetta atom name
        df_pele: np.dataframe
            Dataframe with the bound information for pele
        df_rosetta: np.dataframe
            Dataframe with the bound information for rosetta

        Returns
        -------
        pele_atom_name: str
            The pele atom name equivalent

        """
        # Find bounded atoms for ros_an
        ros_an_bounds_elem = list(df_rosetta.loc[(df_rosetta['AnSource'] == ros_an)]['AnTarget']) + \
                             list(df_rosetta.loc[(df_rosetta['AnTarget'] == ros_an)]['AnSource'])
        ros_an_bounds_elem = sorted([elem.rstrip('1234567890.') for elem in ros_an_bounds_elem])
        pele_an_bounds_elem = {}
        for an in pel_ans:
            n = list(df_pele.loc[(df_pele['AnSource'] == an)]['AnTarget']) + \
                list(df_pele.loc[(df_pele['AnTarget'] == an)]['AnSource'])
            pele_an_bounds_elem[an] = sorted([elem.rstrip('1234567890.') for elem in n])

        # For connections signatures
        equi = []
        dist_ros = list(df_rosetta.loc[(df_rosetta['AnSource'] == ros_an)]['Dist']) + \
                   list(df_rosetta.loc[(df_rosetta['AnTarget'] == ros_an)]['Dist'])
        for an in pel_ans:
            dist_pele = list(df_pele.loc[(df_pele['AnSource'] == an)]['Dist']) + \
                        list(df_pele.loc[(df_pele['AnTarget'] == an)]['Dist'])
            if ros_an_bounds_elem == pele_an_bounds_elem[an] and dist_pele == dist_ros:
                equi.append(an)

        return equi[0] if len(equi) == 1 else pel_ans

    def mapPeleToRosettaParams(self, paramsRosetta, paramsPele, chain):
        """
        Function to map the rosetta atom names with the pele atom names

        Parameters
        ----------
        paramsRosetta: str
            Path to the rosetta params file
        paramsPele: str
            Path to the pele params file
        chain:
            Ligand or chain name

        Returns
        -------
        equival: dict
            Dictionary with the mapped atom names

        """
        if paramsRosetta.endswith('.params'):
            df_rosetta = self._getBondRosetta(paramsRosetta).sort_index()
        elif paramsRosetta.endswith('.pdb'):
            df_rosetta = self._getBondPdb(paramsRosetta).sort_index()
        else:
            raise ValueError('The file extension is not valid. It should be .params or .pdb')
        
        if paramsPele.endswith('.params'):
            df_pele = self._getBondPele(paramsPele, chain).sort_index()
        elif paramsPele.endswith('.pdb'):
            df_pele = self._getBondPdb(paramsPele).sort_index()
        else:
            raise ValueError('The file extension is not valid. It should be .params or .pdb')

        matrix_pele = self._getBondTopology(df_pele)
        matrix_rosetta = self._getBondTopology(df_rosetta)

        # Get the atoms map by their sets
        equival = {}  # Dict[rosetta_an] = pele_an
        for pele_elem in matrix_pele:
            for pele_an in matrix_pele[pele_elem]:
                rosetta_elem = matrix_rosetta[pele_elem]
                pele_set_an = matrix_pele[pele_elem][pele_an]
                for rosetta_an in rosetta_elem:
                    rosetta_set_an = rosetta_elem[rosetta_an]
                    if sorted(rosetta_set_an) == sorted(pele_set_an):
                        if rosetta_an not in equival.keys():
                            equival[rosetta_an] = []
                        equival[rosetta_an].append(pele_an)

        if len(equival) == 0:
            raise ValueError("Can't map any atom_name. Check if the number atoms are the same.")

        # Correct the atoms that have symmetry
        # TODO add distance information from conect lines of the pdb
        prev_pel_an = ''
        not_mapped = {}
        count = 0
        for ros_an in equival.keys():
            # If atoms have more than one equivalent
            if len(equival[ros_an]) > 1:
                pel_an = self._mapSymmetrycal(ros_an, equival[ros_an], df_pele, df_rosetta)
                if isinstance(pel_an, list):
                    not_mapped[ros_an] = equival[ros_an].copy()
                    shallow_copy = dict(equival)
                    del shallow_copy[ros_an]
                    equival = shallow_copy
                    continue
                # If the previous atom name equivalent is the same choose one, because is not important
                if pel_an == prev_pel_an:
                    count += 1
                prev_pel_an = pel_an
                equival[ros_an] = equival[ros_an][count]
            # If atoms only have one equivalent atom
            else:
                count = 0
                equival[ros_an] = equival[ros_an][0]

        if not len(not_mapped) == 0:
            msg = "The following atoms are not mapped, should be revised manualy:\n"
            msg += f"{not_mapped}\n"
            msg += "Will try to assign names by name similarity\n"
            print(msg)
            for ros_an in not_mapped.keys():
                if ros_an in not_mapped[ros_an]:
                    not_mapped[ros_an] = ros_an
            equival = {**equival, **not_mapped}
        else:
            print("All atoms are mapped")

        return equival

    def _readPELECharges(self, pele_params, ligand):
        """
        Read the PELE charges from the pele params file

        Parameters
        ----------
        pele_params: str
            Path to the PELE params file

        Returns
        -------
        charges: dict
            Dictionary containing the pele from charges

        """
        atom_names = {}
        charges = {}
        with open(pele_params) as pele:
            an = False
            nb = False
            count = 1
            for l in pele:
                if l.startswith(ligand):
                    total_an = int(l.split()[1])
                if l.split()[0] == '1':
                    an = True
                if l.startswith('NBON'):
                    an = False
                    nb = True
                    continue
                if l.startswith('BOND'):
                    nb = False
                if an and len(l.split()) == 9 and count <= total_an:
                    atom_names[int(l.split()[0])] = l.split()[4]
                    count += 1
                if nb:
                    atom_name = atom_names[int(l.split()[0])].replace('_', '')
                    charges[atom_name] = float(l.split()[3])
            tc = 0
            for atom in charges:
                tc += charges[atom]

            print('Total PELE charge: %s' % tc)

        return charges

    def _readRosettaCharges(self, rosetta_params):
        """
        Read the rosetta charges from the rosetta params file

        Parameters
        ----------
        rosetta_params: str
            Path to the rosetta params file

        Returns
        -------
        charges: dict
            Dictionary containing the charges from rosetta

        """
        charges = {}
        with open(rosetta_params) as params:
            for l in params:
                if l.startswith('ATOM'):
                    atom_name = l.split()[1]
                    charges[atom_name] = float(l.split()[-1])

            tc = 0
            for atom in charges:
                tc += charges[atom]

            print('Total Rosetta charge: %s' % tc)

        return charges

    def _updateParamsCharges(self, params_file, new_charges):
        """
        Update the charges of the params file

        Parameters
        ----------
        params_file: str
            Path to the params file
        new_charges: dict
            Dictionary with the charges from pele

        """
        with open(params_file) as params:
            with open(params_file + '.tmp', 'w') as tmp:
                for l in params:
                    if l.startswith('ATOM'):
                        ls = l.split()
                        atom_name = l[5:9]
                        atom_type = l[10:15]
                        l = [ls[0]] + [atom_name] + [atom_type] + [ls[3]]
                        l += [' '] + [str(new_charges[atom_name.strip()])]
                        l = ' '.join(l) + '\n'
                    tmp.write(l)
            shutil.move(params_file + '.tmp', params_file)

    def getPELEChargesIntoParams(self, rosetta_params, pele_params, ligand=None):
        """
        Get the pele charges to the rosetta params file
        Parameters
        ----------
        rosetta_params: str
            Path to the rosetta params file
        pele_params: str
            Path to the pele params file
        mapping: dict
            Dictionary with the mapping of the atoms

        """
        if ligand == None:
            ligand = os.path.basename(rosetta_params).rstrip(".params")

        # Get PELE charges
        pele_charges = self._readPELECharges(pele_params, ligand)
        rosetta_charges = self._readRosettaCharges(rosetta_params)

        pele_atoms = set(pele_charges.keys())
        rosetta_atoms = set(rosetta_charges.keys())

        unassigned_pele_atoms = pele_atoms - pele_atoms.intersection(rosetta_atoms)
        unassigned_rosetta_atoms = rosetta_atoms - rosetta_atoms.intersection(pele_atoms)

        mapping = self.mapPeleToRosettaParams(rosetta_params, pele_params, ligand)

        if unassigned_pele_atoms != set() and mapping == None:
            message = 'There are unassinged PELE atoms:\n%s\n' % sorted(unassigned_pele_atoms)
            message += 'There are unassinged Rosetta atoms:\n%s\n' % sorted(unassigned_rosetta_atoms)
            message += 'You can use a mapping dictionary to fix this with the keword mapping'
            raise ValueError(message)

        new_charges = {}
        for a in rosetta_charges:

            if a not in pele_charges and a not in mapping:
                raise ValueError('Missing atom %s not found in mapping dictionary!' % a)
            elif a not in pele_charges and mapping[a] not in pele_charges:
                raise ValueError('Incorrect mapping for atom %s. %s not found in PELE atoms' % (a, mapping[a]))

            if a not in pele_charges:
                new_charges[a] = pele_charges[mapping[a]]
            else:
                new_charges[a] = pele_charges[a]

        self._updateParamsCharges(rosetta_params, new_charges)
        print("Copied the charges from pele to rosetta")
    

    def getBioAtoms(pdb_file):
        """
        Get the Biopython atoms from a certain pdb
        :param pdb_file: pdb input file
        :return:
        """
        parser = PDB.PDBParser()
        structure = parser.get_structure('struc', pdb_file)
        struc_atom = list(structure.get_atoms())
        return struc_atom
    
    def _getBondPDB(pdb):
        """
        Get the bounds from the pdb
        Parameters
        ----------
        pdb: str
            Path to the pdb

        Returns
        -------
        df_mol: pd.dataframe
            Dataframe with the bounds and distance
        atom_names: dict
            Dictionary with the atom numbers and atom names

        """
        from openbabel import openbabel

        with open(pdb) as file:
            # Read pdb to load atom_names and position of the atoms
            atom_names = {}
            for l in file:
                if l.startswith("HETATM"):
                    atom_names[l.split()[1]] = l.split()[2]

            # Make molfile from pdb
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("pdb", "mol2")
            mol = openbabel.OBMol()
            obConversion.ReadFile(mol, pdb)

            # Create the dataframe with the bounds
            df_mol = pd.DataFrame(columns=['AnSource', 'AnTarget'])
            for bond in openbabel.OBMolBondIter(mol):
                df_mol = df_mol.append({'AnSource': atom_names[str(bond.GetBeginAtomIdx())],
                                        'AnTarget': atom_names[str(bond.GetEndAtomIdx())]},
                                    ignore_index=True)
        return df_mol, atom_names

    def getBondDF(pdb, atoms_pdb):
        """
        Calculate the bound dataframe
        :param pdb: input file, pdb format
        :param atoms_pdb: dict with the biopython atom names
        :return: dataframe with the bound of the atom names
        """
        from openbabel import openbabel
        
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "mol2")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, pdb)
        df_mol = pd.DataFrame(columns=['AnSource', 'AnTarget'])

        for bond in openbabel.OBMolBondIter(mol):
            df_mol = df_mol.append({'AnSource': atoms_pdb[bond.GetBeginAtomIdx() - 1].name,
                                    'AnTarget': atoms_pdb[bond.GetEndAtomIdx() - 1].name},
                                   ignore_index=True)

        return df_mol
    
    def writePDBfromMapping(self, pdb, mapping):
        with open(pdb) as params:
            with open(pdb + '.tmp', 'w') as tmp:
                for l in params:
                    if l.startswith('ATOM'):
                        ls = l.split()
                        atom_name = ls[1]
                        new_atom_name = mapping[atom_name]
                        l = l.replace(atom_name, new_atom_name)
                    elif l.startswith('BOND'):
                        ls = l.split()
                        atom_name1 = ls[1]
                        atom_name2 = ls[2]
                        new_atom_name1 = mapping[atom_name1]
                        new_atom_name2 = mapping[atom_name2]
                        l = l.replace(atom_name1, new_atom_name1)
                        l = l.replace(atom_name2, new_atom_name2)
                    elif l.startswith('CHI'):
                        ls = l.split()
                        atom_name1 = ls[2]
                        atom_name2 = ls[3]
                        atom_name3 = ls[4]
                        atom_name4 = ls[5]
                        new_atom_name1 = mapping[atom_name1]
                        new_atom_name2 = mapping[atom_name2]
                        new_atom_name3 = mapping[atom_name3]
                        new_atom_name4 = mapping[atom_name4]
                        l = l.replace(atom_name1, new_atom_name1)
                        l = l.replace(atom_name2, new_atom_name2)
                        l = l.replace(atom_name3, new_atom_name3)
                        l = l.replace(atom_name4, new_atom_name4)
                    elif l.startswith('NBR_ATOM') or l.startswith('FIRST_SIDECHAIN_ATOM'):
                        ls = l.split()
                        atom_name1 = ls[1]
                        new_atom_name1 = mapping[atom_name1]
                        l = l.replace(atom_name1, new_atom_name1)
                    elif l.startswith('LOWER_CONNECT'):
                        lower = l.split()[1]
                    elif l.startswith('UPPER_CONNECT'):
                        upper = l.split()[1]
                    elif l.startswith('ICOOR_INTERNAL'):
                        l = l.replace('UPPER', upper)
                        l = l.replace('LOWER', lower)
                        ls = l.split()
                        atom_name1 = ls[1]
                        atom_name2 = ls[5]
                        atom_name3 = ls[6]
                        atom_name4 = ls[7]
                        new_atom_name1 = mapping[atom_name1]
                        new_atom_name2 = mapping[atom_name2]
                        new_atom_name3 = mapping[atom_name3]
                        new_atom_name4 = mapping[atom_name4]
                        l = l.replace(atom_name1, new_atom_name1)
                        l = l.replace(atom_name2, new_atom_name2)
                        l = l.replace(atom_name3, new_atom_name3)
                        l = l.replace(atom_name4, new_atom_name4)
                    tmp.write(l)
            shutil.move(pdb + '.tmp', pdb)        