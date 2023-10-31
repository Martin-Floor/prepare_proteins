import math
import shutil
import os
import networkx as nx
import pandas as pd
import uuid
import re
import prepare_proteins


class tricks:
    """
    Collection of useful functions to fix PDB formats that, ideally, should not be
    useful.
    """

    def getProteinLigandInputFiles(pele_folder, protein, ligand, separator='-'):
        """
        Returns the paths of the input PDB files of a PELE simulation folder
        for a specific protein and ligand.
        """

        pdb_files = []
        for d in os.listdir(pele_folder):

            if d == 'templates':
                continue

            if separator not in d:
                raise ValueError('Separator %s not found in PELE folder.' % separator)
            if d.count(separator) > 1:
                raise ValueError('Separator %s appears more than one time in the PELE folder!' % separator)

            protein_name = d.split(separator)[0]
            ligand_name = d.split(separator)[1]

            if protein == protein_name and ligand == ligand_name:
                for f in os.listdir(pele_folder + '/' + d):
                    if f.endswith('.pdb'):
                        pdb_files.append(pele_folder + '/' + d + '/' + f)

        return pdb_files

    def changeResidueAtomNames(input_pdb, residue, atom_names, verbose=False):
        """
        Change the atom names of a specific residue in a pdb file.

        Parameters
        ==========
        input_pdb : str
            Path to the target PDB file
        residue : tuple
            Residue to change as (chain_id, resname)
        atom_names : dict
            Mapping the old atom names to the new atom names
        """

        if not isinstance(residue, tuple):
            raise ValueError('The residue must be a two element tuple (chain, resid)')

        found = {}
        for atom in atom_names:
            found[atom] = False

        with open(input_pdb + '.tmp', 'w') as tmp:
            with open(input_pdb) as pdb:
                for l in pdb:
                    if l.startswith('ATOM') or l.startswith('HETATM'):
                        index, name, resname, chain, resid = (int(l[7:12]), l[12:16], l[17:20], l[21], int(l[22:27]))
                        if (chain, resid) == residue:
                            old_atom_name = name
                            if old_atom_name in atom_names:
                                new_atom_name = atom_names[old_atom_name]
                                if len(new_atom_name) != len(old_atom_name):
                                    raise ValueError('The new and old atom names should be the same length.')
                                found[old_atom_name] = True

                                if verbose:
                                    print(f'Replacing atom name "{old_atom_name}" by "{new_atom_name}"')

                                l = l.replace(old_atom_name, new_atom_name)

                    tmp.write(l)
        shutil.move(input_pdb + '.tmp', input_pdb)
        for atom in found:
            if not found[atom]:
                print('Given atom %s was not found in residue %s' % (atom, residue))

    def displaceLigandAtomNames(input_pdb, atom, alignment='right', verbose=False):
        """
        Displace the name of the ligand atom name in the PDB.

        Parameters
        ==========
        input_pdb : str
            Path to the target PDB file
        atom : tuple
            Residue to change as (resname, atom_name)
        """
        if alignment not in ['right', 'left']:
            raise ValueError('Alignment must be either "left" or "right"')

        with open(input_pdb + '.tmp', 'w') as tmp:
            with open(input_pdb) as pdb:
                for l in pdb:
                    if l.startswith('ATOM') or l.startswith('HETATM'):
                        atom_name = l.split()[2]
                        resname = l.split()[3]
                        if (resname, atom_name) == atom:
                            if alignment == 'right':
                                if verbose:
                                    print('Changing atom name %s-%s to the right' % atom)
                                l = l.replace(atom_name + ' ', ' ' + atom_name)
                            elif alignment == 'left':
                                if verbose:
                                    print('Changing atom name %s-%s to the left' % atom)
                                l = l.replace(' ' + atom_name, atom_name + ' ')
                    tmp.write(l)
        shutil.move(input_pdb + '.tmp', input_pdb)

    def displaceResidueAtomNames(input_pdb, atom, alignment='right'):
        """
        Displace the name of the atom name of a specific residue in the PDB.

        Parameters
        ==========
        input_pdb : str
            Path to the target PDB file
        atom : tuple
            Residue to change as (resname, atom_name)
        """
        if alignment not in ['right', 'left']:
            raise ValueError('Alignment must be either "left" or "right"')

        with open(input_pdb + '.tmp', 'w') as tmp:
            with open(input_pdb) as pdb:
                for l in pdb:
                    if l.startswith('ATOM') or l.startswith('HETATM'):
                        atom_name = l.split()[2]
                        resname = l.split()[3]
                        if (resname, atom_name) == atom:
                            if alignment == 'right':
                                l = l.replace(atom_name + ' ', ' ' + atom_name)
                            elif alignment == 'left':
                                l = l.replace(' ' + atom_name, atom_name + ' ')
                    tmp.write(l)
        shutil.move(input_pdb + '.tmp', input_pdb)

    def checkLastEpoch(host, server_path, separator='-'):
        """
        Return the last epoch ran for each pele folder.
check
        Parameters
        ==========
        host : str
            Name of the remote host where pele folder resides. Use localhost for
            a local pele folder.
        server_path : str
            Path to the PELE folder in the remote host.
        separator : str
            Separator used to split the pele folder names into protein and ligand.

        Returns
        =======
        last_epoch : dict
            Last epoch for each protein and ligand combination based on finding the
            specific epoch folder in each pele folder path.

        Caveats:
            - This function does not check the content of the PELE folders, only the presence
              of the epoch folders.
        """

        log_file = '.'+str(uuid.uuid4())+'.log'

        if host == 'localhost':
            os.system('ls '+server_path+'/*/*/output > '+log_file)
        else:
            os.system('ssh '+host+' ls '+server_path+'/*/*/output > '+log_file)

        last_epoch = {}
        with open(log_file) as f:
            for l in f:
                if 'output' in l:
                    model_ligand = l.split('/')[-3]#.split('@')[0]
                    if separator not in model_ligand:
                        raise ValueError(f'Separator "{separator}" not found in pele folder {model_ligand}')
                    elif len(model_ligand.split(separator)) != 2:
                        raise ValueError(f'Separator "{separator}" seems incorrect for pele folder {model_ligand}')

                    model, ligand = model_ligand.split(separator)
                    last_epoch[model, ligand] = None
                else:
                    try: int(l)
                    except: continue
                    last_epoch[model, ligand] = int(l)
        os.remove(log_file)

        return last_epoch

    def checkFailureType(host, server_path, models=None, separator='-'):
        """
        Return the last epoch ran for each pele folder.

        Parameters
        ==========
        host : str
            Name of the remote host where pele folder resides. Use localhost for
            a local pele folder.
        server_path : str
            Path to the PELE folder in the remote host.
        models : list
            List of models name for which to query their error files.
        separator : str
            Separator used to split the pele folder names into protein and ligand.

        Returns
        =======
        last_epoch : dict
            Last epoch for each protein and ligand combination based on finding the
            specific epoch folder in each pele folder path.

        Caveats:
            - This function does not check the content of the PELE folders, only the presence
              of the epoch folders.
        """

        log_file = '.'+str(uuid.uuid4())+'.log'

        if host == 'localhost':
            os.system('ls '+server_path+'/*err > '+log_file)
        else:
            os.system('ssh '+host+' ls '+server_path+'/*err > '+log_file)

        error_types = {}
        with open(log_file) as f:
            for l in f:
                err_file = l.split('/')[-1].strip()
                model = err_file.split('_')[1].split(separator)[0]
                ligand = err_file.split('_')[1].split(separator)[1]

                if models != None and model not in models:
                    continue

                log_err_file = '.'+str(uuid.uuid4())+'.err'
                os.system('ssh '+host+' cat '+server_path+'/'+err_file+' > '+log_err_file)

                with open(log_err_file) as ef:
                    error_type = None
                    for l in ef:
                        if 'Out Of Memory' in l:
                            error_type = 'Out of memory'
                        elif 'ChainFactory::addLink' in l:
                            if l.split()[7] == 'segment':
                                error_type = 'Template for '+l.split()[16].replace(',', '')
                    error_types[(model, ligand)] = error_type

                os.remove(log_err_file)
            os.remove(log_file)

            return error_types

    def _getBondPele(pele_params, ligand_name):
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

    def _getBondRosetta(rosetta_params):
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

    def _getBondTopology(df_pdb):
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
        # dist_pdb = dict(nx.all_pairs_dijkstra_path_length(G))
        matrix_pdb_an = {}

        for node1 in dist_pdb:
            if node1[0] not in matrix_pdb_an.keys():
                matrix_pdb_an[node1[0]] = {}
            # matrix_pdb_an[node1[0]][node1] = set()
            matrix_pdb_an[node1[0]][node1] = []

            for node2 in dist_pdb[node1]:
                # matrix_pdb_an[node1[0]][node1].add(f"{dist_pdb[node1][node2]}{node2[0]}")
                matrix_pdb_an[node1[0]][node1].append(f"{dist_pdb[node1][node2]}{node2[0]}")
            sorted(matrix_pdb_an[node1[0]][node1])

        return matrix_pdb_an

    def _mapSymmetrycal(ros_an, pel_ans, df_pele, df_rosetta):
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

    def mapPeleToRosettaParams(paramsRosetta, paramsPele, chain):
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
        df_pele = self._getBondPele(paramsPele, chain).sort_index()
        df_rosetta = self._getBondRosetta(paramsRosetta).sort_index()
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
                    if rosetta_set_an == pele_set_an:
                        if rosetta_an not in equival.keys():
                            equival[rosetta_an] = []
                        equival[rosetta_an].append(pele_an)

        if len(equival) == 0:
            raise ValueError("Can't map any atom_name. Check if the number atoms are the same.")

        # Correct the atoms that have symmetry
        # Dict[rosetta_an] = pele_an
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

    def _readPELECharges(pele_params, ligand):
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

    def _readRosettaCharges(rosetta_params):
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

    def _updateParamsCharges(params_file, new_charges):
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

    def getPELEChargesIntoParams(rosetta_params, pele_params, mapping, ligand=None):
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


    def changePELEResidueNames(model_folder,output_folder):
        """
        Function to change PELE protonation residue name changes back to normal.

        Parameters
        ----------
        model_folder: str
            Path to the folder with pele models.
        output_folder: str
            Path to the folder to dump new models.
        """
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for f in os.listdir(model_folder):
            new_line = []
            with open(model_folder+'/'+f) as file:
                for line in file:
                    new_line.append(line.replace('HID','HIS').replace('HIE','HIS').replace('HIP','HIS').replace('ASH','ASP').replace('GLH','GLU').replace('CYT','CYS').replace('LYN','LYS'))
            with open(output_folder+'/'+f,'w') as file:
                for line in new_line:
                    file.write(line)
