#!/usr/bin/env python
# coding: utf-8

from . import rosettaScripts as rs
from . import scorefunctions as rs_scorefunctions
from . import movers as rs_movers

import Bio.PDB as PDB
from Bio.PDB.Polypeptide import one_to_three

def bb_only_score_function(membrane=False):

    bb_only = rs_scorefunctions.new_scorefunction('bb_only', 'empty.wts')
    reweights = (('fa_rep', 0.1),
                ('fa_atr', 0.2),
                ('hbond_sr_bb', 2.0),
                ('hbond_lr_bb', 2.0),
                ('rama_prepro', 0.45),
                ('omega', 0.4),
                ('p_aa_pp', 0.6))

    if membrane:
        reweights = list(reweights)
        reweights.append(('fa_mpenv', 0.3))
        reweights.append(('fa_mpsolv', 0.35))
        reweights.append(('fa_mpenv_smooth', 0.5))
        reweights = tuple(reweights)

    for rw in reweights:
        bb_only.addReweight(rw[0],rw[1])

    return bb_only


# In[ ]:


def abinitio_perturbers(loop_residues, loop_remodel=False):

    #define perturbers
    residues1 = loop_residues[:-2]
    atoms1 = ['C']*len(residues1)
    residues2 = loop_residues[1:-1]
    atoms2 = ['N']*len(residues2)
    p_set_dihedrals = rs_movers.generalizedKic.pertubers.setDihedrals(residues1, atoms1, residues2, atoms2, 180.0)
    p_randomize_by_rama = rs_movers.generalizedKic.pertubers.randomizeBackboneByRamaPrepro(loop_residues)

    #Add close bond operation
    if loop_remodel:
        loose_residue = loop_residues[int((len(loop_residues)-2)/2)+1]
    else:
        loose_residue = loop_residues[int((len(loop_residues)-2)/2)]
    close_bond = rs_movers.generalizedKic.closeBond(atom1='C', atom2='N', res1=loose_residue, res2=loose_residue+1)


    return [p_set_dihedrals, p_randomize_by_rama, close_bond]

def montecarlo_perturbers(loop_residues):
    #define perturbers
    atoms1 = ['N']*len(loop_residues)+['CA']*len(loop_residues)
    atoms2 = ['CA']*len(loop_residues)+['C']*len(loop_residues)
    residues = loop_residues*2
    p_perturb_dihedrals = rs_movers.generalizedKic.pertubers.perturbDihedrals(residues, atoms1, residues, atoms2, 15.0)

    return p_perturb_dihedrals


# In[ ]:


def add_missing_residues(insertion_point, sequence):

    peptideStubMover = rs_movers.peptideStubMover()

    #Add first half of the sequence as insert residues
    insert_residues = []
    for i in range(int(len(sequence)/2)):
        insert_residues.append((insertion_point+i, one_to_three(sequence[i])))

    #Add second half of the sequence as prepend residues in reverse order
    prepend_residues = []
    if len(sequence) == 1:
        prepend_point = insertion_point - 1
    else:
        prepend_point = insert_residues[-1][0]
    for i in range(len(sequence)-1,int(len(sequence)/2)-1,-1):
        prepend_residues.append((prepend_point + 2, one_to_three(sequence[i])))

    #Add residues to mover
    peptideStubMover.setResidues(insert_residues, action='insert')
    peptideStubMover.setResidues(prepend_residues, action='prepend')

    addBond = rs_movers.declareBond(atom1='C', atom2='N', res1=prepend_point+1, res2=prepend_point+2)

    return [peptideStubMover,addBond]

def addBond(insertion_point, neighbours=3):

    prepend_point = insertion_point - 1
    addBond = rs_movers.declareBond(atom1='C', atom2='N', res1=prepend_point+1, res2=prepend_point+2)
    return addBond


# In[ ]:


def loopRebuild(xml_script, insertion_point, loop_sequence, hanging_residues=1,
                mc_trials=10, membrane=False, scorefxn='ref2015'):

    # Add missing residues
    add_residues, add_bond = add_missing_residues(insertion_point, loop_sequence)
    xml_script.addMover(add_residues)
    xml_script.addMover(add_bond)

    # Define scorefunctions
    scorefxn = rs_scorefunctions.new_scorefunction(scorefxn, scorefxn)
    if membrane:
        bb_only = bb_only_score_function(membrane=True)
    else:
        bb_only = bb_only_score_function()

    # Add scorefunction
    xml_script.addScorefunction(scorefxn)
    xml_script.addScorefunction(bb_only)

    # Define loop_residues extending one residue at each end of the insertion
    loop_residues = [insertion_point+i for i in range(1,len(loop_sequence)+1)]
    loop_residues = [insertion_point+i for i in range(1,len(loop_sequence)+1)]
    for i in range(hanging_residues):
        loop_residues = [min(loop_residues)-1]+loop_residues+[max(loop_residues)+1]

    rs_loop = rs.residueSelectors.index('loop', loop_residues)
    rs_neighbors = rs.residueSelectors.neighborhood('neighbors', rs_loop.name)
    rs_not_loop = rs.residueSelectors.notSelector('notLoop', rs_loop.name)
    rs_not_neighbors = rs.residueSelectors.notSelector('notNeighbors', rs_neighbors.name)
    rs_loop_and_neighbors = rs.residueSelectors.orSelector('loopAndNeighbors', [rs_loop.name, rs_neighbors.name])
    rs_not_loop_and_neighbors = rs.residueSelectors.andSelector('notLoopAndNeighbors', [rs_not_loop.name, rs_not_neighbors.name])

    # Add residue selectors
    xml_script.addResidueSelector(rs_loop)
    xml_script.addResidueSelector(rs_neighbors)
    xml_script.addResidueSelector(rs_not_loop)
    xml_script.addResidueSelector(rs_not_neighbors)
    xml_script.addResidueSelector(rs_loop_and_neighbors)
    xml_script.addResidueSelector(rs_not_loop_and_neighbors)

    # Define task_operations
    to_repack_loop_neighbors = rs.taskOperations.operateOnResidueSubset('repackLoopAndNeighbors', rs_loop_and_neighbors.name, operation='RestrictToRepackingRLT')
    to_not_repack_else = rs.taskOperations.operateOnResidueSubset('noRepackElse', rs_not_loop_and_neighbors.name, operation='PreventRepackingRLT')
    to_extra_chi = rs.taskOperations.extraRotamersGeneric('extrachi')

    # Add task operations
    xml_script.addTaskOperation(to_repack_loop_neighbors)
    xml_script.addTaskOperation(to_not_repack_else)
    xml_script.addTaskOperation(to_extra_chi)

    #Define ab initio GenKic
    m_genKicAI = rs_movers.generalizedKic('genkicAI', selector='lowest_energy_selector',
                                           selector_scorefunction='bb_only', closure_attempts=5000,
                                           stop_when_n_solutions_found=20)


    m_genKicAI.addKICResidues(loop_residues) # Add loop residues
    p_set_dihedrals, p_randomize_by_rama, close_bond = abinitio_perturbers(loop_residues) # Get abinitio perturbers
    m_genKicAI.addPerturber(p_set_dihedrals) # Add perturbers and filters
    m_genKicAI.addPerturber(p_randomize_by_rama)
    m_genKicAI.addFilter('loop_bump_check')
    m_genKicAI.addCloseBond(close_bond)
    xml_script.addMover(m_genKicAI) # Add abinitio loop modeling mover

    #Define monteCarlo GenKic
    m_genKicMC = rs_movers.generalizedKic('genkicMC', selector='lowest_delta_torsion_selector',
                                          selector_scorefunction='bb_only', stop_when_n_solutions_found=1)
    m_genKicMC.addKICResidues(loop_residues)
    p_perturb_dihedrals = montecarlo_perturbers(loop_residues) # Get montecarlo perturber
    m_genKicMC.addPerturber(p_perturb_dihedrals) # Add perturbers and filters
    xml_script.addMover(m_genKicMC) # Add montecarlo loop sampling mover

    #Define packrotamer mover
    m_packRotamer = rs_movers.packRotamersMover('packRot',
                                                scorefxn=scorefxn,
                                                task_operations=[to_repack_loop_neighbors,
                                                to_not_repack_else,
                                                to_extra_chi])
    # Add packrotamer mover
    xml_script.addMover(m_packRotamer)

    # Define parsed protocol
    m_parsed_protocol = rs_movers.parsedProtocol('mc_moves', movers=[m_genKicMC, m_packRotamer])
    # Add parsed protocol mover
    xml_script.addMover(m_parsed_protocol)

    # Define montecarlo sampling mover
    m_montecarlo = rs_movers.genericMonteCarlo('mc_mover', mover=m_parsed_protocol, trials=mc_trials, scorefxn=scorefxn)
    #Add montecarlo sampling mover
    xml_script.addMover(m_montecarlo)

    movers = [add_residues, add_bond, m_genKicAI, m_montecarlo]

    return movers
