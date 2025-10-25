from openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np
import argparse
import time
import re
import os

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('input_prmtop', help='Path to the input prmtop file.')
parser.add_argument('input_inpcrd', help='Path to the input inpcrd file.')
parser.add_argument('simulation_time', help='Total simulation time in ns.')
parser.add_argument('--chunk_size', default=100, help='Chunk size in simulation time (ns) for output report files (data and dcd).')
parser.add_argument('--temperature', default=300.0, help='Temperature in K.')
parser.add_argument('--collision_rate', default=1.0, help='Langevin integrator collision rate (1/ps).')
parser.add_argument('--time_step', default=0.002, help='Simulation time step in ps.')
parser.add_argument('--dcd_report_time', default=100, help='DCD report interval in ps.')
parser.add_argument('--data_report_time', default=100, help='Simulation data report interval in ps.')
parser.add_argument('--nvt_time', type=float, default=0.1, help='NVT equilibration time in ns.')
parser.add_argument('--npt_time', type=float, default=0.2, help='NPT equilibration time in ns.')
parser.add_argument('--nvt_temp_scaling_steps', type=int, default=50, help='Number of iterations for NVT temperature scaling.')
parser.add_argument('--npt_restraint_scaling_steps', type=int, default=50, help='Number of iterations for NPT restraint scaling.')
parser.add_argument('--restraint_constant', type=float, default=5.0, help='Force constant for positional restraints (kcal/mol/Å²).')
parser.add_argument('--equilibration_data_report_time', type=float, default=1.0, help='Data report interval during equilibration in ps.')
parser.add_argument('--equilibration_dcd_report_time', type=float, default=0.0, help='DCD report interval during equilibration in ps (0 disables reporting).')
parser.add_argument('--seed', help='The seed to be used to generate the intial velocities')
args = parser.parse_args()

def _ns_to_steps(duration_ns, time_step_ps, label):
    """Convert a duration in nanoseconds to integration steps."""
    steps_float = (duration_ns * 1000.0) / time_step_ps
    if steps_float < 1:
        raise ValueError(f"{label} of {duration_ns} ns produces fewer than one integration step with a {time_step_ps} ps time step.")
    steps = int(round(steps_float))
    if steps <= 0:
        raise ValueError(f"{label} results in an invalid number of steps ({steps}).")
    return steps

def _interval_to_steps(interval_ps, time_step_ps, label, allow_zero=False):
    """Convert a reporting interval in picoseconds to integration steps."""
    if allow_zero and interval_ps == 0:
        return None
    steps_float = interval_ps / time_step_ps
    if steps_float < 1:
        raise ValueError(f"{label} of {interval_ps} ps is shorter than the integration time step ({time_step_ps} ps).")
    steps = int(round(steps_float))
    if steps <= 0:
        raise ValueError(f"{label} results in an invalid number of steps ({steps}).")
    return steps

def _split_steps(total_steps, segments, label):
    """Distribute total_steps across a number of segments ensuring each receives at least one step."""
    if segments < 1:
        raise ValueError(f"{label} must be at least 1.")
    if total_steps < segments:
        raise ValueError(f"{label} ({segments}) exceeds the available MD steps ({total_steps}). Reduce the number of segments or increase the simulation length.")
    base, remainder = divmod(total_steps, segments)
    return [base + (1 if i < remainder else 0) for i in range(segments)]

# Validate and parse input parameters
try:
    input_prmtop = args.input_prmtop
    input_inpcrd = args.input_inpcrd
    simulation_time = float(args.simulation_time)
    chunk_size = float(args.chunk_size)
    temperature = float(args.temperature)
    time_step = float(args.time_step)
    collision_rate = float(args.collision_rate)
    dcd_report_time = float(args.dcd_report_time)
    data_report_time = float(args.data_report_time)
    nvt_time = float(args.nvt_time)
    npt_time = float(args.npt_time)
    nvt_temp_scaling_steps = int(args.nvt_temp_scaling_steps)
    npt_restraint_scaling_steps = int(args.npt_restraint_scaling_steps)
    restraint_constant = float(args.restraint_constant)
    equilibration_data_report_time = float(args.equilibration_data_report_time)
    equilibration_dcd_report_time = float(args.equilibration_dcd_report_time)
    seed = int(args.seed) if args.seed is not None else None

    if (simulation_time <= 0 or chunk_size <= 0 or temperature <= 0 or time_step <= 0 or
            collision_rate <= 0 or nvt_time <= 0 or npt_time <= 0 or restraint_constant < 0 or
            nvt_temp_scaling_steps < 1 or npt_restraint_scaling_steps < 1):
        raise ValueError("Simulation time, chunk size, temperature, time step, collision rate, equilibration times, scaling steps, and restraint constant must be positive values.")
    if dcd_report_time <= 0 or data_report_time <= 0 or equilibration_data_report_time <= 0 or equilibration_dcd_report_time < 0:
        raise ValueError("Report intervals must be positive values (set equilibration DCD to 0 to disable).")
except ValueError as e:
    print(f"Parameter error: {e}")
    exit(1)

try:
    total_steps = _ns_to_steps(simulation_time, time_step, "Simulation time")
    chunk_steps = _ns_to_steps(chunk_size, time_step, "Chunk size")
    nvt_steps = _ns_to_steps(nvt_time, time_step, "NVT time")
    npt_steps = _ns_to_steps(npt_time, time_step, "NPT time")
    equilibration_data_report_steps = _interval_to_steps(equilibration_data_report_time, time_step, "Equilibration data report time")
    equilibration_dcd_report_steps = _interval_to_steps(equilibration_dcd_report_time, time_step, "Equilibration DCD report time", allow_zero=True)
    production_data_report_steps = _interval_to_steps(data_report_time, time_step, "Data report time")
    production_dcd_report_steps = _interval_to_steps(dcd_report_time, time_step, "DCD report time")
    nvt_step_schedule = _split_steps(nvt_steps, nvt_temp_scaling_steps, "NVT temperature scaling steps")
    npt_step_schedule = _split_steps(npt_steps, npt_restraint_scaling_steps, "NPT restraint scaling steps")
except ValueError as e:
    print(f"Parameter error: {e}")
    exit(1)


def getHeavyAtoms(topology, exclude_solvent=True):
    """
    Get a list of non-hydrogen atoms in molecules different than HOH.

    Parameters:
    topology (Topology): The topology of the system.
    exclude_solvent (bool): Whether to exclude solvent molecules (HOH).

    Returns:
    list: A list of heavy atoms.
    """
    _hydrogen = re.compile("[123 ]*H.*")
    heavy_atoms = []
    for residue in topology.residues():
        if exclude_solvent and residue.name == 'HOH':
            continue
        for atom in residue.atoms():
            if not _hydrogen.match(atom.name):
                heavy_atoms.append(atom)
    return heavy_atoms

def addHeavyAtomsPositionalRestraintForce(system, positions, topology, restraint_constant):
    """
    Add a positional restraint force for heavy atoms.

    Parameters:
    system (System): The OpenMM system.
    positions (list): The positions of the atoms.
    topology (Topology): The topology of the system.
    restraint_constant (float): The force constant for the restraints.

    Returns:
    CustomExternalForce: The positional restraint force.
    """
    heavy_atoms = getHeavyAtoms(topology)
    posres_force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force_constant = restraint_constant * kilocalories_per_mole / angstroms ** 2
    posres_force.addGlobalParameter("k", force_constant)
    posres_force.addPerParticleParameter("x0")
    posres_force.addPerParticleParameter("y0")
    posres_force.addPerParticleParameter("z0")

    for atom in heavy_atoms:
        posres_force.addParticle(atom.index, positions[atom.index].value_in_unit(nanometers))

    system.addForce(posres_force)
    return posres_force

def removeHeavyAtomPositionalRestraintForce(system):
    """
    Remove the positional restraint force for heavy atoms.

    Parameters:
    system (System): The OpenMM system.
    """
    for i, force in enumerate(system.getForces()):
        if isinstance(force, CustomExternalForce) and 'k' in [force.getGlobalParameterName(j) for j in range(force.getNumGlobalParameters())]:
            system.removeForce(i)
            break

class CustomDataReporter(StateDataReporter):
    """
    Data reporter to report custom data.

    Parameters:
    system (System): The OpenMM system.
    file (str): The file to write the report.
    reportInterval (int): The interval (in steps) at which to report data.
    """
    def __init__(self, system, file, reportInterval, **kwargs):
        super(CustomDataReporter, self).__init__(file, reportInterval, **kwargs)
        self.system = system
        self.force_groups = self.setForceGroups()

    def setForceGroups(self):
        """
        Set force groups for energy reporting.

        Returns:
        dict: A dictionary mapping force names to force group indices.
        """
        force_groups = {}
        for i, force in enumerate(self.system.getForces()):
            force_name = type(force).__name__
            if force_name == 'CustomExternalForce':
                force_name = 'PositionalRestraintForce'
            elif force_name == 'MonteCarloBarostat':
                continue
            elif force_name == 'CMMotionRemover':
                continue  # Skip adding CMMotionRemover to force groups for energy reporting
            force_groups[force_name] = i + 1
            force.setForceGroup(i + 1)
        return force_groups

    def _constructHeaders(self):
        """
        Construct headers for the report file.

        Returns:
        list: A list of headers.
        """
        headers = super()._constructHeaders()
        for force in self.force_groups:
            headers.append(f'{force} [kcal/mol]')
        return headers

    def _constructReportValues(self, simulation, state):
        """
        Construct report values.

        Parameters:
        simulation (Simulation): The OpenMM simulation.
        state (State): The current state of the simulation.

        Returns:
        list: A list of report values.
        """
        values = super()._constructReportValues(simulation, state)
        for force in self.force_groups:
            group = self.force_groups[force]
            energy = simulation.context.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(kilocalories_per_mole)
            values.append(energy)
        return values

def get_last_step_from_report():
    """
    Get the last step from the production data report file.

    Returns:
    tuple: The last step and the last chunk file.
    """
    last_step = 0
    last_chunk_file = None
    for filename in sorted(os.listdir('.'), reverse=True):
        if filename.startswith('production_') and filename.endswith('.data'):
            last_chunk_file = filename
            break

    if last_chunk_file:
        with open(last_chunk_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                last_step = int(line.split(',')[0])

    return last_step, last_chunk_file

def calculate_zfill(num_chunks):
    """
    Calculate the zero-fill length based on the number of chunks.

    Parameters:
    num_chunks (int): The number of chunks.

    Returns:
    int: The zero-fill length.
    """
    return max(2, len(str(num_chunks)))

def check_rename_needed(zfill_length):
    """
    Check if renaming of chunk files is needed based on the zero-fill length.

    Parameters:
    zfill_length (int): The zero-fill length.

    Returns:
    bool: True if renaming is needed, False otherwise.
    """
    for filename in os.listdir('.'):
        if filename.startswith('production_') and (filename.endswith('.data') or filename.endswith('.dcd')):
            parts = filename.split('_')
            chunk_num = parts[1].split('.')[0]
            if len(chunk_num) != zfill_length:
                return True
    return False

def rename_chunk_files(zfill_length):
    """
    Rename chunk files to match the zero-fill length.

    Parameters:
    zfill_length (int): The zero-fill length.
    """
    for filename in os.listdir('.'):
        if filename.startswith('production_') and (filename.endswith('.data') or filename.endswith('.dcd')):
            parts = filename.split('_')
            chunk_num = parts[1].split('.')[0]
            new_chunk_num = chunk_num.zfill(zfill_length)
            new_filename = filename.replace(chunk_num, new_chunk_num)
            os.rename(filename, new_filename)

# Define function to get the current time
def current_time():
    return time.time()

# Define function to print elapsed time
def print_elapsed_time(start, end, description):
    elapsed = end - start
    print(f"{description} took {elapsed:.2f} seconds.")

# Define function to compute ns/day
def compute_ns_per_day(simulation_time_ns, elapsed_seconds):
    return (simulation_time_ns / elapsed_seconds) * 86400

# Main simulation script
try:
    # Load the input files
    prmtop = AmberPrmtopFile(input_prmtop)
    inpcrd = AmberInpcrdFile(input_inpcrd)

    # Create system from the given AMBER prmtop and inpcrd files
    print('Initializing system from input files:')
    print(f'\tprmtop file: {input_prmtop}')
    print(f'\tinpcrd file: {input_inpcrd}')

    # --- detect periodicity and atom count
    has_box = (inpcrd.boxVectors is not None) or (prmtop.topology.getUnitCellDimensions() is not None)
    n_atoms = prmtop.topology.getNumAtoms()
    print(f"System atoms: {n_atoms}")
    print(f"Periodic box present?: {has_box}")

    # --- choose nonbonded method based on periodicity
    nb_method = PME if has_box else NoCutoff
    nb_cutoff = 1 * nanometer if has_box else None

    system_kwargs = dict(nonbondedMethod=nb_method, constraints=HBonds)
    if nb_cutoff is not None:
        system_kwargs["nonbondedCutoff"] = nb_cutoff
    system = prmtop.createSystem(**system_kwargs)

    # Set up integrator
    integrator = LangevinMiddleIntegrator(temperature * kelvin, collision_rate / picosecond, time_step * picoseconds)
    if seed is not None:
        integrator.setRandomNumberSeed(seed)
    simulation = Simulation(prmtop.topology, system, integrator)

    # Initialize simulation
    simulation.context.setPositions(inpcrd.positions)

    checkpoint_file = 'production.chk'

    # Calculate chunking information
    num_chunks = total_steps // chunk_steps
    remaining_steps = total_steps % chunk_steps

    if remaining_steps > 0:
        num_chunks += 1

    # Calculate zfill length and check if renaming is needed
    zfill_length = calculate_zfill(num_chunks)
    if check_rename_needed(zfill_length):
        rename_chunk_files(zfill_length)

    # Check if the checkpoint file exists
    if os.path.exists(checkpoint_file):
        print("Checkpoint file found. Loading checkpoint.")
        simulation.loadCheckpoint(checkpoint_file)

        # Find the last step completed from the latest production data report file
        last_step, last_chunk_file = get_last_step_from_report()
        # Check if the simulation has finished
        if last_step >= total_steps:
            print(f'Simulation has already run for {simulation_time}ns, increase the simulation time to continue...')
            exit()

        # Determine the correct chunk to start from
        chunk = last_step // chunk_steps
    else:
        # Check for the latest chunk checkpoint file if the general checkpoint file is not found
        chunk_checkpoint_found = False
        for chunk in range(num_chunks - 1, -1, -1):
            chunk_suffix = str(chunk + 1).zfill(zfill_length)
            chunk_checkpoint_file = f'production_{chunk_suffix}.chk'
            if os.path.exists(chunk_checkpoint_file):
                print(f'Chunk checkpoint file {chunk_checkpoint_file} found. Loading checkpoint.')
                simulation.loadCheckpoint(chunk_checkpoint_file)
                last_step = (chunk + 1) * chunk_steps
                chunk_checkpoint_found = True
                break

        if not chunk_checkpoint_found:
            print('\tAdding positional restraint force')
            # Add a restraint force to the system
            posres_force = addHeavyAtomsPositionalRestraintForce(system, inpcrd.positions, prmtop.topology, restraint_constant)

            # Set up equilibration reporters
            equilibration_reporter = CustomDataReporter(system, 'equilibration.data', equilibration_data_report_steps, step=True,
                                                        potentialEnergy=True, temperature=True)
            if equilibration_dcd_report_steps:
                equilibration_dcd_reporter = DCDReporter('equilibration.dcd', equilibration_dcd_report_steps)
                simulation.reporters.append(equilibration_dcd_reporter)
            simulation.reporters.append(equilibration_reporter)
            simulation.context.reinitialize(preserveState=True)

            # Setup and run energy minimization
            print('Applying energy minimization.')
            minimization_start = current_time()
            simulation.minimizeEnergy()
            print('\tDone')
            minimization_end = current_time()
            print_elapsed_time(minimization_start, minimization_end, "Energy minimization")

            # Store minimized structure
            print('\tStoring minimized structure')
            with open('minimized.pdb', 'w') as of:
                state = simulation.context.getState(getPositions=True)
                positions = state.getPositions()
                PDBFile.writeFile(prmtop.topology, positions, of)

            # Store energy of the minimized state (step 0)
            state = simulation.context.getState(getEnergy=True)
            equilibration_reporter.report(simulation, state)

            # Run NVT equilibration
            initial_temperature = 5 * kelvin
            simulation.context.setVelocitiesToTemperature(initial_temperature)
            nvt_simulation_steps = nvt_steps
            nvt_simulation_time = nvt_simulation_steps * time_step

            print('NVT equilibration:')
            print(f'\tThe simulation will run for {nvt_simulation_steps} steps with a time step of {time_step} ps ({nvt_simulation_time} ps).')

            nvt_start = current_time()
            T_initial = 5.0
            T_final = temperature
            if nvt_temp_scaling_steps == 1:
                temperature_schedule = [T_final * kelvin]
            else:
                dT = (T_final - T_initial) / (nvt_temp_scaling_steps - 1)
                temperature_schedule = [(T_initial + i * dT) * kelvin for i in range(nvt_temp_scaling_steps)]
            for i, (segment_steps, target_temperature) in enumerate(zip(nvt_step_schedule, temperature_schedule)):
                integrator.setTemperature(target_temperature)
                print(f'\r{" " * 80}', end='\r')
                print(f'\tRunning NVT temperature scaling step {i+1} of {nvt_temp_scaling_steps} at T={target_temperature}', end='\r')
                simulation.step(segment_steps)

            print(f'\r{" " * 80}', end='\r')
            print('\tFinished running NVT equilibration.')
            nvt_end = current_time()
            nvt_elapsed = nvt_end - nvt_start
            ns_per_day = compute_ns_per_day(nvt_time, nvt_elapsed)
            print_elapsed_time(nvt_start, nvt_end, "NVT equilibration")
            print(f"NVT equilibration performance: {ns_per_day:.2f} ns/day")

            # Run NPT equilibration
            npt_simulation_steps = npt_steps
            npt_simulation_time = npt_simulation_steps * time_step

            print('NPT equilibration')
            print('\tInitializing Monte Carlo Barostat:')
            print(f'\tTemperature: {temperature} K')
            print(f'\tPressure: 1 bar')
            barostat = MonteCarloBarostat(1 * bar, temperature * kelvin)
            system.addForce(barostat)
            simulation.context.reinitialize(preserveState=True)
            simulation.context.setVelocitiesToTemperature(temperature * kelvin)

            print(f'\tThe simulation will run for {npt_simulation_steps} steps with a time step of {time_step} ps ({npt_simulation_time / 1000} ns).')
            initial_restraint_constant = restraint_constant
            denom = max(npt_restraint_scaling_steps - 1, 1)
            npt_start = current_time()
            for i, segment_steps in enumerate(npt_step_schedule):
                t = i / denom
                K = initial_restraint_constant * (1 - t)**4 * kilocalories_per_mole / angstroms ** 2
                posres_force.setGlobalParameterDefaultValue(0, K)
                simulation.context.setParameter('k', K)
                print(f'\r{" " * 100}', end='\r')
                print(f'\tRunning NPT restraint scaling step {i+1} of {npt_restraint_scaling_steps} at a restraint constant of {K}', end='\r')
                simulation.step(segment_steps)
            print(f'\r{" " * 100}', end='\r')
            print('\tFinished running NPT equilibration.')
            npt_end = current_time()
            npt_elapsed = npt_end - npt_start
            ns_per_day = compute_ns_per_day(npt_time, npt_elapsed)
            print_elapsed_time(npt_start, npt_end, "NPT equilibration")
            print(f"NPT equilibration performance: {ns_per_day:.2f} ns/day")

            # Remove positional restraint force
            print('Production Run')
            print('\tRemoving positional restraint force')
            removeHeavyAtomPositionalRestraintForce(system)
            simulation.context.reinitialize(preserveState=True)

            last_step = 0  # No steps completed in production run
            chunk = 0

    # Run the production simulation in chunks
    for chunk in range(chunk, num_chunks):
        chunk_start_step = chunk * chunk_steps
        chunk_end_step = min(chunk_start_step + chunk_steps, total_steps)
        chunk_suffix = str(chunk + 1).zfill(zfill_length)

        data_file_exists = os.path.isfile(f'production_{chunk_suffix}.data')
        data_file_mode = 'a' if data_file_exists else 'w'

        production_reporter = CustomDataReporter(system, open(f'production_{chunk_suffix}.data', data_file_mode), production_data_report_steps, step=True,
                                                 potentialEnergy=True, temperature=True)
        production_dcd_reporter = DCDReporter(f'production_{chunk_suffix}.dcd', production_dcd_report_steps, append=data_file_exists)
        production_chk_reporter = CheckpointReporter(f'production_{chunk_suffix}.chk', production_data_report_steps)

        simulation.reporters = [production_reporter, production_dcd_reporter, production_chk_reporter]
        simulation.context.reinitialize(preserveState=True)

        print(f'Running production chunk {chunk_suffix} from step {chunk_start_step} to {chunk_end_step}')
        chunk_start_time = current_time()

        if last_step > chunk_start_step:
            print(f"Resuming from step {last_step}.")

        simulation.step(chunk_end_step - last_step if last_step > chunk_start_step else chunk_steps)
        last_step = chunk_end_step

        # Save the general checkpoint file
        simulation.saveCheckpoint(checkpoint_file)

        chunk_end_time = current_time()
        chunk_elapsed = chunk_end_time - chunk_start_time
        chunk_time = chunk_steps * time_step / 1000
        ns_per_day = compute_ns_per_day(chunk_time, chunk_elapsed)
        print_elapsed_time(chunk_start_time, chunk_end_time, f"Production chunk {chunk_suffix}")
        print(f"Production chunk {chunk_suffix} performance: {ns_per_day} ns/day")

    print('Production run completed.')

    # Save the final state of the system
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open('final_state.pdb', 'w') as f:
        PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    with open('final_state.xml', 'w') as f:
        f.write(XmlSerializer.serialize(state))

except Exception as e:
    print(f"An error occurred: {e}")
