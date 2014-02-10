"""
Example illustrating the parallel tempering facility of repex.

This example loads the PDB file for a terminally-blocked alanine peptide and parameterizes it
using the amber10 collection of forcefield parameters.

"""

#=============================================================================================
# TURN DEBUG LOGGING ON
#=============================================================================================

import logging
logging.basicConfig(level=logging.DEBUG)

#=============================================================================================
# RUN PARALLEL TEMPERING SIMULATION
#=============================================================================================

output_filename = "repex.nc" # name of NetCDF file to store simulation output

# Select simulation platform.
from simtk import openmm
platform = openmm.Platform.getPlatformByName("CUDA")

# If simulation file already exists, try to resume.
import os.path
resume = False
if os.path.exists(output_filename):
    resume = True

if resume:
    try:
        print "Attempting to resume existing simulation..."
        import repex
        simulation = repex.resume(output_filename, platform=platform)
        
        # Extend the simulation by a few iterations.
        niterations_to_extend = 10
        simulation.extend(niterations_to_extend)    

    except Exception as e:
        print "Could not resume existing simulation due to exception:"
        print e
        print ""
        resume = False

if not resume:
    print "Starting new simulation..."

    # Set parallel tempering parameters
    from simtk import unit
    # Temperatures will be exponentially (geometrically) spaced by default
    T_min = 273.0 * unit.kelvin # minimum temperature for parallel tempering ladder
    T_max = 600.0 * unit.kelvin # maximum temperature for parallel tempering ladder
    pressure = 1.0 * unit.atmospheres # external pressure
    n_temps = 10 # number of temperatures

    collision_rate = 20.0 / unit.picosecond # collision rate for Langevin dynamics
    timestep = 2.0 * unit.femtosecond # timestep for Langevin dynamics
    
    # Load forcefield.
    from simtk.openmm import app
    forcefield = app.ForceField("amber10.xml", "tip3p.xml")

    # Load PDB file.
    pdb_filename = 'alanine-dipeptide.pdb'
    pdb = app.PDBFile(pdb_filename)

    # Create a model containing all atoms from PDB file.
    model = app.modeller.Modeller(pdb.topology, pdb.positions)
    model.addSolvent(forcefield, padding=9.0*unit.angstroms) 

    # Create OpenMM system and retrieve atomic positions.
    system = forcefield.createSystem(model.topology, nonbondedMethod=app.CutoffPeriodic, constraints=app.HBonds)
    replica_positions = [model.positions for i in range(n_temps)] # number of replica positions as input must match number of replicas

    # Create parallel tempering simulation object.
    import repex
    mpicomm = repex.dummympi.DummyMPIComm()
    parameters = {"number_of_iterations" : 10}
    parameters = {"collision_rate" : collision_rate}
    from repex import ParallelTempering
    simulation = ParallelTempering.create(system, replica_positions, output_filename, T_min=T_min, T_max=T_max, pressure=pressure, n_temps=n_temps, mpicomm=mpicomm, platform=platform, parameters=parameters)

    # Run the parallel tempering simulation.
    simulation.run()

