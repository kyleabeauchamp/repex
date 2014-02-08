"""
Example illustrating the parallel tempering facility of repex.

This example loads the PDB file for a terminally-blocked alanine peptide and parameterizes it
using the amber10 collection of forcefield parameters.  An implicit solvent (OBC) is used.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import numpy as np
import simtk.unit as unit
from repex.thermodynamics import ThermodynamicState
from repex.parallel_tempering import ParallelTempering
from repex import testsystems
from repex.utils import permute_energies
from repex import dummympi
import repex

import tempfile
from mdtraj.testing import eq


#=============================================================================================
# RUN PARALLEL TEMPERING SIMULATION
#=============================================================================================

output_filename = "repex.nc" # name of NetCDF file to store simulation output

# First, try to resume, if simulation file already exists.
try:
    print "Attempting to resume existing simulation..."
    simulation = repex.resume(output_filename)
    
    # Extend the simulation by a few iterations.
    niterations_to_extend = 10
    simulation.extend(niterations_to_extend)    

except Exception as e:
    print "Could not resume existing simulation, starting new simulation..."
    print e

    # Set parallel tempering parameters
    # Temperatures will be exponentially (geometrically) spaced by default
    T_min = 273.0 * unit.kelvin # minimum temperature for parallel tempering ladder
    T_max = 600.0 * unit.kelvin # maximum temperature for parallel tempering ladder
    n_temps = 10 # number of temperatures

    collision_rate = 20.0 / unit.picosecond # collision rate for Langevin dynamics
    timestep = 2.0 * unit.femtosecond # timestep for Langevin dynamics
    
    # Load forcefield.
    from simtk.openmm import app
    forcefield = app.ForceField("amber10.xml", "amber10_obc.xml")

    # Load PDB file.
    pdb_filename = 'alanine-dipeptide.pdb'
    pdb = app.PDBFile(pdb_filename)

    # Create a model containing all atoms from PDB file.
    model = app.modeller.Modeller(pdb.topology, pdb.positions)

    # Create OpenMM system and retrieve atomic positions.
    system = forcefield.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    replica_positions = [model.positions for i in range(n_temps)] # number of replica positions as input must match number of replicas

    # Create parallel tempering simulation object.
    mpicomm = dummympi.DummyMPIComm()
    parameters = {"number_of_iterations" : 10}
    simulation = ParallelTempering.create(system, replica_positions, output_filename, T_min=T_min, T_max=T_max, n_temps=n_temps, mpicomm=mpicomm, parameters=parameters)

    # Run the parallel tempering simulation.
    replica_exchange.run()

