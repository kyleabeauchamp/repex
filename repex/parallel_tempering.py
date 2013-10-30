#!/usr/local/bin/env python

import os
import sys
import math
import copy
import time
import datetime

import numpy
import numpy.linalg

import simtk.openmm 
import simtk.unit as units

import netCDF4 as netcdf # netcdf4-python is used in place of scipy.io.netcdf for now

from thermodynamics import ThermodynamicState
from replica_exchange import ReplicaExchange

import logging
logger = logging.getLogger(__name__)

from constants import kB

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant



class ParallelTempering(ReplicaExchange):
    """
    Parallel tempering simulation facility.

    DESCRIPTION

    This class provides a facility for parallel tempering simulations.  It is a subclass of ReplicaExchange, but provides
    various convenience methods and efficiency improvements for parallel tempering simulations, so should be preferred for
    this type of simulation.  In particular, the System only need be specified once, while the temperatures (or a temperature
    range) is used to automatically build a set of ThermodynamicState objects for replica-exchange.  Efficiency improvements
    make use of the fact that the reduced potentials are linear in inverse temperature.
    
    EXAMPLES

    Parallel tempering of alanine dipeptide in implicit solvent.
    
    >>> # Create alanine dipeptide test system.
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Initialize parallel tempering on an exponentially-spaced scale
    >>> Tmin = 298.0 * units.kelvin
    >>> Tmax = 600.0 * units.kelvin
    >>> nreplicas = 3
    >>> simulation = ParallelTempering(system, coordinates, store_filename, Tmin=Tmin, Tmax=Tmax, ntemps=nreplicas)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.minimize = False
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    Parallel tempering of alanine dipeptide in explicit solvent at 1 atm.
    
    >>> # Create alanine dipeptide system
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideExplicit()
    >>> # Add Monte Carlo barsostat to system (must be same pressure as simulation).
    >>> import simtk.openmm as openmm
    >>> pressure = 1.0 * units.atmosphere
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Initialize parallel tempering on an exponentially-spaced scale
    >>> Tmin = 298.0 * units.kelvin
    >>> Tmax = 600.0 * units.kelvin    
    >>> nreplicas = 3
    >>> simulation = ParallelTempering(system, coordinates, store_filename, Tmin=Tmin, Tmax=Tmax, pressure=pressure, ntemps=nreplicas)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.minimize = False # don't minimize first
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    """

    def __init__(self, system, coordinates, store_filename, protocol=None, Tmin=None, Tmax=None, ntemps=None, temperatures=None, pressure=None, mm=None, mpicomm=None, metadata=None):
        """
        Initialize a parallel tempering simulation object.

        ARGUMENTS
        
        system (simtk.openmm.System) - the system to simulate
        coordinates (simtk.unit.Quantity of numpy natoms x 3 array of units length, or list) - coordinate set(s) for one or more replicas, assigned in a round-robin fashion
        store_filename (string) -  name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        Tmin, Tmax, ntemps - min and max temperatures, and number of temperatures for exponentially-spaced temperature selection (default: None)
        temperatures (list of simtk.unit.Quantity with units of temperature) - if specified, this list of temperatures will be used instead of (Tmin, Tmax, ntemps) (default: None)
        pressure (simtk.unit.Quantity with units of pressure) - if specified, a MonteCarloBarostat will be added (or modified) to perform NPT simulations
        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict.  Provided keywords will be matched to object variables to replace defaults. (default: None)
        mpicomm (mpi4py communicator) - MPI communicator, if parallel execution is desired (default: None)

        NOTES

        Either (Tmin, Tmax, ntempts) must all be specified or the list of 'temperatures' must be specified.

        """
        # Create thermodynamic states from temperatures.
        if temperatures is not None:
            print "Using provided temperatures"
            self.temperatures = temperatures
        elif (Tmin is not None) and (Tmax is not None) and (ntemps is not None):
            self.temperatures = [ Tmin + (Tmax - Tmin) * (math.exp(float(i) / float(ntemps-1)) - 1.0) / (math.e - 1.0) for i in range(ntemps) ]
        else:
            raise ValueError("Either 'temperatures' or 'Tmin', 'Tmax', and 'ntemps' must be provided.")
        
        states = [ ThermodynamicState(system=system, temperature=self.temperatures[i], pressure=pressure) for i in range(ntemps) ]

        # Initialize replica-exchange simlulation.
        ReplicaExchange.__init__(self, states, coordinates, store_filename, protocol=protocol, mm=mm, mpicomm=mpicomm, metadata=metadata)

        # Override title.
        self.title = 'Parallel tempering simulation created using ParallelTempering class of repex.py on %s' % time.asctime(time.localtime())
        
        return

    def _compute_energies(self):
        """
        Compute reduced potentials of all replicas at all states (temperatures).

        NOTES

        Because only the temperatures differ among replicas, we replace the generic O(N^2) replica-exchange implementation with an O(N) implementation.
        
        """

        start_time = time.time()
        if self.verbose: print "Computing energies..."

        if self.mpicomm:
            # MPI implementation

            # NOTE: This version incurs the overhead of context creation/deletion.
            # TODO: Use cached contexts instead.
            
            # Create an integrator and context.
            state = self.states[0]
            integrator = self.mm.VerletIntegrator(self.timestep)
            context = self.mm.Context(state.system, integrator, self.platform)

            for replica_index in range(self.mpicomm.rank, self.nstates, self.mpicomm.size):
                # Set coordinates.
                context.setPositions(self.replica_coordinates[replica_index])
                # Compute potential energy.
                openmm_state = context.getState(getEnergy=True)            
                potential_energy = openmm_state.getPotentialEnergy()           
                # Compute energies at this state for all replicas.
                for state_index in range(self.nstates):
                    # Compute reduced potential
                    beta = 1.0 / (kB * self.states[state_index].temperature)
                    self.u_kl[replica_index,state_index] = beta * potential_energy

            # Gather energies.
            energies_gather = self.mpicomm.allgather(self.u_kl[self.mpicomm.rank:self.nstates:self.mpicomm.size,:])
            for replica_index in range(self.nstates):
                source = replica_index % self.mpicomm.size # node with trajectory data
                index = replica_index // self.mpicomm.size # index within trajectory batch
                self.u_kl[replica_index,:] = energies_gather[source][index]

            # Clean up.
            del context, integrator
                
        else:
            # Serial implementation.
            # NOTE: This version incurs the overhead of context creation/deletion.
            # TODO: Use cached contexts instead.

            # Create an integrator and context.
            state = self.states[0]
            integrator = self.mm.VerletIntegrator(self.timestep)
            context = self.mm.Context(state.system, integrator, self.platform)
        
            # Compute reduced potentials for all configurations in all states.
            for replica_index in range(self.nstates):
                # Set coordinates.
                context.setPositions(self.replica_coordinates[replica_index])
                # Compute potential energy.
                openmm_state = context.getState(getEnergy=True)            
                potential_energy = openmm_state.getPotentialEnergy()           
                # Compute energies at this state for all replicas.
                for state_index in range(self.nstates):
                    # Compute reduced potential
                    beta = 1.0 / (kB * self.states[state_index].temperature)
                    self.u_kl[replica_index,state_index] = beta * potential_energy

            # Clean up.
            del context, integrator

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy = elapsed_time / float(self.nstates)
        if self.verbose: print "Time to compute all energies %.3f s (%.3f per energy calculation).\n" % (elapsed_time, time_per_energy)

        return

