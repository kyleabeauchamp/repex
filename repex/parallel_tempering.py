#!/usr/local/bin/env python

import os
import sys
import math
import copy
import time
import datetime

import numpy as np
import numpy.linalg

import simtk.openmm as mm
import simtk.unit as units

import netCDF4 as netcdf # netcdf4-python is used in place of scipy.io.netcdf for now

from thermodynamics import ThermodynamicState
from replica_exchange import ReplicaExchange
import netcdf_io

import logging
logger = logging.getLogger(__name__)

from constants import kB

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant



class ParallelTempering(ReplicaExchange):
    """Parallel tempering simulation class.

    This class provides a facility for parallel tempering simulations.  It is a subclass of ReplicaExchange, but provides
    various convenience methods and efficiency improvements for parallel tempering simulations, so should be preferred for
    this type of simulation.  In particular, the System only need be specified once, while the temperatures (or a temperature
    range) is used to automatically build a set of ThermodynamicState objects for replica-exchange.  Efficiency improvements
    make use of the fact that the reduced potentials are linear in inverse temperature.
    
    """

    def _compute_energies(self):
        """Compute reduced potentials of all replicas at all states (temperatures).

        Notes
        -----

        Because only the temperatures differ among replicas, we replace 
        the generic O(N^2) replica-exchange implementation with an O(N) implementation.
        """

        start_time = time.time()
        logger.debug("Computing energies...")

        if self.mpicomm:
            # MPI implementation

            # NOTE: This version incurs the overhead of context creation/deletion.
            # TODO: Use cached contexts instead.
            
            # Create an integrator and context.
            state = self.states[0]
            integrator = mm.VerletIntegrator(self.timestep)
            context = mm.Context(state.system, integrator, self.platform)

            for replica_index in range(self.mpicomm.rank, self.n_states, self.mpicomm.size):
                # Set coordinates.
                context.setPositions(self.replica_coordinates[replica_index])
                # Compute potential energy.
                openmm_state = context.getState(getEnergy=True)            
                potential_energy = openmm_state.getPotentialEnergy()           
                # Compute energies at this state for all replicas.
                for state_index in range(self.n_states):
                    # Compute reduced potential
                    beta = 1.0 / (kB * self.states[state_index].temperature)
                    self.u_kl[replica_index,state_index] = beta * potential_energy

            # Gather energies.
            energies_gather = self.mpicomm.allgather(self.u_kl[self.mpicomm.rank:self.n_states:self.mpicomm.size,:])
            for replica_index in range(self.n_states):
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
            integrator = mm.VerletIntegrator(self.timestep)
            context = mm.Context(state.system, integrator, self.platform)
        
            # Compute reduced potentials for all configurations in all states.
            for replica_index in range(self.n_states):
                # Set coordinates.
                context.setPositions(self.replica_coordinates[replica_index])
                # Compute potential energy.
                openmm_state = context.getState(getEnergy=True)            
                potential_energy = openmm_state.getPotentialEnergy()           
                # Compute energies at this state for all replicas.
                for state_index in range(self.n_states):
                    # Compute reduced potential
                    beta = 1.0 / (kB * self.states[state_index].temperature)
                    self.u_kl[replica_index,state_index] = beta * potential_energy

            # Clean up.
            del context, integrator

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy = elapsed_time / float(self.n_states)
        logger.debug("Time to compute all energies %.3f s (%.3f per energy calculation).\n" % (elapsed_time, time_per_energy))


    @classmethod
    def create_repex(cls, system, coordinates, filename, T_min=None, T_max=None, temperatures=None, n_temps=None, pressure=None, mpicomm=None, **kwargs):
        """Create a new ParallelTempering simulation.
        
        Parameters
        ----------

        system : simtk.openmm.System
            The temperature of the system.
        coordinates : list([simtk.unit.Quantity]), shape=(n_replicas, n_atoms, 3), unit=Length
            The starting coordinates for each replica
        filename : string 
            name of NetCDF file to bind to for simulation output and checkpointing
        T_min : simtk.unit.Quantity, unit=Temperature, default=None
            The lowest temperature of the Parallel Temperature run
        T_max : simtk.unit.Quantity, unit=Temperature, default=None
            The highest temperature of the Parallel Temperature run
        n_temps : int, default=None
            The number of replicas.
        temperatures : list([simtk.unit.Quantity]), unit=Temperature
            Explicit list of each temperature to use
        pressure : simtk.unit.Quantity, unit=Pa, default=None
            If specified, perform NPT simulation at this temperature.
        mpicomm : mpi4py communicator, default=None
            MPI communicator, if parallel execution is desired.      
        kwargs (dict) - Optional parameters to use for specifying simulation
            Provided keywords will be matched to object variables to replace defaults.
            
        Notes
        -----
        
        The parameters of this function are different from  ReplicaExchange.create_repex().
        The optional arguments temperatures is incompatible with (T_min, T_max, and n_temps).  
        Only one of those two groups should be specified.  
        
        If T_min, T_max, and n_temps are specified, temperatures will be exponentially
        spaced between T_min and T_max.
        """

        if temperatures is not None:
            logger.info("Using provided temperatures")
            n_temps = len(temperatures)
        elif (T_min is not None) and (T_max is not None) and (n_temps is not None):
            temperatures = [ T_min + (T_max - T_min) * (np.exp(float(i) / float(n_temps-1)) - 1.0) / (math.e - 1.0) for i in range(n_temps) ]
        else:
            raise ValueError("Either 'temperatures' or 'T_min', 'T_max', and 'n_temps' must be provided.")

        thermodynamic_states = [ ThermodynamicState(system=system, temperature=temperatures[i], pressure=pressure) for i in range(n_temps) ]
    
        if mpicomm is None or (mpicomm.rank == 0):
            database = netcdf_io.NetCDFDatabase(filename, thermodynamic_states, coordinates, **kwargs)  # To do: eventually use factory for looking up database type via filename
        else:
            database = None
        
        repex = cls(thermodynamic_states, coordinates, database, mpicomm=mpicomm, **kwargs)
        # Override title.
        repex.title = 'Parallel tempering simulation created using ParallelTempering class of repex.py on %s' % time.asctime(time.localtime())        

        repex._initialize()
        repex._run_iteration_zero()
        return repex
        


        
