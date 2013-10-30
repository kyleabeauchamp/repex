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

from thermodynamics import ThermodynamicState
from replica_exchange import ReplicaExchange

import logging
logger = logging.getLogger(__name__)

from constants import kB


class HamiltonianExchange(ReplicaExchange):
    """
    Hamiltonian exchange simulation facility.

    DESCRIPTION

    This class provides an implementation of a Hamiltonian exchange simulation based on the ReplicaExchange facility.
    It provides several convenience classes and efficiency improvements, and should be preferentially used for Hamiltonian
    exchange simulations over ReplicaExchange when possible.
    
    EXAMPLES
    
    >>> # Create reference system
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [reference_system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Copy reference system.
    >>> systems = [reference_system for index in range(10)]
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Create reference state.
    >>> from thermodynamics import ThermodynamicState
    >>> reference_state = ThermodynamicState(reference_system, temperature=298.0*units.kelvin)
    >>> # Create simulation.
    >>> simulation = HamiltonianExchange(reference_state, systems, coordinates, store_filename)
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.minimize = False
    >>> # Run simulation.
    >>> simulation.run() # run the simulation
    
    """

    def __init__(self, reference_state, systems, coordinates, store_filename, protocol=None, mm=None, mpicomm=None, metadata=None):
        """
        Initialize a Hamiltonian exchange simulation object.

        ARGUMENTS

        reference_state (ThermodynamicState) - reference state containing all thermodynamic parameters except the system, which will be replaced by 'systems'
        systems (list of simtk.openmm.System) - list of systems to simulate (one per replica)
        coordinates (simtk.unit.Quantity of numpy natoms x 3 with units length) -  coordinates (or a list of coordinates objects) for initial assignment of replicas (will be used in round-robin assignment)
        store_filename (string) - name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict. Provided keywords will be matched to object variables to replace defaults.
        mpicomm (mpi4py communicator) - MPI communicator, if parallel execution is desired (default: None)        

        """

        if systems is None:
            states = None
        else:
            # Create thermodynamic states from systems.        
            states = [ ThermodynamicState(system=system, temperature=reference_state.temperature, pressure=reference_state.pressure, mm=mm) for system in systems ]

        # Initialize replica-exchange simlulation.
        ReplicaExchange.__init__(self, states, coordinates, store_filename, protocol=protocol, mm=mm, mpicomm=mpicomm, metadata=metadata)

        # Override title.
        self.title = 'Hamiltonian exchange simulation created using HamiltonianExchange class of repex.py on %s' % time.asctime(time.localtime())
        
        return

    
