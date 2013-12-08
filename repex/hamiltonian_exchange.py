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
    """Hamiltonian exchange simulation facility.

    HamiltonianExchange provides an implementation of a Hamiltonian exchange simulation based on the ReplicaExchange class.
    It provides several convenience classes and efficiency improvements, and should be preferentially used for Hamiltonian
    exchange simulations over ReplicaExchange when possible.
    
    Notes
    -----
    
    To create a new HamiltonianExchange object, use the `create_repex()`
    class method.  
    
    """

    @classmethod
    def create_repex(cls, reference_state, systems, coordinates, filename, mpicomm=None, **kwargs):
        """Create a new Hamiltonian exchange simulation object.

        Parameters
        ----------

        temperature : simtk.unit.Quantity, optional, units compatible with simtk.unit.kelvin
            The temperature of the system.

        reference_state : ThermodynamicState
            reference state containing all thermodynamic parameters 
            except the system, which will be replaced by 'systems'
        systems : list([simt.openmm.System])
            list of systems to simulate (one per replica)
        coordinates : simtk.unit.Quantity, shape=(n_atoms, 3), unit=Length
            coordinates (or a list of coordinates objects) for initial 
            assignment of replicas (will be used in round-robin assignment)
        filename : string 
            name of NetCDF file to bind to for simulation output and checkpointing
        mpicomm : mpi4py communicator, default=None
            MPI communicator, if parallel execution is desired.      
        kwargs (dict) - Optional parameters to use for specifying simulation
            Provided keywords will be matched to object variables to replace defaults.
            
        Notes
        -----
        
        The parameters of this function are different from  ReplicaExchange.create_repex().

        """
      
        thermodynamic_states = [ ThermodynamicState(system=system, temperature=reference_state.temperature, pressure=reference_state.pressure) for system in systems ]
        return super(cls, HamiltonianExchange).create_repex(thermodynamic_states, coordinates, filename, mpicomm=mpicomm, **kwargs)
