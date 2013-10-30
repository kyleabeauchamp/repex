#!/usr/local/bin/env python

import os
import sys
import copy
import time
import datetime

import numpy as np
import numpy.linalg

import simtk.openmm as mm
import simtk.unit as units

import netCDF4 as netcdf # netcdf4-python is used in place of scipy.io.netcdf for now

from thermodynamics import ThermodynamicState
from constants import kB
from utils import time_and_print, process_kwargs, fix_coordinates
import citations
import netcdf_io

import logging
logger = logging.getLogger(__name__)


class ReplicaExchange(object):
    """
    Replica-exchange simulation facility.

    This base class provides a general replica-exchange simulation facility, allowing any set of thermodynamic states
    to be specified, along with a set of initial coordinates to be assigned to the replicas in a round-robin fashion.
    No distinction is made between one-dimensional and multidimensional replica layout; by default, the replica mixing
    scheme attempts to mix *all* replicas to minimize slow diffusion normally found in multidimensional replica exchange
    simulations.  (Modification of the 'replica_mixing_scheme' setting will allow the tranditional 'neighbor swaps only'
    scheme to be used.)

    While this base class is fully functional, it does not make use of the special structure of parallel tempering or
    Hamiltonian exchange variants of replica exchange.  The ParallelTempering and HamiltonianExchange classes should
    therefore be used for these algorithms, since they are more efficient and provide more convenient ways to initialize
    the simulation classes.

    Stored configurations, energies, swaps, and restart information are all written to a single output file using
    the platform portable, robust, and efficient NetCDF4 library.  Plans for future HDF5 support are pending.    
    
    ATTRIBUTES

    The following parameters (attributes) can be set after the object has been created, but before it has been
    initialized by a call to run():

    * collision_rate (units: 1/time) - the collision rate used for Langevin dynamics (default: 90 ps^-1)
    * constraint_tolerance (dimensionless) - relative constraint tolerance (default: 1e-6)
    * timestep (units: time) - timestep for Langevin dyanmics (default: 2 fs)
    * nsteps_per_iteration (dimensionless) - number of timesteps per iteration (default: 500)
    * number_of_iterations (dimensionless) - number of replica-exchange iterations to simulate (default: 100)
    * number_of_equilibration_iterations (dimensionless) - number of equilibration iterations before beginning exchanges (default: 0)
    * equilibration_timestep (units: time) - timestep for use in equilibration (default: 2 fs)
    * replica_mixing_scheme (string) - scheme used to swap replicas: 'swap-all' or 'swap-neighbors' (default: 'swap-all')
    * online_analysis (boolean) - if True, analysis will occur each iteration (default: False)
    
    TODO

    * Improve speed of Maxwell-Boltzmann velocity assignment.
    * Replace hard-coded Langevin dynamics with general MCMC moves.
    * Allow parallel resource to be used, if available (likely via Parallel Python).
    * Add support for and autodetection of other NetCDF4 interfaces.
    * Add HDF5 support.

    EXAMPLES

    Parallel tempering simulation of alanine dipeptide in implicit solvent (replica exchange among temperatures)
    (This is just an illustrative example; use ParallelTempering class for actual production parallel tempering simulations.)
    
    >>> # Create test system.
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Create thermodynamic states for parallel tempering with exponentially-spaced schedule.
    >>> import simtk.unit as units
    >>> import math
    >>> n_replicas = 3 # number of temperature replicas
    >>> T_min = 298.0 * units.kelvin # minimum temperature
    >>> T_max = 600.0 * units.kelvin # maximum temperature
    >>> T_i = [ T_min + (T_max - T_min) * (np.exp(float(i) / float(n_replicas-1)) - 1.0) / (np.e - 1.0) for i in range(n_replicas) ]
    >>> from thermodynamics import ThermodynamicState
    >>> states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(n_replicas) ]
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # use a temporary file for testing -- you will want to keep this file, since it stores output and checkpoint data
    >>> # Create simulation.
    >>> simulation = ReplicaExchange(states, coordinates, file.name) # initialize the replica-exchange simulation
    >>> simulation.minimize = False
    >>> simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
    >>> simulation.run() # run the simulation

    """    

    # Options to store.
    options_to_store = ['collision_rate', 'constraint_tolerance', 'timestep', 'nsteps_per_iteration', 'number_of_iterations', 'equilibration_timestep', 'number_of_equilibration_iterations', 'title', 'minimize', 'replica_mixing_scheme', 'online_analysis', 'show_mixing_statistics']

    def __init__(self, states, coordinates, filename, **kwargs):
        """
        Initialize replica-exchange simulation facility.

        ARGUMENTS
        
        states (list of ThermodynamicState) - Thermodynamic states to simulate, where one replica is allocated per state.
           Each state must have a system with the same number of atoms, and the same
           thermodynamic ensemble (combination of temperature, pressure, pH, etc.) must
           be defined for each.
        coordinates (Coordinate object or iterable container of Coordinate objects) - One or more sets of initial coordinates
           to be initially assigned to replicas in a round-robin fashion, provided simulation is not resumed from store file.
           Currently, coordinates must be specified as a list of simtk.unit.Quantity-wrapped numpy arrays.
        store_filename (string) - Name of file to bind simulation to use as storage for checkpointing and storage of results.

        OPTIONAL ARGUMENTS
        
        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict. Provided keywords will be matched to object variables to replace defaults.
        mm (implementation of simtk.openmm) - OpenMM API implementation to use (default: simtk.openmm)
        mpicomm (mpi4py communicator) - MPI communicator, if parallel execution is desired (default: None)
        metadata (dict) - metadata to store in a 'metadata' group in store file

        NOTES
        
        If store_filename exists, the simulation will try to resume from this file.
        If thermodynamic state information can be found in the store file, this will be read, and 'states' will be ignored.
        If the store file exists, coordinates will be read from this file, and 'coordinates' will be ignored.        
        The provided 'protocol' will override any options restored from the store file.

        """

        # To allow for parameters to be modified after object creation, class is not initialized until a call to self._initialize().
        self._initialized = False

        # Set MPI communicator (or None if not used).
        options = process_kwargs(kwargs)
        self.mpicomm = kwargs.get("mpicomm")
        
        if (self.mpicomm is not None and self.mpicomm.rank == 0) or self.mpicomm is None:
            self.head_node = True
        else:
            self.head_node = False

        if self.head_node:
            self.database = netcdf_io.NetCDFDatabase(filename, states, coordinates, **kwargs)
            self.resume = self.database.resume
            
        if self.mpicomm:
            self.resume = self.mpicomm.bcast(self.resume, root=0) # use whatever root node decides

        if self.resume == False:
            self.states = states
            self.provided_coordinates = fix_coordinates(coordinates)
            self.options = options
        else:
            if self.head_node:
                self.states == self.database.states
                self.options = self.database.options
                self.provided_coordinates = self.database.coordinates
            elif self.mpicomm:
                self.states = self.mpicomm.bcast(self.states, root=0) # use whatever root node decides
                self.options = self.mpicomm.bcast(self.options, root=0) # use whatever root node decides
                self.provided_coordinates = self.mpicomm.bcast(self.provided_coordinates, root=0) # use whatever root node decides
        
        self.platform = kwargs.get("platform")  # For convenience
        
        self.n_replicas = len(self.states)  # Determine number of replicas from the number of specified thermodynamic states.

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in self.states:
            if not state.is_compatible_with(self.states[0]):
                raise ValueError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")
                
        self.set_attributes()

    def set_attributes(self):
        for key, val in self.options.iteritems():
            setattr(self, key, val)

    def __repr__(self):
        """
        Return a 'formal' representation that can be used to reconstruct the class, if possible.

        """

        # TODO: Can we make this a more useful expression?
        return "<instance of ReplicaExchange>"

    def __str__(self):
        """
        Show an 'informal' human-readable representation of the replica-exchange simulation.

        """
        
        r =  ""
        r += "Replica-exchange simulation\n"
        r += "\n"
        r += "%d replicas\n" % str(self.n_replicas)
        r += "%d coordinate sets provided\n" % len(self.provided_coordinates)
        r += "file store: %s\n" % str(self.store_filename)
        r += "initialized: %s\n" % str(self._initialized)
        r += "\n"
        r += "PARAMETERS\n"
        r += "collision rate: %s\n" % str(self.collision_rate)
        r += "relative constraint tolerance: %s\n" % str(self.constraint_tolerance)
        r += "timestep: %s\n" % str(self.timestep)
        r += "number of steps/iteration: %s\n" % str(self.nsteps_per_iteration)
        r += "number of iterations: %s\n" % str(self.number_of_iterations)
        r += "equilibration timestep: %s\n" % str(self.equilibration_timestep)
        r += "number of equilibration iterations: %s\n" % str(self.number_of_equilibration_iterations)
        r += "\n"
        
        return r

    def run(self):
        """
        Run the replica-exchange simulation.

        Any parameter changes (via object attributes) that were made between object creation and calling this method become locked in
        at this point, and the object will create and bind to the store file.  If the store file already exists, the run will be resumed
        if possible; otherwise, an exception will be raised.

        """

        # Make sure we've initialized everything and bound to a storage file before we begin execution.
        if not self._initialized:
            self._initialize()

        # Main loop
        run_start_time = time.time()              
        run_start_iteration = self.iteration
        while (self.iteration < self.number_of_iterations):
            logger.debug("\nIteration %d / %d" % (self.iteration+1, self.number_of_iterations))
            initial_time = time.time()

            # Attempt replica swaps to sample from equilibrium permuation of states associated with replicas.
            self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states.
            self._compute_energies()

            self._show_energies()

            # Analysis.
            if self.online_analysis:
                self._analysis()

            # Write to storage file.
            self._write_iteration_netcdf()
            
            # Increment iteration counter.
            self.iteration += 1

            # Show mixing statistics.
            if self.show_mixing_statistics:
                self._show_mixing_statistics()

            # Show timing statistics.
            final_time = time.time()
            elapsed_time = final_time - initial_time
            estimated_time_remaining = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.number_of_iterations - self.iteration)
            estimated_total_time = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.number_of_iterations)
            estimated_finish_time = final_time + estimated_time_remaining
            logger.debug("Iteration took %.3f s." % elapsed_time)
            logger.debug("Estimated completion in %s, at %s (consuming total wall clock time %s)." % (str(datetime.timedelta(seconds=estimated_time_remaining)), time.ctime(estimated_finish_time), str(datetime.timedelta(seconds=estimated_total_time))))
            
            # Perform sanity checks to see if we should terminate here.
            self._run_sanity_checks()

        # Clean up and close storage files.
        self._finalize()

    def get_platform(self):

        if self.platform is not None:
            return

        state = self.states[0]
        context = mm.Context(state.system, mm.VerletIntegrator(self.timestep))
        self.platform = context.getPlatform()
        del context

    def _initialize(self):
        """
        Initialize the simulation, and bind to a storage file.

        """

        if self._initialized:
            print "Simulation has already been initialized."
            raise Error

        self.get_platform()

        if self.mpicomm:
            logger.debug("Initialized node %d / %d" % (self.mpicomm.rank, self.mpicomm.size))
  
        self.display_citations()

        # Determine number of alchemical states.
        self.n_states = len(self.states)

        # Create cached Context objects.
        logger.debug("Creating and caching Context objects...")
        MAX_SEED = (1<<31) - 1 # maximum seed value (max size of signed C long)
        seed = int(np.random.randint(MAX_SEED)) # TODO: Is this the right maximum value to use?
        if self.mpicomm:
            # Compile all kernels on master process to avoid nvcc deadlocks.
            # TODO: Fix this when nvcc fix comes out.
            initial_time = time.time()
            if self.mpicomm.rank == 0:
                for state_index in range(self.n_states):
                    logger.debug("Master node compiling kernels for state %d / %d..." % (state_index, self.n_states))
                    state = self.states[state_index]
                    integrator = mm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)                    
                    integrator.setRandomNumberSeed(seed + self.mpicomm.rank) 
                    context = mm.Context(state.system, integrator, self.platform)
                    mm.LocalEnergyMinimizer.minimize(context, 0, 1) # DEBUG
                    del context, integrator
            self.mpicomm.barrier()
            final_time = time.time()
            elapsed_time = final_time - initial_time
            logger.debug("Barrier complete.  Compiling kernels took %.1f s." % elapsed_time)

            # Create cached contexts for only the states this process will handle.
            initial_time = time.time()
            for state_index in range(self.mpicomm.rank, self.n_states, self.mpicomm.size):
                logger.debug("Node %d / %d creating Context for state %d..." % (self.mpicomm.rank, self.mpicomm.size, state_index))
                state = self.states[state_index]
                try:
                    state._integrator = mm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)                    
                    state._integrator.setRandomNumberSeed(seed + self.mpicomm.rank) 
                    initial_context_time = time.time() # DEBUG
                    if self.platform:
                        logger.info("Node %d / %d: Using platform %s" % (self.mpicomm.rank, self.mpicomm.size, self.platform.getName()))
                        state._context = mm.Context(state.system, state._integrator, self.platform)
                    else:
                        logger.info("Node %d / %d: No platform specified." % (self.mpicomm.rank, self.mpicomm.size))
                        state._context = mm.Context(state.system, state._integrator)                    
                    logger.debug("Node %d / %d: Context creation took %.3f s" % (self.mpicomm.rank, self.mpicomm.size, time.time() - initial_context_time))
                except Exception as e:
                    print e
            logger.debug("Note %d / %d: Context creation done.  Waiting for MPI barrier..." % (self.mpicomm.rank, self.mpicomm.size))
            self.mpicomm.barrier()
            logger.debug("Barrier complete.")
        else:
            # Serial version.
            initial_time = time.time()
            for (state_index, state) in enumerate(self.states):  
                logger.debug("Creating Context for state %d..." % state_index)
                state._integrator = mm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)
                state._integrator.setRandomNumberSeed(seed)
                initial_context_time = time.time() # DEBUG
                if self.platform:
                    state._context = mm.Context(state.system, state._integrator, self.platform)
                else:
                    state._context = mm.Context(state.system, state._integrator)
                logger.debug("Context creation took %.3f s" % (time.time() - initial_context_time))
        final_time = time.time()
        elapsed_time = final_time - initial_time
        logger.debug("%.3f s elapsed." % elapsed_time)

        # Determine number of atoms in systems.
        self.natoms = self.states[0].system.getNumParticles()
  
        # Allocate storage.
        self.replica_coordinates = list() # replica_coordinates[i] is the configuration currently held in replica i
        self.replica_box_vectors = list() # replica_box_vectors[i] is the set of box vectors currently held in replica i  
        self.replica_states     = np.zeros([self.n_states], np.int32) # replica_states[i] is the state that replica i is currently at
        self.u_kl               = np.zeros([self.n_states, self.n_states], np.float32)        
        self.swap_Pij_accepted  = np.zeros([self.n_states, self.n_states], np.float32)
        self.Nij_proposed       = np.zeros([self.n_states,self.n_states], np.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1
        self.Nij_accepted       = np.zeros([self.n_states,self.n_states], np.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1

        # Distribute coordinate information to replicas in a round-robin fashion, making a deep copy.
        if not self.resume:
            self.replica_coordinates = [ copy.deepcopy(self.provided_coordinates[replica_index % len(self.provided_coordinates)]) for replica_index in range(self.n_states) ]

        # Assign default box vectors.
        self.replica_box_vectors = list()
        for state in self.states:
            [a,b,c] = state.system.getDefaultPeriodicBoxVectors()
            box_vectors = units.Quantity(np.zeros([3,3], np.float32), units.nanometers)
            box_vectors[0,:] = a
            box_vectors[1,:] = b
            box_vectors[2,:] = c
            self.replica_box_vectors.append(box_vectors)
        
        # Assign initial replica states.
        for replica_index in range(self.n_states):
            self.replica_states[replica_index] = replica_index

        if self.resume:
            # Resume from NetCDF file.
            self._resume_from_netcdf()

            # Show energies.
            if self.show_energies:
                self._show_energies()            
        else:
            # Minimize and equilibrate all replicas.
            self._minimize_and_equilibrate()
            
            # Initialize current iteration counter.
            self.iteration = 0
            
            # TODO: Perform any GPU sanity checks here.
            
            # Compute energies of all alchemical replicas
            self._compute_energies()
            
            # Show energies.
            if self.show_energies:
                self._show_energies()

            # Initialize NetCDF file.
            self._initialize_netcdf()

            # Store initial state.
            self._write_iteration_netcdf()

        # Analysis objet starts off empty.
        # TODO: Use an empty dict instead?
        self.analysis = None

        # Signal that the class has been initialized.
        self._initialized = True

        return

    def _finalize(self):
        """Do anything necessary to finish run except close files.
        """

        if not self.head_node:
            return
        
        self.database.finalize()

    def __del__(self):
        """Clean up, closing files.
        """
        self._finalize()

        if not self.head_node:
            return

        self.database.close()
        
    def display_citations(self):
        """
        Display papers to be cited.

        TODO:

        * Add original citations for various replica-exchange schemes.
        * Show subset of OpenMM citations based on what features are being used.

        """
        
        openmm_citations = """\
        Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing units. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
        Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. Comput. Sci. Eng. 12:34, 2010. DOI: 10.1109/MCSE.2010.27
        Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. J. Comput. Chem. 31:1268, 2010. DOI: 10.1002/jcc.21413
        Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. J. Chem. Theor. Comput. 6:434, 2010. DOI: 10.1021/ct900463w"""
        
        gibbs_citations = """\
        Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs sampling: Simple improvements for enhanced mixing. J. Chem. Phys., in press. arXiv: 1105.5749"""

        mbar_citations = """\
        Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states. J. Chem. Phys. 129:124105, 2008. DOI: 10.1063/1.2978177"""

        logger.info("Please cite the following:")
        logger.info("")
        logger.info("%s" % openmm_citations)
        if self.replica_mixing_scheme == 'swap-all':
            logger.info("%s" % gibbs_citations)
        if self.online_analysis:
            logger.info("%s" % mbar_citations)

    def _propagate_replica(self, replica_index):
        """
        Propagate the replica corresponding to the specified replica index.
        Caching is used.

        ARGUMENTS

        replica_index (int) - the replica to propagate

        RETURNS

        elapsed_time (float) - time (in seconds) to propagate replica

        """

        start_time = time.time()

        # Retrieve state.
        state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
        state = self.states[state_index] # thermodynamic state

        # Retrieve integrator and context from thermodynamic state.
        integrator = state._integrator
        context = state._context

        coordinates = self.replica_coordinates[replica_index]            
        context.setPositions(coordinates)
        setpositions_end_time = time.time()

        box_vectors = self.replica_box_vectors[replica_index]
        context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])

        velocities = generate_maxwell_boltzmann_velocities(state.system, state.temperature) 
        context.setVelocities(velocities)
        setvelocities_end_time = time.time()
        # Run dynamics.
        integrator.step(self.nsteps_per_iteration)
        integrator_end_time = time.time() 
        # Store final coordinates
        getstate_start_time = time.time()
        openmm_state = context.getState(getPositions=True)
        getstate_end_time = time.time()
        self.replica_coordinates[replica_index] = openmm_state.getPositions(asNumpy=True)
        # Store box vectors.
        self.replica_box_vectors[replica_index] = openmm_state.getPeriodicBoxVectors(asNumpy=True)

        end_time = time.time()
        elapsed_time = end_time - start_time
        positions_elapsed_time = setpositions_end_time - start_time
        velocities_elapsed_time = setvelocities_end_time - setpositions_end_time
        integrator_elapsed_time = integrator_end_time - setvelocities_end_time
        getstate_elapsed_time = getstate_end_time - integrator_end_time
        logger.debug("Replica %d/%d: integrator elapsed time %.3f s (positions %.3f s | velocities %.3f s | integrate+getstate %.3f s)." % (replica_index, self.n_replicas, elapsed_time, positions_elapsed_time, velocities_elapsed_time, integrator_elapsed_time+getstate_elapsed_time))

        return elapsed_time

    def _propagate_replicas_mpi(self):
        """
        Propagate all replicas using MPI communicator.

        It is presumed all nodes have the correct configurations in the correct replica slots, but that state indices may be unsynchronized.

        TODO

        * Move synchronization of state information to mix_replicas?
        * Broadcast from root node only?

        """

        # Propagate all replicas.
        logger.debug("Propagating all replicas for %.3f ps..." % (self.nsteps_per_iteration * self.timestep / units.picoseconds))

        # Run just this node's share of states.
        logger.debug("Running trajectories...")
        start_time = time.time()
        # replica_lookup = { self.replica_states[replica_index] : replica_index for replica_index in range(self.n_states) } # replica_lookup[state_index] is the replica index currently at state 'state_index' # requires Python 2.7 features
        replica_lookup = dict( (self.replica_states[replica_index], replica_index) for replica_index in range(self.n_states) ) # replica_lookup[state_index] is the replica index currently at state 'state_index' # Python 2.6 compatible
        replica_indices = [ replica_lookup[state_index] for state_index in range(self.mpicomm.rank, self.n_states, self.mpicomm.size) ] # list of replica indices for this node to propagate
        for replica_index in replica_indices:
            logger.debug("Node %3d/%3d propagating replica %3d state %3d..." % (self.mpicomm.rank, self.mpicomm.size, replica_index, self.replica_states[replica_index]))
            self._propagate_replica(replica_index)
        end_time = time.time()        
        elapsed_time = end_time - start_time
        # Collect elapsed time.
        node_elapsed_times = self.mpicomm.gather(elapsed_time, root=0) # barrier
        if self.mpicomm.rank == 0:
            node_elapsed_times = np.array(node_elapsed_times)
            end_time = time.time()        
            elapsed_time = end_time - start_time
            barrier_wait_times = elapsed_time - node_elapsed_times
            logger.debug("Running trajectories: elapsed time %.3f s (barrier time min %.3f s | max %.3f s | avg %.3f s)" % (elapsed_time, barrier_wait_times.min(), barrier_wait_times.max(), barrier_wait_times.mean()))
            logger.debug("Total time spent waiting for GPU: %.3f s" % (node_elapsed_times.sum()))

        # Send final configurations and box vectors back to all nodes.
        if (self.mpicomm.rank == 0):
            logger.debug("Synchronizing trajectories...")

        start_time = time.time()
        replica_indices_gather = self.mpicomm.allgather(replica_indices)
        replica_coordinates_gather = self.mpicomm.allgather([ self.replica_coordinates[replica_index] for replica_index in replica_indices ])
        replica_box_vectors_gather = self.mpicomm.allgather([ self.replica_box_vectors[replica_index] for replica_index in replica_indices ])
        for (source, replica_indices) in enumerate(replica_indices_gather):
            for (index, replica_index) in enumerate(replica_indices):
                self.replica_coordinates[replica_index] = replica_coordinates_gather[source][index]
                self.replica_box_vectors[replica_index] = replica_box_vectors_gather[source][index]

        end_time = time.time()
        logger.debug("Synchronizing configurations and box vectors: elapsed time %.3f s" % (end_time - start_time))
        
    def _propagate_replicas_serial(self):        
        """
        Propagate all replicas using serial execution.

        """

        logger.debug("Propagating all replicas for %.3f ps..." % (self.nsteps_per_iteration * self.timestep / units.picoseconds))
        for replica_index in range(self.n_states):
            self._propagate_replica(replica_index)

    def _propagate_replicas(self):
        """
        Propagate all replicas.

        TODO

        * Report on efficiency of dyanmics (fraction of time wasted to overhead).

        """
        start_time = time.time()

        if self.mpicomm:
            self._propagate_replicas_mpi()
        else:
            self._propagate_replicas_serial()

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_replica = elapsed_time / float(self.n_states)
        ns_per_day = self.timestep * self.nsteps_per_iteration / time_per_replica * 24*60*60 / units.nanoseconds
        logger.debug("Time to propagate all replicas: %.3f s (%.3f per replica, %.3f ns/day)." % (elapsed_time, time_per_replica, ns_per_day))

        return

    def _minimize_replica(self, replica_index):
        """
        Minimize the specified replica.

        """
        # Retrieve thermodynamic state.
        state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
        state = self.states[state_index] # thermodynamic state
        # Retrieve integrator and context.
        # TODO: This needs to be adapted in case Integrator and Context objects are not cached.
        integrator = state._integrator
        context = state._context
        # Set coordinates.
        coordinates = self.replica_coordinates[replica_index]            
        context.setPositions(coordinates)
        # Set box vectors.
        box_vectors = self.replica_box_vectors[replica_index]
        context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])
        # Minimize energy.
        minimized_coordinates = mm.LocalEnergyMinimizer.minimize(context, self.minimize_tolerance, self.minimize_maxIterations)
        # Store final coordinates
        openmm_state = context.getState(getPositions=True)
        self.replica_coordinates[replica_index] = openmm_state.getPositions(asNumpy=True)
        # Clean up.
        del integrator, context

        return
            
    def _minimize_and_equilibrate(self):
        """
        Minimize and equilibrate all replicas.

        """

        # Minimize
        if self.minimize:
            logger.debug("Minimizing all replicas...")

            if self.mpicomm:
                # MPI implementation.
                
                # Minimize this node's share of replicas.
                start_time = time.time()
                for replica_index in range(self.mpicomm.rank, self.n_states, self.mpicomm.size):
                    logger.debug("node %d / %d : minimizing replica %d / %d" % (self.mpicomm.rank, self.mpicomm.size, replica_index, self.n_states))
                    self._minimize_replica(replica_index)
                end_time = time.time()
                self.mpicomm.barrier()
                logger.debug("Running trajectories: elapsed time %.3f s" % (end_time - start_time))

                # Send final configurations and box vectors back to all nodes.
                logger.debug("Synchronizing trajectories...")
                replica_coordinates_gather = self.mpicomm.allgather(self.replica_coordinates[self.mpicomm.rank:self.n_states:self.mpicomm.size])
                replica_box_vectors_gather = self.mpicomm.allgather(self.replica_box_vectors[self.mpicomm.rank:self.n_states:self.mpicomm.size])        
                for replica_index in range(self.n_states):
                    source = replica_index % self.mpicomm.size # node with trajectory data
                    index = replica_index // self.mpicomm.size # index within trajectory batch
                    self.replica_coordinates[replica_index] = replica_coordinates_gather[source][index]
                    self.replica_box_vectors[replica_index] = replica_box_vectors_gather[source][index]
                logger.debug("Synchronizing configurations and box vectors: elapsed time %.3f s" % (end_time - start_time))

            else:
                # Serial implementation.
                for replica_index in range(self.n_states):
                    self._minimize_replica(replica_index)

        # Equilibrate
        production_timestep = self.timestep
        for iteration in range(self.number_of_equilibration_iterations):
            logger.debug("equilibration iteration %d / %d" % (iteration, self.number_of_equilibration_iterations))
            self._propagate_replicas()
        self.timestep = production_timestep


    def _compute_energies(self):
        """
        Compute energies of all replicas at all states.

        TODO

        * We have to re-order Context initialization if we have variable box volume
        * Parallel implementation
        
        """

        start_time = time.time()
        
        logger.debug("Computing energies...")

        if self.mpicomm:
            # MPI version.

            # Compute energies for this node's share of states.
            for state_index in range(self.mpicomm.rank, self.n_states, self.mpicomm.size):
                for replica_index in range(self.n_states):
                    self.u_kl[replica_index,state_index] = self.states[state_index].reduced_potential(self.replica_coordinates[replica_index], box_vectors=self.replica_box_vectors[replica_index], platform=self.platform)

            # Send final energies to all nodes.
            energies_gather = self.mpicomm.allgather(self.u_kl[:,self.mpicomm.rank:self.n_states:self.mpicomm.size])
            for state_index in range(self.n_states):
                source = state_index % self.mpicomm.size # node with trajectory data
                index = state_index // self.mpicomm.size # index within trajectory batch
                self.u_kl[:,state_index] = energies_gather[source][:,index]

        else:
            # Serial version.
            for state_index in range(self.n_states):
                for replica_index in range(self.n_states):
                    self.u_kl[replica_index,state_index] = self.states[state_index].reduced_potential(self.replica_coordinates[replica_index], box_vectors=self.replica_box_vectors[replica_index], platform=self.platform)

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy= elapsed_time / float(self.n_states)**2 
        logger.debug("Time to compute all energies %.3f s (%.3f per energy calculation)." % (elapsed_time, time_per_energy))

    def _mix_all_replicas(self):
        """
        Attempt exchanges between all replicas to enhance mixing.

        TODO

        * Adjust nswap_attempts based on how many we can afford to do and not have mixing take a substantial fraction of iteration time.
        
        """

        # Determine number of swaps to attempt to ensure thorough mixing.
        # TODO: Replace this with analytical result computed to guarantee sufficient mixing.
        nswap_attempts = self.n_states**5 # number of swaps to attempt (ideal, but too slow!)
        nswap_attempts = self.n_states**3 # best compromise for pure Python?
        
        logger.debug("Will attempt to swap all pairs of replicas, using a total of %d attempts." % nswap_attempts)

        # Attempt swaps to mix replicas.
        for swap_attempt in range(nswap_attempts):
            # Choose replicas to attempt to swap.
            i = np.random.randint(self.n_states) # Choose replica i uniformly from set of replicas.
            j = np.random.randint(self.n_states) # Choose replica j uniformly from set of replicas.

            # Determine which states these resplicas correspond to.
            istate = self.replica_states[i] # state in replica slot i
            jstate = self.replica_states[j] # state in replica slot j

            # Reject swap attempt if any energies are nan.
            if (np.isnan(self.u_kl[i,jstate]) or np.isnan(self.u_kl[j,istate]) or np.isnan(self.u_kl[i,istate]) or np.isnan(self.u_kl[j,jstate])):
                continue

            # Compute log probability of swap.
            log_P_accept = - (self.u_kl[i,jstate] + self.u_kl[j,istate]) + (self.u_kl[i,istate] + self.u_kl[j,jstate])

            #print "replica (%3d,%3d) states (%3d,%3d) energies (%8.1f,%8.1f) %8.1f -> (%8.1f,%8.1f) %8.1f : log_P_accept %8.1f" % (i,j,istate,jstate,self.u_kl[i,istate],self.u_kl[j,jstate],self.u_kl[i,istate]+self.u_kl[j,jstate],self.u_kl[i,jstate],self.u_kl[j,istate],self.u_kl[i,jstate]+self.u_kl[j,istate],log_P_accept)

            # Record that this move has been proposed.
            self.Nij_proposed[istate,jstate] += 1
            self.Nij_proposed[jstate,istate] += 1

            # Accept or reject.
            if (log_P_accept >= 0.0 or (np.random.rand() < np.exp(log_P_accept))):
                # Swap states in replica slots i and j.
                (self.replica_states[i], self.replica_states[j]) = (self.replica_states[j], self.replica_states[i])
                # Accumulate statistics
                self.Nij_accepted[istate,jstate] += 1
                self.Nij_accepted[jstate,istate] += 1

        return

    def _mix_all_replicas_weave(self):
        """
        Attempt exchanges between all replicas to enhance mixing.
        Acceleration by 'weave' from scipy is used to speed up mixing by ~ 400x.
        
        """

        # TODO: Replace this with a different acceleration scheme to achieve better performance?

        # Determine number of swaps to attempt to ensure thorough mixing.
        # TODO: Replace this with analytical result computed to guarantee sufficient mixing.
        # TODO: Alternatively, use timing to figure out how many swaps we can do and still keep overhead to ~ 1% of iteration time?
        # nswap_attempts = self.n_states**5 # number of swaps to attempt (ideal, but too slow!)
        nswap_attempts = self.n_states**4 # number of swaps to attempt
        # Handled in C code below.
        
        logger.debug("Will attempt to swap all pairs of replicas using weave-accelerated code, using a total of %d attempts." % nswap_attempts)

        from scipy import weave

        # TODO: Replace drand48 with numpy random generator.
        code = """
        // Determine number of swap attempts.
        // TODO: Replace this with analytical result computed to guarantee sufficient mixing.        
        //long nswap_attempts = n_states*n_states*n_states*n_states*n_states; // K**5
        //long nswap_attempts = n_states*n_states*n_states; // K**3
        long nswap_attempts = n_states*n_states*n_states*n_states; // K**4

        // Attempt swaps.
        for(long swap_attempt = 0; swap_attempt < nswap_attempts; swap_attempt++) {
            // Choose replicas to attempt to swap.
            int i = (long)(drand48() * n_states); 
            int j = (long)(drand48() * n_states);

            // Determine which states these resplicas correspond to.            
            int istate = REPLICA_STATES1(i); // state in replica slot i
            int jstate = REPLICA_STATES1(j); // state in replica slot j

            // Reject swap attempt if any energies are nan.
            if ((std::isnan(U_KL2(i,jstate)) || std::isnan(U_KL2(j,istate)) || std::isnan(U_KL2(i,istate)) || std::isnan(U_KL2(j,jstate))))
               continue;

            // Compute log probability of swap.
            double log_P_accept = - (U_KL2(i,jstate) + U_KL2(j,istate)) + (U_KL2(i,istate) + U_KL2(j,jstate));

            // Record that this move has been proposed.
            NIJ_PROPOSED2(istate,jstate) += 1;
            NIJ_PROPOSED2(jstate,istate) += 1;

            // Accept or reject.
            if (log_P_accept >= 0.0 || (drand48() < exp(log_P_accept))) {
                // Swap states in replica slots i and j.
                int tmp = REPLICA_STATES1(i);
                REPLICA_STATES1(i) = REPLICA_STATES1(j);
                REPLICA_STATES1(j) = tmp;
                // Accumulate statistics
                NIJ_ACCEPTED2(istate,jstate) += 1;
                NIJ_ACCEPTED2(jstate,istate) += 1;
            }

        }
        """

        # Stage input temporarily.
        n_states = self.n_states
        replica_states = self.replica_states
        u_kl = self.u_kl
        Nij_proposed = self.Nij_proposed
        Nij_accepted = self.Nij_accepted

        # Execute inline C code with weave.
        info = weave.inline(code, ['n_states', 'replica_states', 'u_kl', 'Nij_proposed', 'Nij_accepted'], headers=['<math.h>', '<stdlib.h>'], verbose=2)

        # Store results.
        self.replica_states = replica_states
        self.Nij_proposed = Nij_proposed
        self.Nij_accepted = Nij_accepted

        return

    def _mix_neighboring_replicas(self):
        """
        Attempt exchanges between neighboring replicas only.

        """

        logger.debug("Will attempt to swap only neighboring replicas.")

        # Attempt swaps of pairs of replicas using traditional scheme (e.g. [0,1], [2,3], ...)
        offset = np.random.randint(2) # offset is 0 or 1
        for istate in range(offset, self.n_states-1, 2):
            jstate = istate + 1 # second state to attempt to swap with i

            # Determine which replicas these states correspond to.
            i = None
            j = None
            for index in range(self.n_states):
                if self.replica_states[index] == istate: i = index
                if self.replica_states[index] == jstate: j = index                

            # Reject swap attempt if any energies are nan.
            if (np.isnan(self.u_kl[i,jstate]) or np.isnan(self.u_kl[j,istate]) or np.isnan(self.u_kl[i,istate]) or np.isnan(self.u_kl[j,jstate])):
                continue

            # Compute log probability of swap.
            log_P_accept = - (self.u_kl[i,jstate] + self.u_kl[j,istate]) + (self.u_kl[i,istate] + self.u_kl[j,jstate])

            #print "replica (%3d,%3d) states (%3d,%3d) energies (%8.1f,%8.1f) %8.1f -> (%8.1f,%8.1f) %8.1f : log_P_accept %8.1f" % (i,j,istate,jstate,self.u_kl[i,istate],self.u_kl[j,jstate],self.u_kl[i,istate]+self.u_kl[j,jstate],self.u_kl[i,jstate],self.u_kl[j,istate],self.u_kl[i,jstate]+self.u_kl[j,istate],log_P_accept)

            # Record that this move has been proposed.
            self.Nij_proposed[istate,jstate] += 1
            self.Nij_proposed[jstate,istate] += 1

            # Accept or reject.
            if (log_P_accept >= 0.0 or (np.random.rand() < np.exp(log_P_accept))):
                # Swap states in replica slots i and j.
                (self.replica_states[i], self.replica_states[j]) = (self.replica_states[j], self.replica_states[i])
                # Accumulate statistics
                self.Nij_accepted[istate,jstate] += 1
                self.Nij_accepted[jstate,istate] += 1

    def _mix_replicas(self):
        """
        Attempt to swap replicas according to user-specified scheme.
        
        """

        if (self.mpicomm) and (self.mpicomm.rank != 0):
            # Non-root nodes receive state information.
            self.replica_states = self.mpicomm.bcast(self.replica_states, root=0)
            return

        logger.debug("Mixing replicas...")

        # Reset storage to keep track of swap attempts this iteration.
        self.Nij_proposed[:,:] = 0
        self.Nij_accepted[:,:] = 0

        # Perform swap attempts according to requested scheme.
        start_time = time.time()                    
        if self.replica_mixing_scheme == 'swap-neighbors':
            self._mix_neighboring_replicas()        
        elif self.replica_mixing_scheme == 'swap-all':
            # Try to use weave-accelerated mixing code if possible, otherwise fall back to Python-accelerated code.            
            try:
                self._mix_all_replicas_weave()            
            except:
                self._mix_all_replicas()
        elif self.replica_mixing_scheme == 'none':
            # Don't mix replicas.
            pass
        else:
            raise ValueError("Replica mixing scheme '%s' unknown.  Choose valid 'replica_mixing_scheme' parameter." % self.replica_mixing_scheme)
        end_time = time.time()

        # Determine fraction of swaps accepted this iteration.        
        nswaps_attempted = self.Nij_proposed.sum()
        nswaps_accepted = self.Nij_accepted.sum()
        swap_fraction_accepted = 0.0
        if (nswaps_attempted > 0): swap_fraction_accepted = float(nswaps_accepted) / float(nswaps_attempted);            
        logger.debug("Accepted %d / %d attempted swaps (%.1f %%)" % (nswaps_accepted, nswaps_attempted, swap_fraction_accepted * 100.0))

        # Estimate cumulative transition probabilities between all states.
        #Nij_accepted = self.ncfile.variables['accepted'][:,:,:].sum(0) + self.Nij_accepted        
        #Nij_proposed = self.ncfile.variables['proposed'][:,:,:].sum(0) + self.Nij_proposed
        Nij_accepted = self.database.accepted[:].sum(0) + self.Nij_accepted
        Nij_proposed = self.database.proposed[:].sum(0) + self.Nij_proposed
        swap_Pij_accepted = np.zeros([self.n_states,self.n_states], np.float64)
        for istate in range(self.n_states):
            Ni = Nij_proposed[istate,:].sum()
            if (Ni == 0):
                swap_Pij_accepted[istate,istate] = 1.0
            else:
                swap_Pij_accepted[istate,istate] = 1.0 - float(Nij_accepted[istate,:].sum() - Nij_accepted[istate,istate]) / float(Ni)
                for jstate in range(self.n_states):
                    if istate != jstate:
                        swap_Pij_accepted[istate,jstate] = float(Nij_accepted[istate,jstate]) / float(Ni)

        if self.mpicomm:
            # Root node will share state information with all replicas.
            logger.debug("Sharing state information...")
            self.replica_states = self.mpicomm.bcast(self.replica_states, root=0)

        logger.debug("Mixing of replicas took %.3f s" % (end_time - start_time))


    def _show_mixing_statistics(self):
        """
        Print summary of mixing statistics.

        """

        # Only root node can print.
        if self.mpicomm and (self.mpicomm.rank != 0):
            return

        # Don't print anything until we've accumulated some statistics.
        if self.iteration < 2:
            return

        # Don't print anything if there is only one replica.
        if (self.n_replicas < 2):
            return
        
        initial_time = time.time()

        # Compute statistics of transitions.
        Nij = np.zeros([self.n_states,self.n_states], np.float64)
        for iteration in range(self.iteration - 1):
            for ireplica in range(self.n_states):
                #istate = self.ncfile.variables['states'][iteration,ireplica]
                istate = self.database.states[iteration, ireplica]
                #jstate = self.ncfile.variables['states'][iteration+1,ireplica]
                jstate = self.database.states[iteration + 1, ireplica]
                Nij[istate,jstate] += 0.5
                Nij[jstate,istate] += 0.5
        Tij = np.zeros([self.n_states,self.n_states], np.float64)
        for istate in range(self.n_states):
            Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

        if self.show_mixing_statistics:
            # Print observed transition probabilities.
            PRINT_CUTOFF = 0.001 # Cutoff for displaying fraction of accepted swaps.
            print "Cumulative symmetrized state mixing transition matrix:"
            print "%6s" % "",
            for jstate in range(self.n_states):
                print "%6d" % jstate,
            print ""
            for istate in range(self.n_states):
                print "%-6d" % istate,
                for jstate in range(self.n_states):
                    P = Tij[istate,jstate]
                    if (P >= PRINT_CUTOFF):
                        print "%6.3f" % P,
                    else:
                        print "%6s" % "",
                print ""

        # Estimate second eigenvalue and equilibration time.
        mu = np.linalg.eigvals(Tij)
        mu = -np.sort(-mu) # sort in descending order
        if (mu[1] >= 1):
            print "Perron eigenvalue is unity; Markov chain is decomposable."
        else:
            print "Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1]))

        # Show time consumption statistics.
        final_time = time.time()
        elapsed_time = final_time - initial_time
        logger.info("Time to compute mixing statistics %.3f s" % elapsed_time)

    def output_iteration(self):
        """Write positions, states, and energies of current iteration to arbitrary database.
        """
        if not self.head_node:
            return

        coordinates = np.array([self.replica_coordinates[replica_index] / units.nanometers for replica_index in range(self.n_states)])
        box_vectors = np.array([self.replica_box_vectors[replica_index] for replica_index in range(self.n_states)])
        
        volumes = []
        for replica_index, v in enumerate(box_vectors):
            state_index = self.replica_states[replica_index]
            state = self.states[state_index]
            volumes.append(state._volume(v) / (units.nanometers**3))
        
        volumes = np.array(volumes)

        self.database.output_iteration(iteration=self.iteration, coordinates=coordinates, box_vectors=box_vectors, 
                volumes=volumes, replica_states=self.replica_states, energies=self.u_kl, 
                proposed=self.Nij_proposed, accepted=self.Nij_accepted, time=time.ctime())
    

    def _run_sanity_checks(self):
        """
        Run some checks on current state information to see if something has gone wrong that precludes continuation.

        """

        abort = False

        # Check positions.
        for replica_index in range(self.n_replicas):
            coordinates = self.replica_coordinates[replica_index]
            x = coordinates / units.nanometers
            if np.any(np.isnan(x)):
                print "nan encountered in replica %d coordinates." % replica_index
                abort = True

        # Check energies.
        if np.any(np.isnan(self.u_kl)):
            print "nan encountered in u_kl state energies"
            abort = True

        if abort:
            if self.mpicomm:
                self.mpicomm.Abort()
            else:
                raise Exception("Aborting.")


    def _show_energies(self):
        """
        Show energies (in units of kT) for all replicas at all states.

        """

        # Only root node can print.
        if self.mpicomm and (self.mpicomm.rank != 0):
            return

        # print header
        print "%-24s %16s" % ("reduced potential (kT)", "current state"),
        for state_index in range(self.n_states):
            print " state %3d" % state_index,
        print ""

        # print energies in kT
        for replica_index in range(self.n_states):
            print "replica %-16d %16d" % (replica_index, self.replica_states[replica_index]),
            for state_index in range(self.n_states):
                u = self.u_kl[replica_index,state_index]
                if (u > 1e6):
                    print "%10.3e" % u,
                else:
                    print "%10.1f" % u,
            print ""

