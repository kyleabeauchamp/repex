import time
import datetime

import numpy as np
import pandas as pd
import copy

import simtk.unit as units

import thermodynamics
from utils import find_matching_subclass, dict_to_named_tuple
from mcmc import SamplerState
import mcmc
import citations
import netcdf_io
from version import version as __version__
import dummympi

import logging
logger = logging.getLogger(__name__)


class ReplicaExchange(object):

    default_parameters = {}
    default_parameters["collision_rate"] = 91.0 / units.picosecond 
    default_parameters["constraint_tolerance"] = 1.0e-6 
    default_parameters["timestep"] = 2.0 * units.femtosecond
    default_parameters["nsteps_per_iteration"] = 500
    default_parameters["number_of_iterations"] = 10
    default_parameters["equilibration_timestep"] = 1.0 * units.femtosecond
    default_parameters["number_of_equilibration_iterations"] = 1
    default_parameters["title"] = 'Replica-exchange simulation created using ReplicaExchange class of repex.py on %s' % time.asctime(time.localtime())        
    default_parameters["minimize"] = True 
    default_parameters["minimize_tolerance"] = 1.0 * units.kilojoules_per_mole / units.nanometers # if specified, set minimization tolerance
    default_parameters["minimize_maxIterations"] = 0 # if nonzero, set maximum iterations
    default_parameters["replica_mixing_scheme"] = 'swap-all' # mix all replicas thoroughly
    default_parameters["online_analysis"] = False # if True, analysis will occur each iteration
    default_parameters["show_energies"] = True
    default_parameters["show_mixing_statistics"] = True
    default_parameters["integrator"] = None

    def __init__(self, thermodynamic_states, sampler_states=None, database=None, mpicomm=None, platform=None, parameters={}):
        """
        Create a new ReplicaExchange simulation object.
        
        Parameters
        ----------
        thermodynamic_states : list of ThermodynamicState objects
            List of thermodynamic states to simulate.
        sampler_states : list of SamplerState objects, optional?
            List of MCMC sampler states to initialize replica exchange simulations with.
        database : Database object, optional
            Database to use to write/append simulation data to.
        mpicomm : mpicomm implementation, optional?
            Communicator (real or dummy) implementation to use for parallelization.
        platform : simtk.openmm.Platform, optional, default None
            Platform to use for execution.

        Notes
        -----
        
        Use ReplicaExchange.create() to create a new repex simulation.  
        To resume an existing ReplicaExchange (or subclass) simulation,
        use the `resume()` function.  In general, you will not need to
        directory call the ReplicaExchange() constructor.
        
        """
        
        if mpicomm is None:
            self.mpicomm = dummympi.DummyMPIComm()
        else:
            self.mpicomm = mpicomm

        self.platform = platform
        self.database = database
        self.thermodynamic_states = thermodynamic_states
                
        self.n_states = len(self.thermodynamic_states)
        self.n_atoms = self.thermodynamic_states[0].system.getNumParticles()        
        
        if sampler_states is not None:  # New Repex job
            self.sampler_states = sampler_states
            
            self.parameters = self.process_parameters(parameters)  # Fill in missing parameters with defaults
            
            if self.mpicomm.rank == 0:  # Store the filled-in parameter namedtuple
                self.database.store_parameters(self.parameters)
            
            self.parameters = dict_to_named_tuple(self.parameters)  # Convert to namedtuple for const-ness
            self._check_run_parameter_consistency()
            
            self._allocate_arrays()
        else:  # Resume repex job
            self._broadcast_database()
        
        self.current_timestep = self.parameters.timestep
        
        self.n_replicas = len(self.thermodynamic_states)  # Determine number of replicas from the number of specified thermodynamic states.

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in self.thermodynamic_states:
            if not state.is_compatible_with(self.thermodynamic_states[0]):
                raise ValueError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")
        
        if self.database is not None:
            self.database.ncfile.repex_classname = self.__class__.__name__
            # Eventually, we might want to wrap a setter around the ncfile

        logger.debug("Initialized node %d / %d" % (self.mpicomm.rank, self.mpicomm.size))
        citations.display_citations(self.parameters.replica_mixing_scheme, self.parameters.online_analysis)

    def process_parameters(self, parameters):
        """Process a set of input run parameters and convert to named tuple.
        
        Parameters
        ----------
        
        parameters : dict
            List of user-supplied run parameters.
        
        Returns
        -------
        
        full_parameters : dict
            Dictinoary containing user-input run parameters as well as 
            default parameters for unspecified parameters.
        
        Notes
        -----
        
        The default parameters are stored in the class variable `default_parameters`
        which can be modified in subclasses.
        """
            
        options = {}

        for key in self.default_parameters:
            options[key] = parameters.get(key, self.default_parameters[key])
        
        for key in parameters.keys():
            if not options.has_key(key):
                options[key] = parameters[key]

        return options

    def extend(self, n_iter):
        """Extend an existing repex run and modify its database.
        
        Parameters
        ----------
        
        n_iter : int
            How many repex iterations to append.
        
        Notes
        -----
        
        This function is MPI aware and only makes database changes on the root 
        node.
        
        """
        value = self.parameters.number_of_iterations + n_iter
        if self.mpicomm.rank == 0:
            self.database._store_parameter("number_of_iterations", value)
            self.database.sync()
        
        self.parameters = self.parameters._asdict()  # Make dict for mutability
        self.parameters["number_of_iterations"] = value  # extend
        self.parameters = dict_to_named_tuple(self.parameters)  # Convert to namedtuple for const-ness


    def _check_run_parameter_consistency(self):
        """Check stored run parameters for self consistency.
        
        Notes
        -----
        
        To be implemented!
        """
        pass


    def _broadcast_database(self):
        """Load the positions, replica_states, u_kl, proposed, and accepted from root node database."""
        if self.mpicomm.rank == 0:
            positions = self.database.last_positions
            replica_states = self.database.last_replica_states
            u_kl = self.database.last_u_kl
            Nij_proposed = self.database.last_proposed
            Nij_accepted = self.database.last_accepted
            iteration = self.database.last_iteration
            iteration += 1  # Want to begin with the NEXT step of repex
            parameters = self.database.parameters
        else:
            positions, replica_states, u_kl, Nij_proposed, Nij_accepted, parameters, iteration = None, None, None, None, None, None, None

        positions = self.mpicomm.bcast(positions, root=0)
        self.replica_states = self.mpicomm.bcast(replica_states, root=0)
        self.u_kl = self.mpicomm.bcast(u_kl, root=0)
        self.iteration = self.mpicomm.bcast(iteration, root=0)
        self.Nij_proposed = self.mpicomm.bcast(Nij_proposed, root=0)
        self.Nij_accepted = self.mpicomm.bcast(Nij_accepted, root=0)
        
        self.parameters = self.mpicomm.bcast(parameters, root=0)  # Send out as dictionary
        self.parameters = dict_to_named_tuple(self.parameters)  # Convert to named_tuple for const-ness
        self._check_run_parameter_consistency()
        
        self.sampler_states = [SamplerState(self.thermodynamic_states[k].system, positions[k], self.platform) for k in range(len(self.thermodynamic_states))]


    def run(self):
        """
        Run the replica-exchange simulation.

        Any parameter changes (via object attributes) that were made between object creation and calling this method become locked in
        at this point, and the object will create and bind to the store file.  If the store file already exists, the run will be resumed
        if possible; otherwise, an exception will be raised.

        """

        # Main loop
        run_start_time = time.time()              
        run_start_iteration = self.iteration
        while (self.iteration < self.parameters.number_of_iterations):
            logger.debug("\nIteration %d / %d" % (self.iteration + 1, self.parameters.number_of_iterations))
            initial_time = time.time()

            # Attempt replica swaps to sample from equilibrium permuation of states associated with replicas.
            self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states.
            self._compute_energies()

            self._show_energies()

            # Analysis.
            if self.parameters.online_analysis:
                self._analysis()

            # Write to storage file.
            self.output_iteration()
            
            # Increment iteration counter.
            self.iteration += 1

            self._show_mixing_statistics()

            # Show timing statistics.
            final_time = time.time()
            elapsed_time = final_time - initial_time
            estimated_time_remaining = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.parameters.number_of_iterations - self.iteration)
            estimated_total_time = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.parameters.number_of_iterations)
            estimated_finish_time = final_time + estimated_time_remaining
            logger.debug("Iteration took %.3f s." % elapsed_time)
            logger.debug("Estimated completion in %s, at %s (consuming total wall clock time %s)." % (str(datetime.timedelta(seconds=estimated_time_remaining)), time.ctime(estimated_finish_time), str(datetime.timedelta(seconds=estimated_total_time))))
            
            # Perform sanity checks to see if we should terminate here.
            self._run_sanity_checks()

        # Clean up and close storage files.
        self._finalize()


    def _allocate_arrays(self):
        """Allocate the in-memory numpy arrays."""
  
        self.replica_states     = np.arange(self.n_states)  # replica_states[i] is the state that replica i is currently at
        self.u_kl               = np.zeros([self.n_states, self.n_states], np.float32)        
        self.Nij_proposed       = np.zeros([self.n_states, self.n_states], np.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1
        self.Nij_accepted       = np.zeros([self.n_states, self.n_states], np.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1    
        
        self.iteration = 0

        
    def _run_iteration_zero(self):
            # Minimize and equilibrate all replicas.
            self._minimize_and_equilibrate()
            
            # Initialize current iteration counter.
            self.iteration = 0
            
            # TODO: Perform any GPU sanity checks here.
            
            # Compute energies of all alchemical replicas
            self._compute_energies()
            self._show_energies()

            # Store initial state.
            self.output_iteration()        


    def _finalize(self):
        """Do anything necessary to finish run except close files.
        """

        if not self.mpicomm.rank == 0:
            return
        
        self.database._finalize()

    def __del__(self):
        """Clean up, closing files.
        """
        self._finalize()
        
    def _propagate_replica(self, replica_index):
        """Propagate the replica corresponding to the specified replica index.

        ARGUMENTS

        replica_index (int) - the replica to propagate

        RETURNS

        elapsed_time (float) - time (in seconds) to propagate replica

        """

        start_time = time.time()

        # Retrieve state.
        state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
        thermodynamic_state = self.thermodynamic_states[state_index] # thermodynamic state
        sampler_state = self.sampler_states[replica_index]
        
        # HACK: Use Langevin dynamics move.
        move_set = [mcmc.LangevinDynamicsMove(nsteps=self.parameters.nsteps_per_iteration, timestep=self.current_timestep, collision_rate=self.parameters.collision_rate)]

        sampler = mcmc.MCMCSampler(thermodynamic_state, move_set=move_set, platform=self.platform)
        new_sampler_state = sampler.run(sampler_state)
        
        self.sampler_states[replica_index] = new_sampler_state
        
        end_time = time.time()
        elapsed_time = end_time - start_time
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
        logger.debug("Propagating all replicas for %.3f ps..." % (self.parameters.nsteps_per_iteration * self.parameters.timestep / units.picoseconds))

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
        
        sampler_states_gather = self.mpicomm.allgather([self.sampler_states[replica_index] for replica_index in replica_indices ])
        for (source, replica_indices) in enumerate(replica_indices_gather):
            for (index, replica_index) in enumerate(replica_indices):
                self.sampler_states[replica_index].positions = sampler_states_gather[source][index].positions
                self.sampler_states[replica_index].box_vectors = sampler_states_gather[source][index].box_vectors

        end_time = time.time()
        logger.debug("Synchronizing configurations and box vectors: elapsed time %.3f s" % (end_time - start_time))

        
    def _propagate_replicas(self):
        """
        Propagate all replicas.

        TODO

        * Report on efficiency of dyanmics (fraction of time wasted to overhead).

        """
        start_time = time.time()

        self._propagate_replicas_mpi()

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_replica = elapsed_time / float(self.n_states)
        ns_per_day = self.parameters.timestep * self.parameters.nsteps_per_iteration / time_per_replica * 24*60*60 / units.nanoseconds
        logger.debug("Time to propagate all replicas: %.3f s (%.3f per replica, %.3f ns/day)." % (elapsed_time, time_per_replica, ns_per_day))


    def _minimize_all_replicas(self):
        for replica_index in range(self.n_states):
            self.sampler_states[replica_index].minimize(platform=self.platform)
            
    def _minimize_and_equilibrate(self):
        """
        Minimize and equilibrate all replicas.

        """

        # Minimize
        if self.parameters.minimize:
            logger.debug("Minimizing all replicas...")
            self._minimize_all_replicas()

        # Equilibrate
        self.current_timestep = self.parameters.equilibration_timestep
        
        for iteration in range(self.parameters.number_of_equilibration_iterations):
            logger.debug("equilibration iteration %d / %d" % (iteration, self.parameters.number_of_equilibration_iterations))
            self._propagate_replicas()
        
        self.current_timestep = self.parameters.timestep


    def _compute_energies(self):
        """
        Compute energies of all replicas at all states.

        TODO

        * We have to re-order Context initialization if we have variable box volume
        * Parallel implementation
        
        """

        start_time = time.time()
        
        logger.debug("Computing energies...")
        
        # TODO: Parallel implementation.
        # Compute energies for this node's share of states.
        for state_index in range(self.mpicomm.rank, self.n_states, self.mpicomm.size):
            for replica_index in range(self.n_states):
                self.u_kl[replica_index,state_index] = self.thermodynamic_states[state_index].reduced_potential(self.sampler_states[replica_index].positions, box_vectors=self.sampler_states[replica_index].box_vectors, platform=self.platform)

        # Send final energies to all nodes.
        energies_gather = self.mpicomm.allgather(self.u_kl[:,self.mpicomm.rank:self.n_states:self.mpicomm.size])
        for state_index in range(self.n_states):
            source = state_index % self.mpicomm.size # node with trajectory data
            index = state_index // self.mpicomm.size # index within trajectory batch
            self.u_kl[:,state_index] = energies_gather[source][:,index]

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

    def _mix_all_replicas_weave(self):
        """Attempt exchanges between all replicas to enhance mixing.  Uses 'weave'.
        
        Notes
        -----
        
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

    def _mix_neighboring_replicas(self):
        """Attempt exchanges between neighboring replicas only.
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
        """Attempt to swap replicas according to user-specified scheme.
        """

        if self.mpicomm.rank != 0:
            # Non-root nodes receive state information.
            self.replica_states = self.mpicomm.bcast(self.replica_states, root=0)
            return

        logger.debug("Mixing replicas...")

        # Reset storage to keep track of swap attempts this iteration.
        self.Nij_proposed[:,:] = 0
        self.Nij_accepted[:,:] = 0

        # Perform swap attempts according to requested scheme.
        start_time = time.time()                    
        if self.parameters.replica_mixing_scheme == 'swap-neighbors':
            self._mix_neighboring_replicas()        
        elif self.parameters.replica_mixing_scheme == 'swap-all':
            # Try to use weave-accelerated mixing code if possible, otherwise fall back to Python-accelerated code.            
            try:
                self._mix_all_replicas_weave()            
            except:
                self._mix_all_replicas()
        elif self.parameters.replica_mixing_scheme == 'none':
            # Don't mix replicas.
            pass
        else:
            raise ValueError("Replica mixing scheme '%s' unknown.  Choose valid 'replica_mixing_scheme' parameter." % self.parameters.replica_mixing_scheme)
        end_time = time.time()

        # Determine fraction of swaps accepted this iteration.        
        nswaps_attempted = self.Nij_proposed.sum()
        nswaps_accepted = self.Nij_accepted.sum()
        swap_fraction_accepted = 0.0
        if (nswaps_attempted > 0): swap_fraction_accepted = float(nswaps_accepted) / float(nswaps_attempted);            
        logger.debug("Accepted %d / %d attempted swaps (%.1f %%)" % (nswaps_accepted, nswaps_attempted, swap_fraction_accepted * 100.0))

        # Estimate cumulative transition probabilities between all states.
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
    
        # Root node will share state information with all replicas.
        logger.debug("Sharing state information...")
        self.replica_states = self.mpicomm.bcast(self.replica_states, root=0)

        logger.debug("Mixing of replicas took %.3f s" % (end_time - start_time))


    def _show_mixing_statistics(self):
        """Print summary of mixing statistics.
        """

        # Only root node can print.
        if self.mpicomm.rank != 0 or not self.parameters.show_mixing_statistics:
            return

        # Don't print anything until we've accumulated some statistics.
        if self.iteration < 2:
            return

        # Don't print anything if there is only one replica.
        if (self.n_replicas < 2):
            return

        self.database._show_mixing_statistics()

    def output_iteration(self):
        """Get relevant data from current iteration and store in database.
        
        Notes
        -----
        Will save the following information:
        "iteration", "positions", "box_vectors", "volumes", "energies", "energies", "proposed", "accepted", "timestamp"
        """
        if not self.mpicomm.rank == 0:
            return

        positions = np.array([self.sampler_states[replica_index].positions / units.nanometers for replica_index in range(self.n_states)])
        box_vectors = np.array([self.sampler_states[replica_index].box_vectors / units.nanometers for replica_index in range(self.n_states)])
        
        
        volumes = []
        for replica_index in range(self.n_states):
            v = self.sampler_states[replica_index].box_vectors
            state_index = self.replica_states[replica_index]
            state = self.thermodynamic_states[state_index]
            volumes.append(thermodynamics.volume(v) / (units.nanometers ** 3))
        
        volumes = np.array(volumes)

        self.database.write("positions", positions, self.iteration, sync=False)
        self.database.write("box_vectors", box_vectors, self.iteration, sync=False)
        self.database.write("volumes", volumes, self.iteration, sync=False)
        self.database.write("states", self.replica_states, self.iteration, sync=False)
        self.database.write("energies", self.u_kl, self.iteration, sync=False)
        self.database.write("proposed", self.Nij_proposed, self.iteration, sync=False)
        self.database.write("accepted", self.Nij_accepted, self.iteration, sync=False)
        self.database.write("timestamp", time.time(), self.iteration, sync=False)
        
        self.database.sync()
            

    def _run_sanity_checks(self):
        """Run some checks on current state information to see if something has gone wrong that precludes continuation.
        """

        abort = False

        # Check sampler state (positions and generalized coordinates).
        for replica_index in range(self.n_replicas):
            if self.sampler_states[replica_index].has_nan():
                logger.warn("nan encountered in replica %d coordinates." % replica_index)
                abort = True

        # Check energies.
        if np.any(np.isnan(self.u_kl)):
            logger.warn("nan encountered in u_kl state energies")
            abort = True

        if abort:
            self.mpicomm.Abort()


    def _show_energies(self):
        """Show energies (in units of kT) for all replicas at all states.
        """
        if self.mpicomm.rank != 0 or not self.parameters.show_energies:
            return

        U = pd.DataFrame(self.u_kl)
        logger.info("\n%-24s %16s\n%s" % ("reduced potential (kT)", "current state", U.to_string()))

    @classmethod
    def create(cls, thermodynamic_states, coordinates, filename, mpicomm=None, platform=None, parameters={}):
        """Create a new ReplicaExchange simulation.
        
        Parameters
        ----------

        thermodynamic_states : list([ThermodynamicStates])
            The list of thermodynamic states to simulate in
        coordinates : list([simtk.unit.Quantity]), shape=(n_replicas, n_atoms, 3), unit=Length
            The starting coordinates for each replica
        filename : string 
            name of NetCDF file to bind to for simulation output and checkpointing
        mpicomm : mpi4py communicator, default=None
            MPI communicator, if parallel execution is desired.      
        platform : simtk.openmm.Platform, optional
            Platform to use for simulations, or None if default is to be used.
        parameters (dict) - Optional parameters to use for specifying simulation
            Provided keywords will be matched to object variables to replace defaults.
            
        """    
        coordinates = validate_coordinates(coordinates, thermodynamic_states)

        if mpicomm is None or (mpicomm.rank == 0):
            database = netcdf_io.NetCDFDatabase(filename, thermodynamic_states, coordinates)  # To do: eventually use factory for looking up database type via filename
        else:
            database = None

        sampler_states = [SamplerState(thermodynamic_states[k].system, coordinates[k], platform=platform) for k in range(len(thermodynamic_states))]        
        
        repex = cls(thermodynamic_states, sampler_states, database, mpicomm=mpicomm, platform=platform, parameters=parameters)
        repex._run_iteration_zero()
        return repex
    

def resume(filename, platform=None, mpicomm=None):
    """Resume an existing ReplicaExchange (or subclass) simulation.
    
    Parameters
    ----------

    filename : string 
        name of NetCDF file to bind to for simulation output and checkpointing
    mpicomm : mpi4py communicator, default=None
        MPI communicator, if parallel execution is desired.      
        
    Notes
    -----
    
    This function attempts to find a subclasses of ReplicaExchange whose
    name matches the `repex_classname` attribute in the netCDF database.
    The matching only considers subclasses of ReplicaExchange that have
    been imported in the current python session.  To use a user-defined
    subclass, you must make sure you `import xyz`, where xyz is the python
    module where the subclass is defined.
        
    """
    if mpicomm is None:
        mpicomm = dummympi.DummyMPIComm()
    
    if mpicomm.rank == 0:
        database = netcdf_io.NetCDFDatabase(filename)  # To do: eventually use factory for looking up database type via filename
        thermodynamic_states, repex_classname = database.thermodynamic_states, database.repex_classname
        parameters = database.parameters
    else:
        database, thermodynamic_states, repex_classname, parameters = None, None, None, None

    thermodynamic_states = mpicomm.bcast(thermodynamic_states, root=0)
    repex_classname = mpicomm.bcast(repex_classname, root=0)
    parameters = mpicomm.bcast(parameters, root=0)
    
    cls = find_matching_subclass(ReplicaExchange, repex_classname)

    repex = cls(thermodynamic_states, database=database, mpicomm=mpicomm, platform=platform, parameters=parameters)

    return repex


def validate_coordinates(coordinates, thermodynamic_states):
    n_coord = len(coordinates)
    n_states = len(thermodynamic_states)

    if n_coord == 0 or n_states == 0:
        raise(Exception("Must have at least one state and coordinates."))

    if n_coord > n_states:
        raise(Exception("Cannot input more coordinates than states."))

    elif n_coord < n_states:
        logger.info("Input %d coordinates but %d states, so copying additional coordinates." % (n_coord, n_states))
        new_coordinates = []
        for i in range(n_states):
            new_coordinates.append(copy.deepcopy(coordinates[i % n_coord]))
    
    elif n_coord == n_states:
        new_coordinates = coordinates

    return new_coordinates
