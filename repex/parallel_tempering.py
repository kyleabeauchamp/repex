import time

import numpy as np

from thermodynamics import ThermodynamicState
from replica_exchange import ReplicaExchange
import netcdf_io
from mcmc import MCMCSamplerState

import logging
logger = logging.getLogger(__name__)

from constants import kB


class ParallelTempering(ReplicaExchange):
    """Parallel tempering simulation class.

    This class provides a facility for parallel tempering simulations.  It is a subclass of ReplicaExchange, but provides
    various convenience methods and efficiency improvements for parallel tempering simulations, so should be preferred for
    this type of simulation.  In particular, the System only need be specified once, while the temperatures (or a temperature
    range) is used to automatically build a set of ThermodynamicState objects for replica-exchange.  Efficiency improvements
    make use of the fact that the reduced potentials are linear in inverse temperature.
    
    Notes
    -----
    
    For creating a new ParallelTempering simulation, we recommend the use
    of the `create_repex` function, which provides a convenient way to 
    create PT simulations across a temperature range. 
    
    """

    def __init__(self, thermodynamic_states, sampler_states=None, database=None, mpicomm=None, parameters={}):
        self._check_self_consistency(thermodynamic_states)
        super(ParallelTempering, self).__init__(thermodynamic_states, sampler_states=sampler_states, database=database, mpicomm=mpicomm, parameters=parameters)

    def _check_self_consistency(self, thermodynamic_states):
        """Checks that each state is identical except for the temperature, as required for ParallelTempering."""
        
        for s0 in thermodynamic_states:
            for s1 in thermodynamic_states:
                if s0.pressure != s1.pressure:
                    raise(ValueError("For ParallelTempering, ThermodynamicState objects cannot have different pressures!"))

        for s0 in thermodynamic_states:
            for s1 in thermodynamic_states:
                if s0.system.__getstate__() != s1.system.__getstate__():
                    raise(ValueError("For ParallelTempering, ThermodynamicState objects cannot have different systems!"))


    def _compute_energies(self):
        """Compute reduced potentials of all replicas at all states (temperatures).

        Notes
        -----

        Because only the temperatures differ among replicas, we replace 
        the generic O(N^2) replica-exchange implementation with an O(N) implementation.
        """

        start_time = time.time()
        logger.debug("Computing energies...")
                
        for replica_index in range(self.mpicomm.rank, self.n_states, self.mpicomm.size):
            context = self.sampler_states[replica_index].createContext()
            # Compute potential energy.
            openmm_state = context.getState(getEnergy=True)            
            potential_energy = openmm_state.getPotentialEnergy()           
            # Compute energies at this state for all replicas.
            for state_index in range(self.n_states):
                # Compute reduced potential
                beta = 1.0 / (kB * self.thermodynamic_states[state_index].temperature)
                self.u_kl[replica_index,state_index] = beta * potential_energy

        # Gather energies.
        energies_gather = self.mpicomm.allgather(self.u_kl[self.mpicomm.rank:self.n_states:self.mpicomm.size,:])
        for replica_index in range(self.n_states):
            source = replica_index % self.mpicomm.size # node with trajectory data
            index = replica_index // self.mpicomm.size # index within trajectory batch
            self.u_kl[replica_index,:] = energies_gather[source][index]

        # Clean up.
        del context

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy = elapsed_time / float(self.n_states)
        logger.debug("Time to compute all energies %.3f s (%.3f per energy calculation).\n" % (elapsed_time, time_per_energy))


    @classmethod
    def create(cls, system, coordinates, filename, T_min=None, T_max=None, temperatures=None, n_temps=None, pressure=None, mpicomm=None, parameters={}):
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
        parameters (dict) - Optional parameters to use for specifying simulation
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
            temperatures = [ T_min + (T_max - T_min) * (np.exp(float(i) / float(n_temps-1)) - 1.0) / (np.e - 1.0) for i in range(n_temps) ]
        else:
            raise ValueError("Either 'temperatures' or 'T_min', 'T_max', and 'n_temps' must be provided.")

        thermodynamic_states = [ ThermodynamicState(system=system, temperature=temperatures[i], pressure=pressure) for i in range(n_temps) ]
    
        if mpicomm is None or (mpicomm.rank == 0):
            database = netcdf_io.NetCDFDatabase(filename, thermodynamic_states, coordinates)  # To do: eventually use factory for looking up database type via filename
        else:
            database = None
        
        # UGLY HACK BECAUSE MCMC SAMPLER STATE INITIALIZATION REQURES CONTEXT CREATION
        platform = None
        if 'platform' in parameters: platform = parameters['platform']
        # END UGLY HACK
        sampler_states = [MCMCSamplerState(thermodynamic_states[k].system, coordinates[k], platform=platform) for k in range(len(thermodynamic_states))]
        repex = cls(thermodynamic_states, sampler_states, database, mpicomm=mpicomm, parameters=parameters)
        # Override title.
        repex.title = 'Parallel tempering simulation created using ParallelTempering class of repex.py on %s' % time.asctime(time.localtime())        

        repex._run_iteration_zero()
        return repex
