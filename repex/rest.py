import simtk.openmm as mm
import numpy as np
from thermodynamics import ThermodynamicState
from replica_exchange import ReplicaExchange
from mcmc import SamplerState

import logging
logger = logging.getLogger(__name__)

from constants import kB


class REST(ReplicaExchange):
    """Replica Exchange with Solute Tempering (2.0)

    
    Notes
    -----
    
    To create a new HamiltonianExchange object, use the `create_repex()`
    class method.  
    
    """

    def __init__(self, thermodynamic_states, sampler_states=None, database=None, mpicomm=None, platform=None, parameters={}):
        self._check_self_consistency(thermodynamic_states)
        super(REST, self).__init__(thermodynamic_states, sampler_states=sampler_states, database=database, mpicomm=mpicomm, platform=platform, parameters=parameters)

    def _check_self_consistency(self, thermodynamic_states):
        """Checks that each state has the same temperature and pressure, as required for HamiltonianExchange."""
        
        for s0 in thermodynamic_states:
            for s1 in thermodynamic_states:
                if s0.pressure != s1.pressure:
                    raise(ValueError("For HamiltonianExchange, ThermodynamicState objects cannot have different pressures!"))

        for s0 in thermodynamic_states:
            for s1 in thermodynamic_states:
                if s0.temperature != s1.temperature:
                    raise(ValueError("For HamiltonianExchange, ThermodynamicState objects cannot have different temperatures!"))

#def create(cls, system, coordinates, filename, T_min=None, T_max=None, temperatures=None, n_temps=None, pressure=None, mpicomm=None, platform=None, parameters={}):
    @classmethod
    def create(cls, reference_state, system, coordinates, filename, T_min, T_max, n_temps, hot_atoms, mpicomm=None, platform=None, parameters={}):
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
        
        
        temperatures = [ T_min + (T_max - T_min) * (np.exp(float(i) / float(n_temps-1)) - 1.0) / (np.e - 1.0) for i in range(n_temps) ]
        
        reference_temperature = temperatures[0]  # All systems will be simulated at reference temperature but with softened hamiltonians.
        
        thermodynamic_states = [ ThermodynamicState(system=system, temperature=reference_temperature, pressure=reference_state.pressure) for temperature in temperatures]
        
        for k, state in enumerate(thermodynamic_states):
            cls.perturb_system(state.system, temperature=temperatures[k], reference_temperature=reference_temperature, hot_atoms=hot_atoms)
        
        sampler_states = [SamplerState(thermodynamic_states[k].system, coordinates, platform=platform) for k in range(len(thermodynamic_states))]
        return super(cls, REST).create(thermodynamic_states, [coordinates for i in range(n_temps)], filename, sampler_states=sampler_states, mpicomm=mpicomm, platform=platform, parameters=parameters)
    
    @staticmethod
    def perturb_system(system, temperature, reference_temperature, hot_atoms):
        beta = 1.0 / (kB * temperature)
        beta0 = 1.0 / (kB * reference_temperature)
        rho = beta / beta0
        
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                REST._set_nb(force, hot_atoms, rho)
            if isinstance(force, mm.PeriodicTorsionForce):
                REST._set_torsion(force, hot_atoms, rho)
    
    @staticmethod
    def _set_nb(force, hot_atoms, rho):
        """Modify the NB forces for REST."""
        for atom in hot_atoms:
            q, sigma, epsilon = force.getParticleParameters(atom)
            epsilon = epsilon * rho
            q = q * (rho ** 0.5)
            force.setParticleParameters(atom, q, sigma, epsilon)
    
    @staticmethod
    def _set_torsion(force, hot_atoms, rho):
        """Modify the torsion forces for REST."""
        for k in range(force.getNumTorsions()):
            i0, i1, i2, i3, period, phase, force_constant = force.getTorsionParameters(k)
            if (i0 in hot_atoms) and (i1 in hot_atoms) and (i2 in hot_atoms) and (i3 in hot_atoms):
                force_constant = force_constant * rho
                force.setTorsionParameters(k, i0, i1, i2, i3, period, phase, force_constant)
            
