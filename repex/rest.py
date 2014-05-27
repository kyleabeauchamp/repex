import itertools
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

    @classmethod
    def create(cls, reference_state, system, coordinates, filename, T_min, T_max_list, n_temps_list, hot_atom_lists, mpicomm=None, platform=None, parameters={}):
        """Create a new Replica exchange with Solute Tempering object.

        Parameters
        ----------

        reference_state : ThermodynamicState
            reference state containing all thermodynamic parameters 
            except the system, which will be replaced by 'systems'
        system : simt.openmm.System
            System to simulate
        coordinates : simtk.unit.Quantity, shape=(n_atoms, 3), unit=Length
            coordinates (or a list of coordinates objects) for initial 
            assignment of replicas (will be used in round-robin assignment)
        filename : string 
            name of NetCDF file to bind to for simulation output and checkpointing
        T_min : simtk.unit.Quantity
            The temperature of the reference state, e.g. the lowest temperature.
        T_max_list : list(simtk.unit.Quantity), length=n_atom_groups
            A list of highest temperatures for each atom group.
        hot_atom_lists : list(numpy array like, dtype='int'), length=n_atom_groups
            A list of arrays of atom indices, corresponding to the different atom
            groups to be softened.
        mpicomm : mpi4py communicator, default=None
            MPI communicator, if parallel execution is desired.      
        kwargs (dict) - Optional parameters to use for specifying simulation
            Provided keywords will be matched to object variables to replace defaults.
            
        Notes
        -----
        
        The atom groups in hot_atom_lists should be unique and orthogonal,
        otherwise double-softening will occur.

        """
        
        temperature_lists = [[T_min + (T_max - T_min) * (np.exp(float(i) / float(n_temps-1)) - 1.0) / (np.e - 1.0) for i in range(n_temps) ] for (T_max, n_temps) in zip(T_max_list, n_temps_list)]
        
        temperature_tuples = list(itertools.product(*temperature_lists))  # List of tuples of atom subset temperatures
        
        reference_temperature = T_min  # All systems will be simulated at reference temperature but with softened hamiltonians.
        
        # All thermodynamic states have same temperature and pressure, but will have modified system objects.
        thermodynamic_states = [ ThermodynamicState(system=system, temperature=reference_temperature, pressure=reference_state.pressure) for temperature_tuple in temperature_tuples]
        
        for k, state in enumerate(thermodynamic_states):
            cls.soften_system(state.system, temperature_tuple=temperature_tuples[k], reference_temperature=reference_temperature, hot_atom_lists=hot_atom_lists)
        
        sampler_states = [SamplerState(thermodynamic_states[k].system, coordinates, platform=platform) for k in range(len(thermodynamic_states))]
        return super(cls, REST).create(thermodynamic_states, [coordinates for i in range(n_temps)], filename, sampler_states=sampler_states, mpicomm=mpicomm, platform=platform, parameters=parameters)

    @staticmethod
    def soften_system(system, temperature_tuple, reference_temperature, hot_atom_lists):
        """Soften multiple groups of atoms using different temperatures.

        Parameters
        ----------

        system : simt.openmm.System
            System to soften
        temperature_tuple : tuple(simtk.unit.Quanity) [Kelvin]
            Desired softening temperatures for each atom group
        reference_temperature : simtk.unit.Quanity [Kelvin]
            Reference (or lowest) temperature.
        hot_atom_lists : list(numpy array like, dtype='int'), length=n_atom_groups
            A list of arrays of atom indices, corresponding to the different atom
            groups to be softened by the different temperatures in temperature_tuple
        """
        for temperature, hot_atoms in zip(temperature_tuple, hot_atom_lists):
            REST.soften_system_single_temperature(system, temperature, reference_temperature, hot_atoms)
    
    @staticmethod
    def soften_system_single_temperature(system, temperature, reference_temperature, hot_atoms, soften_torsions=True):
        """Soften a single group of atoms using a single temperature.  Eqn. 3 in Paper.
        
        Parameters
        ----------

        system : simt.openmm.System
            System to soften
        temperature : simtk.unit.Quanity [Kelvin]
            Desired softening temperature for atom group
        reference_temperature : simtk.unit.Quanity [Kelvin]
            Reference (or lowest) temperature.
        hot_atoms : numpy.ndarray-like, dtype='int'
            An array of atom indices to be softened to temperature
        soften_torsions : bool, optional, default=True
            If True, also soften the torsions.
        
        """
        beta = 1.0 / (kB * temperature)
        beta0 = 1.0 / (kB * reference_temperature)
        rho = beta / beta0
        
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                REST._soften_nb(force, hot_atoms, rho)
            if isinstance(force, mm.PeriodicTorsionForce) and soften_torsions:
                REST._soften_torsions(force, hot_atoms, rho)
    
    @staticmethod
    def _soften_nb(force, hot_atoms, rho):
        """Modify the NB forces for REST."""
        for atom in hot_atoms:
            q, sigma, epsilon = force.getParticleParameters(atom)
            epsilon = epsilon * rho
            q = q * (rho ** 0.5)
            force.setParticleParameters(atom, q, sigma, epsilon)
    
    @staticmethod
    def _soften_torsions(force, hot_atoms, rho):
        """Modify the torsion forces for REST."""
        for k in range(force.getNumTorsions()):
            i0, i1, i2, i3, period, phase, force_constant = force.getTorsionParameters(k)
            if (i0 in hot_atoms) and (i1 in hot_atoms) and (i2 in hot_atoms) and (i3 in hot_atoms):
                force_constant = force_constant * rho
                force.setTorsionParameters(k, i0, i1, i2, i3, period, phase, force_constant)
            
