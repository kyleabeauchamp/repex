#!/usr/bin/env python

import time

import numpy as np

import simtk.openmm 
import simtk.unit as units

from mdtraj.utils import ensure_type

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

def generate_maxwell_boltzmann_velocities(system, temperature):
    """
    Generate Maxwell-Boltzmann velocities.

    ARGUMENTS

    system (simtk.openmm.System) - the system for which velocities are to be assigned
    temperature (simtk.unit.Quantity with units temperature) - the temperature at which velocities are to be assigned

    RETURN VALUES

    velocities (simtk.unit.Quantity wrapping numpy array of dimension natoms x 3 with units of distance/time) - drawn from the Maxwell-Boltzmann distribution at the appropriate temperature

    """

    # Get number of atoms
    natoms = system.getNumParticles()

    # Decorate System object with vector of masses for efficiency.
    if not hasattr(system, 'masses'):
        masses = simtk.unit.Quantity(np.zeros([natoms,3], np.float64), units.amu)
        for atom_index in range(natoms):
            mass = system.getParticleMass(atom_index) # atomic mass
            masses[atom_index,:] = mass
        setattr(system, 'masses', masses)

    # Retrieve masses.
    masses = getattr(system, 'masses')

    # Compute thermal energy and velocity scaling factors.
    kT = kB * temperature # thermal energy
    sigma2 = kT / masses

    # Assign velocities from the Maxwell-Boltzmann distribution.
    # TODO: This is wacky because units.sqrt cannot operate on np vectors.
    
    velocity_unit = units.nanometers / units.picoseconds
    velocities = units.Quantity(np.sqrt(sigma2 / (velocity_unit**2)) * np.random.randn(natoms, 3), velocity_unit)
    
    return velocities



def time_and_print(x):
    return x

def fix_coordinates(coordinates):
    if type(coordinates) in [type(list()), type(set())]:
        return [ units.Quantity(np.array(coordinate_set / coordinate_set.unit), coordinate_set.unit) for coordinate_set in coordinates ] 
    else:
        return [ units.Quantity(np.array(coordinates / coordinates.unit), coordinates.unit) ]            



default_options = {}
default_options["collision_rate"] = 91.0 / units.picosecond 
default_options["constraint_tolerance"] = 1.0e-6 
default_options["timestep"] = 2.0 * units.femtosecond
default_options["nsteps_per_iteration"] = 500
default_options["number_of_iterations"] = 10
default_options["equilibration_timestep"] = 1.0 * units.femtosecond
default_options["number_of_equilibration_iterations"] = 1
default_options["title"] = 'Replica-exchange simulation created using ReplicaExchange class of repex.py on %s' % time.asctime(time.localtime())        
default_options["minimize"] = True 
default_options["minimize_tolerance"] = 1.0 * units.kilojoules_per_mole / units.nanometers # if specified, set minimization tolerance
default_options["minimize_maxIterations"] = 0 # if nonzero, set maximum iterations
default_options["platform"] = None
default_options["replica_mixing_scheme"] = 'swap-all' # mix all replicas thoroughly
default_options["online_analysis"] = False # if True, analysis will occur each iteration
default_options["show_energies"] = True
default_options["show_mixing_statistics"] = True
default_options["platform"] = None
default_options["integrator"] = None


def process_kwargs(kwargs):
    options = {}

    for key in default_options:
        options[key] = kwargs.get(key, default_options[key])
    
    for key in kwargs.keys():
        if not options.has_key(key):
            options[key] = kwargs[key]

    return options


def permute_energies(X, s):
    """Re-order an observable X so that u[i, j, k] correponds to frame i, sampled from state j, evaluated in state k.

    Parameters
    ----------

    X : np.ndarray, shape=(n_iter, n_replicas, n_replicas)
        The observable to permute
    s : np.ndarray, shape=(n_iter, n_replicas), dtype='int'
        The thermodynamic state indices of each replica slot.  s[i, k] is the 
        thermodynamic state index of frame i, replica k.  
    """

    X = ensure_type(X, 'float32', 3, "X")
    n_iter, n_replicas, n_replicas = X.shape
    s = ensure_type(s, "int", 2, "s", shape=(n_iter, n_replicas))
    
    u = np.zeros((n_iter, n_replicas, n_replicas))
    for i, si in enumerate(s):
        mapping = dict(zip(range(n_replicas), si))
        inv_map = {v:k for k, v in mapping.items()}
        si_inv = [inv_map[k] for k in range(n_replicas)]
        u[i] = X[i, si_inv]
    
    return u
