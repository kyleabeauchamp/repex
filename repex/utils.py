#!/usr/bin/env python

import time

import numpy as np

import simtk.openmm 
import simtk.unit as units

from mdtraj.utils import ensure_type

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant


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
