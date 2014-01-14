#!/usr/bin/env python

import time
import os

import numpy as np

import simtk.openmm as mm
import simtk.unit as units

from mdtraj.utils import ensure_type

from pkg_resources import resource_filename

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


def str_to_system(system_string):
    """Rebuild an OpenMM System from string representation."""
    system = mm.System() 
    system.__setstate__(system_string)
    return system


def get_data_filename(relative_path):
    """Get the full path to one of the reference files in testsystems.

    In the source distribution, these files are in ``repex/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the repex folder).

    """

    fn = resource_filename('repex', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn


def all_subclasses(cls):
    return [cls] + cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


def find_matching_subclass(cls, name):
    subclasses = all_subclasses(cls)
    for sub in subclasses:
        if sub.__name__ == name:
            return sub
    raise(TypeError("Cannot find matching subclass!"))
