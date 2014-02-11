import os
import collections

import numpy as np

import simtk.openmm as mm
import simtk.unit as units

from mdtraj.utils import ensure_type

from pkg_resources import resource_filename

import logging
logger = logging.getLogger(__name__)


kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant


def fix_coordinates(coordinates):
    if type(coordinates) in [type(list()), type(set())]:
        return [ units.Quantity(np.array(coordinate_set / coordinate_set.unit), coordinate_set.unit) for coordinate_set in coordinates ] 
    else:
        return [ units.Quantity(np.array(coordinates / coordinates.unit), coordinates.unit) ]            


def dict_to_named_tuple(options):
    named_tuple = collections.namedtuple("Parameters", options.keys())(**options)
    return named_tuple
    

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
    """Lists all subclasses (including cls) of cls."""
    return [cls] + cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


def find_matching_subclass(cls, name):
    """Look for a subclass (or the base class) of cls whose name matches name."""
    subclasses = all_subclasses(cls)
    for sub in subclasses:
        if sub.__name__ == name:
            return sub
    raise(TypeError("Cannot find matching subclass!"))

