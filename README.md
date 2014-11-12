[![Build Status](https://travis-ci.org/choderalab/repex.png)](https://travis-ci.org/choderalab/repex)

# repex

Replica-exchange simulation algorithms for OpenMM.

## Description

This package provides a general facility for running replica-exchange simulations, as well as
derived classes for special cases such as parallel tempering (in which the states differ only
in temperature) and Hamiltonian exchange (in which the state differ only by potential function).

This package also provides a number of utilities and tests that are required for Yank and other OpenMM-based projects.

Simulations utilize a generic Markov chain Monte Carlo (MCMC) framework that makes it easy to mix Monte Carlo and molecular dynamics.

Provided classes include:

* `ReplicaExchange` - Base class for general replica-exchange simulations among specified ThermodynamicState objects
* `ParallelTempering` - Convenience subclass of ReplicaExchange for parallel tempering simulations (one System object, many temperatures/pressures)
* `HamiltonianExchange` - Convenience subclass of ReplicaExchange for Hamiltonian exchange simulations (many System objects, same temperature/pressure)

## Dependencies

Use of this module requires the following:

* Python 2.7 or later: 
  * http://www.python.org
* OpenMM with Python wrappers: 
  * http://openmm.org
* NetCDF (compiled with netcdf4 support) and HDF5 (on which NetCDF4 depends): 
  * http://www.unidata.ucar.edu/software/netcdf/
  * http://www.hdfgroup.org/HDF5/
* netcdf4-python (a Python interface for netcdf4)
  * http://code.google.com/p/netcdf4-python/
* numpy and scipy
  * http://www.scipy.org/
* mpi4py (if MPI support is desired)
  * http://mpi4py.scipy.org/
  * Note that mpi4py must be compiled against the appropriate installed MPI implementation.
* mdtraj (optional)
* pandas
* OpenMMTools (https://github.com/choderalab/openmmtools)

## Authors

* John D. Chodera <jchodera@gmail.com>
* Kyle A. Beauchamp <kyleabeauchamp@gmail.com>

## Copyright

Written by John D. Chodera <jchodera@gmail.com> and Kyle A. Beauchamp <kyleabeauchamp@gmail.com> wwhile at the University of California Berkeley and the Memorial Sloan-Kettering Cancer Center.

## License

This code is licensed under the GNU Lesser General Public License (LGPL).

