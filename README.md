repex
=====

Replica Exchange

Replica-exchange simulation algorithms and specific variants.

DESCRIPTION

This package provides a general facility for running replica-exchange simulations, as well as
derived classes for special cases such as parallel tempering (in which the states differ only
in temperature) and Hamiltonian exchange (in which the state differ only by potential function).

This package also provides a number of utilities and tests that are required
for Yank and other OpenMM-based projects.

Provided classes include:

* ReplicaExchange - Base class for general replica-exchange simulations among specified ThermodynamicState objects
* ParallelTempering - Convenience subclass of ReplicaExchange for parallel tempering simulations (one System object, many temperatures/pressures)
* HamiltonianExchange - Convenience subclass of ReplicaExchange for Hamiltonian exchange simulations (many System objects, same temperature/pressure)

DEPENDENCIES

Use of this module requires the following

* Python 2.6 or later

  http://www.python.org

* OpenMM with Python wrappers

  http://simtk.org/home/openmm

* NetCDF (compiled with netcdf4 support) and HDF5 (on which 

  http://www.unidata.ucar.edu/software/netcdf/
  http://www.hdfgroup.org/HDF5/

* netcdf4-python (a Python interface for netcdf4)

  http://code.google.com/p/netcdf4-python/

* numpy and scipy

  http://www.scipy.org/

* mpi4py (if MPI support is desired)

http://mpi4py.scipy.org/

Note that this must be compiled against the appropriate MPI implementation.

NOTES

* MPI is now supported through mpi4py.

COPYRIGHT

Written by John D. Chodera <jchodera@gmail.com> while at the University of California Berkeley.

LICENSE

This code is licensed under the GNU Lesser General Public License.
