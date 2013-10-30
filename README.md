Repex
=====

Replica Exchange


Replica-exchange simulation algorithms and specific variants.

DESCRIPTION

This module provides a general facility for running replica-exchange simulations, as well as
derived classes for special cases such as parallel tempering (in which the states differ only
in temperature) and Hamiltonian exchange (in which the state differ only by potential function).

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

* PyOpenMM Python tools for OpenMM

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

TODO

* Add analysis facility accessible by user.
* Give up on Context caching and revert to serial Context creation/destruction if we run out of GPU memory (issuing an alert).
* Store replica self-energies and/or -ln q(x) for simulation (for analyzing correlation times).
* Add analysis facility.
* Allow user to call initialize() externally and get the NetCDF file handle to add additional data?
* Store / restore parameters and System objects from NetCDF file for resuming and later analysis.
* Sampling support:
  * Short-term: Add support for user to specify a callback to create the Integrator to use ('integrator_factory' or 'integrator_callback').
  * Longer-term: Allow a more complex MCMC sampling scheme consisting of one or more moves to be specified through mcmc.py facility.
* Allow different choices of temperature handling during exchange attempts: 
  * scale velocities (exchanging only on potential energies) - make this the default?
  * randomize velocities (exchanging only on potential energies)
  * exchange on total energies, preserving velocities (requires more replicas)
* Add control over number of times swaps are attempted when mixing replicas, or compute best guess automatically
* Add support for online analysis (potentially by separate threads or while GPU is running?)
* Add another layer of abstraction so that the base class uses generic log probabilities, rather than reduced potentials?
* Add support for HDF5 storage models.
* Support for other NetCDF interfaces via autodetection/fallback:
  * scipy.io.netcdf 
* Use interface-based checking of arguments so that different implementations of the OpenMM API (such as pyopenmm) can be used.
* Eliminate file closures in favor of syncs to avoid closing temporary files in the middle of a run.

COPYRIGHT

Written by John D. Chodera <jchodera@gmail.com> while at the University of California Berkeley.

LICENSE

This code is licensed under the latest available version of the GNU General Public License.
