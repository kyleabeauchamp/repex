##############################################################################
# Repex
#
# Copyright 2012-2013 MSKCC and the Authors
#
# Authors: Kyle A. Beauchamp, John D. Chodera
# Contributors:
#
# Repex is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with -yanktools. If not, see <http://www.gnu.org/licenses/>.
##############################################################################


"""Repex contains a collection of OpenMM tools that can be used
to perform replica exchange molecular dynamics.  Also, repex
contains a number of utilities required for Yank and other packages.
"""

from repex import testsystems
from repex.replica_exchange import resume
from repex.replica_exchange import ReplicaExchange
from repex.parallel_tempering import ParallelTempering
from repex.hamiltonian_exchange import HamiltonianExchange
from repex.netcdf_io import NetCDFDatabase


def _set_logging():
    """Set the logging based on comm.rank.
    
    Notes
    -----
    This function will be hidden from the namespace.
    """
    import logging
    try:
        from mpi4py import MPI # MPI wrapper
        mpicomm = MPI.COMM_WORLD
    except:
        import dummympi
        mpicomm = dummympi.DummyMPIComm()    
    if mpicomm.rank == 0:
        logging.basicConfig(level=logging.DEBUG)  # Change this to INFO for public release!
    else:  # By default, silence output from worker nodes
        logging.basicConfig(level=logging.ERROR)

_set_logging()
