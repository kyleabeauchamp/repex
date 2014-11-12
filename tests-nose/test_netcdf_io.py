import numpy as np
import simtk.unit as unit
from repex.thermodynamics import ThermodynamicState
from repex.parallel_tempering import ParallelTempering
from openmmtools import testsystems
from repex.utils import permute_energies
from repex import dummympi
from repex import resume
import tempfile
from mdtraj.testing import eq, raises
from nose.tools import assert_raises


@raises(IOError)
def test_double_set_thermodynamic_states():
    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_max = 10.0 * unit.kelvin
    n_temps = 3

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions


    coordinates = [positions] * n_temps

    mpicomm = dummympi.DummyMPIComm()
    parameters = {"number_of_iterations":3}
    replica_exchange = ParallelTempering.create(system, coordinates, nc_filename, T_min=T_min, T_max=T_max, n_temps=n_temps, mpicomm=mpicomm, parameters=parameters)
    
    eq(replica_exchange.n_replicas, n_temps)

    replica_exchange.run()

    states = replica_exchange.thermodynamic_states
    replica_exchange.database.thermodynamic_states = states
