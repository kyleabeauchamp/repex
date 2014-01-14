import numpy as np
import simtk.unit as unit
from repex.thermodynamics import ThermodynamicState
from repex import hamiltonian_exchange, replica_exchange, parallel_tempering
from repex import testsystems
from repex import resume
import tempfile
from mdtraj.testing import eq, skipif
import nose

test_mpi = True

try:
    from repex.mpinoseutils import mpitest    
except:
    test_mpi = False

import distutils.spawn
mpiexec = distutils.spawn.find_executable("mpiexec")

if mpiexec is None:
    test_mpi = False

def setup():
    if test_mpi == False:
        raise nose.SkipTest('No MPI detected; skipping MPI tests.')

@mpitest(2)
def test_hrex_save_and_load(mpicomm):

    nc_filename = tempfile.mkdtemp() + "/out.nc"

    temperature = 1 * unit.kelvin

    powers = [2., 2., 4.]
    n_replicas = len(powers)

    oscillators = [testsystems.PowerOscillator(b=powers[i]) for i in range(n_replicas)]

    systems = [ho.system for ho in oscillators]
    positions = [ho.positions for ho in oscillators]

    state = ThermodynamicState(system=systems[0], temperature=temperature)

    rex = hamiltonian_exchange.HamiltonianExchange.create_repex(state, systems, positions, nc_filename, mpicomm=mpicomm, **{})
    rex.number_of_iterations = 5
    rex.run()

    
    rex = resume(nc_filename, mpicomm=mpicomm)
    rex.number_of_iterations = 10
    rex.run()

    eq(rex.__class__.__name__, "HamiltonianExchange")

@mpitest(2)
def test_repex_save_and_load(mpicomm):

    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_i = [T_min, T_min * 10., T_min * 100.]
    n_replicas = len(T_i)

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions

    states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(n_replicas) ]

    coordinates = [positions] * n_replicas

    rex = replica_exchange.ReplicaExchange.create_repex(states, coordinates, nc_filename, mpicomm=mpicomm, **{})
    rex.number_of_iterations = 5
    rex.run()
    
    rex = resume(nc_filename, mpicomm=mpicomm)
    rex.number_of_iterations = 10
    rex.run()

    eq(rex.__class__.__name__, "ReplicaExchange")

@mpitest(2)
def test_parallel_tempering_save_and_load(mpicomm):

    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_max = 10.0 * unit.kelvin
    n_temps = 3

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions


    coordinates = [positions] * n_temps
    
    rex = parallel_tempering.ParallelTempering.create_repex(system, coordinates, nc_filename, T_min=T_min, T_max=T_max, n_temps=n_temps, mpicomm=mpicomm, **{})
    rex.number_of_iterations = 5
    rex.run()
    
    rex = resume(nc_filename, mpicomm=mpicomm)
    rex.number_of_iterations = 10
    rex.run()
    
    eq(rex.__class__.__name__, "ParallelTempering")


