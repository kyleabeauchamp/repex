import numpy as np
import simtk.unit as unit
from repex.thermodynamics import ThermodynamicState
from repex.parallel_tempering import ParallelTempering
from repex import testsystems
from repex.utils import permute_energies
import tempfile
from mdtraj.testing import eq


def test_parallel_tempering():
    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_max = 10.0 * unit.kelvin
    n_temps = 3

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions


    coordinates = [positions] * n_temps

    replica_exchange = ParallelTempering.create_repex(system, coordinates, nc_filename, T_min=T_min, T_max=T_max, n_temps=n_temps, **{})
    
    eq(replica_exchange.n_replicas, n_temps)

    replica_exchange.number_of_iterations = 1000
    replica_exchange.run()

    u_permuted = replica_exchange.database.ncfile.variables["energies"][:]
    s = replica_exchange.database.ncfile.variables["states"][:]

    u = permute_energies(u_permuted, s)

    states = replica_exchange.states
    u0 = np.array([[ho.get_reduced_potential_expectation(s0, s1) for s1 in states] for s0 in states])

    l0 = np.log(u0)  # Compare on log scale because uncertainties are proportional to values
    l1 = np.log(u.mean(0))
    eq(l0, l1, decimal=1)


def test_parallel_tempering_save_and_load():

    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_max = 10.0 * unit.kelvin
    n_temps = 3

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions


    coordinates = [positions] * n_temps

    replica_exchange = ParallelTempering.create_repex(system, coordinates, nc_filename, T_min=T_min, T_max=T_max, n_temps=n_temps, **{})
    replica_exchange.number_of_iterations = 200
    replica_exchange.run()
    
    replica_exchange = ParallelTempering.resume_repex(nc_filename)
    eq(replica_exchange.iteration, 200)
    replica_exchange.number_of_iterations = 300
    replica_exchange.run()


def test_parallel_tempering_explicit_temperature_input():

    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T0 = 1.0 * unit.kelvin
    temperatures = [T0, T0 * 2, T0 * 4]
    n_temps = len(temperatures)

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions

    coordinates = [positions] * n_temps

    replica_exchange = ParallelTempering.create_repex(system, coordinates, nc_filename, temperatures=temperatures, **{})
    replica_exchange.number_of_iterations = 100
    replica_exchange.run()
    
    eq(replica_exchange.n_replicas, n_temps)
