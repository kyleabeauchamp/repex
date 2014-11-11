import numpy as np
import simtk.unit as unit
from repex.thermodynamics import ThermodynamicState
from repex import hamiltonian_exchange, replica_exchange, parallel_tempering
from openmmtools import testsystems
from repex import resume
import tempfile
from mdtraj.testing import eq, skipif
import nose

steps = 3
repeats = 10

def test_hrex_multiple_save_and_load():

    nc_filename = tempfile.mkdtemp() + "/out.nc"

    temperature = 1 * unit.kelvin

    powers = [2., 2., 2.]
    n_replicas = len(powers)

    oscillators = [testsystems.PowerOscillator(b=powers[i]) for i in range(n_replicas)]

    systems = [ho.system for ho in oscillators]
    positions = [ho.positions for ho in oscillators]

    state = ThermodynamicState(system=systems[0], temperature=temperature)
    
    parameters = {"number_of_iterations":steps}
    rex = hamiltonian_exchange.HamiltonianExchange.create(state, systems, positions, nc_filename, parameters=parameters)
    rex.run()

    for repeat in range(repeats):
        replica_states0 = rex.replica_states
        Nij_proposed0 = rex.Nij_proposed
        Nij_accepted0 = rex.Nij_accepted
        sampler_states0 = rex.sampler_states

        rex.extend(steps)

        rex = resume(nc_filename)
        replica_states1 = rex.replica_states
        Nij_proposed1 = rex.Nij_proposed
        Nij_accepted1 = rex.Nij_accepted
        sampler_states1 = rex.sampler_states
        
        eq(replica_states0, replica_states1)
        eq(Nij_proposed0, Nij_proposed1)
        eq(Nij_accepted0, Nij_accepted1)

        rex.run()


def test_repex_multiple_save_and_load():

    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_i = [T_min, T_min * 2.0, T_min * 2.0]
    n_replicas = len(T_i)

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions

    states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(n_replicas) ]

    coordinates = [positions] * n_replicas
    
    parameters = {"number_of_iterations":steps}
    rex = replica_exchange.ReplicaExchange.create(states, coordinates, nc_filename, parameters=parameters)
    rex.run()
    
    for repeat in range(repeats):
        replica_states0 = rex.replica_states
        Nij_proposed0 = rex.Nij_proposed
        Nij_accepted0 = rex.Nij_accepted
        sampler_states0 = rex.sampler_states

        rex.extend(steps)
        
        rex = resume(nc_filename)
        replica_states1 = rex.replica_states
        Nij_proposed1 = rex.Nij_proposed
        Nij_accepted1 = rex.Nij_accepted
        sampler_states1 = rex.sampler_states
        
        eq(replica_states0, replica_states1)
        eq(Nij_proposed0, Nij_proposed1)
        eq(Nij_accepted0, Nij_accepted1)
        
        rex.run()


def test_parallel_tempering_multiple_save_and_load():

    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_max = 2.0 * unit.kelvin
    n_temps = 3

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions

    coordinates = [positions] * n_temps
    
    parameters = {"number_of_iterations":steps}
    rex = parallel_tempering.ParallelTempering.create(system, coordinates, nc_filename, T_min=T_min, T_max=T_max, n_temps=n_temps, parameters=parameters)
    rex.run()
    
    for repeat in range(repeats):
        replica_states0 = rex.replica_states
        Nij_proposed0 = rex.Nij_proposed
        Nij_accepted0 = rex.Nij_accepted
        sampler_states0 = rex.sampler_states

        rex.extend(steps)

        rex = resume(nc_filename)
        replica_states1 = rex.replica_states
        Nij_proposed1 = rex.Nij_proposed
        Nij_accepted1 = rex.Nij_accepted
        sampler_states1 = rex.sampler_states
        
        eq(replica_states0, replica_states1)
        eq(Nij_proposed0, Nij_proposed1)
        eq(Nij_accepted0, Nij_accepted1)

        rex.run()
