import mdtraj as md
import numpy as np
import simtk.unit as unit
from repex.thermodynamics import ThermodynamicState
from repex.replica_exchange import ReplicaExchange
from repex import testsystems
from repex.utils import permute_energies
from repex import dummympi
from repex import resume
import tempfile
from mdtraj.testing import eq, skipif
import logging

logging.basicConfig(level=logging.DEBUG)

@skipif(True)
def test_get_traj():
    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 300.0 * unit.kelvin
    T_i = [T_min, T_min * 1.01, T_min * 1.1]
    n_replicas = len(T_i)

    ho = testsystems.AlanineDipeptideExplicit()

    system = ho.system
    positions = ho.positions

    states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(n_replicas) ]

    coordinates = [positions] * n_replicas

    mpicomm = dummympi.DummyMPIComm()
    parameters = {"number_of_iterations":3}
    replica_exchange = ReplicaExchange.create(states, coordinates, nc_filename, mpicomm=mpicomm, parameters=parameters)
    replica_exchange.run()

    db = replica_exchange.database

    trj0 = md.load("repex/data/alanine-dipeptide-explicit/alanine-dipeptide.pdb")
    db.set_traj(trj0)

    trj1 = db.get_traj(state_index=0)
    trj2 = db.get_traj(replica_index=0)


def test_check_energies():
    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_i = [T_min, T_min * 10., T_min * 100.]
    n_replicas = len(T_i)

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions

    states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(n_replicas) ]

    coordinates = [positions] * n_replicas

    mpicomm = dummympi.DummyMPIComm()
    parameters = {"number_of_iterations":10}
    replica_exchange = ReplicaExchange.create(states, coordinates, nc_filename, mpicomm=mpicomm, parameters=parameters)
    replica_exchange.run()

    db = replica_exchange.database
    db.check_energies()

def test_check_positions():
    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_i = [T_min, T_min * 10., T_min * 100.]
    n_replicas = len(T_i)

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions

    states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(n_replicas) ]

    coordinates = [positions] * n_replicas

    mpicomm = dummympi.DummyMPIComm()
    parameters = {"number_of_iterations":10}
    replica_exchange = ReplicaExchange.create(states, coordinates, nc_filename, mpicomm=mpicomm, parameters=parameters)
    replica_exchange.run()

    db = replica_exchange.database
    db.check_energies()


def test_check_mbar():
    nc_filename = tempfile.mkdtemp() + "/out.nc"

    T_min = 1.0 * unit.kelvin
    T_i = [T_min, T_min * 10., T_min * 100.]
    n_replicas = len(T_i)

    ho = testsystems.HarmonicOscillator()

    system = ho.system
    positions = ho.positions

    states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(n_replicas) ]

    coordinates = [positions] * n_replicas

    mpicomm = dummympi.DummyMPIComm()
    parameters = {"number_of_iterations":10}
    replica_exchange = ReplicaExchange.create(states, coordinates, nc_filename, mpicomm=mpicomm, parameters=parameters)
    replica_exchange.run()

    db = replica_exchange.database
    db.estimate_enthalpies()
    db.run_mbar()
