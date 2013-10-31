import numpy as np
import simtk.unit as u
from simtk.openmm import app
import simtk.openmm as mm
from repex.thermodynamics import ThermodynamicState
from repex.serial_replica_exchange import ReplicaExchange
import repex.netcdf_io
import logging

logging.basicConfig(level=0)

nc_filename = "./out.nc"

n_replicas = 2 # number of temperature replicas
T_min = 298.0 * u.kelvin # minimum temperature
T_max = 300.0 * u.kelvin # maximum temperature

T_i = [ T_min + (T_max - T_min) * (np.exp(float(i) / float(n_replicas-1)) - 1.0) / (np.e - 1.0) for i in range(n_replicas) ]


pdb_filename = "./1vii.pdb"

temperature = 300 * u.kelvin
friction = 0.3 / u.picosecond
timestep = 2.0 * u.femtosecond

forcefield = app.ForceField("amber10.xml", "tip3p.xml")

pdb = app.PDBFile(pdb_filename)

model = app.modeller.Modeller(pdb.topology, pdb.positions)
model.addSolvent(forcefield, padding=0.01 * u.nanometer)

system = forcefield.createSystem(model.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * u.nanometers, constraints=app.HAngles)

states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(n_replicas) ]

coordinates = [model.getPositions()] * n_replicas

#database = repex.netcdf_io.NetCDFDatabase(nc_filename, states, coordinates)

#replica_exchange = ReplicaExchange(states, coordinates, nc_filename)
replica_exchange = ReplicaExchange.create_repex(states, coordinates, nc_filename, **{})
#replica_exchange = ReplicaExchange.resume_repex(nc_filename, **{})
#replica_exchange.number_of_iterations = 20
replica_exchange.run()
