import simtk.unit as u
import math
from simtk.openmm import app
import simtk.openmm as mm
from repex.thermodynamics import ThermodynamicState
from repex.replica_exchange import ReplicaExchange

nc_filename = "repex.nc"

n_replicas = 3 # number of temperature replicas
T_min = 298.0 * u.kelvin # minimum temperature
T_max = 600.0 * u.kelvin # maximum temperature

T_i = [ T_min + (T_max - T_min) * (math.exp(float(i) / float(n_replicas-1)) - 1.0) / (math.e - 1.0) for i in range(n_replicas) ]

pdb_filename = "1VII.pdb"

temperature = 300 * u.kelvin
friction = 0.3 / u.picosecond
timestep = 2.0 * u.femtosecond

forcefield = app.ForceField("amber10.xml", "tip3p.xml")

pdb = app.PDBFile(pdb_filename)

model = app.modeller.Modeller(pdb.topology, pdb.positions)
model.addSolvent(forcefield, padding=0.4 * u.nanometer)

system = forcefield.createSystem(model.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * u.nanometers, constraints=app.HAngles)

states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(n_replicas) ]

simulation = ReplicaExchange(states, [model.getPositions()] * n_replicas, nc_filename) # initialize the replica-exchange simulation
simulation.minimize = False
simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
simulation.timestep = 2.0 * u.femtoseconds # set the timestep for integration
simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
simulation.run() # run the simulation


database = repex.netdf_io.NetCDFDatabase(nc_filename)
simulation = ReplicaExchange(states, [model.getPositions()] * n_replicas, database) # initialize the replica-exchange simulation
