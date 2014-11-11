#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test the MCMC facility.

DESCRIPTION


TODO

COPYRIGHT AND LICENSE

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import sys
import math
import doctest
import numpy
import time

import simtk.unit as units
import simtk.openmm as openmm
from simtk.openmm import app

from openmmtools import testsystems
from openmmtools import integrators 
from repex import mcmc

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
    
#=============================================================================================
# UTILITY SUBROUTINES
#=============================================================================================

def generateMaxwellBoltzmannVelocities(system, temperature):
   """Generate Maxwell-Boltzmann velocities.
   
   ARGUMENTS
   
   system (simtk.openmm.System) - the system for which velocities are to be assigned
   temperature (simtk.unit.Quantity of temperature) - the temperature at which velocities are to be assigned
   
   RETURNS
   
   velocities (simtk.unit.Quantity of numpy Nx3 array, units length/time) - particle velocities
   
   TODO

   This could be sped up by introducing vector operations.
   
   """
   
   # Get number of atoms
   natoms = system.getNumParticles()
   
   # Create storage for velocities.        
   velocities = units.Quantity(numpy.zeros([natoms, 3], numpy.float32), units.nanometer / units.picosecond) # velocities[i,k] is the kth component of the velocity of atom i
   
   # Compute thermal energy and inverse temperature from specified temperature.
   kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
   kT = kB * temperature # thermal energy
   beta = 1.0 / kT # inverse temperature
   
   # Assign velocities from the Maxwell-Boltzmann distribution.
   for atom_index in range(natoms):
      mass = system.getParticleMass(atom_index) # atomic mass
      sigma = units.sqrt(kT / mass) # standard deviation of velocity distribution for each coordinate for this atom
      for k in range(3):
         velocities[atom_index,k] = sigma * numpy.random.normal()

   # Return velocities
   return velocities

def computeHarmonicOscillatorExpectations(K, mass, temperature):
   """
   Compute mean and variance of potential and kinetic energies for harmonic oscillator.

   Numerical quadrature is used.

   ARGUMENTS

   K - spring constant
   mass - mass of particle
   temperature - temperature

   RETURNS

   values

   """

   values = dict()

   # Compute thermal energy and inverse temperature from specified temperature.
   kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
   kT = kB * temperature # thermal energy
   beta = 1.0 / kT # inverse temperature
   
   # Compute standard deviation along one dimension.
   sigma = 1.0 / units.sqrt(beta * K) 

   # Define limits of integration along r.
   r_min = 0.0 * units.nanometers # initial value for integration
   r_max = 10.0 * sigma      # maximum radius to integrate to

   # Compute mean and std dev of potential energy.
   V = lambda r : (K/2.0) * (r*units.nanometers)**2 / units.kilojoules_per_mole # potential in kJ/mol, where r in nm
   q = lambda r : 4.0 * math.pi * r**2 * math.exp(-beta * (K/2.0) * (r*units.nanometers)**2) # q(r), where r in nm
   (IqV2, dIqV2) = scipy.integrate.quad(lambda r : q(r) * V(r)**2, r_min / units.nanometers, r_max / units.nanometers)
   (IqV, dIqV)   = scipy.integrate.quad(lambda r : q(r) * V(r), r_min / units.nanometers, r_max / units.nanometers)
   (Iq, dIq)     = scipy.integrate.quad(lambda r : q(r), r_min / units.nanometers, r_max / units.nanometers)
   values['potential'] = dict()
   values['potential']['mean'] = (IqV / Iq) * units.kilojoules_per_mole
   values['potential']['stddev'] = (IqV2 / Iq) * units.kilojoules_per_mole   
   
   # Compute mean and std dev of kinetic energy.
   values['kinetic'] = dict()
   values['kinetic']['mean'] = (3./2.) * kT
   values['kinetic']['stddev'] = math.sqrt(3./2.) * kT

   return values
   
def statisticalInefficiency(A_n, B_n=None, fast=False, mintime=3):
  """
  Compute the (cross) statistical inefficiency of (two) timeseries.

  REQUIRED ARGUMENTS  
    A_n (numpy array) - A_n[n] is nth value of timeseries A.  Length is deduced from vector.

  OPTIONAL ARGUMENTS
    B_n (numpy array) - B_n[n] is nth value of timeseries B.  Length is deduced from vector.
       If supplied, the cross-correlation of timeseries A and B will be estimated instead of the
       autocorrelation of timeseries A.  
    fast (boolean) - if True, will use faster (but less accurate) method to estimate correlation
       time, described in Ref. [1] (default: False)
    mintime (int) - minimum amount of correlation function to compute (default: 3)
       The algorithm terminates after computing the correlation time out to mintime when the
       correlation function furst goes negative.  Note that this time may need to be increased
       if there is a strong initial negative peak in the correlation function.

  RETURNS
    g is the estimated statistical inefficiency (equal to 1 + 2 tau, where tau is the correlation time).
       We enforce g >= 1.0.

  NOTES 
    The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
    The fast method described in Ref [1] is used to compute g.

  REFERENCES  
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
    histogram analysis method for the analysis of simulated and parallel tempering simulations.
    JCTC 3(1):26-41, 2007.

  EXAMPLES

  Compute statistical inefficiency of timeseries data with known correlation time.  

  >>> import timeseries
  >>> A_n = timeseries.generateCorrelatedTimeseries(N=100000, tau=5.0)
  >>> g = statisticalInefficiency(A_n, fast=True)
  
  """

  # Create numpy copies of input arguments.
  A_n = numpy.array(A_n)
  if B_n is not None:  
    B_n = numpy.array(B_n)
  else:
    B_n = numpy.array(A_n) 
  
  # Get the length of the timeseries.
  N = A_n.size

  # Be sure A_n and B_n have the same dimensions.
  if(A_n.shape != B_n.shape):
    raise ParameterError('A_n and B_n must have same dimensions.')

  # Initialize statistical inefficiency estimate with uncorrelated value.
  g = 1.0
    
  # Compute mean of each timeseries.
  mu_A = A_n.mean()
  mu_B = B_n.mean()

  # Make temporary copies of fluctuation from mean.
  dA_n = A_n.astype(numpy.float64) - mu_A
  dB_n = B_n.astype(numpy.float64) - mu_B

  # Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
  sigma2_AB = (dA_n * dB_n).mean() # standard estimator to ensure C(0) = 1

  # Trap the case where this covariance is zero, and we cannot proceed.
  if(sigma2_AB == 0):
    raise ParameterException('Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency')

  # Accumulate the integrated correlation time by computing the normalized correlation time at
  # increasing values of t.  Stop accumulating if the correlation function goes negative, since
  # this is unlikely to occur unless the correlation function has decayed to the point where it
  # is dominated by noise and indistinguishable from zero.
  t = 1
  increment = 1
  while (t < N-1):

    # compute normalized fluctuation correlation function at time t
    C = sum( dA_n[0:(N-t)]*dB_n[t:N] + dB_n[0:(N-t)]*dA_n[t:N] ) / (2.0 * float(N-t) * sigma2_AB)
    
    # Terminate if the correlation function has crossed zero and we've computed the correlation
    # function at least out to 'mintime'.
    if (C <= 0.0) and (t > mintime):
      break
    
    # Accumulate contribution to the statistical inefficiency.
    g += 2.0 * C * (1.0 - float(t)/float(N)) * float(increment)

    # Increment t and the amount by which we increment t.
    t += increment

    # Increase the interval if "fast mode" is on.
    if fast: increment += 1

  # g must be at least unity
  if (g < 1.0): g = 1.0
   
  # Return the computed statistical inefficiency.
  return g

#=============================================================================================
# MAIN
#=============================================================================================

# Test integrator.
temperature = 298.0 * units.kelvin
pressure = 1.0 * units.atmospheres
kT = kB * temperature

nequil = 10 # number of equilibration iterations
niterations = 100 # number of production iterations

# Select system:
testsystem = testsystems.IdealGas()
#testsystem = testsystems.MolecularIdealGas()
#testsystem = testsystems.AlanineDipeptideVacuum(constraints=None)
#testsystem = testsystems.AlanineDipeptideVacuum(constraints=app.HBonds)
#testsystem = testsystems.AlanineDipeptideImplicit(constraints=app.HBonds)
#testsystem = testsystems.AlanineDipeptideExplicit(constraints=app.HBonds, rigid_water=True)
#testsystem = testsystems.Diatom(constraint=True, use_central_potential=True)
#testsystem = testsystems.ConstraintCoupledHarmonicOscillator()
#testsystem = testsystems.LysozymeImplicit(flexibleConstraints=False, shake=True)
#testsystem = testsystems.HarmonicOscillator()
#testsystem = testsystems.HarmonicOscillatorArray(N=16)
#testsystem = testsystems.WaterBox()

# Retrieve system and positions.
[system, positions] = [testsystem.system, testsystem.positions]

#velocities = generateMaxwellBoltzmannVelocities(system, temperature)
ndof = 3*system.getNumParticles() - system.getNumConstraints()

# Use default constraint tolerance.
#tolerance = integrator.getConstraintTolerance()

# Set constraint tolerance
#integrator.setConstraintTolerance(tolerance)

# Select platform manually.
platform_name = 'CUDA'
platform = openmm.Platform.getPlatformByName(platform_name)
options = dict()
#options = {"OpenCLPrecision": "double", "CudaPrecision": "double"}
#options = {"OpenCLPrecision": "single", "CudaPrecision": "single"}
options = {"OpenCLPrecision": "mixed", "CudaPrecision": "mixed"}
for option in options:
   platform.setPropertyDefaultValue(option, options[option])

# Create Context and set positions and velocities.
#context = openmm.Context(system, integrator, platform, options)
#context.setPositions(positions)
#context.applyConstraints(tolerance)
#print context.getPlatform().getName()

# Create MCMC move set.
from repex.mcmc import HMCMove, GHMCMove, LangevinDynamicsMove, MonteCarloBarostatMove
#move_set = [ GHMCMove(nsteps=10), HMCMove(nsteps=10) ]
move_set = [ GHMCMove(), MonteCarloBarostatMove() ]
#move_set = [ GHMCMove() ]
#move_set = [ LangevinDynamicsMove() ]

# Create thermodynamic state
from repex.thermodynamics import ThermodynamicState
thermodynamic_state = ThermodynamicState(system=testsystem.system, temperature=temperature, pressure=pressure)

# Create MCMC sampler.
from repex.mcmc import MCMCSampler
sampler = MCMCSampler(thermodynamic_state, move_set=move_set, platform=platform)

# Create sampler state.
from repex.mcmc import MCMCSamplerState
sampler_state = MCMCSamplerState(system=testsystem.system, positions=testsystem.positions)

# Equilibrate
for iteration in range(nequil):
    print "equilibration iteration %d / %d" % (iteration, nequil)

    # Update sampler state.
    sampler_state = sampler.run(sampler_state, 1)

# Accumulate statistics.
x_n = numpy.zeros([niterations], numpy.float64) # x_n[i] is the x position of atom 1 after iteration i, in angstroms
potential_n = numpy.zeros([niterations], numpy.float64) # potential_n[i] is the potential energy after iteration i, in kT
kinetic_n = numpy.zeros([niterations], numpy.float64) # kinetic_n[i] is the kinetic energy after iteration i, in kT
temperature_n = numpy.zeros([niterations], numpy.float64) # temperature_n[i] is the instantaneous kinetic temperature from iteration i, in K
for iteration in range(niterations):
    print "iteration %d / %d" % (iteration, niterations)

    # Update sampler state.
    sampler_state = sampler.run(sampler_state, 1)

    # Get statistics.
    potential_energy = sampler_state.potential_energy
    kinetic_energy = sampler_state.kinetic_energy
    total_energy = sampler_state.total_energy
    instantaneous_temperature = kinetic_energy * 2.0 / ndof / (units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA)
    volume = sampler_state.volume

    print "potential %8.1f kT | kinetic %8.1f kT | total %8.1f kT | volume %8.3f nm^3 | instantaneous temperature: %8.1f K" % (potential_energy/kT, kinetic_energy/kT, total_energy/kT, volume/(units.nanometers**3), instantaneous_temperature/units.kelvin)

    # Accumulate statistics.
    x_n[iteration] = sampler_state.positions[0,0] / units.angstroms
    potential_n[iteration] = potential_energy / kT
    kinetic_n[iteration] = kinetic_energy / kT
    temperature_n[iteration] = instantaneous_temperature / units.kelvin

# Compute expected statistics for harmonic oscillator.
K = 100.0 * units.kilocalories_per_mole / units.angstroms**2
beta = 1.0 / kT
x_mean_exact = 0.0 # mean, in angstroms
x_std_exact = 1.0 / units.sqrt(beta * K) / units.angstroms # std dev, in angstroms

# Analyze statistics.
g = 1.0
if potential_n.std() > 0.0:
   g = statisticalInefficiency(potential_n) 
Neff = niterations / g # number of effective samples

x_mean = x_n.mean()
dx_mean = x_n.std() / numpy.sqrt(Neff)
x_mean_error = x_mean - x_mean_exact

x_var = x_n.var()
dx_var = x_var * numpy.sqrt(2. / (Neff-1))

x_std = x_n.std()
dx_std = 0.5 * dx_var / x_std 
x_std_error = x_std - x_std_exact

temperature_mean = temperature_n.mean()
dtemperature_mean = temperature_n.std() / numpy.sqrt(Neff)
temperature_error = temperature_mean - temperature/units.kelvin
nsigma = abs(temperature_error) / dtemperature_mean
nsigma_cutoff = 6.0

# TODO: Rework ugly statistics calculation and add nsigma deviation information.

print "positions"
print "  mean     observed %10.5f +- %10.5f  expected %10.5f  error %10.5f +- %10.5f" % (x_mean, dx_mean, x_mean_exact, x_mean_error, dx_mean)
print "  std      observed %10.5f +- %10.5f  expected %10.5f  error %10.5f +- %10.5f" % (x_std, dx_std, x_std_exact, x_std_error, dx_std)

print "temperature"
if nsigma < nsigma_cutoff:
    print "  mean     observed %10.5f +- %10.5f  expected %10.5f  error %10.5f +- %10.5f (%.1f sigma)" % (temperature_mean, dtemperature_mean, temperature/units.kelvin, temperature_error, dtemperature_mean, nsigma)
else:
    print "  mean     observed %10.5f +- %10.5f  expected %10.5f  error %10.5f +- %10.5f (%.1f sigma) ***" % (temperature_mean, dtemperature_mean, temperature/units.kelvin, temperature_error, dtemperature_mean, nsigma)

print ""

# Report timings.
for move in move_set:
   print move.__class__.__name__
   move.report_timing()
   print ""
