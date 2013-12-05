#!/usr/local/bin/env python

"""
Test barostats to ensure correct ideal gas properties are reproduced.

DESCRIPTION

We check that the average density comes out to be within statistical error of the
values reported in Ref [1], Table 1.

REFERENCES

[1] Shirts MR, Mobley DL, Chodera JD, and Pande VS. Accurate and efficient corrections for
missing dispersion interactions in molecular simulations. JPC B 111:13052, 2007.

[2] Ahn S and Fessler JA. Standard errors of mean, variance, and standard deviation estimators.
Technical Report, EECS Department, The University of Michigan, 2003.

TODO

* Add failure conditions for kinetic temperature.
* Test all Platforms.

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

import os
import os.path
import sys
import math

import numpy

import simtk
import simtk.unit as units
import simtk.openmm as openmm

from test_systems import IdealGas, ThermodynamicState

#=============================================================================================
# Set parameters
#=============================================================================================

NSIGMA_CUTOFF = 6.0 # maximum number of standard deviations away from true value before test fails

# Select platform to test.
platform = openmm.Platform.getPlatformByName('CPU')

# Select run parameters
timestep = 2.0 * units.femtosecond # timestep for integrtion
nsteps = 1 # number of steps per data record
nequiliterations = 1000 # number of equilibration iterations
niterations = 10000 # number of iterations to collect data for

# Set temperature, pressure, and collision rate for stochastic thermostats.
temperature = 298.0 * units.kelvin
pressure = 1.0 * units.atmospheres 
collision_frequency = 91.0 / units.picosecond 
barostat_frequency = 1 # number of steps between MC volume adjustments
nprint = 100
nparticles = 216

# Flag to set verbose debug output
verbose = True

#=============================================================================================
# SUBROUTINES
#=============================================================================================
   
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

def compute_volume(box_vectors):
   """
   Return the volume of the current configuration.
   
   RETURNS
   
   volume (simtk.unit.Quantity) - the volume of the system (in units of length^3), or None if no box positions are defined
   
   """

   # Compute volume of parallelepiped.
   [a,b,c] = box_vectors
   A = numpy.array([a/a.unit, b/a.unit, c/a.unit])
   volume = numpy.linalg.det(A) * a.unit**3
   return volume

def compute_mass(system):
   """
   Returns the total mass of the system in amu.

   RETURNS

   mass (simtk.unit.Quantity) - the mass of the system (in units of amu)

   """

   mass = 0.0 * units.amu
   for i in range(system.getNumParticles()):
      mass += system.getParticleMass(i)
   return mass

#=============================================================================================
# Test thermostats
#=============================================================================================

# Create the test system.
testsystem = IdealGas(nparticles=nparticles, temperature=temperature, pressure=pressure)
[system, positions] = [testsystem.system, testsystem.positions]

# DEBUG
box_vectors = system.getDefaultPeriodicBoxVectors()
print "system volume = %.1f nm^3" % (compute_volume(box_vectors) / units.nanometers**3)

# Determine number of degrees of freedom.
kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
ndof = 3*system.getNumParticles() - system.getNumConstraints()

# Compute total mass.
mass = compute_mass(system).in_units_of(units.gram / units.mole) / units.AVOGADRO_CONSTANT_NA # total system mass in g

# Add Monte Carlo barostat.
barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_frequency)
system.addForce(barostat)
    
# Create integrator.
integrator = openmm.LangevinIntegrator(temperature, collision_frequency, timestep)        

# Create context.
context = openmm.Context(system, integrator, platform)

# Set initial positions.
context.setPositions(positions)

# Set velocities.
context.setVelocitiesToTemperature(temperature)

# Initialize statistics.
data = dict()
data['time'] = units.Quantity(numpy.zeros([niterations], numpy.float64), units.picoseconds)
data['potential'] = units.Quantity(numpy.zeros([niterations], numpy.float64), units.kilocalories_per_mole)
data['kinetic'] = units.Quantity(numpy.zeros([niterations], numpy.float64), units.kilocalories_per_mole)
data['volume'] = units.Quantity(numpy.zeros([niterations], numpy.float64), units.angstroms**3)
data['density'] = units.Quantity(numpy.zeros([niterations], numpy.float64), units.gram / units.centimeters**3)
data['kinetic_temperature'] = units.Quantity(numpy.zeros([niterations], numpy.float64), units.kelvin)

# Equilibrate.
if verbose: print "Equilibrating..."
for iteration in range(nequiliterations):
   # Integrate.
   integrator.step(nsteps)
   
   # Compute properties.
   state = context.getState(getEnergy=True)
   kinetic = state.getKineticEnergy()
   potential = state.getPotentialEnergy()
   box_vectors = state.getPeriodicBoxVectors()
   volume = compute_volume(box_vectors)
   density = (mass / volume).in_units_of(units.gram / units.centimeter**3)
   kinetic_temperature = 2.0 * kinetic / kB / ndof # (1/2) ndof * kB * T = KE
   if verbose and (iteration%nprint)==0:
      print "%6d %9.3f %16.3f %16.3f %16.1f %10.6f" % (iteration, state.getTime() / units.picoseconds, kinetic_temperature / units.kelvin, potential / units.kilocalories_per_mole, volume / units.nanometers**3, density / (units.gram / units.centimeter**3))

# Collect production data.
if verbose: print "Production..."
for iteration in range(niterations):
   # Propagate dynamics.
   integrator.step(nsteps)
   
   # Compute properties.
   state = context.getState(getEnergy=True)
   kinetic = state.getKineticEnergy()
   potential = state.getPotentialEnergy()
   box_vectors = state.getPeriodicBoxVectors()
   volume = compute_volume(box_vectors)
   density = (mass / volume).in_units_of(units.gram / units.centimeter**3)
   kinetic_temperature = 2.0 * kinetic / kB / ndof
   if verbose and (iteration%nprint)==0:
      print "%6d %9.3f %16.3f %16.3f %16.1f %10.6f" % (iteration, state.getTime() / units.picoseconds, kinetic_temperature / units.kelvin, potential / units.kilocalories_per_mole, volume / units.nanometers**3, density / (units.gram / units.centimeter**3))
      
   # Store properties.
   data['time'][iteration] = state.getTime() 
   data['potential'][iteration] = potential 
   data['kinetic'][iteration] = kinetic
   data['volume'][iteration] = volume
   data['density'][iteration] = density
   data['kinetic_temperature'][iteration] = kinetic_temperature
   
#=============================================================================================
# Compute statistical inefficiencies to determine effective number of uncorrelated samples.
#=============================================================================================

#data['g_potential'] = statisticalInefficiency(data['potential'] / units.kilocalories_per_mole)
data['g_potential'] = 1.0 # potential is always zero
data['g_kinetic'] = statisticalInefficiency(data['kinetic'] / units.kilocalories_per_mole)
data['g_volume'] = statisticalInefficiency(data['volume'] / units.angstroms**3)
data['g_density'] = statisticalInefficiency(data['density'] / (units.gram / units.centimeter**3))
data['g_kinetic_temperature'] = statisticalInefficiency(data['kinetic_temperature'] / units.kelvin)

#=============================================================================================
# Compute expectations and uncertainties.
#=============================================================================================

statistics = dict()

# Kinetic energy.
statistics['KE']  = (data['kinetic'] / units.kilocalories_per_mole).mean() * units.kilocalories_per_mole
statistics['dKE'] = (data['kinetic'] / units.kilocalories_per_mole).std() / numpy.sqrt(niterations / data['g_kinetic']) * units.kilocalories_per_mole
statistics['g_KE'] = data['g_kinetic'] * nsteps * timestep 

# Density
unit = (units.gram / units.centimeter**3)
statistics['density']  = (data['density'] / unit).mean() * unit
statistics['ddensity'] = (data['density'] / unit).std() / numpy.sqrt(niterations / data['g_density']) * unit
statistics['g_density'] = data['g_density'] * nsteps * timestep

# Volume
unit = units.nanometer**3
statistics['volume']  = (data['volume'] / unit).mean() * unit
statistics['dvolume'] = (data['volume'] / unit).std() / numpy.sqrt(niterations / data['g_volume']) * unit
statistics['g_volume'] = data['g_volume'] * nsteps * timestep

statistics['std_volume']  = (data['volume'] / unit).std() * unit
statistics['dstd_volume'] = (data['volume'] / unit).std() / numpy.sqrt((niterations / data['g_volume'] - 1) * 2.0) * unit # uncertainty expression from Ref [1].

# Kinetic temperature
unit = units.kelvin
statistics['kinetic_temperature']  = (data['kinetic_temperature'] / unit).mean() * unit
statistics['dkinetic_temperature'] = (data['kinetic_temperature'] / unit).std() / numpy.sqrt(niterations / data['g_kinetic_temperature']) * unit
statistics['g_kinetic_temperature'] = data['g_kinetic_temperature'] * nsteps * timestep

#=============================================================================================
# Print summary statistics
#=============================================================================================

test_pass = True # this gets set to False if test fails

print "Summary statistics (%.3f ns equil, %.3f ns production)" % (nequiliterations * nsteps * timestep / units.nanoseconds, niterations * nsteps * timestep / units.nanoseconds)

print ""

# Kinetic energies
print "average kinetic energy:"
print "%12.3f +- %12.3f  kcal/mol  (g = %12.3f ps)" % (statistics['KE'] / units.kilocalories_per_mole, statistics['dKE'] / units.kilocalories_per_mole, statistics['g_KE'] / units.picoseconds)
      
# Kinetic temperature
print "average kinetic temperature:"
unit = units.kelvin
print "%12.3f +- %12.3f  K         (g = %12.3f ps)" % (statistics['kinetic_temperature'] / unit, statistics['dkinetic_temperature'] / unit, statistics['g_kinetic_temperature'] / units.picoseconds)
      
# Volume
analytical = dict()
#analytical['volume'] = (nparticles+1.0) * (temperature * units.BOLTZMANN_CONSTANT_kB / pressure).in_units_of(units.nanometers**3) 
#analytical['std_volume'] = math.sqrt(nparticles+1.0) * (temperature * units.BOLTZMANN_CONSTANT_kB / pressure).in_units_of(units.nanometers**3) 
analytical['volume'] = testsystem.get_volume_expectation(ThermodynamicState(temperature=temperature, pressure=pressure))
analytical['std_volume'] = testsystem.get_volume_standard_deviation(ThermodynamicState(temperature=temperature, pressure=pressure))
statistics['volume-nsigma'] = abs(statistics['volume'] - analytical['volume']) / statistics['dvolume']
statistics['std_volume-nsigma'] = abs(statistics['std_volume'] - analytical['std_volume']) / statistics['dstd_volume']

print "volume:"
unit = (units.nanometer**3)
print "g = %12.3f ps" % (statistics['g_volume'] / units.picoseconds)
print "mean %12.1f +- %12.1f  nm^3 (analytical %12.1f nm^3)" % (statistics['volume'] / unit, statistics['dvolume'] / unit, analytical['volume'] / unit),
print '  %5.1f sigma' % statistics['volume-nsigma'],
if (statistics['volume-nsigma'] > NSIGMA_CUTOFF):
   print ' ***',
   test_pass = False
print ''

print "std  %12.1f +- %12.1f  nm^3 (analytical %12.1f nm^3)" % (statistics['std_volume'] / unit, statistics['dstd_volume'] / unit, analytical['std_volume'] / unit),
print '  %5.1f sigma' % statistics['std_volume-nsigma'],
if (statistics['std_volume-nsigma'] > NSIGMA_CUTOFF):
   print ' ***',
   test_pass = False
print ''

#=============================================================================================
# Report pass or fail in exit code
#=============================================================================================

if test_pass:
   sys.exit(0)
else:
   sys.exit(1)

   
