#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test a variety of custom integrators.

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

import test_systems as testsystems

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# INTEGRATORS
#=============================================================================================

def VelocityVerletIntegrator(timestep):
    """
    Construct a velocity Verlet integrator.

    ARGUMENTS

    timestep (numpy.unit.Quantity compatible with femtoseconds) - the integration timestep

    RETURNS

    integrator (simtk.openmm.CustomIntegrator) - a velocity Verlet integrator

    NOTES

    This code is verbatim from Peter Eastman's example.

    """
    
    integrator = openmm.CustomIntegrator(timestep)

    integrator.addPerDofVariable("x1", 0)

    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
    integrator.addConstrainVelocities()
    
    return integrator

def AndersenVelocityVerletIntegrator(timestep, friction, temperature):
    """
    Construct a velocity Verlet integrator.

    ARGUMENTS

    timestep (numpy.unit.Quantity compatible with femtoseconds) - the integration timestep

    RETURNS

    integrator (simtk.openmm.CustomIntegrator) - a velocity Verlet integrator

    NOTES

    This code is verbatim from Peter Eastman's example.

    """
    
    integrator = openmm.CustomIntegrator(timestep)

    #
    # Integrator setup.
    #
    kT = kB * temperature
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addPerDofVariable("sigma_v", 0) # velocity distribution stddev for Maxwell-Boltzmann (set later)
    integrator.addPerDofVariable("x1", 0) # for constraints

    #
    # Update velocities from Maxwell-Boltzmann distribution.
    #
    integrator.addComputePerDof("sigma_v", "sqrt(kT/m)")
    integrator.addComputePerDof("v", "sigma_v*gaussian")

    #
    # Velocity Verlet
    #
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
    integrator.addConstrainVelocities()

    return integrator

def MetropolisMonteCarloIntegrator(timestep, temperature=298.0*units.kelvin, sigma=0.01*units.angstroms):
    """
    Create a simple Metropolis Monte Carlo integrator that uses Gaussian displacement trials.

    ARGUMENTS

    timestep (numpy.unit.Quantity compatible with femtoseconds) - the integration timestep
    temperature (numpy.unit.Quantity compatible with kelvin) - the temperature
    sigma (numpy.unit.Quantity compatible with nanometers) - the displacement standard deviation for each degree of freedom

    RETURNS

    integrator (simtk.openmm.CustomIntegrator) - a Metropolis Monte Carlo integrator

    WARNING

    This integrator does not respect constraints.

    NOTES

    Velocities are drawn from a Maxwell-Boltzmann distribution each timestep to generate correct (x,v) statistics.
    Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and accepted, respectively.
    
    """
    
    integrator = openmm.CustomIntegrator(timestep)

    kT = kB * temperature
    
    integrator.addGlobalVariable("naccept", 0) # number accepted
    integrator.addGlobalVariable("ntrials", 0) # number of Metropolization trials

    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addPerDofVariable("sigma_x", sigma) # perturbation size
    integrator.addPerDofVariable("sigma_v", 0) # velocity distribution stddev for Maxwell-Boltzmann (set later)
    integrator.addPerDofVariable("xold", 0) # old positions
    integrator.addGlobalVariable("Eold", 0) # old energy
    integrator.addGlobalVariable("Enew", 0) # new energy
    integrator.addGlobalVariable("accept", 0) # accept or reject

    #
    # Context state update.
    #
    integrator.addUpdateContextState();    

    #
    # Update velocities from Maxwell-Boltzmann distribution.
    #
    integrator.addComputePerDof("sigma_v", "sqrt(kT/m)")
    integrator.addComputePerDof("v", "sigma_v*gaussian")
    integrator.addConstrainVelocities();

    #
    # propagation steps
    #
    # Store old positions and energy.
    integrator.addComputePerDof("xold", "x")
    integrator.addComputeGlobal("Eold", "energy")
    # Gaussian particle displacements.
    integrator.addComputePerDof("x", "x + sigma_x*gaussian")
    # Accept or reject with Metropolis criteria.
    integrator.addComputeGlobal("accept", "step(exp(-(energy-Eold)/kT) - uniform)")
    integrator.addComputePerDof("x", "(1-accept)*xold + x*accept")
    # Accumulate acceptance statistics.
    integrator.addComputeGlobal("naccept", "naccept + accept")
    integrator.addComputeGlobal("ntrials", "ntrials + 1")   
    

    return integrator

def HMCIntegrator(timestep, temperature=298.0*units.kelvin, nsteps=10):
    """
    Create a hybrid Monte Carlo (HMC) integrator.

    ARGUMENTS

    timestep (numpy.unit.Quantity compatible with femtoseconds) - the integration timestep
    temperature (numpy.unit.Quantity compatible with kelvin) - the temperature
    nsteps (int) - the number of velocity Verlet steps to take per HMC trial

    RETURNS

    integrator (simtk.openmm.CustomIntegrator) - a hybrid Monte Carlo integrator

    WARNING

    Because 'nsteps' sets the number of steps taken, a call to integrator.step(1) actually takes 'nsteps' steps.

    NOTES
    
    The velocity is drawn from a Maxwell-Boltzmann distribution, then 'nsteps' steps are taken,
    and the new configuration is either accepted or rejected.

    Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
    accepted, respectively.

    TODO

    Currently, the simulation timestep is only advanced by 'timestep' each step, rather than timestep*nsteps.  Fix this.

    """

    kT = kB * temperature
        
    integrator = openmm.CustomIntegrator(timestep)

    integrator.addGlobalVariable("naccept", 0) # number accepted
    integrator.addGlobalVariable("ntrials", 0) # number of Metropolization trials

    integrator.addGlobalVariable("kT", kB*temperature) # thermal energy
    integrator.addPerDofVariable("sigma", 0) 
    integrator.addGlobalVariable("ke", 0) # kinetic energy
    integrator.addPerDofVariable("xold", 0) # old positions
    integrator.addGlobalVariable("Eold", 0) # old energy
    integrator.addGlobalVariable("Enew", 0) # new energy
    integrator.addGlobalVariable("accept", 0) # accept or reject
    integrator.addPerDofVariable("x1", 0) # for constraints

    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    #
    # Allow Context updating here.
    #
    integrator.addUpdateContextState(); 

    # 
    # Draw new velocity.
    #
    integrator.addComputePerDof("v", "sigma*gaussian")
    integrator.addConstrainVelocities();

    #
    # Store old position and energy.
    #
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Eold", "ke + energy")
    integrator.addComputePerDof("xold", "x")

    #
    # Inner symplectic steps using velocity Verlet.
    #
    for step in range(nsteps):
        integrator.addUpdateContextState()
        integrator.addComputePerDof("v", "v+0.5*dt*f/m")
        integrator.addComputePerDof("x", "x+dt*v")
        integrator.addComputePerDof("x1", "x")
        integrator.addConstrainPositions()
        integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        integrator.addConstrainVelocities()

    #
    # Accept/reject step.
    #
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Enew", "ke + energy")
    integrator.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
    integrator.addComputePerDof("x", "x*accept + xold*(1-accept)")

    #
    # Accumulate statistics.
    #
    integrator.addComputeGlobal("naccept", "naccept + accept")
    integrator.addComputeGlobal("ntrials", "ntrials + 1")   

    return integrator

def GHMCIntegrator(timestep, temperature=298.0*units.kelvin, gamma=50.0/units.picoseconds):
    """
    Create a generalized hybrid Monte Carlo (GHMC) integrator.
    
    ARGUMENTS

    timestep (numpy.unit.Quantity compatible with femtoseconds) - the integration timestep
    temperature (numpy.unit.Quantity compatible with kelvin) - the temperature
    gamma (numpy.unit.Quantity compatible with 1/picoseconds) - the collision rate

    RETURNS

    integrator (simtk.openmm.CustomIntegrator) - a GHMC integrator

    NOTES
    
    This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
    Metrpolization step to ensure sampling from the appropriate distribution.

    Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
    accepted, respectively.

    TODO

    Move initialization of 'sigma' to setting the per-particle variables.

    """

    kT = kB * temperature
        
    integrator = openmm.CustomIntegrator(timestep)

    integrator.addGlobalVariable("kT", kB*temperature) # thermal energy
    integrator.addGlobalVariable("b", numpy.exp(-gamma*timestep)) # velocity mixing parameter
    integrator.addPerDofVariable("sigma", 0) 
    integrator.addGlobalVariable("ke", 0) # kinetic energy
    integrator.addPerDofVariable("vold", 0) # old velocities
    integrator.addPerDofVariable("xold", 0) # old positions
    integrator.addGlobalVariable("Eold", 0) # old energy
    integrator.addGlobalVariable("Enew", 0) # new energy
    integrator.addGlobalVariable("accept", 0) # accept or reject
    integrator.addGlobalVariable("naccept", 0) # number accepted
    integrator.addGlobalVariable("ntrials", 0) # number of Metropolization trials
    integrator.addPerDofVariable("x1", 0) # position before application of constraints
    
    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    #
    # Allow context updating here.
    #
    integrator.addUpdateContextState();

    # 
    # Velocity perturbation.
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();
    
    #
    # Metropolized symplectic step.
    #
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Eold", "ke + energy")
    integrator.addComputePerDof("xold", "x")
    integrator.addComputePerDof("vold", "v")
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    integrator.addComputePerDof("x", "x + v*dt")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions();
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
    integrator.addConstrainVelocities();
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Enew", "ke + energy")
    integrator.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
    integrator.addComputePerDof("x", "x*accept + xold*(1-accept)")
    integrator.addComputePerDof("v", "v*accept - vold*(1-accept)")

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();

    #
    # Accumulate statistics.
    #
    integrator.addComputeGlobal("naccept", "naccept + accept")
    integrator.addComputeGlobal("ntrials", "ntrials + 1")   

    return integrator

def VVVRIntegrator(timestep, temperature=298.0*units.kelvin, gamma=50.0/units.picoseconds):
    """
    Create a velocity verlet with velocity randomization (VVVR) integrator.
    
    ARGUMENTS

    timestep (numpy.unit.Quantity compatible with femtoseconds) - the integration timestep
    temperature (numpy.unit.Quantity compatible with kelvin) - the temperature
    gamma (numpy.unit.Quantity compatible with 1/picoseconds) - the collision rate

    RETURNS

    integrator (simtk.openmm.CustomIntegrator) - a VVVR integrator

    NOTES
    
    This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
    timestep correction to ensure that the field-free diffusion constant is timestep invariant.

    The global 'pseudowork' keeps track of the pseudowork accumulated during integration, and can be
    used to correct the sampled statistics or in a Metropolization scheme.
    
    TODO

    Move initialization of 'sigma' to setting the per-particle variables.
    We can ditch pseudowork and instead use total energy difference - heat.
    
    """

    kT = kB * temperature
    
    integrator = openmm.CustomIntegrator(timestep)
    
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("b", numpy.exp(-gamma*timestep)) # velocity mixing parameter
    integrator.addPerDofVariable("sigma", 0) 
    integrator.addGlobalVariable("ke_old", 0) # kinetic energy
    integrator.addGlobalVariable("ke_new", 0) # kinetic energy
    integrator.addGlobalVariable("ke", 0) # kinetic energy
    integrator.addGlobalVariable("Eold", 0) # old energy
    integrator.addGlobalVariable("Enew", 0) # new energy
    integrator.addGlobalVariable("accept", 0) # accept or reject
    integrator.addGlobalVariable("naccept", 0) # number accepted
    integrator.addGlobalVariable("ntrials", 0) # number of Metropolization trials
    integrator.addPerDofVariable("x1", 0) # position before application of constraints

    integrator.addGlobalVariable("pseudowork", 0) # accumulated pseudowork
    integrator.addGlobalVariable("heat", 0) # accumulated heat
    
    #
    # Allow context updating here.
    #
    integrator.addUpdateContextState();

    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    # 
    # Velocity perturbation.
    #
    integrator.addComputeSum("ke_old", "0.5*m*v*v")    
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();
    integrator.addComputeSum("ke_new", "0.5*m*v*v")
    integrator.addComputeGlobal("heat", "heat + (ke_new - ke_old)")    
    
    #
    # Metropolized symplectic step.
    #
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Eold", "ke + energy")
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    integrator.addComputePerDof("x", "x + v*dt")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions();
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
    integrator.addConstrainVelocities();
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Enew", "ke + energy")

    #
    # Accumulate statistics.
    #
    integrator.addComputeGlobal("pseudowork", "pseudowork + (Enew-Eold)") # accumulate pseudowork
    integrator.addComputeGlobal("naccept", "naccept + 1")
    integrator.addComputeGlobal("ntrials", "ntrials + 1")   

    #
    # Velocity randomization
    #
    integrator.addComputeSum("ke_old", "0.5*m*v*v")
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();
    integrator.addComputeSum("ke_new", "0.5*m*v*v")
    integrator.addComputeGlobal("heat", "heat + (ke_new - ke_old)")    

    return integrator
    
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
timestep = 1.0 * units.femtosecond
temperature = 298.0 * units.kelvin
kT = kB * temperature
friction = 20.0 / units.picosecond

nsteps = 1000
niterations = 100

# Select system:
testsystem = testsystems.MolecularIdealGas()
#testsystem = testsystems.AlanineDipeptideImplicit(flexibleConstraints=False, shake=True)
#testsystem = testsystems.LysozymeImplicit(flexibleConstraints=False, shake=True)
#testsystem = testsystems.HarmonicOscillator()
#testsystem = testsystems.HarmonicOscillatorArray(N=16)
#testsystem = testsystems.AlanineDipeptideExplicit(flexibleConstraints=False, shake=True)

# Retrieve system and positions.
[system, positions] = [testsystem.system, testsystem.positions]

velocities = generateMaxwellBoltzmannVelocities(system, temperature)
ndof = 3*system.getNumParticles() - system.getNumConstraints()

# Select integrator:
integrator = openmm.LangevinIntegrator(temperature, friction, timestep)
#integrator = AndersenVelocityVerletIntegrator(temperature, timestep)
#integrator = MetropolisMonteCarloIntegrator(timestep, temperature=temperature)
#integrator = HMCIntegrator(timestep, temperature=temperature)
#integrator = VVVRIntegrator(timestep, temperature=temperature)
#integrator = GHMCIntegrator(timestep, temperature=temperature)
#integrator = VelocityVerletIntegrator(timestep)
#integrator = openmm.VerletIntegrator(timestep)

# Create Context and set positions and velocities.
context = openmm.Context(system, integrator)
context.setPositions(positions)
context.setVelocities(velocities) 

print context.getPlatform().getName()

# Minimize
#openmm.LocalEnergyMinimizer.minimize(context)

# Accumulate statistics.
x_n = numpy.zeros([niterations], numpy.float64) # x_n[i] is the x position of atom 1 after iteration i, in angstroms
potential_n = numpy.zeros([niterations], numpy.float64) # potential_n[i] is the potential energy after iteration i, in kT
kinetic_n = numpy.zeros([niterations], numpy.float64) # kinetic_n[i] is the kinetic energy after iteration i, in kT
temperature_n = numpy.zeros([niterations], numpy.float64) # temperature_n[i] is the instantaneous kinetic temperature from iteration i, in K
for iteration in range(niterations):
    print "iteration %d / %d : propagating for %d steps..." % (iteration, niterations, nsteps)

    state = context.getState(getEnergy=True)
    initial_potential_energy = state.getPotentialEnergy()
    initial_kinetic_energy = state.getKineticEnergy()
    initial_total_energy = initial_kinetic_energy + initial_potential_energy
    
    initial_time = time.time()

    integrator.step(nsteps)

    state = context.getState(getEnergy=True, getPositions=True)
    final_potential_energy = state.getPotentialEnergy()
    final_kinetic_energy = state.getKineticEnergy()
    final_total_energy = final_kinetic_energy + final_potential_energy

    final_time = time.time()
    elapsed_time = final_time - initial_time
    
    delta_total_energy = final_total_energy - initial_total_energy
    instantaneous_temperature = final_kinetic_energy * 2.0 / ndof / (units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA)

    print "total energy: initial %8.1f kT | final %8.1f kT | delta = %8.3f kT | instantaneous temperature: %8.1f K | time %.3f s" % (initial_total_energy/kT, final_total_energy/kT, delta_total_energy/kT, instantaneous_temperature/units.kelvin, elapsed_time)

    #pseudowork = integrator.getGlobalVariable(0) * units.kilojoules_per_mole / kT
    #b = integrator.getGlobalVariable(2)
    #c = integrator.getGlobalVariable(3)
    #print (pseudowork, b, c)

#    global_variables = { integrator.getGlobalVariableName(index) : index for index in range(integrator.getNumGlobalVariables()) }
#    naccept = integrator.getGlobalVariable(global_variables['naccept'])
#    ntrials = integrator.getGlobalVariable(global_variables['ntrials'])
#    print "accepted %d / %d (%.3f %%)" % (naccept, ntrials, float(naccept)/float(ntrials)*100.0)

    # Accumulate statistics.
    x_n[iteration] = state.getPositions(asNumpy=True)[0,0] / units.angstroms
    potential_n[iteration] = final_potential_energy / kT
    kinetic_n[iteration] = final_kinetic_energy / kT
    temperature_n[iteration] = instantaneous_temperature / units.kelvin
    


# Compute expected statistics for harmonic oscillator.
K = 100.0 * units.kilocalories_per_mole / units.angstroms**2
beta = 1.0 / kT
x_mean_exact = 0.0 # mean, in angstroms
x_std_exact = 1.0 / units.sqrt(beta * K) / units.angstroms # std dev, in angstroms

# Analyze statistics.
g = statisticalInefficiency(x_n) 
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
