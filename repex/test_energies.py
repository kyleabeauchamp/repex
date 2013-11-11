#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test that all OpenMM systems in simtk.pyopenmm.extras.testsystems.* give the expected potential
energies and can stably run a short dynamics simulation.

DESCRIPTION

This script tests a number of simple model test systems, available in the package
simtk.pyopenmm.extras.testsystems, to make sure they reproduce known potential energies.

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

import simtk.unit as units
import simtk.openmm as openmm

import test_systems as testsystems

#=============================================================================================
# Expected potential energies for each test system
#=============================================================================================

testsystem_energies = {
   'AlanineDipeptideExplicit' : -24654.9876211 * units.kilojoules_per_mole,
   'AlanineDipeptideImplicit' : -137.437357167 * units.kilojoules_per_mole,
   'ConstraintCoupledHarmonicOscillator' : 0.0 * units.kilojoules_per_mole,
   'Diatom' : 0.0 * units.kilojoules_per_mole,
   'HarmonicOscillator' : 0.0 * units.kilojoules_per_mole,
   'HarmonicOscillatorArray' : 0.0 * units.kilojoules_per_mole,
   'LennardJonesCluster' : 4.10034520364 * units.kilojoules_per_mole, 
   'LennardJonesFluid' : -653.16317781 * units.kilojoules_per_mole, 
   'CustomLennardJonesFluid' : -653.162946612 * units.kilojoules_per_mole,
   'CustomGBForceSystem' : -78203.4777545 * units.kilojoules_per_mole,   
   'SodiumChlorideCrystal' : -455.766773418 * units.kilojoules_per_mole,
   'WaterBox' : -7316.86673998 * units.kilojoules_per_mole,
   'LysozymeImplicit' : -25593.6293016 * units.kilojoules_per_mole,
   'IdealGas' : 0.0 * units.kilocalories_per_mole,
   'MethanolBox' : 1331.1307688 * units.kilojoules_per_mole,
   'MolecularIdealGas' : 1357.65080814 * units.kilojoules_per_mole,   
}

#=============================================================================================
# UTILITIES
#=============================================================================================

ENERGY_TOLERANCE = 0.06*units.kilocalories_per_mole

def assert_approximately_equal(computed_potential, expected_potential, tolerance=ENERGY_TOLERANCE):
    """
    Check whether computed potential is acceptably close to expected value, using an error tolerance.

    ARGUMENTS

    computed_potential (simtk.unit.Quantity in units of energy) - computed potential energy
    expected_potential (simtk.unit.Quantity in units of energy) - expected
    
    OPTIONAL ARGUMENTS

    tolerance (simtk.unit.Quantity in units of energy) - acceptable tolerance

    EXAMPLES

    >>> assert_approximately_equal(0.0000 * units.kilocalories_per_mole, 0.0001 * units.kilocalories_per_mole, tolerance=0.06*units.kilocalories_per_mole)
        
    """

    # Compute error.
    error = (computed_potential - expected_potential)

    # Raise an exception if the error is larger than the tolerance.
    if abs(error) > tolerance:
        raise Exception("Computed potential %s, expected %s.  Error %s is larger than acceptable tolerance of %s." % (computed_potential, expected_potential, error, tolerance))
    
    return

#=============================================================================================
# MAIN
#=============================================================================================

# Run doctests on all systems.
#doctest.testmod(testsystems, extraglobs={'platform' : platform})

platform = openmm.Platform.getPlatformByName('CUDA')

# Compute energies and run a short bit of dynamics for each test system.
tests_passed = 0
tests_failed = 0
for cls in testsystems.testsystem_classes:
   system_name = cls.__name__
   print '*' * 80
   print system_name
   
   # Set failure flag.
   failure = False
   
   # Create system.
   print "Constructing system..."
   testsystem = cls()
   [system, positions] = [testsystem.system, testsystem.positions]

   # Create integrator and context.
   temperature = 298.0 * units.kelvin
   collision_rate = 91.0 / units.picosecond
   timestep = 1.0 * units.femtosecond    
   integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
   context = openmm.Context(system, integrator, platform)

   # Set positions
   context.setPositions(positions)

   # Evaluate the potential energy.
   print "Computing potential energy..."
   state = context.getState(getEnergy=True)
   potential = state.getPotentialEnergy()

   # If we have an expected result, check to make sure this is approximately satisfied.
   if system_name in testsystem_energies:
      try:
         expected_potential = testsystem_energies[system_name]
         assert_approximately_equal(potential, expected_potential)
      except Exception as exception:
         print str(exception)
         failure = True
   else:
      print "'%s' : %s * units.kilojoules_per_mole," % (system_name, str(potential / units.kilojoules_per_mole))
      # Check that energy is not 'nan'.
      if numpy.isnan(potential / units.kilojoules_per_mole):
         print "Potential energy is 'nan'."
         failure = True

   # Integrate a few steps of dynamics to see if system remains stable.
   nsteps = 10 # number of steps to integrate
   print "Running %d steps of dynamics..." % nsteps   
   integrator.step(nsteps)   

   # Retrieve configuration to make sure no positions are nan.
   state = context.getState(getPositions=True)
   positions = state.getPositions(asNumpy=True)
   if numpy.any(numpy.isnan(positions / units.nanometers)):
      print 'Some positions are nan after integration.'
      failure = True

   # Accumulate passes and fails.
   if failure:
      tests_failed += 1
   else:
      tests_passed += 1
      
print '*' * 80
print "%d tests passed" % tests_passed
print "%d tests failed" % tests_failed

# Exit.
if tests_failed > 0:
   # signal failure
   sys.exit(1)   
else:
   sys.exit(0)
