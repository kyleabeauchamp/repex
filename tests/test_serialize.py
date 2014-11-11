#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test that all testsystems can be correctly serialized/deserialized.

DESCRIPTION

COPYRIGHT

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

TODO

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import os.path
import sys
import math

import simtk.unit as units
import simtk.openmm as openmm

from openmmtools import testsystems

#=============================================================================================
# SUBROUTINES
#=============================================================================================

# These settings control what tolerance is allowed between platforms and the Reference platform.
ENERGY_TOLERANCE = 0.06*units.kilocalories_per_mole # energy difference tolerance
FORCE_RMSE_TOLERANCE = 0.06*units.kilocalories_per_mole/units.angstrom # per-particle force root-mean-square error tolerance

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
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest

    debug = False # Don't display extra debug information.

    # Make a list of all test system classes.
    testsystem_classes = testsystems.TestSystem.__subclasses__()
    systems = [ (cls.__name__, cls) for cls in testsystem_classes ]

    # Make a count of how often set tolerance is exceeded.
    tests_failed = 0 # number of times tolerance is exceeded
    tests_passed = 0 # number of times tolerance is not exceeded
    print "%16s          %16s          %16s" % ("error", "force mag", "rms error")    
    for (name, constructor) in systems:
        print "%s" % name

        testsystem = constructor()
        [system, positions] = [testsystem.system, testsystem.positions]

        # Compute initial potential and force.
        # Create a Context.        
        timestep = 1.0 * units.femtoseconds
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(system, integrator)
        # Set positions
        context.setPositions(positions)
        # Evaluate the potential energy.
        state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        initial_potential = state.getPotentialEnergy()
        initial_force = state.getForces(asNumpy=True)
        # Clean up
        del context, integrator

        from simtk.openmm import XmlSerializer

        # Serialize.
        system_xml = XmlSerializer.serialize(system)
        state_xml = XmlSerializer.serialize(state)
        
        # Deserialize.
        system = XmlSerializer.deserialize(system_xml)
        state = XmlSerializer.deserialize(state_xml)

        # Compute final potential and force.
        # Create a Context.        
        timestep = 1.0 * units.femtoseconds
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(system, integrator)
        # Set positions
        context.setPositions(positions)
        # Evaluate the potential energy.
        state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        final_potential = state.getPotentialEnergy()
        final_force = state.getForces(asNumpy=True)
        # Clean up
        del context, integrator

        # Compute error in potential.
        potential_error = final_potential - initial_potential

        # Compute per-atom RMS (magnitude) and RMS error in force.
        force_unit = units.kilocalories_per_mole / units.nanometers
        natoms = system.getNumParticles()
        force_mse = (((initial_force - final_force) / force_unit)**2).sum() / natoms * force_unit**2
        force_rmse = units.sqrt(force_mse)

        force_ms = ((initial_force / force_unit)**2).sum() / natoms * force_unit**2
        force_rms = units.sqrt(force_ms)

        print "%16.6f kcal/mol %16.6f kcal/mol %16.6f kcal/mol" % (potential_error / units.kilocalories_per_mole, force_rms / force_unit, force_rmse / force_unit)

        # Mark whether tolerance is exceeded or not.
        test_success = True
        if abs(potential_error) > ENERGY_TOLERANCE:
            test_success = False
            print "%32s WARNING: Potential energy error (%.6f kcal/mol) exceeds tolerance (%.6f kcal/mol).  Test failed." % ("", potential_error/units.kilocalories_per_mole, ENERGY_TOLERANCE/units.kilocalories_per_mole)
        if abs(force_rmse) > FORCE_RMSE_TOLERANCE:
            test_success = False
            print "%32s WARNING: Force RMS error (%.6f kcal/mol) exceeds tolerance (%.6f kcal/mol).  Test failed." % ("", force_rmse/force_unit, FORCE_RMSE_TOLERANCE/force_unit)
            
            
        if test_success:
            tests_passed += 1
        else:
            tests_failed += 1

    print "%d tests failed" % tests_failed
    print "%d tests passed" % tests_passed
            
    if (tests_failed > 0):
        # Signal failure of test.
        sys.exit(1)
    else:
        sys.exit(0)
