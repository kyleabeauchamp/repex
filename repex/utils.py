#!/usr/local/bin/env python

import os
import sys
import math
import copy
import time
import datetime

import numpy
import numpy.linalg

import simtk.openmm 
import simtk.unit as units

import netCDF4 as netcdf # netcdf4-python is used in place of scipy.io.netcdf for now

from thermodynamics import ThermodynamicState

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant


def generate_maxwell_boltzmann_velocities(system, temperature):
    """
    Generate Maxwell-Boltzmann velocities.

    ARGUMENTS

    system (simtk.openmm.System) - the system for which velocities are to be assigned
    temperature (simtk.unit.Quantity with units temperature) - the temperature at which velocities are to be assigned

    RETURN VALUES

    velocities (simtk.unit.Quantity wrapping numpy array of dimension natoms x 3 with units of distance/time) - drawn from the Maxwell-Boltzmann distribution at the appropriate temperature

    """

    # Get number of atoms
    natoms = system.getNumParticles()

    # Decorate System object with vector of masses for efficiency.
    if not hasattr(system, 'masses'):
        masses = simtk.unit.Quantity(numpy.zeros([natoms,3], numpy.float64), units.amu)
        for atom_index in range(natoms):
            mass = system.getParticleMass(atom_index) # atomic mass
            masses[atom_index,:] = mass
        setattr(system, 'masses', masses)

    # Retrieve masses.
    masses = getattr(system, 'masses')

    # Compute thermal energy and velocity scaling factors.
    kT = kB * temperature # thermal energy
    sigma2 = kT / masses

    # Assign velocities from the Maxwell-Boltzmann distribution.
    # TODO: This is wacky because units.sqrt cannot operate on numpy vectors.
    
    velocity_unit = units.nanometers / units.picoseconds
    velocities = units.Quantity(numpy.sqrt(sigma2 / (velocity_unit**2)) * numpy.random.randn(natoms, 3), velocity_unit)
    
    return velocities

