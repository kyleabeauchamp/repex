#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Markov chain Monte Carlo simulation framework.

DESCRIPTION

This module provides a framework for equilibrium sampling from a given thermodynamic state of
a biomolecule using a Markov chain Monte Carlo scheme.

CAPABILITIES
* Langevin dynamics [assumed to be free of integration error; use at your own risk]
* hybrid Monte Carlo
* generalized hybrid Monte Carlo

NOTES

This is still in development.

REFERENCES

[1] Jun S. Liu. Monte Carlo Strategies in Scientific Computing. Springer, 2008.

EXAMPLES

Construct a simple MCMC simulation using Langevin dynamics moves.

>>> # Create a test system
>>> import testsystems
>>> test = testsystems.AlanineDipeptideVacuum()
>>> # Create a thermodynamic state.
>>> import simtk.unit as u
>>> from thermodynamics import ThermodynamicState
>>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
>>> # Create a sampler state.
>>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
>>> # Create a move set.
>>> move_set = [ HMCMove(nsteps=10), LangevinDynamicsMove(nsteps=10) ]
>>> # Create MCMC sampler
>>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)
>>> # Run a number of iterations of the sampler.
>>> updated_sampler_state = sampler.run(sampler_state, 10)

TODO

* Split this into a separate package, with individual files for each move type.

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

TODO
----
* Recognize when MonteCarloBarostat is in use with system.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import math
import numpy
import copy
import time

import simtk
import simtk.openmm as mm
import simtk.unit as u

from repex import integrators

from abc import abstractmethod 

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

#=============================================================================================
# MCMC sampler state
#=============================================================================================

class MCMCSamplerState(object):
    """
    Sampler state for MCMC move representing everything that may be allowed to change during
    the simulation.

    Parameters
    ----------
    positions : array of simtk.unit.Quantity compatible with nanometers
       Particle positions.
    velocities : optional, array of simtk.unit.Quantity compatible with nanometers/picoseconds
       Particle velocities.
    box_vectors : optional, 3x3 array of simtk.unit.Quantity compatible with nanometers
       Current box vectors.
    system : optional, simtk.openmm.System 
       Current system specifying force calculations.    
    
    Examples
    --------

    Create a sampler state for a system with box vectors.

    >>> # Create a test system
    >>> import testsystems
    >>> test = testsystems.LennardJonesFluid()
    >>> # Create a sampler state manually.
    >>> box_vectors = test.system.getDefaultPeriodicBoxVectors()
    >>> sampler_state = MCMCSamplerState(positions=test.positions, box_vectors=box_vectors, system=test.system)

    Create a sampler state for a system without box vectors.

    >>> # Create a test system
    >>> import testsystems
    >>> test = testsystems.LennardJonesCluster()
    >>> # Create a sampler state manually.
    >>> sampler_state = MCMCSamplerState(positions=test.positions, system=test.system)

    """
    def __init__(self, positions, velocities=None, box_vectors=None, system=None):
        self.positions = positions
        self.velocities = velocities
        self.box_vectors = box_vectors
        self.system = system

    @classmethod
    def createFromContext(cls, context):
        """
        Create an MCMCSamplerState object from the information in a current OpenMM Context object.
        
        Parameters
        ----------
        context : simtk.openmm.Context
           The Context object from which to create a sampler state.
           
        Returns
        -------
        sampler_state : MCMCSamplerState
           The sampler state containing positions, velocities, and box vectors.

        Examples
        --------

        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = mm.VerletIntegrator(1.0 * u.femtoseconds)
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> context = mm.Context(test.system, integrator, platform)
        >>> # Set positions and velocities.
        >>> context.setPositions(test.positions)
        >>> context.setVelocitiesToTemperature(298 * u.kelvin)
        >>> # Create a sampler state from the Context.
        >>> sampler_state = MCMCSamplerState.createFromContext(context)
        >>> # Clean up.
        >>> del context, integrator

        """
        openmm_state = context.getState(getPositions=True, getVelocities=True)
        
        positions = openmm_state.getPositions(asNumpy=True)
        velocities = openmm_state.getVelocities(asNumpy=True)
        box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=True)
        system = context.getSystem()

        return MCMCSamplerState(positions=positions, velocities=velocities, box_vectors=box_vectors, system=system)

    def createContext(self, integrator, platform=None):
        """
        Create an OpenMM Context object from the current sampler state.

        Parameters
        ----------
        integrator : simtk.openmm.Integrator
           The integrator to use for Context creation.
        platform : simtk.openmm.Platform, optional, default=None
           If specified, the Platform to use for context creation.

        Returns
        -------
        context : simtk.openmm.Context
           The created OpenMM Context object

        Notes
        -----
        If the selected or default platform fails, the CPU and Reference platforms will be tried, in that order.        

        Examples
        --------

        Create a context for a system with periodic box vectors.
        
        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.LennardJonesFluid()
        >>> # Create a sampler state manually.
        >>> box_vectors = test.system.getDefaultPeriodicBoxVectors()
        >>> sampler_state = MCMCSamplerState(positions=test.positions, box_vectors=box_vectors, system=test.system)
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = mm.VerletIntegrator(1.0*u.femtoseconds)
        >>> context = sampler_state.createContext(integrator)
        >>> # Clean up.
        >>> del context

        Create a context for a system without periodic box vectors.

        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.LennardJonesCluster()
        >>> # Create a sampler state manually.
        >>> sampler_state = MCMCSamplerState(positions=test.positions, system=test.system)
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = mm.VerletIntegrator(1.0*u.femtoseconds)
        >>> context = sampler_state.createContext(integrator)
        >>> # Clean up.
        >>> del context
        
        """

        if not self.system:
            raise Exception("MCMCSamplerState must have a 'system' object specified to create a Context")

        # TODO: Make this less hacky, and introduce a fallback chain based on platform speeds.
        # TODO: Eliminate debug output
        try:
            if platform:
                context = mm.Context(self.system, integrator, platform)
            else:
                context = mm.Context(self.system, integrator)
        except Exception as e:
            print "Exception occurred in creating Context: '%s'" % str(e)

            try:
                platform_name = 'CPU'
                print "Attempting to use fallback platform '%s'..." % platform_name
                platform = mm.Platform.getPlatformByName(platform_name)
                context = mm.Context(self.system, integrator, platform)            
            except Exception as e:
                print "Exception occurred in creating Context: '%s'" % str(e)

                platform_name = 'Reference'
                print "Attempting to use fallback platform '%s'..." % platform_name
                platform = mm.Platform.getPlatformByName(platform_name)
                context = mm.Context(self.system, integrator, platform)            

        context.setPositions(self.positions)
        if self.velocities: context.setVelocities(self.velocities)
        if self.box_vectors: 
            try:
                # try tuple of box vectors
                context.setPeriodicBoxVectors(self.box_vectors[0], self.box_vectors[1], self.box_vectors[2])
            except:
                # try numpy 3x3 matrix of box vectors
                context.setPeriodicBoxVectors(self.box_vectors[0,:], self.box_vectors[1,:], self.box_vectors[2,:])

        return context

#=============================================================================================
# Monte Carlo Move abstract base class
#=============================================================================================

class MCMCMove(object):
    """
    Markov chain Monte Carlo (MCMC) move abstract base class.

    Markov chain Monte Carlo (MCMC) simulations are constructed from a set of derived objects.
    
    """

    @abstractmethod
    def apply(self, thermodynamic_state, sampler_state, platform=None):
        """
        Apply the MCMC move.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use when applying the MCMC move
        sampler_state : MCMCSamplerState
           The sampler state to apply the move to
        platform : simtk.openmm.Platform, optional, default = None
           The platform to use.

        Returns
        -------
        updated_sampler_state : MCMCSamplerState
           The updated sampler state


        """
        pass

#=============================================================================================
# Markov chain Monte Carlo sampler
#=============================================================================================

class MCMCSampler(object):
    """
    Markov chain Monte Carlo sampler.

    >>> # Create a test system
    >>> import testsystems
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a thermodynamic state.
    >>> import simtk.unit as u
    >>> from thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
    >>> # Create a sampler state.
    >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
    >>> # Create a move set specifying probabilities fo each type of move.
    >>> move_set = { HMCMove(nsteps=10) : 0.5, LangevinDynamicsMove(nsteps=10) : 0.5 }
    >>> # Create MCMC sampler
    >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)
    >>> # Run a number of iterations of the sampler.
    >>> updated_sampler_state = sampler.run(sampler_state, 10)
    
    """

    def __init__(self, thermodynamic_state, move_set=None, platform=None):
        """
        Initialize a Markov chain Monte Carlo sampler.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            Thermodynamic state to sample during MCMC run.
        move_set : container of MarkovChainMonteCarloMove objects
            Moves to attempt during MCMC run.
            If list or tuple, will run all moves each iteration in specified sequence. (e.g. [move1, move2, move3])
            if dict, will use specified unnormalized weights (e.g. { move1 : 0.3, move2 : 0.5, move3, 0.9 })
        platform : simtk.openmm.Platform, optional, default = None
            If specified, the Platform to use for simulations.

        Examples
        --------

        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a thermodynamic state.
        >>> import simtk.unit as u
        >>> from thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a sampler state.
        >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)

        Create a move set specifying probabilities for each type of move.

        >>> move_set = { HMCMove() : 0.5, LangevinDynamicsMove() : 0.5 }
        >>> # Create MCMC sampler
        >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)

        Create a move set specifying an order of moves.

        >>> move_set = [ HMCMove(), LangevinDynamicsMove(), HMCMove() ]
        >>> # Create MCMC sampler
        >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)

        """

        # Store thermodynamic state.
        self.thermodynamic_state = thermodynamic_state

        # Store the move set.
        if type(move_set) not in [list, dict]:
            raise Exception("move_set must be list or dict")
        # TODO: Make deep copy of the move set?
        self.move_set = move_set
        self.platform = platform

        return

    def run(self, sampler_state, niterations):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        sampler_state : SamplerState
            The current state of the sampler.
        niterations : int
            Number of iterations of the sampler to run.
        
        Examples
        --------

        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a thermodynamic state.
        >>> import simtk.unit as u
        >>> from thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a sampler state.
        >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
        >>> # Create a move set specifying probabilities fo each type of move.
        >>> move_set = { HMCMove(nsteps=10) : 0.5, LangevinDynamicsMove(nsteps=10) : 0.5 }
        >>> # Create MCMC sampler
        >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)
        >>> # Run a number of iterations of the sampler.
        >>> updated_sampler_state = sampler.run(sampler_state, 10)

        """

        # Make a deep copy of the sampler state so that initial state is unchanged.
        sampler_state = copy.deepcopy(sampler_state)

        # Generate move sequence.
        move_sequence = list()
        if type(self.move_set) == list:
            # Sequential moves.
            for iteration in range(niterations):
                for move in self.move_set:
                    move_sequence.append(move)
        elif type(self.move_set) == dict:
            # Random moves.
            moves = self.move_set.keys()
            weights = numpy.array([self.move_set[move] for move in moves])
            weights /= weights.sum() # normalize
            move_sequence = numpy.random.choice(moves, size=niterations, p=weights)
        
        # Apply move sequence.
        for move in move_sequence:
            move.apply(self.thermodynamic_state, sampler_state, platform=self.platform)
                
        # Return the updated sampler state.
        return sampler_state

#=============================================================================================
# Langevin dynamics move
#=============================================================================================

class LangevinDynamicsMove(MCMCMove):
    """
    Langevin dynamics segment as a (pseudo) Monte Carlo move.

    This move assigns a velocity from the Maxwell-Boltzmann distribution and executes a number
    of Maxwell-Boltzmann steps to propagate dynamics.  This is not a *true* Monte Carlo move,
    in that the generation of the correct distribution is only exact in the limit of infinitely
    small timestep; in other words, the discretization error is assumed to be negligible. Use
    HybridMonteCarloMove instead to ensure the exact distribution is generated.

    Warning
    -------
    No Metropolization is used to ensure the correct phase space distribution is sampled.
    This means that timestep-dependent errors will remain uncorrected, and are amplified with larger timesteps.
    Use this move at your own risk!

    Examples
    --------

    >>> # Create a test system
    >>> import testsystems
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a sampler state.
    >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> from thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
    >>> # Create a LangevinDynamicsMove
    >>> move = LangevinDynamicsMove(nsteps=10)
    >>> # Perform one update of the sampler state.
    >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

    """

    def __init__(self, timestep=1.0*simtk.unit.femtosecond, collision_rate=10.0/simtk.unit.picoseconds, nsteps=1000, reassign_velocities=False):
        """
        Parameters
        ----------
        timestep : simtk.unit.Quantity compatible with femtoseconds, optional, default = 1*simtk.unit.femtoseconds
            The timestep to use for Langevin integration.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, optional, default = 10/simtk.unit.picoseconds
            The collision rate with fictitious bath particles.
        nsteps : int, optional, default = 1000 
            The number of integration timesteps to take each time the move is applied.
        reassign_velocities : bool, optional, default = False
            If True, the velocities will be reassigned from the Maxwell-Boltzmann distribution at the beginning of the move.

        Note
        ----
        The temperature of the thermodynamic state is used in Langevin dynamics.

        Examples
        --------
        
        Create a Langevin move with default parameters.

        >>> move = LangevinDynamicsMove()

        Create a Langevin move with specified parameters.

        >>> move = LangevinDynamicsMove(timestep=0.5*u.femtoseconds, collision_rate=20.0/u.picoseconds, nsteps=100)

        """

        self.timestep = timestep
        self.collision_rate = collision_rate
        self.nsteps = nsteps
        self.reassign_velocities = reassign_velocities

        return

    def apply(self, thermodynamic_state, sampler_state, platform=None):
        """
        Apply the Langevin dynamics MCMC move.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use when applying the MCMC move
        sampler_state : MCMCSamplerState
           The sampler state to apply the move to
        platform : simtk.openmm.Platform, optional, default = None
           If not None, the specified platform will be used.

        Returns
        -------
        updated_sampler_state : MCMCSamplerState
           The updated sampler state

        Examples
        --------
        
        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a LangevinDynamicsMove
        >>> move = LangevinDynamicsMove(nsteps=10, timestep=0.5*u.femtoseconds, collision_rate=20.0/u.picoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        """
        
        # Create integrator.
        integrator = mm.LangevinIntegrator(thermodynamic_state.temperature, self.collision_rate, self.timestep)

        # Create context.
        context = sampler_state.createContext(integrator, platform=platform)

        if self.reassign_velocities:
            # Assign Maxwell-Boltzmann velocities.        
            context.setVelocitiesToTemperature(thermodynamic_state.temperature)
        
        # Run dynamics.
        integrator.step(self.nsteps)
        
        # Get updated sampler state.
        updated_sampler_state = MCMCSamplerState.createFromContext(context)

        # Clean up.
        del context

        return updated_sampler_state
    
#=============================================================================================
# Genaralized Hybrid Monte Carlo (GHMC, a form of Metropolized Langevin dynamics) move
#=============================================================================================

class GHMCMove(MCMCMove):
    """
    Generalized hybrid Monte Carlo (GHMC) Markov chain Monte Carlo move

    This move uses generalized Hybrid Monte Carlo (GHMC), a form of Metropolized Langevin
    dynamics, to propagate the system.

    References
    ----------
    Lelievre T, Stoltz G, and Rousset M. Free Energy Computations: A Mathematical Perspective
    http://www.amazon.com/Free-Energy-Computations-Mathematical-Perspective/dp/1848162472

    Examples
    --------

    >>> # Create a test system
    >>> import testsystems
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a sampler state.
    >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> from thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
    >>> # Create a GHMC move
    >>> move = GHMCMove(nsteps=10)
    >>> # Perform one update of the sampler state.
    >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

    """

    def __init__(self, timestep=1.0*simtk.unit.femtosecond, collision_rate=10.0/simtk.unit.picoseconds, nsteps=1000):
        """
        Parameters
        ----------
        timestep : simtk.unit.Quantity compatible with femtoseconds, optional, default = 1*simtk.unit.femtoseconds
            The timestep to use for Langevin integration.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, optional, default = 10/simtk.unit.picoseconds
            The collision rate with fictitious bath particles.
        nsteps : int, optional, default = 1000 
            The number of integration timesteps to take each time the move is applied.

        Note
        ----
        The temperature of the thermodynamic state is used.

        Examples
        --------
        
        Create a GHMC move with default parameters.

        >>> move = GHMCMove()

        Create a GHMC move with specified parameters.

        >>> move = GHMCMove(timestep=0.5*u.femtoseconds, collision_rate=20.0/u.picoseconds, nsteps=100)

        """

        # User parameters.
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.nsteps = nsteps

        self.reset_statistics()

        return

    def reset_statistics(self):
        """
        Reset the internal statistics of number of accepted and attempted moves.

        Examples
        --------
        
        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a LangevinDynamicsMove
        >>> move = GHMCMove(nsteps=10, timestep=1.0*u.femtoseconds, collision_rate=20.0/u.picoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        Reset statistics.
        
        >>> move.reset_statistics()

        """

        self.naccepted = 0 # number of accepted steps
        self.nattempted = 0 # number of attempted steps

        return
    
    def get_statistics(self):
        """
        Return the current acceptance/rejection statistics of the sampler.

        Returns
        -------
        naccepted : int
           The number of accepted steps
        nattempted : int
           The number of attempted steps
        faction_accepted : float
           The fraction of steps accepted.

        Examples
        --------
        
        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a LangevinDynamicsMove
        >>> move = GHMCMove(nsteps=10, timestep=1.0*u.femtoseconds, collision_rate=20.0/u.picoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        Get statistics.
        
        >>> [naccepted, nattempted, fraction_accepted] = move.get_statistics()

        """
        return (self.naccepted, self.nattempted, float(self.naccepted) / float(self.nattempted))
        

    def apply(self, thermodynamic_state, sampler_state, platform=None):
        """
        Apply the GHMC MCMC move.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use when applying the MCMC move
        sampler_state : MCMCSamplerState
           The sampler state to apply the move to
        platform : simtk.openmm.Platform, optional, default = None
           If not None, the specified platform will be used.

        Returns
        -------
        updated_sampler_state : MCMCSamplerState
           The updated sampler state

        Examples
        --------
        
        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a LangevinDynamicsMove
        >>> move = GHMCMove(nsteps=10, timestep=1.0*u.femtoseconds, collision_rate=20.0/u.picoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        """
        
        # Create integrator.
        integrator = integrators.GHMCIntegrator(temperature=thermodynamic_state.temperature, collision_rate=self.collision_rate, timestep=self.timestep)

        # Create context.
        context = sampler_state.createContext(integrator, platform=platform)

        # Run dynamics.
        integrator.step(self.nsteps)
        
        # Get updated sampler state.
        updated_sampler_state = MCMCSamplerState.createFromContext(context)
        
        # Accumulate acceptance statistics.
        ghmc_global_variables = { integrator.getGlobalVariableName(index) : index for index in range(integrator.getNumGlobalVariables()) }
        naccepted = integrator.getGlobalVariable(ghmc_global_variables['naccept'])
        nattempted = integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
        self.naccepted += naccepted
        self.nattempted += nattempted

        # Clean up.
        del context

        return updated_sampler_state
    
#=============================================================================================
# Hybrid Monte Carlo move
#=============================================================================================

class HMCMove(MCMCMove):
    """
    Hybrid Monte Carlo dynamics.

    This move assigns a velocity from the Maxwell-Boltzmann distribution and executes a number
    of velocity Verlet steps to propagate dynamics.  

    Examples
    --------

    >>> # Create a test system
    >>> import testsystems
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a sampler state.
    >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> from thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
    >>> # Create an HMC move.
    >>> move = HMCMove(nsteps=10)
    >>> # Perform one update of the sampler state.
    >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

    """

    def __init__(self, timestep=1.0*simtk.unit.femtosecond, nsteps=1000):
        """
        Parameters
        ----------
        timestep : simtk.unit.Quantity compatible with femtoseconds, optional, default = 1*femtosecond
           The timestep to use for HMC dynamics (which uses velocity Verlet following velocity randomization)
        nsteps : int, optional, default = 1000
           The number of dynamics steps to take before Metropolis acceptance/rejection.

        Examples
        --------

        Create an HMC move with default timestep and number of steps.

        >>> move = HMCMove()

        Create an HMC move with specified timestep and number of steps.

        >>> move = HMCMove(timestep=0.5*u.femtoseconds, nsteps=500)

        """

        # Set defaults
        self.timestep = 1.0 * u.femtosecond
        self.nsteps = nsteps

        return

    def apply(self, thermodynamic_state, sampler_state, platform=None):
        """
        Apply the MCMC move.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use when applying the MCMC move
        sampler_state : MCMCSamplerState
           The sampler state to apply the move to
        platform : simtk.openmm.Platform, optional, default = None
           If not None, the specified platform will be used.

        Returns
        -------
        updated_sampler_state : MCMCSamplerState
           The updated sampler state

        Examples
        --------
        
        >>> # Create a test system
        >>> import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = MCMCSamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create an HMC move.
        >>> move = HMCMove(nsteps=10, timestep=0.5*u.femtoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        """
        
        # Create integrator.
        integrator = integrators.HMCIntegrator(temperature=thermodynamic_state.temperature, timestep=self.timestep, nsteps=self.nsteps)

        # Create context.
        context = sampler_state.createContext(integrator, platform=platform)

        # Run dynamics.
        # Note that ONE step of this integrator is equal to self.nsteps of velocity Verlet dynamics followed by Metropolis accept/reject.
        integrator.step(1)
        
        # Get sampler state.
        updated_sampler_state = MCMCSamplerState.createFromContext(context)

        # Clean up.
        del context

        # Return updated sampler state.
        return updated_sampler_state
    
#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
