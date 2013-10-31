#!/usr/bin/env python

import os
import math
import copy
import time
import datetime

import numpy as np
import numpy.linalg

import simtk.openmm as mm
import simtk.unit as units

import netCDF4 as netcdf

from thermodynamics import ThermodynamicState
from utils import time_and_print, process_kwargs, fix_coordinates
from constants import kB

import logging
logger = logging.getLogger(__name__)

__version__ = 0.0

class NetCDFDatabase(object):
    options_to_store = ['collision_rate', 'constraint_tolerance', 'timestep', 'nsteps_per_iteration', 'number_of_iterations', 'equilibration_timestep', 'number_of_equilibration_iterations', 'title', 'minimize', 'replica_mixing_scheme', 'online_analysis', 'show_mixing_statistics']

    def __init__(self, filename, states=None, coordinates=None, **kwargs):

        # Check if netcdf file exists.
        self.resume = os.path.exists(filename) and (os.path.getsize(filename) > 0)
        
        self.ncfile = netcdf.Dataset(filename, 'w', version='NETCDF4')
        self.title = "No Title."
        
        if self.resume: 
            logger.info("Attempting to resume by reading thermodynamic states and options...")
            self.initial_states = self.load_thermodynamic_states()
            self.options = self.load_options()
            self.check_self_consistency(states, coordinates, options)
        else:
            self.initial_states = states
            self.options = process_kwargs(kwargs)
            self.coordinates = coordinates
            self._initialize_netcdf()

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in self.initial_states:
            if not state.is_compatible_with(self.initial_states[0]):
                raise ValueError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")

        if not self.resume:
            self.coordinates = fix_coordinates(coordinates)

    def check_self_consistency(self, states, coordinates, options):
        """Raises a ValueError is input and loaded data disagree."""
        pass

    def _initialize_netcdf(self):
        """
        Initialize NetCDF file for storage.
        
        """
        
        self.n_replicas = len(self.initial_states)
        self.n_atoms = len(self.coordinates[0])
        self.n_states = len(self.initial_states)
        assert self.n_replicas == len(self.coordinates), "Error: inconsistent number of replicas."
        
        # Create dimensions.
        self.ncfile.createDimension('iteration', 0) # unlimited number of iterations
        self.ncfile.createDimension('replica', self.n_replicas) # number of replicas
        self.ncfile.createDimension('atom', self.n_atoms) # number of atoms in system
        self.ncfile.createDimension('spatial', 3) # number of spatial dimensions

        # Set global attributes.
        setattr(self.ncfile, 'title', self.title)
        setattr(self.ncfile, 'application', 'YANK')
        setattr(self.ncfile, 'program', 'yank.py')
        setattr(self.ncfile, 'programVersion', __version__)
        setattr(self.ncfile, 'Conventions', 'YANK')
        setattr(self.ncfile, 'ConventionVersion', '0.1')
        
        # Create variables.
        ncvar_positions = self.ncfile.createVariable('positions', 'f', ('iteration','replica','atom','spatial'))
        ncvar_states    = self.ncfile.createVariable('states', 'i', ('iteration','replica'))
        ncvar_energies  = self.ncfile.createVariable('energies', 'f', ('iteration','replica','replica'))        
        ncvar_proposed  = self.ncfile.createVariable('proposed', 'l', ('iteration','replica','replica'))
        ncvar_accepted  = self.ncfile.createVariable('accepted', 'l', ('iteration','replica','replica'))                
        ncvar_box_vectors = self.ncfile.createVariable('box_vectors', 'f', ('iteration','replica','spatial','spatial'))        
        ncvar_volumes  = self.ncfile.createVariable('volumes', 'f', ('iteration','replica'))
        
        # Define units for variables.
        setattr(ncvar_positions, 'units', 'nm')
        setattr(ncvar_states,    'units', 'none')
        setattr(ncvar_energies,  'units', 'kT')
        setattr(ncvar_proposed,  'units', 'none')
        setattr(ncvar_accepted,  'units', 'none')                
        setattr(ncvar_box_vectors, 'units', 'nm')
        setattr(ncvar_volumes, 'units', 'nm**3')

        # Define long (human-readable) names for variables.
        setattr(ncvar_positions, "long_name", "positions[iteration][replica][atom][spatial] is position of coordinate 'spatial' of atom 'atom' from replica 'replica' for iteration 'iteration'.")
        setattr(ncvar_states,    "long_name", "states[iteration][replica] is the state index (0..n_states-1) of replica 'replica' of iteration 'iteration'.")
        setattr(ncvar_energies,  "long_name", "energies[iteration][replica][state] is the reduced (unitless) energy of replica 'replica' from iteration 'iteration' evaluated at state 'state'.")
        setattr(ncvar_proposed,  "long_name", "proposed[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")
        setattr(ncvar_accepted,  "long_name", "accepted[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")
        setattr(ncvar_box_vectors, "long_name", "box_vectors[iteration][replica][i][j] is dimension j of box vector i for replica 'replica' from iteration 'iteration-1'.")
        setattr(ncvar_volumes, "long_name", "volume[iteration][replica] is the box volume for replica 'replica' from iteration 'iteration-1'.")

        # Create timestamp variable.
        ncvar_timestamp = self.ncfile.createVariable('timestamp', str, ('iteration',))

        # Create group for performance statistics.
        ncgrp_timings = self.ncfile.createGroup('timings')
        ncvar_iteration_time = ncgrp_timings.createVariable('iteration', 'f', ('iteration',)) # total iteration time (seconds)
        ncvar_iteration_time = ncgrp_timings.createVariable('mixing', 'f', ('iteration',)) # time for mixing
        ncvar_iteration_time = ncgrp_timings.createVariable('propagate', 'f', ('iteration','replica')) # total time to propagate each replica
        
        # Store thermodynamic states.
        self._store_thermodynamic_states(self.initial_states)

        # Store run options
        self._store_options()

        # Force sync to disk to avoid data loss.
        self.ncfile.sync()
    
    @time_and_print
    def _write_iteration_netcdf(self):
        """
        Write positions, states, and energies of current iteration to NetCDF file.
        
        """

        initial_time = time.time()

        # Store replica positions.
        for replica_index in range(self.n_states):
            coordinates = self.replica_coordinates[replica_index]
            x = coordinates / units.nanometers
            self.ncfile.variables['positions'][self.iteration,replica_index,:,:] = x[:,:]
            
        # Store box vectors and volume.
        for replica_index in range(self.n_states):
            state_index = self.replica_states[replica_index]
            state = self.initial_states[state_index]
            box_vectors = self.replica_box_vectors[replica_index]
            for i in range(3):
                self.ncfile.variables['box_vectors'][self.iteration,replica_index,i,:] = (box_vectors[i] / units.nanometers)
            volume = state._volume(box_vectors)
            self.ncfile.variables['volumes'][self.iteration,replica_index] = volume / (units.nanometers**3)

        # Store state information.
        self.ncfile.variables['states'][self.iteration,:] = self.replica_states[:]

        # Store energies.
        self.ncfile.variables['energies'][self.iteration,:,:] = self.u_kl[:,:]

        # Store mixing statistics.
        # TODO: Write mixing statistics for this iteration?
        self.ncfile.variables['proposed'][self.iteration,:,:] = self.Nij_proposed[:,:]
        self.ncfile.variables['accepted'][self.iteration,:,:] = self.Nij_accepted[:,:]        

        # Store timestamp this iteration was written.
        self.ncfile.variables['timestamp'][self.iteration] = time.ctime()

        self.ncfile.sync()  # Force sync to disk to avoid data loss.

    @time_and_print
    def _store_thermodynamic_states(self, states):
        """
        Store the thermodynamic states in a NetCDF file.

        """
        logger.debug("Storing thermodynamic states in NetCDF file...")
            
        # Create a group to store state information.
        ncgrp_stateinfo = self.ncfile.createGroup('thermodynamic_states')

        # Get number of states.
        ncvar_n_states = ncgrp_stateinfo.createVariable('n_states', int)
        ncvar_n_states.assignValue(self.n_states)


        # Temperatures.
        ncvar_temperatures = ncgrp_stateinfo.createVariable('temperatures', 'f', ('replica',))
        setattr(ncvar_temperatures, 'units', 'K')
        setattr(ncvar_temperatures, 'long_name', "temperatures[state] is the temperature of thermodynamic state 'state'")
        for state_index in range(self.n_states):
            ncvar_temperatures[state_index] = self.initial_states[state_index].temperature / units.kelvin

        # Pressures.
        if self.initial_states[0].pressure is not None:
            ncvar_temperatures = ncgrp_stateinfo.createVariable('pressures', 'f', ('replica',))
            setattr(ncvar_temperatures, 'units', 'atm')
            setattr(ncvar_temperatures, 'long_name', "pressures[state] is the external pressure of thermodynamic state 'state'")
            for state_index in range(self.n_states):
                ncvar_temperatures[state_index] = self.initial_states[state_index].pressure / units.atmospheres                

        # TODO: Store other thermodynamic variables store in ThermodynamicState?  Generalize?
                
        # Systems.
        ncvar_serialized_states = ncgrp_stateinfo.createVariable('systems', str, ('replica',), zlib=True)
        setattr(ncvar_serialized_states, 'long_name', "systems[state] is the serialized OpenMM System corresponding to the thermodynamic state 'state'")
        for state_index in range(self.n_states):
            logger.debug("Serializing state %d..." % state_index)
            serialized = self.initial_states[state_index].system.__getstate__()
            logger.debug("Serialized state is %d B | %.3f KB | %.3f MB" % (len(serialized), len(serialized) / 1024.0, len(serialized) / 1024.0 / 1024.0))
            ncvar_serialized_states[state_index] = serialized


    def load_thermodynamic_states(self):
        """
        Restore the thermodynamic states from a NetCDF file.

        """
        logging.debug("Restoring thermodynamic states from NetCDF file...")
           
        # Make sure this NetCDF file contains thermodynamic state information.
        if not 'thermodynamic_states' in self.ncfile.groups:
            raise(ValueError("thermodynamics_states not found in database!"))

        # Create a group to store state information.
        ncgrp_stateinfo = self.ncfile.groups['thermodynamic_states']

        # Get number of states.
        n_states = int(ncgrp_stateinfo.variables['n_states'][0])

        # Read state information.
        states = list()
        for state_index in range(n_states):
            # Populate a new ThermodynamicState object.
            state = ThermodynamicState()
            # Read temperature.
            state.temperature = ncgrp_stateinfo.variables['temperatures'][state_index] * units.kelvin
            # Read pressure, if present.
            if 'pressures' in ncgrp_stateinfo.variables:
                state.pressure = ncgrp_stateinfo.variables['pressures'][state_index] * units.atmospheres
            # Reconstitute System object.
            state.system = mm.System() 
            state.system.__setstate__(str(ncgrp_stateinfo.variables['systems'][state_index]))
            # Store state.
            states.append(state)
        
        return states

    def _store_options(self):
        """
        Store run parameters in NetCDF file.

        """

        logger.debug("Storing run parameters in NetCDF file...")

        # Create scalar dimension if not already present.
        if 'scalar' not in self.ncfile.dimensions:
            self.ncfile.createDimension('scalar', 1) # scalar dimension
        
        # Create a group to store state information.
        ncgrp_options = self.ncfile.createGroup('options')

        # Store run parameters.
        for option_name in self.options_to_store:
            # Get option value.
            #option_value = getattr(self, option_name)
            option_value = self.options.get(option_name)
            # If Quantity, strip off units first.
            option_unit = None
            if type(option_value) == units.Quantity:
                option_unit = option_value.unit
                option_value = option_value / option_unit
            # Store the Python type.
            option_type = type(option_value)
            # Handle booleans
            if type(option_value) == bool:
                option_value = int(option_value)
            # Store the variable.
            logger.debug("Storing option: %s -> %s (type: %s)" % (option_name, option_value, str(option_type)))
            if type(option_value) == str:
                ncvar = ncgrp_options.createVariable(option_name, type(option_value), 'scalar')
                packed_data = np.empty(1, 'O')
                packed_data[0] = option_value
                ncvar[:] = packed_data
            else:
                ncvar = ncgrp_options.createVariable(option_name, type(option_value))
                ncvar.assignValue(option_value)
            if option_unit: setattr(ncvar, 'units', str(option_unit))
            setattr(ncvar, 'type', option_type.__name__) 

    def load_options(self):
        """Return a dictionary of options from a loaded NetCDF file.
        """
        
        logger.debug("Attempting to restore options from NetCDF file...")
        
        # Make sure this NetCDF file contains option information
        if not 'options' in self.ncfile.groups:
            # Not found, signal failure.
            raise(ValueError("Options not found in NetCDF file!"))

        # Find the group.
        ncgrp_options = self.ncfile.groups['options']
        
        options = {}

        # Load run parameters.
        for option_name in ncgrp_options.variables.keys():
            # Get NetCDF variable.
            option_ncvar = ncgrp_options.variables[option_name]
            # Get option value.
            option_value = option_ncvar[0]
            # Cast to Python type.
            type_name = getattr(option_ncvar, 'type') 
            option_value = eval(type_name + '(' + repr(option_value) + ')')
            # If Quantity, assign units.
            if hasattr(option_ncvar, 'units'):
                option_unit_name = getattr(option_ncvar, 'units')
                if option_unit_name[0] == '/': option_unit_name = '1' + option_unit_name
                option_unit = eval(option_unit_name, vars(units))
                option_value = units.Quantity(option_value, option_unit)
            # Store option.
            logger.debug("Restoring option: %s -> %s (type: %s)" % (option_name, str(option_value), type(option_value)))
            #setattr(self, option_name, option_value)
            options[option_name] = option_value
            
        return options

    def _resume_from_netcdf(self):
        """Resume execution by reading current positions and energies from a NetCDF file.
        
        """

        # TODO: Perform sanity check on file before resuming

        iteration = self.ncfile.variables['positions'].shape[0] - 1 
        n_states = self.ncfile.variables['positions'].shape[1]
        n_atoms = self.ncfile.variables['positions'].shape[2]
        
        logging.debug("iteration = %d, n_states = %d, n_atoms = %d" % (iteration, n_states, n_atoms))

        # Restore positions.
        replica_coordinates = list()
        for replica_index in range(n_states):
            x = self.ncfile.variables['positions'][iteration, replica_index, :, :].astype(np.float64).copy()
            coordinates = units.Quantity(x, units.nanometers)
            replica_coordinates.append(coordinates)
        
        # Restore box vectors.
        replica_box_vectors = list()
        for replica_index in range(n_states):
            x = self.ncfile.variables['box_vectors'][iteration,replica_index,:,:].astype(np.float64).copy()
            box_vectors = units.Quantity(x, units.nanometers)
            replica_box_vectors.append(box_vectors)

        # Restore state information.
        replica_states = self.ncfile.variables['states'][iteration,:].copy()

        # Restore energies.
        u_kl = self.ncfile.variables['energies'][iteration,:,:].copy()
        
        # We will work on the next iteration.
        iteration += 1
        
        return replica_coordinates, replica_box_vectors, u_kl, iteration


    def output_iteration(self, **kwargs):
        """To do: use mdtraj.utils.ensure_type to ensure correct shapes and dtypes!"""
        
        required_keys = ["iteration", "coordinates", "box_vectors", "volumes", "replica_states", "energies", "proposed", "accepted", "time"]
        assert set(required_keys) == set(kwargs.keys()), "Wrong keys provided to output_iteration!"

        iteration = kwargs["iteration"]
        
        self.ncfile.variables["positions"][iteration] = kwargs["coordinates"]
        self.ncfile.variables['box_vectors'][iteration] = kwargs["box_vectors"]
        self.ncfile.variables['volumes'][iteration] = kwargs["volumes"]    
        self.ncfile.variables['states'][iteration] = kwargs["replica_states"]
        self.ncfile.variables['energies'][iteration] = kwargs["energies"]
        self.ncfile.variables['proposed'][iteration] = kwargs["proposed"]
        self.ncfile.variables['accepted'][iteration] = kwargs["accepted"]
        self.ncfile.variables['timestamp'][iteration] = kwargs["time"]

        self.ncfile.sync()

    @property
    def proposed(self):
        """Proposed moves
        """
        return self.ncfile.variables['proposed'][:]
        
    @property
    def accepted(self):
        """Return accepted moves"""
        return self.ncfile.variables['accepted'][:]

    @property
    def states(self):
        """Return accepted moves"""
        return self.ncfile.variables['states'][:]

    
    def finalize(self):
        logger.warn("WARNING: database finalize() has not yet been implemented.")

    def close(self):
        logger.warn("WARNING: database close() has not yet been implemented.")
