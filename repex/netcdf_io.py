#!/usr/bin/env python

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

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

def time_and_print(x):
    return x

class NetCDFIO(object):
    options_to_store = ['collision_rate', 'constraint_tolerance', 'timestep', 'nsteps_per_iteration', 'number_of_iterations', 'equilibration_timestep', 'number_of_equilibration_iterations', 'title', 'minimize', 'replica_mixing_scheme', 'online_analysis', 'verbose', 'show_mixing_statistics']

    def __init__(self, states=None, coordinates=None, store_filename=None, protocol=None, mm=None, mpicomm=None, metadata=None):

        self.store_filename = store_filename

        # Check if netcdf file exists.
        self.resume = os.path.exists(self.store_filename) and (os.path.getsize(self.store_filename) > 0)
        if self.mpicomm: self.resume = self.mpicomm.bcast(self.resume, root=0) # use whatever root node decides

        # Try to restore thermodynamic states and run options from the NetCDF file.
        states_restored = False 
        options_restored = False
        if self.resume: 
            print "Attempting to resume by reading thermodynamic states and options..." 
            ncfile = netcdf.Dataset(self.store_filename, 'r')            
            states_restored = self._restore_thermodynamic_states(ncfile)
            options_restored = self._restore_options(ncfile)
            ncfile.close()            

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in self.states:
            if not state.is_compatible_with(self.states[0]):
                raise ParameterError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")

        if not self.resume:
            # Distribute coordinate information to replicas in a round-robin fashion.
            # We have to explicitly check to see if z is a list or a set here because it turns out that numpy 2D arrays are iterable as well.
            # TODO: Handle case where coordinates are passed in as a list of tuples, or list of lists, or list of Vec3s, etc.
            if type(coordinates) in [type(list()), type(set())]:
                self.provided_coordinates = [ units.Quantity(numpy.array(coordinate_set / coordinate_set.unit), coordinate_set.unit) for coordinate_set in coordinates ] 
            else:
                self.provided_coordinates = [ units.Quantity(numpy.array(coordinates / coordinates.unit), coordinates.unit) ]            
                    

    def _initialize_netcdf(self):
        """
        Initialize NetCDF file for storage.
        
        """    
        
        # Open NetCDF 4 file for writing.
        #ncfile = netcdf.NetCDFFile(self.store_filename, 'w', version=2)
        #ncfile = netcdf.Dataset(self.store_filename, 'w', version=2)        
        ncfile = netcdf.Dataset(self.store_filename, 'w', version='NETCDF4')
        
        # Create dimensions.
        ncfile.createDimension('iteration', 0) # unlimited number of iterations
        ncfile.createDimension('replica', self.nreplicas) # number of replicas
        ncfile.createDimension('atom', self.natoms) # number of atoms in system
        ncfile.createDimension('spatial', 3) # number of spatial dimensions

        # Set global attributes.
        setattr(ncfile, 'title', self.title)
        setattr(ncfile, 'application', 'YANK')
        setattr(ncfile, 'program', 'yank.py')
        setattr(ncfile, 'programVersion', __version__)
        setattr(ncfile, 'Conventions', 'YANK')
        setattr(ncfile, 'ConventionVersion', '0.1')
        
        # Create variables.
        ncvar_positions = ncfile.createVariable('positions', 'f', ('iteration','replica','atom','spatial'))
        ncvar_states    = ncfile.createVariable('states', 'i', ('iteration','replica'))
        ncvar_energies  = ncfile.createVariable('energies', 'f', ('iteration','replica','replica'))        
        ncvar_proposed  = ncfile.createVariable('proposed', 'l', ('iteration','replica','replica'))
        ncvar_accepted  = ncfile.createVariable('accepted', 'l', ('iteration','replica','replica'))                
        ncvar_box_vectors = ncfile.createVariable('box_vectors', 'f', ('iteration','replica','spatial','spatial'))        
        ncvar_volumes  = ncfile.createVariable('volumes', 'f', ('iteration','replica'))
        
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
        setattr(ncvar_states,    "long_name", "states[iteration][replica] is the state index (0..nstates-1) of replica 'replica' of iteration 'iteration'.")
        setattr(ncvar_energies,  "long_name", "energies[iteration][replica][state] is the reduced (unitless) energy of replica 'replica' from iteration 'iteration' evaluated at state 'state'.")
        setattr(ncvar_proposed,  "long_name", "proposed[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")
        setattr(ncvar_accepted,  "long_name", "accepted[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")
        setattr(ncvar_box_vectors, "long_name", "box_vectors[iteration][replica][i][j] is dimension j of box vector i for replica 'replica' from iteration 'iteration-1'.")
        setattr(ncvar_volumes, "long_name", "volume[iteration][replica] is the box volume for replica 'replica' from iteration 'iteration-1'.")

        # Create timestamp variable.
        ncvar_timestamp = ncfile.createVariable('timestamp', str, ('iteration',))

        # Create group for performance statistics.
        ncgrp_timings = ncfile.createGroup('timings')
        ncvar_iteration_time = ncgrp_timings.createVariable('iteration', 'f', ('iteration',)) # total iteration time (seconds)
        ncvar_iteration_time = ncgrp_timings.createVariable('mixing', 'f', ('iteration',)) # time for mixing
        ncvar_iteration_time = ncgrp_timings.createVariable('propagate', 'f', ('iteration','replica')) # total time to propagate each replica
        
        # Store thermodynamic states.
        self._store_thermodynamic_states(ncfile)

        # Store run options
        self._store_options(ncfile)

        # Store metadata.
        if self.metadata:
            self._store_metadata(ncfile, 'metadata', self.metadata)

        # Force sync to disk to avoid data loss.
        ncfile.sync()

        # Store netcdf file handle.
        self.ncfile = ncfile
        
        return
    
    @time_and_print
    def _write_iteration_netcdf(self):
        """
        Write positions, states, and energies of current iteration to NetCDF file.
        
        """

        initial_time = time.time()

        # Store replica positions.
        for replica_index in range(self.nstates):
            coordinates = self.replica_coordinates[replica_index]
            x = coordinates / units.nanometers
            self.ncfile.variables['positions'][self.iteration,replica_index,:,:] = x[:,:]
            
        # Store box vectors and volume.
        for replica_index in range(self.nstates):
            state_index = self.replica_states[replica_index]
            state = self.states[state_index]
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

        # Force sync to disk to avoid data loss.
        presync_time = time.time()
        self.ncfile.sync()


    def _store_thermodynamic_states(self, ncfile):
        """
        Store the thermodynamic states in a NetCDF file.

        """
        if self.verbose: print "Storing thermodynamic states in NetCDF file..."
        initial_time = time.time()
            
        # Create a group to store state information.
        ncgrp_stateinfo = ncfile.createGroup('thermodynamic_states')

        # Get number of states.
        ncvar_nstates = ncgrp_stateinfo.createVariable('nstates', int)
        ncvar_nstates.assignValue(self.nstates)


        # Temperatures.
        ncvar_temperatures = ncgrp_stateinfo.createVariable('temperatures', 'f', ('replica',))
        setattr(ncvar_temperatures, 'units', 'K')
        setattr(ncvar_temperatures, 'long_name', "temperatures[state] is the temperature of thermodynamic state 'state'")
        for state_index in range(self.nstates):
            ncvar_temperatures[state_index] = self.states[state_index].temperature / units.kelvin

        # Pressures.
        if self.states[0].pressure is not None:
            ncvar_temperatures = ncgrp_stateinfo.createVariable('pressures', 'f', ('replica',))
            setattr(ncvar_temperatures, 'units', 'atm')
            setattr(ncvar_temperatures, 'long_name', "pressures[state] is the external pressure of thermodynamic state 'state'")
            for state_index in range(self.nstates):
                ncvar_temperatures[state_index] = self.states[state_index].pressure / units.atmospheres                

        # TODO: Store other thermodynamic variables store in ThermodynamicState?  Generalize?
                
        # Systems.
        ncvar_serialized_states = ncgrp_stateinfo.createVariable('systems', str, ('replica',), zlib=True)
        setattr(ncvar_serialized_states, 'long_name', "systems[state] is the serialized OpenMM System corresponding to the thermodynamic state 'state'")
        for state_index in range(self.nstates):
            if self.verbose: print "Serializing state %d..." % state_index
            serialized = self.states[state_index].system.__getstate__()
            if self.verbose: print "Serialized state is %d B | %.3f KB | %.3f MB" % (len(serialized), len(serialized) / 1024.0, len(serialized) / 1024.0 / 1024.0)
            ncvar_serialized_states[state_index] = serialized
        final_time = time.time() 
        elapsed_time = final_time - initial_time

        if self.verbose: print "Serializing thermodynamic states took %.3f s." % elapsed_time
        
        return

    def _restore_thermodynamic_states(self, ncfile):
        """
        Restore the thermodynamic states from a NetCDF file.

        """
        if self.verbose_root: print "Restoring thermodynamic states from NetCDF file..."
           
        # Make sure this NetCDF file contains thermodynamic state information.
        if not 'thermodynamic_states' in ncfile.groups:
            # Not found, signal failure.
            return False

        # Create a group to store state information.
        ncgrp_stateinfo = ncfile.groups['thermodynamic_states']

        # Get number of states.
        self.nstates = int(ncgrp_stateinfo.variables['nstates'][0])

        # Read state information.
        self.states = list()
        for state_index in range(self.nstates):
            # Populate a new ThermodynamicState object.
            state = ThermodynamicState()
            # Read temperature.
            state.temperature = ncgrp_stateinfo.variables['temperatures'][state_index] * units.kelvin
            # Read pressure, if present.
            if 'pressures' in ncgrp_stateinfo.variables:
                state.pressure = ncgrp_stateinfo.variables['pressures'][state_index] * units.atmospheres
            # Reconstitute System object.
            state.system = self.mm.System() 
            state.system.__setstate__(str(ncgrp_stateinfo.variables['systems'][state_index]))
            # Store state.
            self.states.append(state)

        return True

    def _store_options(self, ncfile):
        """
        Store run parameters in NetCDF file.

        """

        if self.verbose_root: print "Storing run parameters in NetCDF file..."

        # Create scalar dimension if not already present.
        if 'scalar' not in ncfile.dimensions:
            ncfile.createDimension('scalar', 1) # scalar dimension
        
        # Create a group to store state information.
        ncgrp_options = ncfile.createGroup('options')

        # Store run parameters.
        for option_name in self.options_to_store:
            # Get option value.
            option_value = getattr(self, option_name)
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
            if self.verbose_root: print "Storing option: %s -> %s (type: %s)" % (option_name, option_value, str(option_type))
            if type(option_value) == str:
                ncvar = ncgrp_options.createVariable(option_name, type(option_value), 'scalar')
                packed_data = numpy.empty(1, 'O')
                packed_data[0] = option_value
                ncvar[:] = packed_data
            else:
                ncvar = ncgrp_options.createVariable(option_name, type(option_value))
                ncvar.assignValue(option_value)
            if option_unit: setattr(ncvar, 'units', str(option_unit))
            setattr(ncvar, 'type', option_type.__name__) 

        return

    def _restore_options(self, ncfile):
        """
        Restore run parameters from NetCDF file.
        """

        if self.verbose_root: print "Attempting to restore options from NetCDF file..."

        # Make sure this NetCDF file contains option information
        if not 'options' in ncfile.groups:
            # Not found, signal failure.
            return False

        # Find the group.
        ncgrp_options = ncfile.groups['options']

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
            if self.verbose_root: print "Restoring option: %s -> %s (type: %s)" % (option_name, str(option_value), type(option_value))
            setattr(self, option_name, option_value)
            
        # Signal success.
        return True

    def _store_metadata(self, ncfile, groupname, metadata):
        """
        Store metadata in NetCDF file.
        
        """

        # Create group.
        ncgrp = ncfile.createGroup(groupname)

        # Store metadata.
        for (key, value) in metadata.iteritems():
            # TODO: Handle more sophisticated types.
            ncvar = ncgrp.createVariable(key, type(value))
            ncvar.assignValue(value)

        return

    def _resume_from_netcdf(self):
        """
        Resume execution by reading current positions and energies from a NetCDF file.
        
        """

        # Open NetCDF file for reading
        if self.verbose: print "Reading NetCDF file '%s'..." % self.store_filename
        #ncfile = netcdf.NetCDFFile(self.store_filename, 'r') # Scientific.IO.NetCDF
        ncfile = netcdf.Dataset(self.store_filename, 'r') # netCDF4
        
        # TODO: Perform sanity check on file before resuming

        # Get current dimensions.
        self.iteration = ncfile.variables['positions'].shape[0] - 1 
        self.nstates = ncfile.variables['positions'].shape[1]
        self.natoms = ncfile.variables['positions'].shape[2]
        if self.verbose: print "iteration = %d, nstates = %d, natoms = %d" % (self.iteration, self.nstates, self.natoms)

        # Restore positions.
        self.replica_coordinates = list()
        for replica_index in range(self.nstates):
            x = ncfile.variables['positions'][self.iteration,replica_index,:,:].astype(numpy.float64).copy()
            coordinates = units.Quantity(x, units.nanometers)
            self.replica_coordinates.append(coordinates)
        
        # Restore box vectors.
        self.replica_box_vectors = list()
        for replica_index in range(self.nstates):
            x = ncfile.variables['box_vectors'][self.iteration,replica_index,:,:].astype(numpy.float64).copy()
            box_vectors = units.Quantity(x, units.nanometers)
            self.replica_box_vectors.append(box_vectors)

        # Restore state information.
        self.replica_states = ncfile.variables['states'][self.iteration,:].copy()

        # Restore energies.
        self.u_kl = ncfile.variables['energies'][self.iteration,:,:].copy()
        
        # Close NetCDF file.
        ncfile.close()        

        # We will work on the next iteration.
        self.iteration += 1

        if (self.mpicomm is None) or (self.mpicomm.rank == 0):
            # Reopen NetCDF file for appending, and maintain handle.
            self.ncfile = netcdf.Dataset(self.store_filename, 'a')
        else:
            self.ncfile = None
        
        return
