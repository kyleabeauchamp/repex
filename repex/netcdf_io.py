import os

import numpy as np

import simtk.unit as units

import netCDF4 as netcdf

from thermodynamics import ThermodynamicState
from utils import str_to_system
from version import version as __version__
from analysis import Analyzer

import logging
logger = logging.getLogger(__name__)


class NetCDFDatabase(Analyzer):
    """A netCDF based database for Repex simulations.
    
    
    Parameters
    ----------
    filename : str
        The filename of the database.  Use ".nc" as the suffix.
    thermodynamic_states : list(ThermodynamicState), default=None
        A list of thermodynamic states for the replica exchange run.  
        These will be saved to disk.  If None, will be loaded from disk.
    positions : list(simtk.unit)
        A list of coordinates for each repex slot.  If None, will be 
        loaded from disk.

    Notes
    -----
    
    This class inherits the Analyzer class, which provides member functions
    related to MBar and trajectory analysis.
    
    """
    def __init__(self, filename, thermodynamic_states=None, positions=None):

        # Check if netcdf file exists.
        resume = os.path.exists(filename) and (os.path.getsize(filename) > 0)
        
        if resume:
            assert thermodynamic_states is None and positions is None, "Cannot input thermodynamic_states and positions if you are resuming from disk."
            self.ncfile = netcdf.Dataset(filename, 'a', version='NETCDF4')

        else:
            assert thermodynamic_states is not None and positions is not None, "Must input thermodynamic_states and coordinates if no existing database."
            assert len(thermodynamic_states) == len(positions), "Must have same number of thermodynamic_states and coordinate sets."
            
            self.ncfile = netcdf.Dataset(filename, 'w', version='NETCDF4')
        
        self.title = "No Title."
        
        if resume:
            logger.info("Attempting to resume by reading thermodynamic states and options...")
            self.parameters = self._load_parameters()
        else:
            self._initialize_netcdf(thermodynamic_states, positions)

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in self.thermodynamic_states:
            if not state.is_compatible_with(self.thermodynamic_states[0]):
                raise ValueError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")


    def _initialize_netcdf(self, thermodynamic_states, positions):
        """Initialize NetCDF file for storage by allocating arrays on disk.
        
        Parameters
        ----------

        thermodynamic_states : list(ThermodynamicState), default=None
            A list of thermodynamic states for the replica exchange run.  
            These will be saved to disk.  If None, will be loaded from disk.
        positions : list(simtk.unit)
            A list of coordinates for each repex slot.  If None, will be 
            loaded from disk.
        
        """
        
        n_replicas = len(thermodynamic_states)
        n_atoms = len(positions[0])
        n_states = len(thermodynamic_states)
        
        
        # Create dimensions.
        self.ncfile.createDimension('iteration', 0) # unlimited number of iterations
        self.ncfile.createDimension('replica', n_replicas) # number of replicas
        self.ncfile.createDimension('atom', n_atoms) # number of atoms in system
        self.ncfile.createDimension('spatial', 3) # number of spatial dimensions

        # Set global attributes.
        self.ncfile.title = self.title
        self.ncfile.application = 'Repex'
        #self.ncfile.program = 'yank.py'
        self.ncfile.programVersion = __version__
        self.ncfile.Conventions = 'Repex'
        self.ncfile.ConventionVersion = '0.1'
        
        # Allocate variable for the ReplicaExchange (sub)class name.
        self.ncfile.repex_classname = "Unknown"
        
        # Create variables.
        ncvar_positions = self.ncfile.createVariable('positions', 'f', ('iteration','replica','atom','spatial'))
        ncvar_states    = self.ncfile.createVariable('states', 'i', ('iteration','replica'))
        ncvar_energies  = self.ncfile.createVariable('energies', 'f', ('iteration','replica','replica'))        
        ncvar_proposed  = self.ncfile.createVariable('proposed', 'l', ('iteration','replica','replica'))
        ncvar_accepted  = self.ncfile.createVariable('accepted', 'l', ('iteration','replica','replica'))                
        ncvar_box_vectors = self.ncfile.createVariable('box_vectors', 'f', ('iteration','replica','spatial','spatial'))        
        ncvar_volumes  = self.ncfile.createVariable('volumes', 'f', ('iteration','replica'))
        
        # Define units for variables.
        ncvar_positions.units = 'nm'
        ncvar_states.units = 'none'
        ncvar_energies.units = 'kT'
        ncvar_proposed.units = 'none'
        ncvar_accepted.units = 'none'
        ncvar_box_vectors.units = 'nm'
        ncvar_volumes.units = 'nm**3'

        # Define long (human-readable) names for variables.
        ncvar_positions.long_name = "positions[iteration][replica][atom][spatial] is position of coordinate 'spatial' of atom 'atom' from replica 'replica' for iteration 'iteration'."
        ncvar_states.long_name = "states[iteration][replica] is the state index (0..n_states-1) of replica 'replica' of iteration 'iteration'."
        ncvar_energies.long_name = "energies[iteration][replica][state] is the reduced (unitless) energy of replica 'replica' from iteration 'iteration' evaluated at state 'state'."
        ncvar_proposed.long_name = "proposed[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'."
        ncvar_accepted.long_name = "accepted[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'."
        ncvar_box_vectors.long_name = "box_vectors[iteration][replica][i][j] is dimension j of box vector i for replica 'replica' from iteration 'iteration-1'."
        ncvar_volumes.long_name = "volume[iteration][replica] is the box volume for replica 'replica' from iteration 'iteration-1'."

        # Create timestamp variable.
        ncvar_timestamp = self.ncfile.createVariable('timestamp', "f", ('iteration',))

        # Create group for performance statistics.
        ncgrp_timings = self.ncfile.createGroup('timings')
        ncvar_iteration_time = ncgrp_timings.createVariable('iteration', 'f', ('iteration',)) # total iteration time (seconds)
        ncvar_iteration_time = ncgrp_timings.createVariable('mixing', 'f', ('iteration',)) # time for mixing
        ncvar_iteration_time = ncgrp_timings.createVariable('propagate', 'f', ('iteration','replica')) # total time to propagate each replica
        
        # Store thermodynamic states using @property setter
        self.thermodynamic_states = thermodynamic_states

        # Force sync to disk to avoid data loss.
        self.ncfile.sync()


    @property
    def thermodynamic_states(self):
        """Return the thermodynamic states from a NetCDF file.
        
        Returns
        -------
        thermodynamic_states : list
            List of thermodynamic states for repex.
        """
        
        logging.debug("Restoring thermodynamic states from NetCDF file...")
           
        # Make sure this NetCDF file contains thermodynamic state information.
        if not 'thermodynamic_states' in self.ncfile.groups:
            raise(ValueError("thermodynamics_states not found in database!"))

        # Create a group to store state information.
        ncgrp_stateinfo = self.ncfile.groups['thermodynamic_states']

        # Get number of states.
        n_states = int(ncgrp_stateinfo.variables['n_states'][:])

        # Read state information.
        thermodynamic_states = list()
        for state_index in range(n_states):
                        
            temperature = ncgrp_stateinfo.variables['temperatures'][state_index] * units.kelvin
            
            pressure = None
            if 'pressures' in ncgrp_stateinfo.variables:
                pressure = ncgrp_stateinfo.variables['pressures'][state_index] * units.atmospheres
            
            # Reconstitute System object.
            system = str(ncgrp_stateinfo.variables['systems'][state_index])
            system = str_to_system(system)
            
            state = ThermodynamicState(system=system, temperature=temperature, pressure=pressure)

            thermodynamic_states.append(state)
        
        return thermodynamic_states


    @thermodynamic_states.setter
    def thermodynamic_states(self, thermodynamic_states):
        """Store the thermodynamic states in a NetCDF file.
        """
        logger.debug("Storing thermodynamic states in NetCDF file...")

        if self.ncfile.groups.has_key("thermodynamic_states"):
            raise(IOError("Thermodynamic states have already been set!"))

        # Create a group to store state information.
        ncgrp_stateinfo = self.ncfile.createGroup('thermodynamic_states')

        # Get number of states.
        ncvar_n_states = ncgrp_stateinfo.createVariable('n_states', int)
        ncvar_n_states.assignValue(self.n_states)


        # Temperatures.
        ncvar_temperatures = ncgrp_stateinfo.createVariable('temperatures', 'f', ('replica',))
        ncvar_temperatures.units = 'K'
        ncvar_temperatures.long_name = "temperatures[state] is the temperature of thermodynamic state 'state'"
        for state_index in range(self.n_states):
            ncvar_temperatures[state_index] = thermodynamic_states[state_index].temperature / units.kelvin

        # Pressures.
        if thermodynamic_states[0].pressure is not None:
            ncvar_temperatures = ncgrp_stateinfo.createVariable('pressures', 'f', ('replica',))
            ncvar_temperatures.units = 'atm'
            ncvar_temperatures.long_name = "pressures[state] is the external pressure of thermodynamic state 'state'"
            for state_index in range(self.n_states):
                ncvar_temperatures[state_index] = thermodynamic_states[state_index].pressure / units.atmospheres

        # TODO: Store other thermodynamic variables store in ThermodynamicState?  Generalize?
                
        # Systems.
        ncvar_serialized_states = ncgrp_stateinfo.createVariable('systems', str, ('replica',), zlib=True)
        ncvar_serialized_states.long_name = "systems[state] is the serialized OpenMM System corresponding to the thermodynamic state 'state'"
        for state_index in range(self.n_states):
            logger.debug("Serializing state %d..." % state_index)
            serialized = thermodynamic_states[state_index].system.__getstate__()
            logger.debug("Serialized state is %d B | %.3f KB | %.3f MB" % (len(serialized), len(serialized) / 1024.0, len(serialized) / 1024.0 / 1024.0))
            ncvar_serialized_states[state_index] = serialized


    def store_parameters(self, parameters):
        """Store run parameters in NetCDF file.
        
        Parameters
        ----------
        parameters : dict
            A dict containing ALL run parameters
            
        Notes
        -----
        
        The user supplies a dictionary of run parameters to Repex.  
        Repex (or a subclass) then fills in any missing parameters.  
        This dictionary is the correct input here.  Eventually, this dictionary
        will be converted to a namedtuple, at which point it is immutable.  
        
        """

        logger.debug("Storing run parameters in NetCDF file...")

        # Create scalar dimension if not already present.
        if 'scalar' not in self.ncfile.dimensions:
            self.ncfile.createDimension('scalar', 1) # scalar dimension
        
        # Create a group to store state information.
        ncgrp_options = self.ncfile.createGroup('options')

        for option_name, option_value in parameters.iteritems():
            self._store_parameter(option_name, option_value)

    def _store_parameter(self, option_name, option_value):
        # If Quantity, strip off units first.
        option_unit = None
        if type(option_value) == units.Quantity:
            option_unit = option_value.unit
            option_value = option_value / option_unit
        # Store the Python type.
        option_type = type(option_value)
        
        # Handle NoneType
        if type(option_value) == type(None):
            option_value = ""
        # Handle booleans
        if type(option_value) == bool:
            option_value = int(option_value)
        # Store the variable.
        logger.debug("Storing option: %s -> %s (type: %s)" % (option_name, option_value, str(option_type)))
        if type(option_value) == str:
            if option_name in self.ncfile.groups['options'].variables:
                ncvar = self.ncfile.groups['options'].variables[option_name]
            else:
                ncvar = self.ncfile.groups['options'].createVariable(option_name, type(option_value), 'scalar')
            packed_data = np.empty(1, 'O')
            packed_data[0] = option_value
            ncvar[:] = packed_data
        else:
            if option_name in self.ncfile.groups['options'].variables:
                ncvar = self.ncfile.groups['options'].variables[option_name]
            else:
                ncvar = self.ncfile.groups['options'].createVariable(option_name, type(option_value))
            ncvar.assignValue(option_value)
        if option_unit: setattr(ncvar, 'units', str(option_unit))
        setattr(ncvar, 'type', option_type.__name__)

    def _load_parameter(self, option_name):
        """Load and return a single run parameter from netCDF file."""
        
        option_ncvar = self.ncfile.groups['options'].variables[option_name]
        # Get option value.
        
        try:
            option_value = option_ncvar[0]  # If the option is a np.array(['string']), this turns it into a scalar.
        except IndexError:
            option_value = option_ncvar[:]  # If the option is a float or int, this works and gives a scalar

        # Cast to Python type.
        type_name = getattr(option_ncvar, 'type')
        
        if type_name == "NoneType":
            return None
        
        else:
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
            return option_value 

    def _load_parameters(self):
        """Return a dictionary of options from a loaded NetCDF file.
        """
        
        logger.debug("Attempting to restore options from NetCDF file...")
        
        # Make sure this NetCDF file contains option information
        if not 'options' in self.ncfile.groups:
            # Not found, signal failure.
            raise(ValueError("Options not found in NetCDF file!"))

        ncgrp_options = self.ncfile.groups['options']
        
        options = {}
        for option_name in ncgrp_options.variables.keys():
            options[option_name] = self._load_parameter(option_name)

        return options


    def write(self, key, value, iteration, sync=True):
        """Write a variable to the database and sync."""
        
        self.ncfile.variables[key][iteration] = value
        
        if sync == True:
            self.sync()


    def sync(self):
        """Sync the database."""
        self.ncfile.sync()
    
    def _finalize(self):
        self.sync()

    def __del__(self):
        self._finalize()
        self.ncfile.close()

    @property
    def positions(self):
        """Return the positions."""
        return self.ncfile.variables['positions']

    @property
    def box_vectors(self):
        """Return the box vectors."""
        return self.ncfile.variables['box_vectors']
                    
    @property
    def volumes(self):
        """Return the volumes."""
        return self.ncfile.variables['volumes']

    @property
    def states(self):
        """Return the state indices."""
        return self.ncfile.variables['states']
        
    @property
    def energies(self):
        """Return the energies."""
        return self.ncfile.variables['energies']
                                    
    @property
    def proposed(self):
        """Return the proposed moves."""
        return self.ncfile.variables['proposed']
        
    @property
    def accepted(self):
        """Return the accepted moves."""
        return self.ncfile.variables['accepted']

    @property
    def timestamp(self):
        """Return the timestamp."""
        return self.ncfile.variables['timestamp']

    @property
    def repex_classname(self):
        return self.ncfile.repex_classname

    @property
    def last_proposed(self):
        iteration = self.last_iteration
        return self.proposed[iteration]

    @property
    def last_accepted(self):
        iteration = self.last_iteration
        return self.accepted[iteration]

    @property
    def last_box_vectors(self):
        """Return data from the last iteration saved in a database.
        
        Returns
        -------
        
        replica_coordinates : list
            The coordinates of each replica
        """
        
        iteration = self.last_iteration
        
        replica_box_vectors = list()
        for replica_index in range(self.n_states):
            x = self.ncfile.variables['box_vectors'][iteration, replica_index,:,:].astype(np.float64).copy()
            box_vectors = units.Quantity(x, units.nanometers)
            replica_box_vectors.append(box_vectors)        
        
        return replica_box_vectors
        

    @property
    def last_u_kl(self):
        """Return data from the last iteration saved in a database.
        
        Returns
        -------
        
        replica_coordinates : list
            The coordinates of each replica
        """
        iteration = self.last_iteration
        return self.energies[iteration,:].copy()

    
    @property
    def last_iteration(self):
        """Return data from the last iteration saved in a database.
        
        Returns
        -------
        
        replica_coordinates : list
            The coordinates of each replica
        """
        return self.positions.shape[0] - 1


    @property
    def last_positions(self):
        """Return positions from the last iteration saved in a database.
        
        Returns
        -------
        
        replica_coordinates : list
            The coordinates of each replica
            
        Notes
        -----
        Formatted for input to Repex
        """    
        
        iteration = self.last_iteration
        
        replica_coordinates = list()
        for replica_index in range(self.n_states):
            x = self.positions[iteration, replica_index, :, :].astype(np.float64).copy()
            coordinates = units.Quantity(x, units.nanometers)
            replica_coordinates.append(coordinates)
        
        return replica_coordinates


    @property
    def last_replica_states(self):
        
        iteration = self.last_iteration
        return self.states[iteration,:].copy()


    @property
    def n_states(self):
        return self.positions.shape[1]


    @property
    def n_atoms(self):
        return self.positions.shape[2]
