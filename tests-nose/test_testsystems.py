import numpy as np
import simtk.unit as u
from repex.thermodynamics import ThermodynamicState
from repex.parallel_tempering import ParallelTempering
from repex import testsystems
from repex import dummympi
import tempfile
from mdtraj.testing import eq, skipif
import logging

def test_doctest():
    import doctest
    from repex import testsystems
    doctest.testmod(testsystems)

def test_minimizer_all_testsystems():
    testsystem_classes = testsystems.TestSystem.__subclasses__()
    
    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__
        logging.info("Testing minimization with testsystem %s" % class_name)
        
        testsystem = testsystem_class()

        from repex import mcmc
        sampler_state = mcmc.SamplerState(testsystem.system, testsystem.positions)

        # Check if NaN.
        if np.isnan(sampler_state.potential_energy / u.kilocalories_per_mole):
            raise Exception("Initial energy of system %s yielded NaN" % class_name)        

        # Minimize
        sampler_state.minimize()
        
        # Check if NaN.
        if np.isnan(sampler_state.potential_energy / u.kilocalories_per_mole):
            raise Exception("Minimization of system %s yielded NaN" % class_name)        

def test_properties_all_testsystems():
    testsystem_classes = testsystems.TestSystem.__subclasses__()
    print "Testing analytical property computation:"
    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__
        testsystem = testsystem_class()
        property_list = testsystem.analytical_properties
        state = ThermodynamicState(temperature=300.0*u.kelvin, pressure=1.0*u.atmosphere, system=testsystem.system)
        if len(property_list) > 0:
            for property_name in property_list:
                method = getattr(testsystem, 'get_' + property_name)
                logging.info("%32s . %32s : %32s" % (class_name, property_name, str(method(state))))


fast_testsystems = ["HarmonicOscillator", "PowerOscillator", "Diatom", "ConstraintCoupledHarmonicOscillator", "HarmonicOscillatorArray", "SodiumChlorideCrystal", "LennardJonesCluster", "LennardJonesFluid", "IdealGas", "AlanineDipeptideVacuum"]

def test_parallel_tempering_all_testsystems():
    T_min = 1.0 * u.kelvin
    T_max = 10.0 * u.kelvin
    n_temps = 3

    testsystem_classes = testsystems.TestSystem.__subclasses__()
    
    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__
        if class_name in fast_testsystems:
            logging.info("Testing replica exchange with testsystem %s" % class_name)
        else:
            logging.info("Skipping replica exchange with testsystem %s." % class_name)
            continue
            
        testsystem = testsystem_class()
        
        system = testsystem.system
        positions = [testsystem.positions] * n_temps
        
        nc_filename = tempfile.mkdtemp() + "/out.nc"
        parameters = {"number_of_iterations":3}
        replica_exchange = ParallelTempering.create(system, positions, nc_filename, T_min=T_min, T_max=T_max, n_temps=n_temps, parameters=parameters)
        replica_exchange.run()
