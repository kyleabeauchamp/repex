import numpy

import simtk.openmm as openmm
import simtk.unit as units

import repex.testsystems

from pymbar import timeseries

from repex.mcmc import HMCMove, GHMCMove, LangevinDynamicsMove, MonteCarloBarostatMove

# Test various combinations of systems and MCMC schemes
analytical_testsystems = [
    ( "HarmonicOscillatorArray", [ HMCMove() ] ), 
    ( "HarmonicOscillatorArray", [ GHMCMove() ]),
    ("HarmonicOscillatorArray", { GHMCMove() : 0.5, HMCMove() : 0.5 }),
    ("HarmonicOscillatorArray", [ LangevinDynamicsMove() ]),
    ("IdealGas", [ GHMCMove(), MonteCarloBarostatMove() ])
    ]

NSIGMA_CUTOFF = 6.0 # cutoff for significance testing

debug = False # set to True only for manual debugging of this nose test

def test_mcmc_expectations():
    # Select system:
    for [system_name, move_set] in analytical_testsystems:
        testsystem_class = getattr(repex.testsystems, system_name)
        testsystem = testsystem_class()
        subtest_mcmc_expectation(testsystem, move_set)

def subtest_mcmc_expectation(testsystem, move_set):
    if debug: 
        print testsystem.__class__.__name__
        print str(move_set)

    # Test settings.
    temperature = 298.0 * units.kelvin
    pressure = 1.0 * units.atmospheres
    nequil = 10 # number of equilibration iterations
    niterations = 100 # number of production iterations

    # Retrieve system and positions.
    [system, positions] = [testsystem.system, testsystem.positions]
    
    # Compute properties.
    kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    ndof = 3*system.getNumParticles() - system.getNumConstraints()

    # Create thermodynamic state
    from repex.thermodynamics import ThermodynamicState
    thermodynamic_state = ThermodynamicState(system=testsystem.system, temperature=temperature, pressure=pressure)

    # Create MCMC sampler.
    from repex.mcmc import MCMCSampler
    sampler = MCMCSampler(thermodynamic_state, move_set=move_set)

    # Create sampler state.
    from repex.mcmc import SamplerState
    sampler_state = SamplerState(system=testsystem.system, positions=testsystem.positions)

    # Equilibrate
    for iteration in range(nequil):
        #print "equilibration iteration %d / %d" % (iteration, nequil)

        # Update sampler state.
        sampler_state = sampler.run(sampler_state, 1)

    # Accumulate statistics.
    x_n = numpy.zeros([niterations], numpy.float64) # x_n[i] is the x position of atom 1 after iteration i, in angstroms
    potential_n = numpy.zeros([niterations], numpy.float64) # potential_n[i] is the potential energy after iteration i, in kT
    kinetic_n = numpy.zeros([niterations], numpy.float64) # kinetic_n[i] is the kinetic energy after iteration i, in kT
    temperature_n = numpy.zeros([niterations], numpy.float64) # temperature_n[i] is the instantaneous kinetic temperature from iteration i, in K
    volume_n = numpy.zeros([niterations], numpy.float64) # volume_n[i] is the volume from iteration i, in K
    for iteration in range(niterations):
        if debug: print "iteration %d / %d" % (iteration, niterations)

        # Update sampler state.
        sampler_state = sampler.run(sampler_state, 1)

        # Get statistics.
        potential_energy = sampler_state.potential_energy
        kinetic_energy = sampler_state.kinetic_energy
        total_energy = sampler_state.total_energy
        instantaneous_temperature = kinetic_energy * 2.0 / ndof / (units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA)
        volume = sampler_state.volume
        
        #print "potential %8.1f kT | kinetic %8.1f kT | total %8.1f kT | volume %8.3f nm^3 | instantaneous temperature: %8.1f K" % (potential_energy/kT, kinetic_energy/kT, total_energy/kT, volume/(units.nanometers**3), instantaneous_temperature/units.kelvin)

        # Accumulate statistics.
        x_n[iteration] = sampler_state.positions[0,0] / units.angstroms
        potential_n[iteration] = potential_energy / kT
        kinetic_n[iteration] = kinetic_energy / kT
        temperature_n[iteration] = instantaneous_temperature / units.kelvin
        volume_n[iteration] = volume / (units.nanometers**3)

    # Compute expected statistics.
    if ('get_potential_expectation' in dir(testsystem)):
        # Skip this check if the std dev is zero.
        skip_test = False
        if (potential_n.std() == 0.0):
            skip_test = True
            if debug: print "Skipping potential test since variance is zero."
        if not skip_test:
            potential_expectation = testsystem.get_potential_expectation(thermodynamic_state) / kT
            potential_mean = potential_n.mean()            
            g = timeseries.statisticalInefficiency(potential_n, fast=True)
            dpotential_mean = potential_n.std() / numpy.sqrt(niterations / g)
            potential_error = potential_mean - potential_expectation
            nsigma = abs(potential_error) / dpotential_mean
            test_passed = True
            if (nsigma > NSIGMA_CUTOFF):
                test_passed = False

            if debug or (test_passed is False):
                print "Potential energy expectation"
                print "observed %10.5f +- %10.5f kT | expected %10.5f | error %10.5f +- %10.5f (%.1f sigma)" % (potential_mean, dpotential_mean, potential_expectation, potential_error, dpotential_mean, nsigma)
                if test_passed:
                    print "TEST PASSED"
                else:                
                    print "TEST FAILED"
                print "----------------------------------------------------------------------------"

    if ('get_volume_expectation' in dir(testsystem)):
        # Skip this check if the std dev is zero.
        skip_test = False
        if (volume_n.std() == 0.0):
            skip_test = True
            if debug: print "Skipping volume test."
        if not skip_test:
            volume_expectation = testsystem.get_volume_expectation(thermodynamic_state) / (units.nanometers**3)
            volume_mean = volume_n.mean()            
            g = timeseries.statisticalInefficiency(volume_n, fast=True)
            dvolume_mean = volume_n.std() / numpy.sqrt(niterations / g)
            volume_error = volume_mean - volume_expectation
            nsigma = abs(volume_error) / dvolume_mean
            test_passed = True
            if (nsigma > NSIGMA_CUTOFF):
                test_passed = False

            if debug or (test_passed is False):
                print "Volume expectation"
                print "observed %10.5f +- %10.5f kT | expected %10.5f | error %10.5f +- %10.5f (%.1f sigma)" % (volume_mean, dvolume_mean, volume_expectation, volume_error, dvolume_mean, nsigma)
                if test_passed:
                    print "TEST PASSED"
                else:                
                    print "TEST FAILED"
                print "----------------------------------------------------------------------------"

                

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    test_mcmc_expectations()
