import numpy

import simtk.openmm as openmm
import simtk.unit as units

import repex.testsystems

from pymbar import timeseries

analytical_testsystems = ["HarmonicOscillatorArray", "IdealGas"] # systems with analytical results to test

NSIGMA_CUTOFF = 6.0 # cutoff for significance testing

def test_mcmc_expectations():
    # Select system:
    for system_name in analytical_testsystems:
        testsystem_class = getattr(repex.testsystems, system_name)
        testsystem = testsystem_class()
        test_mcmc_expectation(testsystem)

def test_mcmc_expectation(testsystem):
    # Test settings.
    temperature = 298.0 * units.kelvin
    pressure = 1.0 * units.atmospheres
    nequil = 10 # number of equilibration iterations
    niterations = 100 # number of production iterations
    platform_name = "CUDA" # platform to try

    # Retrieve system and positions.
    [system, positions] = [testsystem.system, testsystem.positions]
    
    # Compute properties.
    kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    ndof = 3*system.getNumParticles() - system.getNumConstraints()

    # Select platform manually.
    platform = openmm.Platform.getPlatformByName(platform_name)

    # Create MCMC move set.
    from repex.mcmc import HMCMove, GHMCMove, LangevinDynamicsMove, MonteCarloBarostatMove
    move_set = [ GHMCMove(), MonteCarloBarostatMove() ]

    # Create thermodynamic state
    from repex.thermodynamics import ThermodynamicState
    thermodynamic_state = ThermodynamicState(system=testsystem.system, temperature=temperature, pressure=pressure)

    # Create MCMC sampler.
    from repex.mcmc import MCMCSampler
    sampler = MCMCSampler(thermodynamic_state, move_set=move_set, platform=platform)

    # Create sampler state.
    from repex.mcmc import MCMCSamplerState
    sampler_state = MCMCSamplerState(system=testsystem.system, positions=testsystem.positions)

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
    for iteration in range(niterations):
        #print "iteration %d / %d" % (iteration, niterations)

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

    # Compute expected statistics.
    if ('potential_expectation' in dir(testsystem)):
        # Skip this check if the std dev is zero.
        skip_test = False
        if ('potential_standard_deviation' in dir(testsystem)):
            potential_standard_deviation = testsystem.get_potential_standard_deviation() / kT
            if potential_standard_deviation == 0.0:
                skip_test = True
        if not skip_test:
            potential_expectation = testsystem.get_potential_expectation() / kT
            potential_mean = potential_n.mean()            
            g = timeseries.statisticalInefficiency(potential_n, fast=True)
            dpotential_mean = potential_n.std() / numpy.sqrt(niterations / g)
            potential_error = potential_mean - potential_expectation
            nsigma = abs(potential_error) / dpotential_mean
            # Test if error exceeds threshhold.
            if (nsigma > NSIGMA_CUTOFF):
                print "Potential energy expectation"
                print "observed %10.5 +- %10.5f kT | expected %10.5f | error %10.5f +- %10.5f (%.1f sigma)" % (potential_mean, dpotential_mean, potential_expectation, potential_error, dpotential_mean, nsigma)
                print "TEST FAILED"
                print "----------------------------------------------------------------------------"


#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    test_mcmc_expectations()
