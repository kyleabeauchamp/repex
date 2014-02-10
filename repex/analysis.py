import os
import numpy as np
import pandas as pd

import mdtraj as md

from pymbar import MBAR, timeseries

import logging
logger = logging.getLogger(__name__)


class Analyzer(object):
    """A MixIn class that gives MBAR and other analysis functions to the database.
    """


    def set_traj(self, traj):
        """Set a trajectory object for extracting trajectory slices.
        
        Parameters
        ----------
        
        traj : mdtraj.Trajectory
            A trajectory object whose topology is identical to your system.
        """
        if self.n_atoms != traj.n_atoms:
            raise(ValueError("Trajectory must have %d atoms, found %d!" % (self.n_atoms, traj.n_atoms)))

        self._traj = traj
        
    def get_traj(self, state_index=None, replica_index=None):
        """Extract a trajectory slice along a replica or state index.
        
        Parameters
        ----------
        state_index : int, optional
            Extract trajectory from this thermodynamic state
        replica_index : int, optional
            Extract trajectory from this replica slot.
        
        Returns
        -------
        traj : mdtraj.Trajectory
            A trajectory object containing the desired slice of data.            
        
        Notes
        -----
        You must specify exactly ONE of `state_index` or `replica_index`.
        
        This function is a memory hog.
        
        To do: allow Trajectory as input to __init__ and automatically call set_traj().
        To do: state_index code might be slow, possibly amenable to cython.
        """
        if not hasattr(self, "_traj"):
            raise(IOError("You must first specify a compatible trajectory with set_traj()."))

        if not (type(state_index) is int or type(replica_index) is int):
            raise(ValueError("Must input either state_index or replica_index (as integer)."))
        
        if state_index is not None and replica_index is not None:
            raise(ValueError("Cannot input both state_index and replica_index"))
        
        if state_index is not None and (state_index < 0 or state_index >= self.n_states):
            raise(ValueError("Input must be between 0 and %d" % self.n_states))

        if replica_index is not None and (replica_index < 0 or replica_index >= self.n_states):
            raise(ValueError("Input must be between 0 and %d" % self.n_states))
        
        if replica_index is not None:
            xyz = self.positions[:, replica_index]
            box_vectors = self.box_vectors[:, replica_index]

        if state_index is not None:
            replica_indices = self.states[:].argsort()[:, state_index]
            n = len(replica_indices)
            xyz = np.array([self.positions[i, replica_indices[i]] for i in range(n)])  # Have to do this because netcdf4py breaks numpy fancy indexing and calls it a "feature" because it looks "more like fortran".
            box_vectors = np.array([self.box_vectors[i, replica_indices[i]] for i in range(n)])

        traj = md.Trajectory(xyz, self._traj.top)
        traj.unitcell_vectors = box_vectors

        return traj


    def extract_reordered(self, trace):
        """Return a given quantity re-sorted by thermodynamic state, rather than replica.
        
        Parameters
        ----------
        trace : database property
            The netcdf variable to sort by state.

        Returns
        -------
        reordered : np.ndarray
            The reordered variable, in memory.
        
        Notes
        -----
        
        1.  This is a memory hog.
        2.  Be aware that there are funny slicing differences between netCDF and numpy.  
        This function *should* work for both ndarray and netcdf inputs, but 
        be aware.
        """
        replica_indices = self.states[:].argsort()
        n = len(replica_indices)
        reordered = np.array([trace[i, replica_indices[i]] for i in range(n)])  # Have to do this because netcdf4py breaks numpy fancy indexing and calls it a "feature" because it looks "more like fortran".
        
        return reordered


    def check_energies(self):
        """Examine energy history for signs of instability (nans)."""

        # Get current dimensions.
        n_iterations = self.energies.shape[0]

        logger.info("Reading energies...")
        u_kln_replica = np.zeros([self.n_states, self.n_states, n_iterations], np.float64)
        for n in range(n_iterations):
            u_kln_replica[:, :, n] = self.energies[n]
        logger.info("Done.")

        logger.info("Deconvoluting replicas...")
        u_kln = np.zeros([self.n_states, self.n_states, n_iterations], np.float64)
        for iteration in range(n_iterations):
            state_indices = self.states[iteration]
            u_kln[state_indices, :, iteration] = self.energies[iteration]
        logger.info("Done.")

        # If no energies are 'nan', we're clean.
        if not np.any(np.isnan(self.energies[:])):
            return

        # There are some energies that are 'nan', so check if the first iteration has nans in their *own* energies:
        u_k = np.diag(self.energies[0])
        if np.any(np.isnan(u_k)):
            logger.info("First iteration has exploded replicas.  Check to make sure structures are minimized before dynamics")
            logger.info("Energies for all replicas after equilibration:")
            logger.info(u_k)
            return

        # There are some energies that are 'nan' past the first iteration.  Find the first instances for each replica and write PDB files.
        first_nan_k = np.zeros([self.n_states], np.int32)
        for iteration in range(n_iterations):
            for k in range(self.n_states):
                if np.isnan(self.energies[iteration, k, k]) and first_nan_k[k] == 0:
                    first_nan_k[k] = iteration
        
        if not all(first_nan_k == 0):
            logger.info("Some replicas exploded during the simulation.")
            logger.info("Iterations where explosions were detected for each replica:")
            logger.info(first_nan_k)
            logger.info("Writing PDB files immediately before explosions were detected...")
            for replica in range(self.n_states):            
                if (first_nan_k[replica] > 0):
                    state = self.states[iteration,replica]
                    iteration = first_nan_k[replica] - 1
                    filename = 'replica-%d-before-explosion.pdb' % replica

                    # TO DO: output frame as pdb
                    #write_pdb(atoms, filename, iteration, replica, title, ncfile)

        # There are some energies that are 'nan', but these are energies at foreign lambdas.  We'll just have to be careful with MBAR.
        # Raise a warning.
        logger.info("WARNING: Some energies at foreign lambdas are 'nan'.  This is recoverable.")            


    def check_positions(self):
        """Make sure no positions are nan."""

        # Get current dimension.
        n_iterations = self.positions.shape[0]

        for iteration in range(n_iterations):
            for replica in range(self.n_states):
                positions = self.positions[iteration, replica]
                # Check for nan
                if np.any(np.isnan(positions)):
                    # Nan found -- raise error
                    logger.info("Iteration %d, state %d - nan found in positions." % (iteration, replica))
                    # Report coordinates
                    for atom_index in range(self.n_atoms):
                        logger.info("%16.3f %16.3f %16.3f" % (positions[atom_index, 0], positions[atom_index, 1], positions[atom_index, 2]))
                        if np.any(np.isnan(positions[atom_index])):
                            logger.info("nan detected in positions")


    def get_u_kln(self):
        """Extract energies by thermodynamic state.


        Returns
        -------
        
        u_kln : np.ndarray, shape=(n_states, n_states, n_iterations)
            The statewise energies
        """

        # Get current dimensions.
        n_iterations = self.positions.shape[0]

        u_kln_replica = np.transpose(self.energies, (1, 2, 0)).astype('float64')
        u_kln = np.transpose(self.extract_reordered(self.energies), (1, 2, 0)).astype('float64')

        # Compute total negative log probability over all iterations.
        u_n = u_kln.diagonal(axis1=0, axis2=1).sum(1)
        
        return u_kln_replica, u_kln, u_n
        
    def run_mbar(self, ndiscard=0, nuse=None):
        """Estimate free energies of all alchemical states.

        Parameters
        ----------
        ndiscard : int, optinoal, default=0
            number of iterations to discard to equilibration
        nuse : int, optional, default=None
            maximum number of iterations to use (after discarding)

        Returns
        -------
        
        Deltaf_ij : np.ndarray, shape=(n_states, n_states)
            The statewise free energy differences

        dDeltaf_ij : np.ndarray, shape=(n_states, n_states)
            The statewise free energy difference uncertainties

        """    
        
        u_kln_replica, u_kln, u_n = self.get_u_kln()

        u_kln_replica, u_kln, u_n, N_k, N = self.equilibrate_and_subsample(u_kln_replica, u_kln, u_n, ndiscard=ndiscard, nuse=nuse)

        logger.info("Initialing MBAR and computing free energy differences...")
        mbar = MBAR(u_kln, N_k, verbose = False, method = 'self-consistent-iteration', maximum_iterations = 50000) # use slow self-consistent-iteration (the default)

        # Get matrix of dimensionless free energy differences and uncertainty estimate.
        logger.info("Computing covariance matrix...")
        (Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
       
        logger.info("\n%-24s %16s\n%s" % ("Deltaf_ij", "current state", pd.DataFrame(Deltaf_ij).to_string()))
        logger.info("\n%-24s %16s\n%s" % ("Deltaf_ij", "current state", pd.DataFrame(dDeltaf_ij).to_string()))        
                
        return (Deltaf_ij, dDeltaf_ij)

    def equilibrate_and_subsample(self, u_kln_replica, u_kln, u_n, ndiscard=0, nuse=None):
        """Equilibrate, truncate, and subsample uncorrelated samples.

        Parameters
        ----------
        ndiscard : int, optinoal, default=0
            number of iterations to discard to equilibration
        nuse : int, optional, default=None
            maximum number of iterations to use (after discarding)
        
        Returns
        -------
        """
        
        logger.info("Discarding initial data as equilibration (ndiscard = %d)" % ndiscard)
        u_kln_replica = u_kln_replica[:,:,ndiscard:]
        u_kln = u_kln[:,:, ndiscard:]
        u_n = u_n[ndiscard:]

        
        if nuse is not None:
            logger.info("Truncating to number of specified conforamtions to use(nuse = %d)" % nuse)
            u_kln_replica = u_kln_replica[:,:,0:nuse]
            u_kln = u_kln[:,:,0:nuse]
            u_n = u_n[0:nuse]
        
        logger.info("Subsample data to obtain uncorrelated samples")
        N_k = np.zeros(self.n_states, np.int32)    
        indices = timeseries.subsampleCorrelatedData(u_n) # indices of uncorrelated samples

        N = len(indices) # number of uncorrelated samples
        N_k[:] = N      
        u_kln[:, :, 0:N] = u_kln[:, :, indices]
        logger.info("number of uncorrelated samples:")
        logger.info(N_k)
        logger.info("")
        
        return u_kln_replica, u_kln, u_n, N_k, N


    def estimate_enthalpies(self, ndiscard=0, nuse=None):
        """Estimate average enthalpies of all alchemical states.

        Parameters
        ----------
        ndiscard : int, optional, default=0
            number of iterations to discard to equilibration
        nuse : int, optional, default=None
            maximum number of iterations to use (after discarding)
        
        """
        
        n_iterations = self.positions.shape[0]
        
        u_kln_replica, u_kln, u_n = self.get_u_kln()
        u_kln_replica, u_kln, u_n, N_k, N = self.equilibrate_and_subsample(u_kln_replica, u_kln, u_n, ndiscard=ndiscard, nuse=nuse)

        # Compute average enthalpies.
        H_k = np.zeros([self.n_states], np.float64) # H_i[i] is estimated enthalpy of state i
        dH_k = np.zeros([self.n_states], np.float64)
        for k in range(self.n_states):
            H_k[k] = u_kln[k,k,:].mean()
            dH_k[k] = u_kln[k,k,:].std() / np.sqrt(N)

        return (H_k, dH_k)


    def _accumulate_mixing_statistics(self):
        """Return the mixing transition matrix Tij."""
        try:
            return self._accumulate_mixing_statistics_update()
        except AttributeError:
            pass
        except ValueError:
            logger.info("Inconsistent transition count matrix detected, recalculating from scratch.")

        return self._accumulate_mixing_statistics_full()

    def _accumulate_mixing_statistics_full(self):
        """Compute statistics of transitions iterating over all iterations of repex."""
        self._Nij = np.zeros([self.n_states, self.n_states], np.float64)
        for iteration in range(self.states.shape[0] - 1):
            for ireplica in range(self.n_states):
                istate = self.states[iteration, ireplica]
                jstate = self.states[iteration + 1, ireplica]
                self._Nij[istate, jstate] += 0.5
                self._Nij[jstate, istate] += 0.5
        
        Tij = np.zeros([self.n_states, self.n_states], np.float64)
        for istate in range(self.n_states):
            Tij[istate] = self._Nij[istate] / self._Nij[istate].sum()
        
        return Tij
    
    def _accumulate_mixing_statistics_update(self):
        """Compute statistics of transitions updating Nij of last iteration of repex."""

        if self._Nij.sum() != (self.states.shape[0] - 2) * self.n_states:  # n_iter - 2 = (n_iter - 1) - 1.  Meaning that you have exactly one new iteration to process.
            raise(ValueError("Inconsistent transition count matrix detected.  Perhaps you tried updating twice in a row?"))

        for ireplica in range(self.n_states):
            istate = self.states[-2, ireplica]
            jstate = self.states[-1, ireplica]
            self._Nij[istate, jstate] += 0.5
            self._Nij[jstate, istate] += 0.5

        Tij = np.zeros([self.n_states, self.n_states], np.float64)
        for istate in range(self.n_states):
            Tij[istate] = self._Nij[istate] / self._Nij[istate].sum()
        
        return Tij


    def _show_mixing_statistics(self):
        Tij = self._accumulate_mixing_statistics()

        P = pd.DataFrame(Tij)
        logger.info("\nCumulative symmetrized state mixing transition matrix:\n%s" % P.to_string())

        # Estimate second eigenvalue and equilibration time.
        mu = np.linalg.eigvals(Tij)
        mu = -np.sort(-mu) # sort in descending order
        if (mu[1] >= 1):
            logger.info("\nPerron eigenvalue is unity; Markov chain is decomposable.")
        else:
            logger.info("\nPerron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1])))



    def output_diagnostics(self, diagnostics_path):
        """Create a directory, markdown file, and PDF containing replica exchange diagnostics.
        """
        try:
            import matplotlib
            # Force matplotlib to not use any Xwindows backend.
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except:
            print("Cannot import matplotlib, skipping diagnostics output.")
            return
        
        if not os.path.exists(diagnostics_path):
            try:
                os.mkdir(diagnostics_path)
            except OSError:
                print("Cannot create directory %s, skipping diagnostics output." % (diagnostics_path))
                return

        try:
            os.mkdir(diagnostics_path + "/figures/")
        except OSError:
            print("Cannot create directory %s, skipping diagnostics output." % (diagnostics_path + "/figures/"))
            return

        plt.plot(range(10))
        plt.savefig(diagnostics_path + "/figures/test.png")
        
        plt.plot(range(20))
        plt.savefig(diagnostics_path + "/figures/test2.png")
        
        s = """
        ============================
        Replica Exchange Diagnostics
        ============================
        
        .. image:: %s/test.png
        
        Still more 

        .. image:: %s/test2.png

        """ % ("figures/", "figures/")
        
        f = open(diagnostics_path + "/diagnostics.rst", "w")
        f.write(s)
        f.close()

        current = os.getcwd()
        try:
            os.chdir(diagnostics_path)
            logger.info("Temporarily changing directory from %s to %s to run pandoc." % (current, diagnostics_path))
            os.system("pandoc diagnostics.rst -o diagnostics.pdf")
            logger.info("Finished running pandoc")
        except:
            pass
        finally:
            logger.info("Changing back from %s to %s." % (diagnostics_path, current))
            os.chdir(current)
        
