#!/usr/local/bin/env python

import logging
logger = logging.getLogger(__name__)


openmm_citations = """\
Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing units. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. Comput. Sci. Eng. 12:34, 2010. DOI: 10.1109/MCSE.2010.27
Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. J. Comput. Chem. 31:1268, 2010. DOI: 10.1002/jcc.21413
Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. J. Chem. Theor. Comput. 6:434, 2010. DOI: 10.1021/ct900463w"""

gibbs_citations = """\
Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs sampling: Simple improvements for enhanced mixing. J. Chem. Phys., in press. arXiv: 1105.5749"""

mbar_citations = """\
Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states. J. Chem. Phys. 129:124105, 2008. DOI: 10.1063/1.2978177"""


def display_citations(replica_mixing_scheme, online_analysis):
    """
    Display papers to be cited.

    TODO:

    * Add original citations for various replica-exchange schemes.
    * Show subset of OpenMM citations based on what features are being used.

    """

    logger.info("Please cite the following:")
    logger.info("")
    logger.info("%s" % openmm_citations)
    if replica_mixing_scheme == 'swap-all':
        logger.info("%s" % gibbs_citations)
    if online_analysis:
        logger.info("%s" % mbar_citations)
