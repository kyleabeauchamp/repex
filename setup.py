"""repex: a python mpi-enabled replica exchange driver.

"""

from __future__ import print_function
DOCLINES = __doc__.split("\n")

import os
import sys
import shutil
import tempfile
import subprocess
from distutils.ccompiler import new_compiler
from setuptools import setup, Extension
import sys

import numpy
try:
    from Cython.Distutils import build_ext
    setup_kwargs = {'cmdclass': {'build_ext': build_ext}}
    cython_extension = 'pyx'
except ImportError:
    setup_kwargs = {}
    cython_extension = 'c'



try:
    # add an optional command line flag --no-install-deps to setup.py
    # to turn off setuptools automatic downloading of dependencies
    sys.argv.remove('--no-install-deps')
    no_install_deps = True
except ValueError:
    no_install_deps = False


##########################
VERSION = "0.1"
ISRELEASED = False
__version__ = VERSION
##########################


CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

extensions = []

setup(name='repex',
      author='Kyle A. Beauchamp',
      author_email='kyleabeauchamp@gmail.com',
      description=DOCLINES[0],
      long_description="\n".join(DOCLINES[2:]),
      version=__version__,
      license='GPLv3+',
      url='http://github.com/ChoderaLab/repex',
      platforms=['Linux', 'Mac OS-X', 'Unix'],
      classifiers=CLASSIFIERS.splitlines(),
      packages=["repex"],
      package_dir={'mdtraj': 'MDTraj', 'mdtraj.scripts': 'scripts'},
      install_requires=['numpy', 'nose', 'nose-exclude', 'pymbar'],
      zip_safe=False,
      scripts=[],
      ext_modules=extensions,
      package_data={'repex': ['data/*/*']},  # Install all data directories of the form testsystems/data/X/
      **setup_kwargs
      )
