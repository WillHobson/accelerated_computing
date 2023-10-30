from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='ll', ext_modules=cythonize("cy_a2.pyx"), include_dirs=[numpy.get_include()])
