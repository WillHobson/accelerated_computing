from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "par_cy",
        ["par_cy.pyx"],
        extra_compile_args=['-fopenmp',
            '-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/'],
        extra_link_args=['-lgomp', '-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/13/',
            '-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/']
    )
]

setup(name="par_cy",include_dirs=[numpy.get_include()],
      ext_modules=cythonize(ext_modules))