from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules=[
    Extension("cresample",
              ["cresample.pyx"],
              libraries=["m"]) # Unix-like specific
]

setup(
  name = "Resample",
  cmdclass = {"build_ext": build_ext},
  include_dirs = [np.get_include()], 
  ext_modules = ext_modules
)