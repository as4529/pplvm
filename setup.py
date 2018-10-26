#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='vipoint',
      version='0.0.1',
      description='State space models with point process observations',
      author='Anuj Sharma',
      packages=['pplvm', 'pplvm.likelihoods', 'pplvm.message_passing',
                'pplvm.models', 'pplvm.utils'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
      )
