# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'cvxopt',
    'dill',
    'h5py'
]

setup(name='vst_sim',
      version='0.0.1',
      description='Versatile simulator project code',
      author='Hui Zhang',
      author_email='hui.zhang@kuleuven.be',
      package_dir = {'': 'src'},
      packages=['vstsim'],
      install_requires=requirements,
      test_suite='test'
     )
