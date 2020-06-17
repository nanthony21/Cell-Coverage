# Copyright 2018-2020 Nick Anthony, Backman Biophotonics Lab, Northwestern University

# -*- coding: utf-8 -*-
"""
This file is used to install the cell coverage package. for example navigate in your terminal to the directory containing this
file and type `pip install .`.
"""
from setuptools import setup, find_packages
import os.path as osp
import os


setup(name='cellcoverage',
      version='0.1',
      description='A program for analyzing cell coverage.',
      author='Nick Anthony',
      author_email='nicholas.anthony@northwestern.edu',
      url='https://bitbucket.org/backmanlab/pwspython/src/master/',
      python_requires='>3.7',
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'h5py',
                        'opencv-python', #opencv is required but naming differences between conda and pip seem to cause issues. Maybe should be commented out?
                        'PyQt5',
                        'pwspy',  # Backmanlab PWS codebase. available on the backmanlab anaconda cloud channel.
                        'imageio',
                        'Pillow'],
      package_dir={'': 'src'},
      package_data={'cellcoverage': ['masks/6WellPlate/*']},
      packages=find_packages('src')
	)
