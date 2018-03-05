"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.org'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lp-visu',
    version='1.0',
    description='A module to visualize 2D (integer) linear programming problem solving',
    long_description=long_description,
    url='https://github.com/tofgarion/lp-visu',
    author='Christophe Garion',
    license='GPLv3',
    packages=find_packages(),
    install_requires=['matplotlib', 'numpy'],
    project_urls={
        'Bug Reports': 'https://github.com/tofgarion/lp-visu/issues',
        'Source': 'https://github.com/tofgarion/lp-visu/',
    },
)
