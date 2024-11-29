
from setuptools import setup, find_packages

setup(
    name='farmingpy',
    version='0.1.0',
    url='https://github.com/TwinYields/farmingpy',
    license='MIT',
    author='Matti Pastell, Annimari Hartikainen, Juha Backman',
    author_email='matti.pastell@luke.fi',
    description='farmingpy: a Python package for developing Digital Twins of precision farming systems',
    packages=find_packages(),
    include_package_data=True,
)