
from setuptools import setup, find_packages

setup(
    name='farmingpy',
    version='0.1.0',
    url='https://github.com/TwinYields/farmingpy',
    license='MIT',
    author='Matti Pastell',
    author_email='matti.pastell@luke.fi',
    description='Package for reading ISOBUS task files and processing data',
    packages=find_packages(),
    include_package_data=True,
)