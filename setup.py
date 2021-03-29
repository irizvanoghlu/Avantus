"""
Use `python setup.py develop --uninstall` to uninstall development
Use `python setup.py develop` to install in development mode
See https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html#running-setup-py
for more
"""
from setuptools import setup, find_packages
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='dervet',
      version='1.0.0',
      description='',
      url='der-vet.com',
      long_description=README,
      author='Halley Nathwani',
      author_email='hnathwani@epri.com',
      license='EPRI',
      classifiers=[
              "License :: EPRI License",
              "Programming Language :: Python"
      ],
      # python_requires="==3.6",
      packages=find_packages(),
      zip_safe=False)
