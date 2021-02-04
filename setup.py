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

setup(name='storagevet',
      version='0.1',
      description='Testing installation of Package',
      url='#',
      long_description=README,
      # scripts=['run_Storagevet.py'],
      author='auth',
      author_email='author@email.com',
      license='EPRI',
      classifiers=[
              "License :: EPRI License",
              "Programming Language :: Python"
      ],
      packages=find_packages(),
      zip_safe=False)
