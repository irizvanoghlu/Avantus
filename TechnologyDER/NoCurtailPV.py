"""
NoCurtailPV.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

from Technology import Technology
import cvxpy as cvx


class NoCurtailPV(Technology.Technology):
    """ Pre_IEEE 1547 2018 standards. Generates all or nothing. Assumes perfect foresight.

    Note: not currently used

    """

    def __init__(self, name,  financial, params, tech_params, time_series):
        """ Initializes a PV class where perfect foresight of generation is assumed.
        It inherits from the technology class. Additionally, it sets the type and physical constraints of the
        technology.

        Args:
            params (dict): params dictionary from dataframe for one case
            time_series (series): time series dataframe
        """
        Technology.Technology.__init__(self, name, params, 'Perfect Foresight PV')
        self.generation = time_series['pv_generation']
        # self.load = np.zeros(len(time_series))

    def add_vars(self, size, binary, slack, startup):
        """ Adds optimization variables to dictionary

        Variables added:
            pv_out (Variable): A cvxpy variable for the ac eq power outputted by the PV system

        Args:
            size (Int): Length of optimization variables to create
            slack (bool): True if any pre-dispatch services are turned on, else False
            binary (bool): True if user wants to implement binary variables in optimization, else False
            startup (bool): True if user wants to implement startup variables in optimization, else False

        Returns:
            Dictionary of optimization variables
        """
        variables = {'pv_out': cvx.Parameter(shape=size, name='pv_out', value=self.generation.value)}

        return variables
