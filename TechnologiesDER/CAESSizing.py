"""
CAESSizing.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Thien Nguyen', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import storagevet
import cvxpy as cvx
import logging
import pandas as pd
import numpy as np
import storagevet.Constraint as Const

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class CAESSizing(storagevet.CAESTech):
    """ CAES class that inherits from Storage.

    """

    def __init__(self, name, opt_agg, params):
        """ Initializes CAES class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            opt_agg (Series): time series data determined by optimization window size (total Series length is 8760)
            params (dict): params dictionary from dataframe for one case
            cycle_life (DataFrame): Cycle life information
        """

        # create generic storage object
        super().__init__(name, opt_agg, params)

        self.size_constraints = []

        self.optimization_variables = {}

    def sizing_summary(self):
        """
        TODO: CAESSizing is waiting to be implemented, it is currently mimicking BatterySizing's method

        Returns: A datafram indexed by the terms that describe this DER's size and captial costs.

        """
        # obtain the size of the CAES, these may or may not be optimization variable
        # therefore we check to see if it is by trying to get its value attribute in a try-except statement.
        # If there is an error, then we know that it was user inputted and we just take that value instead.
        try:
            energy_rated = self.ene_max_rated.value
        except AttributeError:
            energy_rated = self.ene_max_rated

        try:
            ch_max_rated = self.ch_max_rated.value
        except AttributeError:
            ch_max_rated = self.ch_max_rated

        try:
            dis_max_rated = self.dis_max_rated.value
        except AttributeError:
            dis_max_rated = self.dis_max_rated

        index = pd.Index([self.name], name='DER')
        sizing_results = pd.DataFrame({'CAES Energy Rating (kWh)': energy_rated,
                                       'CAES Charge Rating (kW)': ch_max_rated,
                                       'CAES Discharge Rating (kW)': dis_max_rated,
                                       'CAES Duration (hours)': energy_rated / dis_max_rated,
                                       'CAES Capital Cost ($)': self.ccost,
                                       'CAES Capital Cost ($/kW)': self.ccost_kw,
                                       'CAES Capital Cost ($/kWh)': self.ccost_kwh}, index=index)
        return sizing_results





