"""
Diesel

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import cvxpy as cvx
import numpy as np
import pandas as pd
from storagevet.Technology.Diesel import Diesel


class DieselSizing(Diesel):
    """ An ICE generator

    """

    def __init__(self, name, params):
        """ Initialize all technology with the following attributes.

        Args:
            name (str): A unique string name for the technology being added, also works as category.
            params (dict): Dict of parameters for initialization
        """
        # create generic technology object
        Diesel.__init__(self, name, params)
        self.n_min = params['n_min']  # generators
        self.n_max = params['n_max']  # generators
        self.n = cvx.Variable(shape=1, integer=True, name='generators')
        self.capex = self.capital_cost * self.n + self.capital_cost * self.p_max

    def build_master_constraints(self, variables, mask, reservations, mpc_ene=None):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            variables (Dict): Dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            reservations (Dict): Dictionary of energy and power reservations required by the services being
                preformed with the current optimization subset
            mpc_ene (float): value of energy at end of last opt step (for mpc opt)

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = Diesel.build_master_constraints(self, variables, mask, reservations, mpc_ene)
        ice_gen = variables['ice_gen']

        constraint_list += [cvx.NonPos(self.n_min - self.n)]
        constraint_list += [cvx.NonPos(self.n - self.n_max)]
        constraint_list += [cvx.NonPos(ice_gen - self.n * self.p_max)]

        return constraint_list

    def sizing_summary(self):
        """

        Returns: A datafram indexed by the terms that describe this DER's size and captial costs.

        """
        sizing_data = [self.p_max/self.n,
                       self.capital_cost,
                       self.ccost_kw,
                       self.n]
        index = pd.Index(['Power Capacity (kW)',
                          'Capital Cost ($)',
                          'Capital Cost ($/kW)',
                          'Quantity'], name='Size and Costs')
        sizing_results = pd.DataFrame({self.name: sizing_data}, index=index)
        return sizing_results
