"""
CurtailPVPV.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import cvxpy as cvx
import pandas as pd
import storagevet


class CurtailPVSizing(storagevet.CurtailPV):
    """ Pre_IEEE 1547 2018 standards. Assumes perfect foresight. Ability to curtail PV generation, unlike ChildPV.

    """

    def __init__(self, name, params):
        """ Initializes a PV class where perfect foresight of generation is assumed.
        It inherits from the technology class. Additionally, it sets the type and physical constraints of the
        technology.

        Args:
            name (str): A unique string name for the technology being added, also works as category.
            params (dict): Dict of parameters
        """
        # create generic technology object
        storagevet.CurtailPV.__init__(self, name, params)

        self.size_constraints = []

        if not self.rated_capacity:
            self.rated_capacity = cvx.Variable(name='PV rating', integer=True)
            self.size_constraints += [cvx.NonPos(-self.rated_capacity)]
            self.capex = self.cost_per_kW * self.rated_capacity

    def sizing_summary(self):
        """

        Returns: A datafram indexed by the terms that describe this DER's size and captial costs.

        """
        # obtain the size of the battery, these may or may not be optimization variable
        # therefore we check to see if it is by trying to get its value attribute in a try-except statement.
        # If there is an error, then we know that it was user inputted and we just take that value instead.
        try:
            rated_capacity = self.rated_capacity.value
        except AttributeError:
            rated_capacity = self.rated_capacity
        sizing_data = [rated_capacity,
                       self.cost_per_kW]
        index = pd.Index(['Power Capacity (kW)',
                          'Capital Cost ($/kW)'], name='Size and Costs')
        sizing_results = pd.DataFrame({self.name: sizing_data}, index=index)
        return sizing_results

    def objective_constraints(self, variables, mask, reservations, mpc_ene=None):
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
        constraints = storagevet.CurtailPV.objective_constraints(self, variables, mask, reservations, mpc_ene)

        constraints += self.size_constraints
        return constraints
