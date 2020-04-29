"""
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

import logging
import cvxpy as cvx
from MicrogridDER.Sizing import Sizing
from storagevet.Technology import BatteryTech


u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')
DEBUG = False


class BatterySizing(BatteryTech.Battery, Sizing):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, params):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            params (dict): params dictionary from dataframe for one case
        """

        # create generic storage object
        BatteryTech.Battery.__init__(self, params)
        Sizing.__init__(self)

        self.user_duration = params['duration_max']

        # if the user inputted the energy rating as 0, then size for energy rating
        if not self.ene_max_rated:
            self.ene_max_rated = cvx.Variable(name='Energy_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ene_max_rated)]

        # if both the discharge and charge ratings are 0, then size for both and set them equal to each other
        if not self.ch_max_rated and not self.dis_max_rated:
            self.ch_max_rated = cvx.Variable(name='power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]
            self.dis_max_rated = self.ch_max_rated

        elif not self.ch_max_rated:  # if the user inputted the discharge rating as 0, then size discharge rating
            self.ch_max_rated = cvx.Variable(name='charge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]

        elif not self.dis_max_rated:  # if the user inputted the charge rating as 0, then size for charge
            self.dis_max_rated = cvx.Variable(name='discharge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.dis_max_rated)]

        if self.user_duration:
            self.size_constraints += [cvx.NonPos((self.ene_max_rated / self.dis_max_rated) - self.user_duration)]

    def constraints(self, mask):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """

        constraint_list = super().constraints(mask)

        constraint_list += self.size_constraints

        return constraint_list

    def objective_function(self, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                    the entire project lifetime (only to be set iff sizing, else alpha should not affect the aobject function)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        costs = super().objective_function(mask, annuity_scalar)

        costs.update({self.name + 'capex': self.get_capex()})
        return costs

    def sizing_summary(self):
        """

        Returns: A dataframe indexed by the terms that describe this DER's size and captial costs.

        """
        # obtain the size of the battery, these may or may not be optimization variable
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

        sizing_results = {
            'DER': self.name,
            'Energy Rating (kWh)': energy_rated,
            'Charge Rating (kW)': ch_max_rated,
            'Discharge Rating (kW)': dis_max_rated,
            'Round Trip Efficiency (%)': self.rte,
            'Lower Limit on SOC (%)': self.llsoc,
            'Upper Limit on SOC (%)': self.ulsoc,
            'Duration (hours)': energy_rated/dis_max_rated,
            'Capital Cost ($)': self.capital_cost_function[0],
            'Capital Cost ($/kW)': self.capital_cost_function[1],
            'Capital Cost ($/kWh)': self.capital_cost_function[2]}
        if sizing_results['Duration (hours)'] > 24:
            u_logger.error(f'The duration of {self.name} is greater than 24 hours!')
        return sizing_results

    def calculate_duration(self):
        """ Determines the duration of the storage (after solving for the size)

        Returns:
        Note: unused
        """
        try:
            energy_rated = self.ene_max_rated.value
        except AttributeError:
            energy_rated = self.ene_max_rated

        try:
            dis_max_rated = self.dis_max_rated.value
        except AttributeError:
            dis_max_rated = self.dis_max_rated
        return energy_rated / dis_max_rated
