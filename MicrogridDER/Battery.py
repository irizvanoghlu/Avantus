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
from MicrogridDER.DERExtension import DERExtension


u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')
DEBUG = False


class Battery(BatteryTech.Battery, Sizing, DERExtension):
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
        DERExtension.__init__(self, params)
        Sizing.__init__(self)

        self.user_duration = params['duration_max']

        # if the user inputted the energy rating as 0, then size for energy rating
        if not self.ene_max_rated:
            self.ene_max_rated = cvx.Variable(name='Energy_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ene_max_rated)]
            # recalculate the effective SOE limits s.t. they are CVXPY expressions
            self.effective_soe_min = self.llsoc * self.ene_max_rated
            self.effective_soe_max = self.ulsoc * self.ene_max_rated

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

    def discharge_capacity(self, solution=False):
        """

        Returns: the maximum discharge that can be attained

        """
        if not solution:
            return self.dis_max_rated
        else:
            try:
                dis_max_rated = self.dis_max_rated.value
            except AttributeError:
                dis_max_rated = self.dis_max_rated
            return dis_max_rated

    def charge_capacity(self, solution=False):
        """

        Returns: the maximum charge that can be attained

        """
        if not solution:
            return self.dis_max_rated
        else:
            try:
                ch_max_rated = self.ch_max_rated.value
            except AttributeError:
                ch_max_rated = self.ch_max_rated
            return ch_max_rated

    def energy_capacity(self, solution=False):
        """

        Returns: the maximum energy that can be attained

        """
        if not solution:
            return self.ene_max_rated
        else:
            try:
                max_rated = self.ene_max_rated.value
            except AttributeError:
                max_rated = self.ene_max_rated
            return max_rated

    def operational_max_energy(self, solution=False):
        """

        Returns: the maximum energy that should stored in this DER based on user inputs

        """
        if not solution:
            return self.effective_soe_max
        else:
            try:
                effective_soe_max = self.effective_soe_max.value
            except AttributeError:
                effective_soe_max = self.effective_soe_max
            return effective_soe_max

    def operational_min_energy(self, solution=False):
        """

        Returns: the minimum energy that should stored in this DER based on user inputs
        """
        if not solution:
            return self.effective_soe_min
        else:
            try:
                effective_soe_min = self.effective_soe_min.value
            except AttributeError:
                effective_soe_min = self.effective_soe_min
            return effective_soe_min

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

        sizing_results = {
            'DER': self.name,
            'Energy Rating (kWh)': energy_rated,
            'Charge Rating (kW)': self.charge_capacity(),
            'Discharge Rating (kW)': self.discharge_capacity(),
            'Round Trip Efficiency (%)': self.rte,
            'Lower Limit on SOC (%)': self.llsoc,
            'Upper Limit on SOC (%)': self.ulsoc,
            'Duration (hours)': self.calculate_duration(),
            'Capital Cost ($)': self.capital_cost_function[0],
            'Capital Cost ($/kW)': self.capital_cost_function[1],
            'Capital Cost ($/kWh)': self.capital_cost_function[2]}
        if sizing_results['Duration (hours)'] > 24:
            u_logger.error(f'The duration of {self.name} is greater than 24 hours!')
        return sizing_results

    def calculate_duration(self):
        """ Determines the duration of the storage (after solving for the size)

        Returns:
        """
        try:
            energy_rated = self.ene_max_rated.value
        except AttributeError:
            energy_rated = self.ene_max_rated

        return energy_rated / self.discharge_capacity(solution=True)

    def update_for_evaluation(self, input_dict):
        """ Updates price related attributes with those specified in the input_dictionary

        Args:
            input_dict: hold input data, keys are the same as when initialized

        """
        super().update_for_evaluation(input_dict)
        fixed_om = input_dict.get('fixedOM')
        if fixed_om is not None:
            self.fixedOM_perKW = fixed_om

        variable_om = input_dict.get('OMexpenses')
        if variable_om is not None:
            self.variable_om = variable_om*100

        if self.incl_startup:
            p_start_ch = input_dict.get('p_start_ch')
            if p_start_ch is not None:
                self.p_start_ch = p_start_ch * 100

            p_start_dis = input_dict.get('p_start_dis')
            if p_start_dis is not None:
                self.p_start_dis = p_start_dis * 100
