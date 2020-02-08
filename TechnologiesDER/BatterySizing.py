"""
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import storagevet
import logging
import cvxpy as cvx
import pandas as pd
import numpy as np
import storagevet.Constraint as Const
import copy
import re
import sys

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class BatterySizing(storagevet.BatteryTech):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, name, opt_agg, params, cycle_life):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            opt_agg (Series): time series data determined by optimization window size (total Series length is 8760)
            params (dict): params dictionary from dataframe for one case
            cycle_life (DataFrame): Cycle life information
        """

        # create generic storage object
        super().__init__(name, opt_agg, params, cycle_life)

        self.size_constraints = []

        self.optimization_variables = {}
        ess_id = self.unique_ess_id()

        # if the user inputted the energy rating as 0, then size for duration
        if not self.ene_max_rated:
            self.ene_max_rated = cvx.Variable(name='Energy_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ene_max_rated)]
            self.optimization_variables[ess_id + 'ene_max_rated'] = self.ene_max_rated

        # if both the discharge and charge ratings are 0, then size for both and set them equal to each other
        if not self.ch_max_rated and not self.dis_max_rated:
            self.ch_max_rated = cvx.Variable(name='power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]
            self.dis_max_rated = self.ch_max_rated
            self.optimization_variables[ess_id + 'ch_max_rated'] = self.ch_max_rated
            self.optimization_variables[ess_id + 'dis_max_rated'] = self.dis_max_rated

        elif not self.ch_max_rated:  # if the user inputted the discharge rating as 0, then size discharge rating
            self.ch_max_rated = cvx.Variable(name='charge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]
            self.optimization_variables[ess_id + 'ch_max_rated'] = self.ch_max_rated

        elif not self.dis_max_rated:  # if the user inputted the charge rating as 0, then size for charge
            self.dis_max_rated = cvx.Variable(name='discharge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.dis_max_rated)]
            self.optimization_variables[ess_id + 'dis_max_rated'] = self.dis_max_rated

        self.physical_constraints = {
            'ene_min_rated': Const.Constraint('ene_min_rated', self.name, self.llsoc * self.ene_max_rated),
            'ene_max_rated': Const.Constraint('ene_max_rated', self.name, self.ulsoc * self.ene_max_rated),
            'ch_min_rated': Const.Constraint('ch_min_rated', self.name, self.ch_min_rated),
            'ch_max_rated': Const.Constraint('ch_max_rated', self.name, self.ch_max_rated),
            'dis_min_rated': Const.Constraint('dis_min_rated', self.name, self.dis_min_rated),
            'dis_max_rated': Const.Constraint('dis_max_rated', self.name, self.dis_max_rated)}

    def objective_function(self, variables, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                    the entire project lifetime (only to be set iff sizing, else alpha should not affect the aobject function)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        ess_id = self.unique_ess_id()
        super().objective_function(variables, mask, annuity_scalar)

        self.costs.update({ess_id + 'capex': self.capex()*annuity_scalar})
        return self.costs

    def sizing_summary(self):
        """

        Returns: A datafram indexed by the terms that describe this DER's size and captial costs.

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

        index = pd.Index([self.name], name='DER')
        sizing_results = pd.DataFrame({'Energy Rating (kWh)': energy_rated,
                                       'Charge Rating (kW)': ch_max_rated,
                                       'Discharge Rating (kW)': dis_max_rated,
                                       'Duration (hours)': energy_rated/dis_max_rated,
                                       'Capital Cost ($)': self.capital_costs['flat'],
                                       'Capital Cost ($/kW)': self.capital_costs['ccost_kW'],
                                       'Capital Cost ($/kWh)': self.capital_costs['ccost_kWh']}, index=index)
        return sizing_results

    def objective_constraints(self, variables, mask, reservations, mpc_ene=None, sizing=False):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            variables (Dict): Dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            reservations (Dict): Dictionary of energy and power reservations required by the services being
                preformed with the current optimization subset
            mpc_ene (float): value of energy at end of last opt step (for mpc opt)
            sizing (bool): flag that tells indicates whether the technology is being sized

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """

        constraint_list = super().objective_constraints(variables, mask, reservations, mpc_ene, True)
        constraint_list += self.size_constraints

        return constraint_list

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            ene (Variable): A cvxpy variable for Energy at the end of the time step
            dis (Variable): A cvxpy variable for Discharge Power, kW during the previous time step
            ch (Variable): A cvxpy variable for Charge Power, kW during the previous time step
            ene_max_slack (Variable): A cvxpy variable for energy max slack
            ene_min_slack (Variable): A cvxpy variable for energy min slack
            ch_max_slack (Variable): A cvxpy variable for charging max slack
            ch_min_slack (Variable): A cvxpy variable for charging min slack
            dis_max_slack (Variable): A cvxpy variable for discharging max slack
            dis_min_slack (Variable): A cvxpy variable for discharging min slack

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        variables = super().add_vars(size)

        variables.update(self.optimization_variables)

        return variables

