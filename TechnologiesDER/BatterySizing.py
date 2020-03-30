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
DEBUG = False


class BatterySizing(storagevet.BatteryTech):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, opt_agg, params):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            opt_agg (Series): time series data determined by optimization window size (total Series length is 8760)
            params (dict): params dictionary from dataframe for one case
        """

        # create generic storage object
        storagevet.BatteryTech.__init__(self, opt_agg, params)

        self.user_duration = params['duration_max']
        self.sizing_properties = []

    def add_vars(self, size):
        """
        BatterySizing Variables possibly added:
            ene_max_rated (Variable): A cvxpy variable for energy cap (kWh)
            ch_max_rated (Variable): A cvxpy variable for charge power cap (kW)
            dis_max_rated (Variable): A cvxpy variable for discharge power cap (kW)

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        super().add_vars(size)

        ess_id = self.unique_ess_id()
        tech_id = self.unique_tech_id()

        # if the user inputted the energy rating as 0, then size for energy rating
        if not self.ene_max_rated:
            self.ene_max_rated = cvx.Variable(shape=size, name=ess_id + 'energy_cap', integer=True)
            self.variables_dict.update({tech_id + ess_id + 'ene_max_rated': self.ene_max_rated})
            self.sizing_properties.append(['ene_max_rated'])
            self.variable_names.update([tech_id + ess_id + 'ene_max_rated'])

        # if both the discharge and charge ratings are 0, then size for both and set them equal to each other
        if not self.ch_max_rated and not self.dis_max_rated:
            self.ch_max_rated = cvx.Variable(shape=size, name=ess_id + 'ch_power_cap', integer=True)
            self.dis_max_rated = cvx.Variable(shape=size, name=ess_id + 'dis_power_cap', integer=True)
            self.variables_dict.update({tech_id + ess_id + 'ch_max_rated': self.ch_max_rated})
            self.variables_dict.update({tech_id + ess_id + 'dis_max_rated': self.dis_max_rated})
            self.sizing_properties.append(['ch_max_rated', 'dis_max_rated'])
            self.variable_names.update([tech_id + ess_id + 'ch_max_rated'])
            self.variable_names.update([tech_id + ess_id + 'dis_max_rated'])

        elif not self.ch_max_rated:  # if the user inputted the charge rating as 0, then size charge rating
            self.ch_max_rated = cvx.Variable(shape=size, name=ess_id + 'ch_power_cap', integer=True)
            self.variables_dict.update({tech_id + ess_id + 'ch_max_rated': self.ch_max_rated})
            self.sizing_properties.append(['ch_max_rated'])
            self.variable_names.update([tech_id + ess_id + 'ch_max_rated'])

        elif not self.dis_max_rated:  # if the user inputted the discharge rating as 0, then size for discharge
            self.dis_max_rated = cvx.Variable(shape=size, name=ess_id + 'dis_power_cap', integer=True)
            self.variables_dict.update({tech_id + ess_id + 'dis_max_rated': self.dis_max_rated})
            self.sizing_properties.append(['dis_max_rated'])
            self.variable_names.update([tech_id + ess_id + 'dis_max_rated'])

    def calculate_duration(self):
        try:
            energy_rated = self.ene_max_rated.value
        except AttributeError:
            energy_rated = self.ene_max_rated

        try:
            dis_max_rated = self.dis_max_rated.value
        except AttributeError:
            dis_max_rated = self.dis_max_rated
        return energy_rated/dis_max_rated

    def objective_function(self, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                    the entire project lifetime (only to be set iff sizing, else alpha should not affect the aobject function)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        ess_id = self.unique_ess_id()
        tech_id = self.unique_tech_id()
        super().objective_function(mask, annuity_scalar)

        capex = self.capital_costs['flat'] + (self.capital_costs['/kW'] * self.dis_max_rated) + (self.capital_costs['/kWh'] * self.ene_max_rated)
        self.costs.update({tech_id + ess_id + 'capex': capex * annuity_scalar})
        return self.costs

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

        index = pd.Index([self.name], name='DER')
        sizing_results = pd.DataFrame({'Energy Rating (kWh)': energy_rated,
                                       'Charge Rating (kW)': ch_max_rated,
                                       'Discharge Rating (kW)': dis_max_rated,
                                       'Round Trip Efficiency (%)': self.rte,
                                       'Lower Limit on SOC (%)': self.llsoc,
                                       'Upper Limit on SOC (%)': self.ulsoc,
                                       'Duration (hours)': energy_rated/dis_max_rated,
                                       'Capital Cost ($)': self.capital_costs['flat'],
                                       'Capital Cost ($/kW)': self.capital_costs['/kW'],
                                       'Capital Cost ($/kWh)': self.capital_costs['/kWh']}, index=index)
        if (sizing_results['Duration (hours)'] > 24).any():
            print('The duration of an Energy Storage System is greater than 24 hours!')
        return sizing_results

    def objective_constraints(self, mask, mpc_ene=None, sizing=True):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            mpc_ene (float): value of energy at end of last opt step (for mpc opt)
            sizing (bool): flag that tells indicates whether the technology is being sized

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """

        constraint_list = super().objective_constraints(mask, mpc_ene, sizing)

        if 'ene_max_rated' in self.sizing_properties:
            constraint_list += [cvx.NonPos(-self.ene_max_rated)]
        if 'ch_max_rated' in self.sizing_properties:
            constraint_list += [cvx.NonPos(-self.ch_max_rated)]
        if 'dis_max_rated' in self.sizing_properties:
            constraint_list += [cvx.NonPos(-self.dis_max_rated)]
        if self.user_duration:
            constraint_list += [cvx.NonPos((self.ene_max_rated / self.dis_max_rated) - self.user_duration)]

        return constraint_list

    def proforma_report(self, opt_years, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            results (DataFrame): DataFrame with all the optimization variable solutions

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

            Creates a dataframe with only the years that we have data for. Since we do not label the column,
            it defaults to number the columns with a RangeIndex (starting at 0) therefore, the following
            DataFrame has only one column, labeled by the int 0

        """
        # recalculate capex before reporting proforma
        self.capex = self.capital_costs['flat'] + (self.capital_costs['/kW'] * self.dis_max_rated) + (self.capital_costs['/kWh'] * self.ene_max_rated)
        proforma = super().proforma_report(opt_years, results)
        return proforma

    def being_sized(self):
        """ checks itself to see if this instance is being sized

        Returns: true if being sized, false if not being sized

        """
        return bool(len(self.size_constraints))
