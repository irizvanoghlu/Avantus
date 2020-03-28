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

        capex = self.capital_costs['flat'] + (self.capital_costs['/kW'] * self.dis_max_rated) + (self.capital_costs['kWh'] * self.ene_max_rated)
        self.costs.update({tech_id + ess_id + 'capex': capex * annuity_scalar})
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
                                       'Round Trip Efficiency (%)': self.rte,
                                       'Lower Limit on SOC (%)': self.llsoc,
                                       'Upper Limit on SOC (%)': self.ulsoc,
                                       'Duration (hours)': energy_rated/dis_max_rated,
                                       'Capital Cost ($)': self.capital_costs['flat'],
                                       'Capital Cost ($/kW)': self.capital_costs['ccost_kW'],
                                       'Capital Cost ($/kWh)': self.capital_costs['ccost_kWh']}, index=index)
        if (sizing_results['Duration (hours)'] > 24).any():
            print('The duration of an Energy Storage System is greater than 24 hours!')
        return sizing_results

    # TODO: rework this method, almost mimicking Storage objective_constraints method
    #  Control_constraints are currently created in Controller identify_system_requirements method
    #  Revise this method with developer team
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

        # Note: does this method need to inherit the objective_constraints method of Battery class?
        # Storage self.physical_constraints variable was not used in the Battery/POI class in run_StorageVET

        ess_id = self.unique_ess_id()
        tech_id = self.unique_tech_id()

        constraint_list = []
        size = int(np.sum(mask))
        ene_target = self.soc_target * self.ene_max_rated

        self.physical_constraints = {
            'ene_min_rated': Const.Constraint('ene_min_rated', self.name, self.llsoc * self.ene_max_rated),
            'ene_max_rated': Const.Constraint('ene_max_rated', self.name, self.ulsoc * self.ene_max_rated),
            'ch_min_rated': Const.Constraint('ch_min_rated', self.name, self.ch_min_rated),
            'ch_max_rated': Const.Constraint('ch_max_rated', self.name, self.ch_max_rated),
            'dis_min_rated': Const.Constraint('dis_min_rated', self.name, self.dis_min_rated),
            'dis_max_rated': Const.Constraint('dis_max_rated', self.name, self.dis_max_rated)}

        if 'ene_max_rated' in self.sizing_properties:
            constraint_list += [cvx.NonPos(-self.ene_max_rated)]
        if 'ch_max_rated' in self.sizing_properties:
            constraint_list += [cvx.NonPos(-self.ch_max_rated)]
        if 'dis_max_rated' in self.sizing_properties:
            constraint_list += [cvx.NonPos(-self.dis_max_rated)]
        if self.user_duration:
            constraint_list += [cvx.NonPos((self.ene_max_rated / self.dis_max_rated) - self.user_duration)]


        # optimization variables
        ene = variables['ene']
        dis = variables['dis']
        ch = variables['ch']
        on_c = variables['on_c']
        on_d = variables['on_d']
        try:
            pv_gen = variables['pv_out']
        except KeyError:
            pv_gen = np.zeros(size)
        try:
            ice_gen = variables['ice_gen']
        except KeyError:
            ice_gen = np.zeros(size)

        if 'ene_max' in self.control_constraints.keys():
            ene_max = self.control_constraints['ene_max'].value[mask].values
            ene_max_t = ene_max[:-1]
            ene_max_n = ene_max[-1]
        else:
            ene_max = self.physical_constraints['ene_max_rated'].value
            ene_max_t = ene_max
            ene_max_n = ene_max

        if 'ene_min' in self.control_constraints.keys():
            ene_min = self.control_constraints['ene_min'].value[mask].values
            ene_min_t = ene_min[1:]
            ene_min_n = ene_min[-1]
        else:
            ene_min = self.physical_constraints['ene_min_rated'].value
            ene_min_t = ene_min
            ene_min_n = ene_min

        if 'ch_max' in self.control_constraints.keys():
            ch_max = self.control_constraints['ch_max'].value[mask].values
        else:
            ch_max = self.physical_constraints['ch_max_rated'].value

        if 'ch_min' in self.control_constraints.keys():
            ch_min = self.control_constraints['ch_min'].value[mask].values
        else:
            ch_min = self.physical_constraints['ch_min_rated'].value

        if 'dis_max' in self.control_constraints.keys():
            dis_max = self.control_constraints['dis_max'].value[mask].values
        else:
            dis_max = self.physical_constraints['dis_max_rated'].value

        if 'dis_min' in self.control_constraints.keys():
            dis_min = self.control_constraints['dis_min'].value[mask].values
        else:
            dis_min = self.physical_constraints['dis_min_rated'].value

        # energy at the end of the last time step
        constraint_list += [cvx.Zero((ene_target - ene[-1]) - (self.dt * ch[-1] * self.rte) + (self.dt * dis[-1]) - reservations['E'][-1] + (self.dt * ene[-1] * self.sdr * 0.01))]

        # energy generally for every time step
        constraint_list += [cvx.Zero(ene[1:] - ene[:-1] - (self.dt * ch[:-1] * self.rte) + (self.dt * dis[:-1]) - reservations['E'][:-1] + (self.dt * ene[:-1] * self.sdr * 0.01))]

        # energy at the beginning of the optimization window
        if mpc_ene is None:
            constraint_list += [cvx.Zero(ene[0] - ene_target)]
        else:
            constraint_list += [cvx.Zero(ene[0] - mpc_ene)]

        # Keep energy in bounds determined in the constraints configuration function
        constraint_list += [cvx.NonPos(ene_target - ene_max_n + reservations['E_upper'][-1] - variables['ene_max_slack'][-1])]  # TODO: comment out if putting energy user constraint and infeasible result
        constraint_list += [cvx.NonPos(ene[:-1] - ene_max_t + reservations['E_upper'][:-1] - variables['ene_max_slack'][:-1])]

        constraint_list += [cvx.NonPos(-ene_target + ene_min_n - (pv_gen[-1]*self.dt) - (ice_gen[-1]*self.dt) - reservations['E_lower'][-1] - variables['ene_min_slack'][-1])]
        constraint_list += [cvx.NonPos(ene_min_t - (pv_gen[1:]*self.dt) - (ice_gen[1:]*self.dt) - ene[1:] + reservations['E_lower'][:-1] - variables['ene_min_slack'][:-1])]

        # Keep charge and discharge power levels within bounds
        constraint_list += [cvx.NonPos(ch - cvx.multiply(ch_max, on_c) - variables['ch_max_slack'])]
        constraint_list += [cvx.NonPos(ch - ch_max + reservations['C_max'] - variables['ch_max_slack'])]

        constraint_list += [cvx.NonPos(cvx.multiply(ch_min, on_c) - ch - variables['ch_min_slack'])]
        constraint_list += [cvx.NonPos(ch_min - ch + reservations['C_min'] - variables['ch_min_slack'])]

        constraint_list += [cvx.NonPos(dis - cvx.multiply(dis_max, on_d) - variables['dis_max_slack'])]
        constraint_list += [cvx.NonPos(dis - dis_max + reservations['D_max'] - variables['dis_max_slack'])]

        constraint_list += [cvx.NonPos(cvx.multiply(dis_min, on_d) - dis - variables['dis_min_slack'])]
        constraint_list += [cvx.NonPos(dis_min - dis + reservations['D_min'] - variables['dis_min_slack'])]
        # constraints to keep slack variables positive
        if self.incl_slack:
            constraint_list += [cvx.NonPos(-variables['ch_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['ch_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['dis_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['dis_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['ene_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['ene_min_slack'])]

        if self.incl_binary:
            # when dis_min or ch_min has been overwritten (read: increased) by predispatch services, need to force technology to be on
            # TODO better way to do this???
            if 'dis_min' in self.control_constraints.keys():
                ind_d = [i for i in range(size) if self.control_constraints['dis_min'].value[mask].values[i] > self.physical_constraints['dis_min_rated'].value]
                if len(ind_d) > 0:
                    constraint_list += [on_d[ind_d] == 1]  # np.ones(len(ind_d))
            if 'ch_min' in self.control_constraints.keys():
                ind_c = [i for i in range(size) if self.control_constraints['ch_min'].value[mask].values[i] > self.physical_constraints['ch_min_rated'].value]
                if len(ind_c) > 0:
                    constraint_list += [on_c[ind_c] == 1]  # np.ones(len(ind_c))

            # note: cannot operate startup without binary
            if self.incl_startup:
                # startup variables are positive
                constraint_list += [cvx.NonPos(-variables['start_d'])]
                constraint_list += [cvx.NonPos(-variables['start_c'])]
                # difference between binary variables determine if started up in previous interval
                constraint_list += [cvx.NonPos(cvx.diff(on_d) - variables['start_d'][1:])]  # first variable not constrained
                constraint_list += [cvx.NonPos(cvx.diff(on_c) - variables['start_c'][1:])]  # first variable not constrained

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
        # recacluate capex before reporting proforma
        self.capex = self.ccost + (self.ccost_kw * self.dis_max_rated) + (self.ccost_kwh * self.ene_max_rated)
        proforma = super().proforma_report(opt_years, results)
        return proforma

    # def physical_properties(self):
    #     """
    #
    #     Returns: a dictionary of physical properties that define the ess
    #         includes 'charge max', 'discharge max, 'operation soc min', 'operation soc max', 'rte', 'energy cap'
    #
    #     """
    #     try:
    #         energy_rated = self.ene_max_rated.value
    #     except AttributeError:
    #         energy_rated = self.ene_max_rated
    #
    #     try:
    #         ch_max_rated = self.ch_max_rated.value
    #     except AttributeError:
    #         ch_max_rated = self.ch_max_rated
    #
    #     try:
    #         dis_max_rated = self.dis_max_rated.value
    #     except AttributeError:
    #         dis_max_rated = self.dis_max_rated
    #
    #     ess_properties = {'charge max': ch_max_rated,
    #                       'discharge max': dis_max_rated,
    #                       'rte': self.rte,
    #                       'energy cap': energy_rated,
    #                       'operation soc min': self.llsoc,
    #                       'operation soc max': self.ulsoc}
    #     return ess_properties

    def being_sized(self):
        """ checks itself to see if this instance is being sized

        Returns: true if being sized, false if not being sized

        """
        return bool(len(self.size_constraints))
