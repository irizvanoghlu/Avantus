"""
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

from storagevet.Technology.BatteryTech import BatteryTech
import logging
import cvxpy as cvx
import pandas as pd
import numpy as np

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class BatterySizing(BatteryTech):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, name,  opt_agg, params, cycle_life):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            opt_agg (DataFrame): Initalized Financial Class
            params (dict): params dictionary from dataframe for one case
            cycle_life (DataFrame): Cycle life information
        """

        # create generic storage object
        BatteryTech.__init__(self, name,  opt_agg, params, cycle_life)

        # if the user inputted the energy rating as 0, then size for duration
        if not self.ene_max_rated:
            self.ene_max_rated = cvx.Variable(name='Energy_cap')

        # if both the discharge and charge ratings are 0, then size for both and set them equal to each other
        if not self.ch_max_rated and not self.dis_max_rated:
            self.ch_max_rated = cvx.Variable(name='power_cap')
            self.dis_max_rated = self.ch_max_rated
        elif not self.ch_max_rated:  # if the user inputted the discharge rating as 0, then size discharge rating
            self.ch_max_rated = cvx.Variable(name='charge_power_cap')
        elif not self.dis_max_rated:  # if the user inputted the charge rating as 0, then size for charge
            self.dis_max_rated = cvx.Variable(name='discharge_power_cap')

    def objective_function(self, variables, mask):
        BatteryTech.objective_function(self, variables, mask)
        # Calculate and add the annuity required to pay off the capex of the storage system. A more detailed financial model is required in the future
        capex = self.ene_max_rated * self.ccost_kwh + self.dis_max_rated * self.ccost_kw + self.ccost  # TODO: This is hard coded for battery storage
        # n = self.end_year - self.start_year
        annualized_capex = (capex * .11)  # TODO: Hardcoded ratio - need to calculate annuity payment and fit into a multiyear optimization framework
        self.capex = annualized_capex
        self.costs.update({'capex': annualized_capex})
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

        sizing_data = [energy_rated,
                       ch_max_rated,
                       dis_max_rated,
                       energy_rated / dis_max_rated,
                       self.ccost,
                       self.ccost_kw,
                       self.ccost_kwh]
        index = pd.Index(['Energy Rating (kWh)',
                          'Charge Rating (kW)',
                          'Discharge Rating (kW)',
                          'Duration (hours)',
                          'Capital Cost ($)',
                          'Capital Cost ($/kW)',
                          'Capital Cost ($/kWh)'], name='Size and Costs')
        sizing_results = pd.DataFrame({self.name: sizing_data}, index=index)
        return sizing_results

    # def build_master_constraints(self, variables, mask, reservations, mpc_ene=None):
    #     """ Builds the master constraint list for the subset of timeseries data being optimized.
    #
    #     Args:
    #         variables (Dict): Dictionary of variables being optimized
    #         mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
    #             in the subs data set
    #         reservations (Dict): Dictionary of energy and power reservations required by the services being
    #             preformed with the current optimization subset
    #         mpc_ene (float): value of energy at end of last opt step (for mpc opt)
    #
    #     Returns:
    #         A list of constraints that corresponds the battery's physical constraints and its service constraints
    #     """
    #
    #     constraint_list = []
    #
    #     size = int(np.sum(mask))
    #
    #     curr_e_cap = self.physical_constraints['ene_max_rated'].value
    #     ene_target = self.soc_target * curr_e_cap
    #
    #     # optimization variables
    #     ene = variables['ene']
    #     dis = variables['dis']
    #     ch = variables['ch']
    #     on_c = variables['on_c']
    #     on_d = variables['on_d']
    #     try:
    #         pv_gen = variables['pv_out']
    #     except KeyError:
    #         pv_gen = np.zeros(size)
    #     try:
    #         ice_gen = variables['ice_gen']
    #     except KeyError:
    #         ice_gen = np.zeros(size)
    #
    #     # create cvx parameters of control constraints (this improves readability in cvx costs and better handling)
    #     # ene_max = cvx.Parameter(size, value=self.control_constraints['ene_max'].value[mask].values, name='ene_max')
    #     ene_min = cvx.Parameter(size, value=self.control_constraints['ene_min'].value[mask].values, name='ene_min')
    #     # ch_max = cvx.Parameter(size, value=self.control_constraints['ch_max'].value[mask].values, name='ch_max')
    #     ch_min = cvx.Parameter(size, value=self.control_constraints['ch_min'].value[mask].values, name='ch_min')
    #     # dis_max = cvx.Parameter(size, value=self.control_constraints['dis_max'].value[mask].values, name='dis_max')
    #     dis_min = cvx.Parameter(size, value=self.control_constraints['dis_min'].value[mask].values, name='dis_min')
    #
    #     ene_max = self.ene_max_rated
    #     ch_max = self.ch_max_rated
    #     dis_max = self.dis_max_rated
    #     ch_min = 0
    #     dis_min = 0
    #
    #     # energy at the end of the last time step
    #     constraint_list += [cvx.Zero((ene_target - ene[-1]) - (self.dt * ch[-1] * self.rte) + (self.dt * dis[-1]) - reservations['E'][-1] + (self.dt * ene[-1] * self.sdr * 0.01))]
    #
    #     # energy generally for every time step
    #     constraint_list += [cvx.Zero(ene[1:] - ene[:-1] - (self.dt * ch[:-1] * self.rte) + (self.dt * dis[:-1]) - reservations['E'][:-1] + (self.dt * ene[:-1] * self.sdr * 0.01))]
    #
    #     # energy at the beginning of the optimization window
    #     if mpc_ene is None:
    #         constraint_list += [cvx.Zero(ene[0] - ene_target)]
    #     else:
    #         constraint_list += [cvx.Zero(ene[0] - mpc_ene)]
    #
    #     # # Keep energy in bounds determined in the constraints configuration function
    #     constraint_list += [cvx.NonPos(ene_target - ene_max + reservations['E_upper'][-1] - variables['ene_max_slack'][-1])]
    #     constraint_list += [cvx.NonPos(ene[1:] - ene_max + reservations['E_upper'][:-1] - variables['ene_max_slack'][:-1])]
    #
    #     constraint_list += [cvx.NonPos(-ene_target + ene_min[-1] - (pv_gen[-1]*self.dt) - (ice_gen[-1]*self.dt) - reservations['E_lower'][-1] - variables['ene_min_slack'][-1])]
    #     constraint_list += [cvx.NonPos(ene_min[1:] - (pv_gen[1:]*self.dt) - (ice_gen[1:]*self.dt) - ene[1:] + reservations['E_lower'][:-1] - variables['ene_min_slack'][:-1])]
    #
    #     # Keep charge and discharge power levels within bounds
    #     constraint_list += [cvx.NonPos(ch - cvx.multiply(ch_max, on_c) - variables['ch_max_slack'])]
    #     constraint_list += [cvx.NonPos(ch - ch_max + reservations['C_max'] - variables['ch_max_slack'])]
    #
    #     constraint_list += [cvx.NonPos(cvx.multiply(ch_min, on_c) - ch - variables['ch_min_slack'])]
    #     constraint_list += [cvx.NonPos(ch_min - ch + reservations['C_min'] - variables['ch_min_slack'])]
    #
    #     constraint_list += [cvx.NonPos(dis - cvx.multiply(dis_max, on_d) - variables['dis_max_slack'])]
    #     constraint_list += [cvx.NonPos(dis - dis_max + reservations['D_max'] - variables['dis_max_slack'])]
    #
    #     constraint_list += [cvx.NonPos(cvx.multiply(dis_min, on_d) - dis - variables['dis_min_slack'])]
    #     constraint_list += [cvx.NonPos(dis_min - dis + reservations['D_min'] - variables['dis_min_slack'])]
    #     # constraints to keep slack variables positive
    #     if self.incl_slack:
    #         constraint_list += [cvx.NonPos(-variables['ch_max_slack'])]
    #         constraint_list += [cvx.NonPos(-variables['ch_min_slack'])]
    #         constraint_list += [cvx.NonPos(-variables['dis_max_slack'])]
    #         constraint_list += [cvx.NonPos(-variables['dis_min_slack'])]
    #         constraint_list += [cvx.NonPos(-variables['ene_max_slack'])]
    #         constraint_list += [cvx.NonPos(-variables['ene_min_slack'])]
    #
    #     if self.incl_binary:
    #         # when dis_min or ch_min has been overwritten (read: increased) by predispatch services, need to force technology to be on
    #         # TODO better way to do this???
    #         ind_d = [i for i in range(size) if self.control_constraints['dis_min'].value[mask].values[i] > self.physical_constraints['dis_min_rated'].value]
    #         ind_c = [i for i in range(size) if self.control_constraints['ch_min'].value[mask].values[i] > self.physical_constraints['ch_min_rated'].value]
    #         if len(ind_d) > 0:
    #             constraint_list += [on_d[ind_d] == 1]  # np.ones(len(ind_d))
    #         if len(ind_c) > 0:
    #             constraint_list += [on_c[ind_c] == 1]  # np.ones(len(ind_c))
    #
    #         # note: cannot operate startup without binary
    #         if self.incl_startup:
    #             # startup variables are positive
    #             constraint_list += [cvx.NonPos(-variables['start_d'])]
    #             constraint_list += [cvx.NonPos(-variables['start_c'])]
    #             # difference between binary variables determine if started up in previous interval
    #             constraint_list += [cvx.NonPos(cvx.diff(on_d) - variables['start_d'][1:])]  # first variable not constrained
    #             constraint_list += [cvx.NonPos(cvx.diff(on_c) - variables['start_c'][1:])]  # first variable not constrained
    #
    #     return constraint_list
