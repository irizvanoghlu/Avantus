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

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class BatterySizing(storagevet.BatteryTech):
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
        storagevet.BatteryTech.__init__(self, name,  opt_agg, params, cycle_life)

        self.size_constraints = []

        self.optimization_variables = {}

        # if the user inputted the energy rating as 0, then size for duration
        if not self.ene_max_rated:
            self.ene_max_rated = cvx.Variable(name='Energy_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ene_max_rated)]
            self.optimization_variables['ene_max_rated'] = self.ene_max_rated

        # if both the discharge and charge ratings are 0, then size for both and set them equal to each other
        if not self.ch_max_rated and not self.dis_max_rated:
            self.ch_max_rated = cvx.Variable(name='power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]
            self.dis_max_rated = self.ch_max_rated
            self.optimization_variables['ch_max_rated'] = self.ch_max_rated
            self.optimization_variables['dis_max_rated'] = self.dis_max_rated

        elif not self.ch_max_rated:  # if the user inputted the discharge rating as 0, then size discharge rating
            self.ch_max_rated = cvx.Variable(name='charge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]
            self.optimization_variables['ch_max_rated'] = self.ch_max_rated

        elif not self.dis_max_rated:  # if the user inputted the charge rating as 0, then size for charge
            self.dis_max_rated = cvx.Variable(name='discharge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.dis_max_rated)]
            self.optimization_variables['dis_max_rated'] = self.dis_max_rated

        self.capex = self.ccost + (self.ccost_kw * self.dis_max_rated) + (self.ccost_kwh * self.ene_max_rated)
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
        storagevet.BatteryTech.objective_function(self, variables, mask, annuity_scalar)

        self.costs.update({'capex': self.capex})
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
                                       'Capital Cost ($)': self.ccost,
                                       'Capital Cost ($/kW)': self.ccost_kw,
                                       'Capital Cost ($/kWh)': self.ccost_kwh}, index=index)
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

        constraint_list = []

        size = int(np.sum(mask))
        ene_target = self.soc_target * self.ene_max_rated

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

        # create cvx parameters of control constraints (this improves readability in cvx costs and better handling)
        # ene_max = cvx.Parameter(size, value=self.control_constraints['ene_max'].value[mask].values, name='ene_max')
        # ene_min = cvx.Parameter(size, value=self.control_constraints['ene_min'].value[mask].values, name='ene_min')
        # ch_max = cvx.Parameter(size, value=self.control_constraints['ch_max'].value[mask].values, name='ch_max')
        # ch_min = cvx.Parameter(size, value=self.control_constraints['ch_min'].value[mask].values, name='ch_min')
        # dis_max = cvx.Parameter(size, value=self.control_constraints['dis_max'].value[mask].values, name='dis_max')
        # dis_min = cvx.Parameter(size, value=self.control_constraints['dis_min'].value[mask].values, name='dis_min')
        ene_max = self.physical_constraints['ene_max_rated'].value
        ene_min = self.control_constraints['ene_min'].value[mask].values
        ch_max = self.physical_constraints['ch_max_rated'].value  #TODO: this will break if we have any max charge/discharge constraints
        ch_min = self.control_constraints['ch_min'].value[mask].values
        dis_max = self.physical_constraints['dis_max_rated'].value
        dis_min = self.control_constraints['dis_min'].value[mask].values

        # ene_max = self.ene_max_rated
        # ch_max = self.ch_max_rated
        # dis_max = self.dis_max_rated
        # ch_min = 0
        # dis_min = 0

        # energy at the end of the last time step
        constraint_list += [cvx.Zero((ene_target - ene[-1]) - (self.dt * ch[-1] * self.rte) + (self.dt * dis[-1]) - reservations['E'][-1] + (self.dt * ene[-1] * self.sdr * 0.01))]

        # energy generally for every time step
        constraint_list += [cvx.Zero(ene[1:] - ene[:-1] - (self.dt * ch[:-1] * self.rte) + (self.dt * dis[:-1]) - reservations['E'][:-1] + (self.dt * ene[:-1] * self.sdr * 0.01))]

        # energy at the beginning of the optimization window
        if mpc_ene is None:
            constraint_list += [cvx.Zero(ene[0] - ene_target)]
        else:
            constraint_list += [cvx.Zero(ene[0] - mpc_ene)]

        # Keep energy in bounds determined in the constraints configuration function -- making sure our storage meets control constraints
        constraint_list += [cvx.NonPos(ene_target - ene_max + reservations['E_upper'][-1] - variables['ene_max_slack'][-1])]
        constraint_list += [cvx.NonPos(ene[:-1] - ene_max + reservations['E_upper'][:-1] - variables['ene_max_slack'][:-1])]

        constraint_list += [cvx.NonPos(-ene_target + ene_min[-1] + reservations['E_lower'][-1] - variables['ene_min_slack'][-1])]
        constraint_list += [cvx.NonPos(ene_min[1:] - ene[1:] + reservations['E_lower'][:-1] - variables['ene_min_slack'][:-1])]

        # Keep charge and discharge power levels within bounds
        constraint_list += [cvx.NonPos(-ch_max + ch - dis + reservations['D_min'] + reservations['C_max'] - variables['ch_max_slack'])]
        constraint_list += [cvx.NonPos(-ch + dis + reservations['C_min'] + reservations['D_max'] - dis_max - variables['dis_max_slack'])]

        constraint_list += [cvx.NonPos(ch - cvx.multiply(ch_max, on_c))]
        constraint_list += [cvx.NonPos(dis - cvx.multiply(dis_max, on_d))]

        # removing the band in between ch_min and dis_min that the battery will not operate in
        constraint_list += [cvx.NonPos(cvx.multiply(ch_min, on_c) - ch + reservations['C_min'])]
        constraint_list += [cvx.NonPos(cvx.multiply(dis_min, on_d) - dis + reservations['D_min'])]
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
            ind_d = [i for i in range(size) if self.control_constraints['dis_min'].value[mask].values[i] > self.physical_constraints['dis_min_rated'].value]
            ind_c = [i for i in range(size) if self.control_constraints['ch_min'].value[mask].values[i] > self.physical_constraints['ch_min_rated'].value]
            if len(ind_d) > 0:
                constraint_list += [on_d[ind_d] == 1]  # np.ones(len(ind_d))
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

        constraint_list += self.size_constraints

        return constraint_list

    def calculate_control_constraints(self, datetimes):
        """ Generates a list of master or 'control constraints' from physical constraints and all
        predispatch service constraints.

        Args:
            datetimes (list): The values of the datetime column within the initial time_series data frame.

        Returns:
            Array of datetimes where the control constraints conflict and are infeasible. If all feasible return None.

        Note: the returned failed array returns the first infeasibility found, not all feasibilities.
        TODO: come back and check the user inputted constraints --HN
        """
        # create temp dataframe with values from physical_constraints
        temp_constraints = pd.DataFrame(index=datetimes)
        # create a df with all physical constraint values
        for constraint in self.physical_constraints.values():
            temp_constraints[re.search('^.+_.+_', constraint.name).group(0)[0:-1]] = copy.deepcopy(constraint.value)

        # change physical constraint with predispatch service constraints at each timestep
        # predispatch service constraints should be absolute constraints
        for service in self.predispatch_services.values():
            for constraint in service.constraints.values():
                if constraint.value is not None:
                    strp = constraint.name.split('_')
                    const_name = strp[0]
                    const_type = strp[1]
                    name = const_name + '_' + const_type
                    absolute_const = constraint.value.values  # constraint values
                    absolute_index = constraint.value.index  # the datetimes for which the constraint applies

                    current_const = temp_constraints.loc[absolute_index, name].values  # value of the current constraint

                    if const_type == "min":
                        # if minimum constraint, choose higher constraint value
                        try:
                            temp_constraints.loc[absolute_index, name] = np.max(absolute_const, current_const)
                        except TypeError:
                            temp_constraints.loc[absolute_index, name] = absolute_const
                        # temp_constraints.loc[constraint.value.index, name] += constraint.value.values

                        # if the minimum value needed is greater than the physical maximum, infeasible scenario
                        max_value = self.physical_constraints[const_name + '_max' + '_rated'].value
                        try:
                            constraint_violation = any(temp_constraints[name] > max_value)
                        except (ValueError, TypeError):
                            constraint_violation = False
                        if constraint_violation:
                            return temp_constraints[temp_constraints[name] > max_value].index

                    else:
                        # if maximum constraint, choose lower constraint value
                        try:
                            temp_constraints.loc[absolute_index, name] = np.min(absolute_const, current_const)
                        except TypeError:
                            temp_constraints.loc[absolute_index, name] = absolute_const
                        # temp_constraints.loc[constraint.value.index, name] -= constraint.value.values

                        # if the maximum energy needed is less than the physical minimum, infeasible scenario
                        min_value = self.physical_constraints[const_name + '_min' + '_rated'].value
                        try:
                            constraint_violation = any(temp_constraints[name] > min_value)
                        except (ValueError, TypeError):
                            constraint_violation = False
                        if (const_name == 'ene') & constraint_violation:

                            return temp_constraints[temp_constraints[name] > max_value].index
                        else:
                            # it is ok to floor at zero since negative power max values will be handled in power min
                            # i.e negative ch_max means dis_min should be positive and ch_max should be 0)
                            temp_constraints[name] = temp_constraints[name].clip(lower=0)
            if service.name == 'UserConstraints':
                user_inputted_constraint = service.user_constraints
                for user_constraint_name in user_inputted_constraint:
                    # determine if the user inputted constraint is a max or min constraint
                    user_constraint = user_inputted_constraint[user_constraint_name]
                    const_type = user_constraint_name.split('_')[1]
                    if const_type == 'max':
                        # iterate through user inputted constraint Series
                        for i in user_constraint.index:
                            # update temp_constraints df if user inputted a lower max constraint
                            if temp_constraints[user_constraint_name].loc[i] > user_constraint.loc[i]:
                                temp_constraints[user_constraint_name].loc[i] = user_constraint.loc[i]
                    elif const_type == 'min':
                        # iterate through user inputted constraint Series
                        for i in user_constraint.index:
                            # update temp_constraints df if user inputted a higher min constraint
                            if temp_constraints[user_constraint_name].loc[i] < user_constraint.loc[i]:
                                temp_constraints[user_constraint_name].loc[i] = user_constraint.loc[i]
                    else:
                        dLogger.error("User has inputted an invalid constraint for Storage. Please change and run again.")
                        e_logger.error("User has inputted an invalid constraint for Storage. Please change and run again.")
                        sys.exit()

        # now that we have a new list of constraints, create Constraint objects and store as 'control constraint'
        self.control_constraints = {'ene_min': Const.Constraint('ene_min', self.name, temp_constraints['ene_min']),
                                    'ene_max': Const.Constraint('ene_max', self.name, temp_constraints['ene_max']),
                                    'ch_min': Const.Constraint('ch_min', self.name, temp_constraints['ch_min']),
                                    'ch_max': Const.Constraint('ch_max', self.name, temp_constraints['ch_max']),
                                    'dis_min': Const.Constraint('dis_min', self.name, temp_constraints['dis_min']),
                                    'dis_max': Const.Constraint('dis_max', self.name, temp_constraints['dis_max'])}
        return None

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            bat_ene (Variable): A cvxpy variable for Energy at the end of the time step
            bat_dis (Variable): A cvxpy variable for Discharge Power, kW during the previous time step
            bat_ch (Variable): A cvxpy variable for Charge Power, kW during the previous time step
            bat_ene_max_slack (Variable): A cvxpy variable for energy max slack
            bat_ene_min_slack (Variable): A cvxpy variable for energy min slack
            bat_ch_max_slack (Variable): A cvxpy variable for charging max slack
            bat_ch_min_slack (Variable): A cvxpy variable for charging min slack
            bat_dis_max_slack (Variable): A cvxpy variable for discharging max slack
            bat_dis_min_slack (Variable): A cvxpy variable for discharging min slack

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        variables = {'bat_ene': cvx.Variable(shape=size, name='bat_ene'),
                     'bat_dis': cvx.Variable(shape=size, name='bat_dis'),
                     'bat_ch': cvx.Variable(shape=size, name='bat_ch'),
                     'bat_ene_max_slack': cvx.Parameter(shape=size, name='bat_ene_max_slack', value=np.zeros(size)),
                     'bat_ene_min_slack': cvx.Parameter(shape=size, name='bat_ene_min_slack', value=np.zeros(size)),
                     'bat_dis_max_slack': cvx.Parameter(shape=size, name='bat_dis_max_slack', value=np.zeros(size)),
                     'bat_dis_min_slack': cvx.Parameter(shape=size, name='bat_dis_min_slack', value=np.zeros(size)),
                     'bat_ch_max_slack': cvx.Parameter(shape=size, name='bat_ch_max_slack', value=np.zeros(size)),
                     'bat_ch_min_slack': cvx.Parameter(shape=size, name='bat_ch_min_slack', value=np.zeros(size)),
                     'bat_on_c': cvx.Parameter(shape=size, name='bat_on_c', value=np.ones(size)),
                     'bat_on_d': cvx.Parameter(shape=size, name='bat_on_d', value=np.ones(size)),
                     }

        if self.incl_slack:
            self.variable_names.update(['bat_ene_max_slack', 'bat_ene_min_slack', 'bat_dis_max_slack', 'bat_dis_min_slack', 'bat_ch_max_slack', 'bat_ch_min_slack'])
            variables.update({'bat_ene_max_slack': cvx.Variable(shape=size, name='bat_ene_max_slack'),
                              'bat_ene_min_slack': cvx.Variable(shape=size, name='bat_ene_min_slack'),
                              'bat_dis_max_slack': cvx.Variable(shape=size, name='bat_dis_max_slack'),
                              'bat_dis_min_slack': cvx.Variable(shape=size, name='bat_dis_min_slack'),
                              'bat_ch_max_slack': cvx.Variable(shape=size, name='bat_ch_max_slack'),
                              'bat_ch_min_slack': cvx.Variable(shape=size, name='bat_ch_min_slack')})
        if self.incl_binary:
            self.variable_names.update(['bat_on_c', 'bat_on_d'])
            variables.update({'bat_on_c': cvx.Variable(shape=size, boolean=True, name='bat_on_c'),
                              'bat_on_d': cvx.Variable(shape=size, boolean=True, name='bat_on_d')})
            if self.incl_startup:
                self.variable_names.update(['bat_start_c', 'bat_start_d'])
                variables.update({'bat_start_c': cvx.Variable(shape=size, name='bat_start_c'),
                                  'bat_start_d': cvx.Variable(shape=size, name='bat_start_d')})

        variables.update(self.optimization_variables)

        return variables