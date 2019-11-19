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
import re
import copy
import sys

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class CAESSizing(storagevet.CAESTech):
    """ CAES class that inherits from Storage.

    """

    def __init__(self, name,  opt_agg, params, cycle_life):
        """ Initializes CAES class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            opt_agg (DataFrame): Initialized Financial Class
            params (dict): params dictionary from dataframe for one case
            cycle_life (DataFrame): Cycle life information
        """

        # create generic storage object
        storagevet.CAESTech.__init__(self, name,  opt_agg, params, cycle_life)

        self.size_constraints = []

        self.optimization_variables = {}

        self.physical_constraints = {
            'caes_ene_min_rated': Const.Constraint('caes_ene_min_rated', self.name, self.llsoc * self.ene_max_rated),
            'caes_ene_max_rated': Const.Constraint('caes_ene_max_rated', self.name, self.ulsoc * self.ene_max_rated),
            'caes_ch_min_rated': Const.Constraint('caes_ch_min_rated', self.name, self.ch_min_rated),
            'caes_ch_max_rated': Const.Constraint('caes_ch_max_rated', self.name, self.ch_max_rated),
            'caes_dis_min_rated': Const.Constraint('caes_dis_min_rated', self.name, self.dis_min_rated),
            'caes_dis_max_rated': Const.Constraint('caes_dis_max_rated', self.name, self.dis_max_rated)}

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            caes_ene (Variable): A cvxpy variable for Energy at the end of the time step
            caes_dis (Variable): A cvxpy variable for Discharge Power, kW during the previous time step
            caes_ch (Variable): A cvxpy variable for Charge Power, kW during the previous time step
            caes_ene_max_slack (Variable): A cvxpy variable for energy max slack
            caes_ene_min_slack (Variable): A cvxpy variable for energy min slack
            caes_ch_max_slack (Variable): A cvxpy variable for charging max slack
            caes_ch_min_slack (Variable): A cvxpy variable for charging min slack
            caes_dis_max_slack (Variable): A cvxpy variable for discharging max slack
            caes_dis_min_slack (Variable): A cvxpy variable for discharging min slack

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        variables = {'caes_ene': cvx.Variable(shape=size, name='caes_ene'),
                     'caes_dis': cvx.Variable(shape=size, name='caes_dis'),
                     'caes_ch': cvx.Variable(shape=size, name='caes_ch'),
                     'caes_ene_max_slack': cvx.Parameter(shape=size, name='caes_ene_max_slack', value=np.zeros(size)),
                     'caes_ene_min_slack': cvx.Parameter(shape=size, name='caes_ene_min_slack', value=np.zeros(size)),
                     'caes_dis_max_slack': cvx.Parameter(shape=size, name='caes_dis_max_slack', value=np.zeros(size)),
                     'caes_dis_min_slack': cvx.Parameter(shape=size, name='caes_dis_min_slack', value=np.zeros(size)),
                     'caes_ch_max_slack': cvx.Parameter(shape=size, name='caes_ch_max_slack', value=np.zeros(size)),
                     'caes_ch_min_slack': cvx.Parameter(shape=size, name='caes_ch_min_slack', value=np.zeros(size)),
                     'caes_on_c': cvx.Parameter(shape=size, name='caes_on_c', value=np.ones(size)),
                     'caes_on_d': cvx.Parameter(shape=size, name='caes_on_d', value=np.ones(size)),
                     }

        if self.incl_slack:
            self.variable_names.update(['caes_ene_max_slack', 'caes_ene_min_slack', 'caes_dis_max_slack', 'caes_dis_min_slack', 'caes_ch_max_slack', 'caes_ch_min_slack'])
            variables.update({'caes_ene_max_slack': cvx.Variable(shape=size, name='caes_ene_max_slack'),
                              'caes_ene_min_slack': cvx.Variable(shape=size, name='caes_ene_min_slack'),
                              'caes_dis_max_slack': cvx.Variable(shape=size, name='caes_dis_max_slack'),
                              'caes_dis_min_slack': cvx.Variable(shape=size, name='caes_dis_min_slack'),
                              'caes_ch_max_slack': cvx.Variable(shape=size, name='caes_ch_max_slack'),
                              'caes_ch_min_slack': cvx.Variable(shape=size, name='caes_ch_min_slack')})
        if self.incl_binary:
            self.variable_names.update(['caes_on_c', 'caes_on_d'])
            variables.update({'caes_on_c': cvx.Variable(shape=size, boolean=True, name='caes_on_c'),
                              'caes_on_d': cvx.Variable(shape=size, boolean=True, name='caes_on_d')})
            if self.incl_startup:
                self.variable_names.update(['caes_start_c', 'caes_start_d'])
                variables.update({'caes_start_c': cvx.Variable(shape=size, name='caes_start_c'),
                                  'caes_start_d': cvx.Variable(shape=size, name='caes_start_d')})

        variables.update(self.optimization_variables)

        return variables

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

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = storagevet.CAESTech.timeseries_report(self)
        results[self.name + ' CAES Discharge (kW)'] = self.variables['caes_dis']
        results[self.name + ' CAES Charge (kW)'] = self.variables['caes_ch']
        results[self.name + ' CAES Power (kW)'] = self.variables['caes_dis'] - self.variables['caes_ch']
        results[self.name + ' CAES State of Energy (kWh)'] = self.variables['caes_ene']

        try:
            energy_rate = self.ene_max_rated.value
        except AttributeError:
            energy_rate = self.ene_max_rated

        results['CAES SOC (%)'] = self.variables['caes_ene'] / energy_rate
        results['CAES Fuel Price ($)'] = self.fuel_price

        return results

    def objective_function(self, variables, mask, tech_id, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.power_kw
            tech_id (str): name of the technology associated with the active service
                        (lower case letters identifying the optimization variables identified in the technology)
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                    the entire project lifetime (only to be set iff sizing, else annuity_scalar should not affect the aobject function)

        Returns:
            self.costs (Dict): Dict of objective costs
        """

        storagevet.CAESTech.objective_function(self, variables, mask, tech_id, annuity_scalar=1)

        # create objective expression for variable om based on discharge activity
        var_om = cvx.sum(variables[tech_id + '_dis']) * self.OMexpenses * self.dt * 1e-3 * annuity_scalar

        self.costs = {
            tech_id + '_fixed_om': self.fixed_om * annuity_scalar,
            tech_id + '_var_om': var_om}

        # add slack objective costs. These try to keep the slack variables as close to 0 as possible
        if self.incl_slack:
            self.costs.update({
                tech_id + '_ene_max_slack': cvx.sum(self.kappa_ene_max * variables[tech_id + '_ene_max_slack']),
                tech_id + '_ene_min_slack': cvx.sum(self.kappa_ene_min * variables[tech_id + '_ene_min_slack']),
                tech_id + '_dis_max_slack': cvx.sum(self.kappa_dis_max * variables[tech_id + '_dis_max_slack']),
                tech_id + '_dis_min_slack': cvx.sum(self.kappa_dis_min * variables[tech_id + '_dis_min_slack']),
                tech_id + '_ch_max_slack': cvx.sum(self.kappa_ch_max * variables[tech_id + '_ch_max_slack']),
                tech_id + '_ch_min_slack': cvx.sum(self.kappa_ch_min * variables[tech_id + '_ch_min_slack'])})

        # add startup objective costs
        if self.incl_startup:
            self.costs.update({
                          tech_id + '_ch_startup': cvx.sum(variables[tech_id + '_start_c']) * self.p_start_ch * annuity_scalar,
                          tech_id + '_dis_startup': cvx.sum(variables[tech_id + '_start_d']) * self.p_start_dis * annuity_scalar})

        return self.costs

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
                        e_logger.error("User has inputted an invalid constraint for Storage. Please change and run again.")
                        sys.exit()

        # now that we have a new list of constraints, create Constraint objects and store as 'control constraint'
        self.control_constraints = {'caes_ene_min': Const.Constraint('caes_ene_min', self.name, temp_constraints['caes_ene_min']),
                                    'caes_ene_max': Const.Constraint('caes_ene_max', self.name, temp_constraints['caes_ene_max']),
                                    'caes_ch_min': Const.Constraint('caes_ch_min', self.name, temp_constraints['caes_ch_min']),
                                    'caes_ch_max': Const.Constraint('caes_ch_max', self.name, temp_constraints['caes_ch_max']),
                                    'caes_dis_min': Const.Constraint('caes_dis_min', self.name, temp_constraints['caes_dis_min']),
                                    'caes_dis_max': Const.Constraint('caes_dis_max', self.name, temp_constraints['caes_dis_max'])}
        return None

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
        ene = variables['caes_ene']
        dis = variables['caes_dis']
        ch = variables['caes_ch']
        on_c = variables['caes_on_c']
        on_d = variables['caes_on_d']
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
        ene_max = self.physical_constraints['caes_ene_max_rated'].value
        ene_min = self.control_constraints['caes_ene_min'].value[mask].values
        ch_max = self.physical_constraints['caes_ch_max_rated'].value  #TODO: this will break if we have any max charge/discharge constraints
        ch_min = self.control_constraints['caes_ch_min'].value[mask].values
        dis_max = self.physical_constraints['caes_dis_max_rated'].value
        dis_min = self.control_constraints['caes_dis_min'].value[mask].values

        # ene_max = self.ene_max_rated
        # ch_max = self.ch_max_rated
        # dis_max = self.dis_max_rated
        # ch_min = 0
        # dis_min = 0

        # energy at the end of the last time step
        constraint_list += [cvx.Zero((ene_target - ene[-1]) - (self.dt * ch[-1] * self.rte) + (self.dt * dis[-1]) - reservations['CAES']['CAES E'][-1] + (self.dt * ene[-1] * self.sdr * 0.01))]

        # energy generally for every time step
        constraint_list += [cvx.Zero(ene[1:] - ene[:-1] - (self.dt * ch[:-1] * self.rte) + (self.dt * dis[:-1]) - reservations['CAES']['CAES E'][:-1] + (self.dt * ene[:-1] * self.sdr * 0.01))]

        # energy at the beginning of the optimization window
        if mpc_ene is None:
            constraint_list += [cvx.Zero(ene[0] - ene_target)]
        else:
            constraint_list += [cvx.Zero(ene[0] - mpc_ene)]

        # Keep energy in bounds determined in the constraints configuration function
        constraint_list += [cvx.NonPos(ene_target - ene_max + reservations['CAES']['CAES E_upper'][-1] - variables['caes_ene_max_slack'][-1])]
        constraint_list += [cvx.NonPos(ene[1:] - ene_max + reservations['CAES']['CAES E_upper'][:-1] - variables['caes_ene_max_slack'][:-1])]

        constraint_list += [cvx.NonPos(-ene_target + ene_min[-1] - (pv_gen[-1]*self.dt) - (ice_gen[-1]*self.dt) - reservations['CAES']['CAES E_lower'][-1] - variables['caes_ene_min_slack'][-1])]
        constraint_list += [cvx.NonPos(ene_min[1:] - (pv_gen[1:]*self.dt) - (ice_gen[1:]*self.dt) - ene[1:] + reservations['CAES']['CAES E_lower'][:-1] - variables['caes_ene_min_slack'][:-1])]

        # Keep charge and discharge power levels within bounds
        constraint_list += [cvx.NonPos(ch - cvx.multiply(ch_max, on_c) - variables['caes_ch_max_slack'])]
        constraint_list += [cvx.NonPos(ch - ch_max + reservations['CAES']['CAES C_max'] - variables['caes_ch_max_slack'])]

        constraint_list += [cvx.NonPos(cvx.multiply(ch_min, on_c) - ch - variables['caes_ch_min_slack'])]
        constraint_list += [cvx.NonPos(ch_min - ch + reservations['CAES']['CAES C_min'] - variables['caes_ch_min_slack'])]

        constraint_list += [cvx.NonPos(dis - cvx.multiply(dis_max, on_d) - variables['caes_dis_max_slack'])]
        constraint_list += [cvx.NonPos(dis - dis_max + reservations['CAES']['CAES D_max'] - variables['caes_dis_max_slack'])]

        constraint_list += [cvx.NonPos(cvx.multiply(dis_min, on_d) - dis - variables['caes_dis_min_slack'])]
        constraint_list += [cvx.NonPos(dis_min - dis + reservations['CAES']['CAES D_min'] - variables['caes_dis_min_slack'])]
        # constraints to keep slack variables positive
        if self.incl_slack:
            constraint_list += [cvx.NonPos(-variables['caes_ch_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['caes_ch_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['caes_dis_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['caes_dis_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['caes_ene_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['caes_ene_min_slack'])]

        if self.incl_binary:
            # when dis_min or ch_min has been overwritten (read: increased) by predispatch services, need to force technology to be on
            # TODO better way to do this???
            ind_d = [i for i in range(size) if self.control_constraints['caes_dis_min'].value[mask].values[i] > self.physical_constraints['caes_dis_min_rated'].value]
            ind_c = [i for i in range(size) if self.control_constraints['caes_ch_min'].value[mask].values[i] > self.physical_constraints['caes_ch_min_rated'].value]
            if len(ind_d) > 0:
                constraint_list += [on_d[ind_d] == 1]  # np.ones(len(ind_d))
            if len(ind_c) > 0:
                constraint_list += [on_c[ind_c] == 1]  # np.ones(len(ind_c))

            # note: cannot operate startup without binary
            if self.incl_startup:
                # startup variables are positive
                constraint_list += [cvx.NonPos(-variables['caes_start_d'])]
                constraint_list += [cvx.NonPos(-variables['caes_start_c'])]
                # difference between binary variables determine if started up in previous interval
                constraint_list += [cvx.NonPos(cvx.diff(on_d) - variables['caes_start_d'][1:])]  # first variable not constrained
                constraint_list += [cvx.NonPos(cvx.diff(on_c) - variables['caes_start_c'][1:])]  # first variable not constrained

        constraint_list += self.size_constraints

        return constraint_list

