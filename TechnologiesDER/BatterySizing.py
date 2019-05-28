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

from dervet.storagevet.Technology.Storage import Storage
import copy
import logging
import storagevet.Constraint as Const
import re
import sys
import numpy as np
import pandas as pd
import rainflow
import cvxpy as cvx

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')


class BatterySizing(Storage):
    """ Battery class that inherits from Technology.

    """

    def __init__(self, name,  opt_agg, params, tech_params, cycle_life):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            opt_agg (Analysis): Initalized Financial Class
            tech_params (dict): params dictionary from dataframe for one case
            cycle_life (DataFrame): Cycle life information
        """
        # update ene_max, ch_max, and dis_max into cvxpy Variables
        tech_params['ene_max_rated'] = cvx.Variable(name='Energy_cap')

        power_capacity = cvx.Variable(name='power_cap')
        tech_params['dis_max_rated'] = power_capacity
        tech_params['dis_min_rated'] = 0
        tech_params['ch_max_rated'] = power_capacity
        tech_params['ch_min_rated'] = 0

        # create generic technology object
        Storage.__init__(self, name, tech_params)

        # add degradation information
        self.cycle_life = cycle_life
        self.degrade_data = pd.DataFrame(index=opt_agg)

        # calculate current degrade_perc since installation
        if tech_params['incl_cycle_degrade']:
            start_dttm = opt_agg[0]
            degrade_perc = self.calc_degradation(self.install_date, start_dttm)
            self.degrade_data['degrade_perc'] = degrade_perc
            self.degrade_data['eff_e_cap'] = self.apply_degradation(degrade_perc)

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
            A list of constraints that corresponds the battery's physical constraints and its
            service constraints.
        """

        constraint_list = []
        size = int(np.sum(mask))
        curr_e_cap = self.physical_constraints['ene_max_rated'].value
        ene_target = self.soc_target * curr_e_cap

        # optimization variables
        ene = variables['ene']
        dis = variables['dis']
        ch = variables['ch']
        on_c = variables['on_c']
        on_d = variables['on_d']
        pv_gen = variables['pv_out']

        # create cvx parameters of control constraints (this improves readability in cvx expressions and better handling)
        # ene_max = cvx.Parameter(size, value=self.control_constraints['ene_max'].value[mask].values, name='ene_max')
        ene_max = self.ene_max_rated
        ene_min = cvx.Parameter(size, value=self.control_constraints['ene_min'].value[mask].values, name='ene_min')
        # # ch_max = cvx.Parameter(size, value=self.control_constraints['ch_max'].value[mask].values, name='ch_max')
        # ch_max = self.ch_max_rated
        # ch_min = cvx.Parameter(size, value=self.control_constraints['ch_min'].value[mask].values, name='ch_min')
        # # dis_max = cvx.Parameter(size, value=self.control_constraints['dis_max'].value[mask].values, name='dis_max')
        # dis_max = self.dis_max_rated
        # dis_min = cvx.Parameter(size, value=self.control_constraints['dis_min'].value[mask].values, name='dis_min')

        # energy at the end of the first time step
        constraint_list += [cvx.Zero((ene_target - ene[0]) + (self.dt * ch[0] * self.rte) - (self.dt * dis[0]) - reservations['E'][0] - (self.dt * ene_target * self.sdr * 0.01))]
        # energy after every time step
        constraint_list += [cvx.Zero((ene[:-1] - ene[1:]) + (self.dt * ch[1:] * self.rte) - (self.dt * dis[1:]) - reservations['E'][1:] - (self.dt * ene[1:] * self.sdr * 0.01))]

        # energy at the end of the optimization window
        constraint_list += [cvx.Zero(ene[-1] - ene_target)]

        # Keep energy in bounds determined in the constraints configuration function
        constraint_list += [cvx.NonPos(ene_target - ene_max + reservations['E_upper'][0] - variables['ene_max_slack'][0])]
        constraint_list += [cvx.NonPos(ene[:-1] - ene_max + reservations['E_upper'][1:] - variables['ene_max_slack'][1:])]

        constraint_list += [cvx.NonPos(-ene_target + ene_min[0] - (pv_gen[0]*self.dt) - reservations['E_lower'][0] - variables['ene_min_slack'][0])]
        constraint_list += [cvx.NonPos(ene_min[1:] - (pv_gen[1:]*self.dt) - ene[:-1] - reservations['E_lower'][1:] - variables['ene_min_slack'][1:])]

        # Keep charge and discharge power levels within bounds
        constraint_list += [cvx.NonPos(ch - cvx.multiply(self.ch_max_rated, on_c) + reservations['C_max'] - variables['ch_max_slack'])]
        constraint_list += [cvx.NonPos(cvx.multiply(self.ch_min_rated, on_c) - ch + reservations['C_min'] - variables['ch_min_slack'])]
        constraint_list += [cvx.NonPos(dis - cvx.multiply(self.dis_max_rated, on_d) + reservations['D_max'] - variables['dis_max_slack'])]
        constraint_list += [cvx.NonPos(cvx.multiply(self.dis_min_rated, on_d) - dis + reservations['D_min'] - variables['dis_min_slack'])]

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

            # add constraint that battery can not charge and discharge in the same timestep
            constraint_list += [variables['on_c'] + variables['on_d'] <= 1]

            # note: cannot operate startup without binary
            if self.incl_startup:
                # startup variables are positive
                constraint_list += [cvx.NonPos(-variables['start_d'])]
                constraint_list += [cvx.NonPos(-variables['start_c'])]
                # difference between binary variables determine if started up in previous interval
                constraint_list += [cvx.NonPos(cvx.diff(on_d) - variables['start_d'][1:])]  # first variable not constrained
                constraint_list += [cvx.NonPos(cvx.diff(on_c) - variables['start_c'][1:])]  # first variable not constrained

        return constraint_list

    def calculate_control_constraints(self, datetimes, user_inputted_constraint=pd.DataFrame):
        """ Generates a list of master or 'control constraints' from physical constraints and all
        predispatch service constraints.

        Args:
            datetimes (list): The values of the datetime column within the initial time_series data frame.
            user_inputted_constraint (DataFrame): timeseries of any user inputed constraints.

        Returns:
            Array of datetimes where the control constraints conflict and are infeasible. If all feasible return None.

        Note: the returned failed array returns the first infeasibility found, not all feasibilities.
        TODO: come back and check the user inputted constraints --HN
        """
        # create temp dataframe with values from physical_constraints
        temp_constraints = pd.DataFrame(index=datetimes)
        # create a df with all physical constraint values  #todo: this does not look at physical technology contraints anymore, only predispatch
        for constraint in self.physical_constraints.values():
            temp_constraints[re.search('^.+_.+_', constraint.name).group(0)[0:-1]] = np.zeros(len(datetimes))

        # change physical constraint with predispatch service constraints at each timestep
        # predispatch service constraints can add to minimums or subtract from maximums (formulated this way so they can stack)
        for service in self.predispatch_services.values():
            for constraint in service.constraints.values():
                if constraint.value is not None:
                    strp = constraint.name.split('_')
                    const_name = strp[0]
                    const_type = strp[1]
                    name = const_name + '_' + const_type

                    if const_type == "min":
                        # if minimum constraint, add predispatch constraint value
                        temp_constraints.loc[constraint.value.index, name] += constraint.value.values
                    else:
                        # if maximum constraint, subtract predispatch constraint value
                        temp_constraints.loc[constraint.value.index, name] -= constraint.value.values

        # add handle for user inputted constraints here
        if not user_inputted_constraint.empty:
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
                    print("User has inputted an invalid constraint. Please change and run again.")
                    sys.exit()

        # now that we have a new list of constraints, create Constraint objects and store as 'control constraint'
        self.control_constraints = {'ene_min': Const.Constraint('ene_min', self.name, temp_constraints['ene_min']),
                                    'ene_max': Const.Constraint('ene_max', self.name, temp_constraints['ene_max']),
                                    'ch_min': Const.Constraint('ch_min', self.name, temp_constraints['ch_min']),
                                    'ch_max': Const.Constraint('ch_max', self.name, temp_constraints['ch_max']),
                                    'dis_min': Const.Constraint('dis_min', self.name, temp_constraints['dis_min']),
                                    'dis_max': Const.Constraint('dis_max', self.name, temp_constraints['dis_max'])}
        return None

    def calc_degradation(self, opt_period, start_dttm, end_dttm, energy_series=None):
        """ calculate degradation percent based on yearly degradation and cycle degradation

        Args:
            start_dttm (DateTime): Start timestamp to calculate degradation
            end_dttm (DateTime): End timestamp to calculate degradation
            energy_series (Series): time series of energy values

        Returns:
            A percent that represented the energy capacity degradation
        """

        # time difference between time stamps converted into years multiplied by yearly degrate rate
        # TODO dont hard code 365 (leap year)
        if self.incl_cycle_degrade:
            time_degrade = min((end_dttm - start_dttm).days/365*self.yearly_degrade/100, 1)

            # if given energy data and user wants cycle degradation
            if (energy_series is not None) and self.incl_cycle_degrade:
                # use rainflow counting algorithm to get cycle counts
                cycle_counts = rainflow.count_cycles(energy_series, ndigits=4)

                # sort cycle counts into user inputed cycle life bins
                digitized_cycles = np.searchsorted(self.cycle_life['Cycle Depth Upper Limit'],
                                                   [min(i[0]/self.ene_max_rated, 1) for i in cycle_counts], side='left')

                # sum up number of cycles for all cycle counts in each bin
                cycle_sum = copy.deepcopy(self.cycle_life)
                cycle_sum['cycles'] = 0
                for i in range(len(cycle_counts)):
                    cycle_sum.loc[digitized_cycles[i], 'cycles'] += cycle_counts[i][1]

                # sum across bins to get total degrade percent
                # 1/cycle life value is degrade percent for each cycle
                cycle_degrade = np.dot(1/cycle_sum['Cycle Life Value'], cycle_sum.cycles)
            else:
                cycle_degrade = 0

            degrade_percent = time_degrade + cycle_degrade
            if opt_period:
                self.degrade_data.loc[opt_period, 'degrade_perc'] = degrade_percent + self.degrade_perc
            else:
                self.degrade_perc = degrade_percent + self.degrade_perc

    def apply_degradation(self, degrade_percent, datetimes=None):
        """ Updates ene_max_rated and control constraints based on degradation percent

        Args:
            degrade_percent (Series): percent energy capacity should decrease
            datetimes (DateTime): Vector of timestamp to recalculate control_constraints. Default is None which results in control constraints not updated

        Returns:
            Degraded energy capacity
        """

        # apply degrade percent to rated energy capacity
        new_ene_max = max(self.ulsoc*self.ene_max_rated*(1-degrade_percent), 0)

        # update physical constraint
        self.physical_constraints['ene_max_rated'].value = new_ene_max

        failure = None
        if datetimes is not None:
            # update control constraints
            failure = self.calculate_control_constraints(datetimes)
        if failure is not None:
            # possible that degredation caused infeasible scenario
            print('Degradation results in infeasible scenario')
            quit()
        return new_ene_max

    def objective_function(self, variables, mask):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.power_kw

        Returns:
            self.expressions (Dict): Dict of objective costs
        """
        Storage.objective_function(self, variables, mask)
        # Calculate and add the annuity required to pay off the capex of the storage system. A more detailed financial model is required in the future
        capex = self.ene_max_rated * self.ccost_kwh + self.dis_max_rated * self.ccost_kw + self.ccost  # TODO: This is hard coded for battery storage
        # n = self.end_year - self.start_year
        annualized_capex = (capex * .11)  # TODO: Hardcoded ratio - need to calculate annuity payment and fit into a multiyear optimization framework

        self.expressions.update({'capex': annualized_capex})
        return self.expressions
