"""
Technology.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import copy
import Constraint as Const
import cvxpy as cvx
import numpy as np
import pandas as pd
import re
import svet_helper as sh
import sys
#from profilehooks import profile


class TechnologySizing:
    """ A general template for technology object

    We define a "technology" as anything that can affect the quantity of load/power being delivered or used. Specific
    types of technologies are subclasses. The technology subclass should be called. The technology class should never
    be called directly.

    """

    def __init__(self, name, params, category):
        """ Initialize all technology with the following attributes.

        Args:
            name (str): A unique string name for the technology being added.
            params (dict): Dict of parameters
            category (str): A string identification for the type of technology.
        """

        # input params
        # note: these should never be changed in simulation (i.e from degradation)
        self.name = name
        self.install_date = params['install_date']
        self.rte = params['rte']
        self.sdr = params['sdr']
        # self.ene_max_rated = params['ene_max_rated']
        # self.dis_max_rated = params['dis_max_rated']
        # self.dis_min_rated = params['dis_min_rated']
        # self.ch_max_rated = params['ch_max_rated']
        # self.ch_min_rated = params['ch_min_rated']
        self.ene_max_rated = cvx.Variable(name='Energy_cap')  # TODO: change this to be determined by user input
        self.dis_max_rated = cvx.Variable(name='power_cap')
        self.dis_min_rated = 0
        self.ch_max_rated = self.dis_max_rated
        self.ch_min_rated = 0
        self.ulsoc = params['ulsoc']
        self.llsoc = params['llsoc']
        self.soc_target = params['soc_target']
        self.yearly_degrade = params['yearly_degrade']
        self.incl_cycle_degrade = int(params['incl_cycle_degrade'])  # this is a true / false -EG
        self.ccost = params['ccost']
        self.ccost_kw = params['ccost_kw']
        self.ccost_kwh = params['ccost_kwh']
        self.fixedOM = params['fixedOM']
        self.OMexpenses = params['OMexpenses']
        if params['startup']:
            self.p_start_ch = params['p_start_ch']
            self.p_start_dis = params['p_start_dis']
        if params['slack']:
            self.kappa_ene_max = params['kappa_ene_max']
            self.kappa_ene_min = params['kappa_ene_min']
            self.kappa_ch_max = params['kappa_ch_max']
            self.kappa_ch_min = params['kappa_ch_min']
            self.kappa_dis_max = params['kappa_dis_max']
            self.kappa_dis_min = params['kappa_dis_min']

        # create physical constraints from input parameters
        # note: these can be changed throughout simulation (i.e. from degradation)
        self.physical_constraints = {'ene_min_rated': Const.Constraint('ene_min_rated', self.name, self.llsoc*self.ene_max_rated),
                                     'ene_max_rated': Const.Constraint('ene_max_rated', self.name, self.ulsoc*self.ene_max_rated),
                                     'ch_min_rated': Const.Constraint('ch_min_rated', self.name, self.ch_min_rated),
                                     'ch_max_rated': Const.Constraint('ch_max_rated', self.name, self.ch_max_rated),
                                     'dis_min_rated': Const.Constraint('dis_min_rated', self.name, self.dis_min_rated),
                                     'dis_max_rated': Const.Constraint('dis_max_rated', self.name, self.dis_max_rated)}

        # initialize internal attributes
        self.control_constraints = {}
        self.services = {}
        self.predispatch_services = {}
        self.type = category
        self.expressions = {}
        self.degrade_data = None

        # not currently used
        self.load = 0
        self.generation = 0
        # TODO: Hardcoded - inherit from case (in the end, we need to tie degradation to this)
        self.start_year = 2017
        self.end_year = 2017+15
        self.r = .07

    def add_service(self, service, predispatch=False):
        """ Adds a service to the list of services provided by the technology.

        Args:
            service (:obj, Service): A Service class object
            predispatch (Boolean): Flag to add predispatch or dispatch service
        """
        if predispatch:
            self.predispatch_services[service.name] = service
        else:
            self.services[service.name] = service

    def objective_function(self, variables, mask, dt, slack, startup):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.opt_results
            dt (float): optimization timestep (hours)
            slack (bool): True if user wants to implement slack variables in optimization, else False
            startup (bool): True if user wants to implement startup variables in optimization, else False

        Returns:
            self.expressions (Dict): Dict of objective expressions
        """

        # time difference over optimization window
        time_diff = mask[mask].index[-1] - (mask[mask].index[0]-pd.Timedelta(dt, unit='h'))

        # TODO dont hard code 365 (leap year)
        # create constant objective expression for fixed_om
        fixed_om = self.fixedOM*time_diff.days/365*self.physical_constraints['dis_max_rated'].value
        # fixed_om = cvx.expressions.constants.Constant(fixed_om)  # uses the CXVpy type and allows .value attribute
        # create objective expression for variable om based on discharge activity
        var_om = cvx.sum(variables['dis']*self.OMexpenses*dt*1e-3)

        self.expressions = {'fixed_om': fixed_om,
                            'var_om': var_om}

        # add slack objective expressions. These try to keep the slack variables as close to 0 as possible
        if slack:
            # TODO write a helper function to clean this up also takes awhile MBL ***
            self.expressions.update({
                'ene_max_slack': cvx.sum(self.kappa_ene_max * variables['ene_max_slack']),
                'ene_min_slack': cvx.sum(self.kappa_ene_min * variables['ene_min_slack']),
                'dis_max_slack': cvx.sum(self.kappa_dis_max * variables['dis_max_slack']),
                'dis_min_slack': cvx.sum(self.kappa_dis_min * variables['dis_min_slack']),
                'ch_max_slack': cvx.sum(self.kappa_ch_max * variables['ch_max_slack']),
                'ch_min_slack': cvx.sum(self.kappa_ch_min * variables['ch_min_slack'])})

        # add startup objective expressions
        if startup:
            self.expressions.update({
                          'ch_startup': cvx.sum(variables['start_c']*self.p_start_ch),
                          'dis_startup': cvx.sum(variables['start_d']*self.p_start_dis)})

        return self.expressions

    def build_master_constraints(self, variables, dt, mask, reservations, binary, slack, startup):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            variables (Dict): Dictionary of variables being optimized
            dt (float): Timestep size where dt=1 means 1 hour intervals, while dt=.25 means 15 min intervals
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            reservations (Dict): Dictionary of energy and power reservations required by the services being
                preformed with the current optimization subset
            slack (bool): True if any pre-dispatch services are turned on, else False
            binary (bool): True if user wants to implement binary variables in optimization, else False
            startup (bool): True if user wants to implement startup variables in optimization, else False

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
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
        # ch_max = cvx.Parameter(size, value=self.control_constraints['ch_max'].value[mask].values, name='ch_max')
        ch_max = self.ch_max_rated
        ch_min = cvx.Parameter(size, value=self.control_constraints['ch_min'].value[mask].values, name='ch_min')
        # dis_max = cvx.Parameter(size, value=self.control_constraints['dis_max'].value[mask].values, name='dis_max')
        dis_max = self.dis_max_rated
        dis_min = cvx.Parameter(size, value=self.control_constraints['dis_min'].value[mask].values, name='dis_min')

        # energy at the end of the first time step
        constraint_list += [cvx.Zero((ene_target - ene[0]) + (dt * ch[0] * self.rte) - (dt * dis[0]) - reservations['E'][0] - (dt * ene_target * self.sdr * 0.01))]
        # energy after every time step
        constraint_list += [cvx.Zero((ene[:-1] - ene[1:]) + (dt * ch[1:] * self.rte) - (dt * dis[1:]) - reservations['E'][1:] - (dt * ene[1:] * self.sdr * 0.01))]

        # energy at the end of the optimization window
        constraint_list += [cvx.Zero(ene[-1] - ene_target)]

        # Keep energy in bounds determined in the constraints configuration function
        # keep SOE low enough to never run out of headroom
        constraint_list += [cvx.NonPos(ene_target - ene_max + reservations['E_upper'][0] - variables['ene_max_slack'][0])]
        constraint_list += [cvx.NonPos(ene[:-1] - ene_max + reservations['E_upper'][1:] - variables['ene_max_slack'][1:])]

        # keep SOE high enough to never run out of stored energy
        constraint_list += [cvx.NonPos(-ene_target + ene_min[0] - (pv_gen[0]*dt) - reservations['E_lower'][0] - variables['ene_min_slack'][0])]
        constraint_list += [cvx.NonPos(ene_min[1:] - (pv_gen[1:]*dt) - ene[:-1] - reservations['E_lower'][1:] - variables['ene_min_slack'][1:])]

        # Keep charge and discharge power levels within bounds
        constraint_list += [cvx.NonPos(ch - cvx.multiply(self.ch_max_rated, on_c) + reservations['C_max'] - variables['ch_max_slack'])]
        constraint_list += [cvx.NonPos(cvx.multiply(self.ch_min_rated, on_c) - ch + reservations['C_min'] - variables['ch_min_slack'])]
        constraint_list += [cvx.NonPos(dis - cvx.multiply(self.dis_max_rated, on_d) + reservations['D_max'] - variables['dis_max_slack'])]
        constraint_list += [cvx.NonPos(cvx.multiply(self.dis_min_rated, on_d) - dis + reservations['D_min'] - variables['dis_min_slack'])]

        # constraints to keep slack variables positive
        if slack:
            constraint_list += [cvx.NonPos(-variables['ch_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['ch_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['dis_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['dis_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['ene_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['ene_min_slack'])]

        if binary:
            # when dis_min or ch_min has been overwritten (read: increased) by predispatch services, need to force technology to be on
            # TODO better way to do this???
            ind_d = [i for i in range(size) if self.control_constraints['dis_min'].value[mask].values[i] > self.physical_constraints['dis_min_rated'].value]
            ind_c = [i for i in range(size) if self.control_constraints['ch_min'].value[mask].values[i] > self.physical_constraints['ch_min_rated'].value]
            if len(ind_d) > 0:
                constraint_list += [on_d[ind_d] == 1]  # np.ones(len(ind_d))
            if len(ind_c) > 0:
                constraint_list += [on_c[ind_c] == 1]  # np.ones(len(ind_c))

        # note: cannot operate startup without binary
        if startup:
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

                        # if the minimum value needed is greater than the physical maximum, infeasible scenario
                        # max_value = self.physical_constraints[const_name + '_max' + '_rated'].value
                        # if any(temp_constraints[name] >= max_value):
                        #     return temp_constraints[temp_constraints[name] > max_value].index

                    else:
                        # if maximum constraint, subtract predispatch constraint value
                        # min_value = self.physical_constraints[const_name + '_min' + '_rated'].value
                        temp_constraints.loc[constraint.value.index, name] -= constraint.value.values

                        # if (const_name == 'ene') & any(temp_constraints[name] <= min_value):
                        #     # if the maximum energy needed is less than the physical minimum, infeasible scenario
                        #     return temp_constraints[temp_constraints[name] > max_value].index
                        # else:
                        #     # it is ok to floor at zero since negative power max values will be handled in power min
                        #     # i.e negative ch_max means dis_min should be positive and ch_max should be 0)
                        #     temp_constraints[name] = temp_constraints[name].clip(lower=0)
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

    def calc_degradation(self, start_dttm, end_dttm, energy_series=None):
        """ Default is zero degradation
        Args:
            start_dttm (DateTime): Start timestamp to calculate degradation
            end_dttm (DateTime): End timestamp to calculate degradation
            energy_series (Series): time series of energy values

        Returns:
            A percent that represented the energy capacity degradation
        """
        return 0

    def apply_degradation(self, degrade_percent, datetimes):
        """ Default is no degradation effect

        Args:
            degrade_percent (Series): percent energy capacity should decrease
            datetimes (DateTime): Vector of timestamp to recalculate control_constraints

        Returns:
            Degraded energy capacity
        """
        pass

    @staticmethod
    def add_vars(size, binary, slack, startup):
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
            slack (bool): True if any pre-dispatch services are turned on, else False
            binary (bool): True if user wants to implement binary variables in optimization, else False
            startup (bool): True if user wants to implement startup variables in optimization, else False

        Returns:
            Dictionary of optimization variables
        """

        variables = {'ene': cvx.Variable(shape=size, name='ene'),
                     'dis': cvx.Variable(shape=size, name='dis'),
                     'ch': cvx.Variable(shape=size, name='ch'),
                     'ene_max_slack': cvx.Parameter(shape=size, name='ene_max_slack', value=np.zeros(size)),
                     'ene_min_slack': cvx.Parameter(shape=size, name='ene_min_slack', value=np.zeros(size)),
                     'dis_max_slack': cvx.Parameter(shape=size, name='dis_max_slack', value=np.zeros(size)),
                     'dis_min_slack': cvx.Parameter(shape=size, name='dis_min_slack', value=np.zeros(size)),
                     'ch_max_slack': cvx.Parameter(shape=size, name='ch_max_slack', value=np.zeros(size)),
                     'ch_min_slack': cvx.Parameter(shape=size, name='ch_min_slack', value=np.zeros(size)),
                     'on_c': cvx.Parameter(shape=size, name='on_c', value=np.ones(size)),
                     'on_d': cvx.Parameter(shape=size, name='on_d', value=np.ones(size)),
                     }

        if slack:
            variables.update({'ene_max_slack': cvx.Variable(shape=size, name='ene_max_slack'),
                              'ene_min_slack': cvx.Variable(shape=size, name='ene_min_slack'),
                              'dis_max_slack': cvx.Variable(shape=size, name='dis_max_slack'),
                              'dis_min_slack': cvx.Variable(shape=size, name='dis_min_slack'),
                              'ch_max_slack': cvx.Variable(shape=size, name='ch_max_slack'),
                              'ch_min_slack': cvx.Variable(shape=size, name='ch_min_slack')})
        if binary:
            variables.update({'on_c': cvx.Variable(shape=size, boolean=True, name='on_c'),
                              'on_d': cvx.Variable(shape=size, boolean=True, name='on_d')})
            if startup:
                variables.update({'start_c': cvx.Variable(shape=size, name='start_c'),
                                  'start_d': cvx.Variable(shape=size, name='start_d')})

        return variables

    def __eq__(self, other, compare_init=False):
        """ Determines whether Technology object equals another Technology object. Compare_init = True will do an
        initial comparison ignoring any attributes that are changed in the course of running a case.

        Args:
            other (Technology): Technology object to compare
            compare_init (bool): Flag to ignore attributes that change after initialization

        Returns:
            bool: True if objects are close to equal, False if not equal.
        """
        return sh.compare_class(self, other, compare_init)











