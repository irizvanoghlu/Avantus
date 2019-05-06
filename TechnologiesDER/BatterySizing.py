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

from TechnologiesDER import TechnologySizing
import copy
import numpy as np
import pandas as pd
import rainflow
# import cvxpy as cvx


class BatterySizing(TechnologySizing.TechnologySizing):
    """ Battery class that inherits from Technology.

    """

    def __init__(self, name,  financial, params, tech_params, cycle_life):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            financial (Analysis): Initalized Financial Class
            tech_params (dict): params dictionary from dataframe for one case
            cycle_life (DataFrame): Cycle life information
        """

        # create generic technology object
        TechnologySizing.TechnologySizing.__init__(self, name, tech_params, 'Battery')

        # add degradation information
        self.cycle_life = cycle_life
        self.degrade_data = pd.DataFrame(index=financial.obj_val.index)

        # calculate current degrade_perc since installation
        if tech_params['incl_cycle_degrade']:
            start_dttm = financial.fin_inputs.index[0]
            degrade_perc = self.calc_degradation(self.install_date, start_dttm)
            self.degrade_data['degrade_perc'] = degrade_perc
            self.degrade_data['eff_e_cap'] = self.apply_degradation(degrade_perc)

    def build_master_constraints(self, variables, dt, mask, reservations, binary, slack, startup):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            variables (Dict): Dictionary of variables being optimized
            dt (float): Timestep size where dt=1 means 1 hour intervals, while dt=.25 means 15 min intervals
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            reservations (Dict): Dictionary of energy and power reservations required by the services being
                preformed with the current optimization subset
            binary (bool): True if user wants to implement binary variables in optimization, else False
            slack (bool): True if user wants to implement slack variables in optimization, else False
            startup (bool): True if user wants to implement startup variables in optimization, else False

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its
            service constraints.
        """

        # create default list of constraints
        constraint_list = TechnologySizing.TechnologySizing.build_master_constraints(self, variables, dt, mask, reservations, binary, slack, startup)
        # add constraint that battery can not charge and discharge in the same timestep
        if binary:
            # can only be on or off
            constraint_list += [variables['on_c'] + variables['on_d'] <= 1]

            # # when trying non binary
            # constraint_list += [0 <= on_c]
            # constraint_list += [0 <= on_d]

            # # NL formulation of binary variables
            # constraint_list += [cvx.square(on_c) - on_c == 0]
            # constraint_list += [cvx.square(on_d) - on_d == 0]

        return constraint_list

    def calc_degradation(self, start_dttm, end_dttm, energy_series=None):
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
        return degrade_percent

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

    def objective_function(self, variables, mask, dt, slack, startup):
        TechnologySizing.TechnologySizing.objective_function(self, variables, mask, dt, slack, startup)
        # Calculate and add the annuity required to pay off the capex of the storage system. A more detailed financial model is required in the future
        capex = self.ene_max_rated * self.ccost_kwh + self.dis_max_rated * self.ccost_kw + self.ccost  # TODO: This is hard coded for battery storage
        n = self.end_year - self.start_year
        annualized_capex = (capex * .11)  # TODO: Hardcoded ratio - need to calculate annuity payment and fit into a multiyear optimization framework

        self.expressions.update({'capex': annualized_capex})
        return self.expressions
