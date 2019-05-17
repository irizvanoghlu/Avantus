"""
CurtailPVPV.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

from Technology.DER import DER
from TechnologiesDER.TechnologySizing import TechnologySizing
import cvxpy as cvx
import pandas as pd


class CurtailPV(TechnologySizing):
    """ Pre_IEEE 1547 2018 standards. Assumes perfect foresight. Ability to curtail PV generation, unlike ChildPV.

    """

    def __init__(self, name,  financial, params, tech_params, time_series):
        """ Initializes a PV class where perfect foresight of generation is assumed.
        It inherits from the technology class. Additionally, it sets the type and physical constraints of the
        technology.

        Args:
            params (dict): params dictionary from dataframe for one case
            time_series (series): time series dataframe
        """
        TechnologySizing.__init__(self, name, tech_params, 'PV with controls')
        self.no_export = tech_params['no_export']
        self.gen_per_rated = time_series['PV_gen/rated']
        self.load = time_series['site_load']
        self.rated_capacity = params['rated_capacity']
        self.cost_per_kW = params['cost_per_kW']
        if not self.rated_capacity:
            self.rated_capacity = cvx.Variable(shape=1, name='PV rating', integer=True, nonneg=True)

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
        constraint_list = [cvx.NonPos(variables['pv_out'] - cvx.multiply(self.gen_per_rated[mask], self.rated_capacity))]
        # constraint_list += [cvx.NonPos(-variables['pv_out'])]
        # constraint_list += [cvx.NonPos(-self.rated_capacity)]

        if self.no_export:
            constraint_list += [cvx.NonPos(variables['dis']-variables['ch']+variables['pv_out']-self.load[mask])]
        return constraint_list

    @staticmethod
    def add_vars(size, binary, slack, startup):
        """ Adds optimization variables to dictionary

        Variables added:
            pv_out (Variable): A cvxpy variable for the ac eq power outputted by the PV system

        Args:
            size (Int): Length of optimization variables to create
            slack (bool): True if any pre-dispatch services are turned on, else False
            binary (bool): True if user wants to implement binary variables in optimization, else False
            startup (bool): True if user wants to implement startup variables in optimization, else False

        Returns:
            Dictionary of optimization variables
        """

        variables = {'pv_out': cvx.Variable(shape=size, name='pv_out', nonneg=True)}

        return variables

    def calculate_control_constraints(self, datetimes, user_inputted_constraint=pd.DataFrame):
        """ Generates a list of master or 'control constraints' from physical constraints and all
        predispatch service constraints.

        Args:
            datetimes (list): The values of the datetime column within the initial time_series data frame.
            user_inputted_constraint (DataFrame): timeseries of any user inputed constraints.

        Returns:
            Array of datetimes where the control constraints conflict and are infeasible. If all feasible return None.

        Note: the returned failed array returns the first infeasibility found, not all feasibilities.
        """
        return None

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
        self.expressions = {'PV capital cost': self.cost_per_kW*self.rated_capacity}
        self.capex = self.cost_per_kW*self.rated_capacity
        return self.expressions
