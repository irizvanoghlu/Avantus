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
import cvxpy as cvx


class CurtailPV(DER):
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
        DER.__init__(self, name, tech_params)
        self.no_export = tech_params['no_export']
        self.generation = time_series['PV_Gen (kW)']
        self.load = time_series['Site_Load (kW)']

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
        constraint_list = [cvx.NonPos(variables['pv_out'] - self.generation[mask])]
        constraint_list += [cvx.NonPos(-variables['pv_out'])]

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

        variables = {'pv_out': cvx.Variable(shape=size, name='pv_out')}

        return variables
