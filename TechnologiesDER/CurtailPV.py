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

    def __init__(self, name, params):
        """ Initializes a PV class where perfect foresight of generation is assumed.
        It inherits from the technology class. Additionally, it sets the type and physical constraints of the
        technology.

        Args:
            name (str): A unique string name for the technology being added, also works as category.
            params (dict): Dict of parameters
        """
        DER.__init__(self, name)
        self.no_export = params['no_export']
        self.no_import = params['no_import']
        self.charge_from_solar = params['charge_from_solar']

        self.gen_per_rated = params['PV_gen/rated']
        self.load = params['site_load']
        self.rated_capacity = params['rated_capacity']
        self.cost_per_kW = params['cost_per_kW']

        if not self.rated_capacity:
            self.rated_capacity = cvx.Variable(shape=1, name='PV rating', integer=True)

    def build_master_constraints(self, variables, mask, reservations, mpc_ene=None):
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

        if self.no_export:
            constraint_list += [cvx.NonPos(variables['dis']-variables['ch']+variables['pv_out']-self.load[mask])]

        if self.no_import:
            constraint_list += [cvx.NonPos(-variables['dis'] + variables['ch'] - self.generation[mask] + self.load[mask])]

        if self.charge_from_solar:
            constraint_list += [cvx.NonPos(variables['ch'] - self.generation[mask])]

        return constraint_list

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            pv_out (Variable): A cvxpy variable for the ac eq power outputted by the PV system

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        variables = {'pv_out': cvx.Variable(shape=size, name='pv_out', nonneg=True)}

        return variables

    def objective_function(self, variables, mask):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.opt_results

        Returns:
            self.expressions (Dict): Dict of objective expressions
        """
        self.expressions = {'PV capital cost': self.cost_per_kW*self.rated_capacity}
        self.capex = self.cost_per_kW*self.rated_capacity
        return self.expressions
