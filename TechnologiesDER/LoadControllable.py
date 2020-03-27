"""
Load

This Python class contains methods and attributes specific for technology analysis within DERVET.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = 'beta'

import cvxpy as cvx
from storagevet.Technology.Load import Load


class ControllableLoad(Load):
    """ An Load object

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """
        # create generic technology object
        Load.__init__(self, params)
        # input params  UNITS ARE COMMENTED TO THE RIGHT
        self.rated_power = params['power_rated']  # kW
        self.duration = params['duration']  # hour
        self.energy_max = self.rated_power * self.duration
        self.variables_dict = {}
        if self.duration:  # if DURATION is not 0
            self.variable_names = {'power', 'ene'}

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            power (Variable): A cvxpy variable equivalent to dis and ch in batteries/CAES

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        self.variables_dict = {}
        if self.duration:
            self.variables_dict = {'power': cvx.Variable(shape=size, name='power'),
                                   'ene': cvx.Variable(shape=size, name='ene', nonneg=True)}
        return self.variables_dict

    def get_charge(self, mask):
        return self.site_load[mask] - self.variables_dict['power']

    def get_energy(self, mask):
        return self.variables_dict['ene']

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
        if self.duration:
            power = variables['power']
            energy = variables['ene']

            # SOE EVALUATION EQUATIONS (one set for every day)
            sub = mask.loc[mask]
            for day in sub.index.dayofyear.unique():
                day_mask = (day == sub.index.dayofyear)
                # general
                constraint_list += [cvx.Zero(energy[day_mask][:-1] + (power[day_mask][:-1] * self.dt) - energy[day_mask][1:])]
                # start of first timestep of the day
                constraint_list += [cvx.Zero(energy[day_mask][0] == self.energy_max)]
                # end of the last timestep of the day
                constraint_list += [cvx.Zero(energy[day_mask][-1] + (power[day_mask][-1] * self.dt) - self.energy_max)]

        return constraint_list

    def effective_load(self, mask):
        """ Returns the

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
        """
        if self.duration:
            return self.site_load.loc[mask] - self.variables.loc[mask, 'power']
        else:
            return self.site_load.loc[mask]
