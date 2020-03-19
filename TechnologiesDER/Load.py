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
import numpy as np
import pandas as pd
from storagevet.Technology.DER import DER


class Load(DER):
    """ An Load object

    """

    def __init__(self, name, params):
        """ Initialize all technology with the following attributes.

        Args:
            name (str): A unique string name for the technology being added, also works as category.
            params (dict): Dict of parameters for initialization
        """
        # create generic technology object
        DER.__init__(self, params['name'], 'Load', params)
        # input params  UNITS ARE COMMENTED TO THE RIGHT
        self.dt = params['dt']
        self.rated_power = params['power_rated']  # kW
        self.duration = params['duration']  # hour
        self.site_load = params['site_load']
        self.energy_max = self.rated_power * self.duration

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
        variables = {'power': cvx.Variable(shape=size, name='power'),
                     'ene': cvx.Variable(shape=size, name='ene', nonneg=True)}
        return variables

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
        power = variables['power']
        energy = variables['ene']

        constraint_list = []
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

    # def timeseries_report(self):
    #     """ Summaries the optimization results for this DER.
    #
    #     Returns: A timeseries dataframe with user-friendly column headers that summarize the results
    #         pertaining to this instance
    #
    #     """
    #     results = pd.DataFrame(index=self.variables.index)
    #     if self.duration:
    #         results["Controllable Load (kw)"] = self.site_load - self.variables['power']
    #     else:
    #         results["Load (kw)"] = self.site_load
    #     return results
