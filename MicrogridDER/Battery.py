"""
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

import cvxpy as cvx
from storagevet.Technology import BatteryTech
from MicrogridDER.ESSSizing import ESSSizing
from ErrorHandelling import *
import numpy as np

DEBUG = False


class Battery(BatteryTech.Battery, ESSSizing):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, params):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            params (dict): params dictionary from dataframe for one case
        """

        # create generic storage object
        BatteryTech.Battery.__init__(self, params)
        ESSSizing.__init__(self, self.tag, params)

        self.user_duration = params['duration_max']
        self.user_ch_rated_max = params['user_ch_rated_max']
        self.user_ch_rated_min = params['user_ch_rated_min']
        self.user_dis_rated_max = params['user_dis_rated_max']
        self.user_dis_rated_min = params['user_dis_rated_min']
        self.user_ene_rated_max = params['user_ene_rated_max']
        self.user_ene_rated_min = params['user_ene_rated_min']

        # TODO: move to ESSSizing
        # if the user inputted the energy rating as 0, then size for energy rating
        if not self.ene_max_rated:
            self.ene_max_rated = cvx.Variable(name='Energy_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ene_max_rated)]
            # recalculate the effective SOE limits s.t. they are CVXPY expressions
            self.effective_soe_min = self.llsoc * self.ene_max_rated
            self.effective_soe_max = self.ulsoc * self.ene_max_rated
            if self.incl_energy_limits and self.limit_energy_max is not None:
                LogError.error(f'Ignoring energy max time series because {self.tag}-{self.name} sizing for energy capacity')
                self.limit_energy_max = None
            if self.user_ene_rated_min:
                self.size_constraints += [cvx.NonPos(self.user_ene_rated_min - self.ene_max_rated)]
            if self.user_ene_rated_max:
                self.size_constraints += [cvx.NonPos(self.ene_max_rated - self.user_ene_rated_max)]

        # if both the discharge and charge ratings are 0, then size for both and set them equal to each other
        if not self.ch_max_rated and not self.dis_max_rated:
            self.ch_max_rated = cvx.Variable(name='power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]
            if self.user_ch_rated_max:
                self.size_constraints += [cvx.NonPos(self.ch_max_rated - self.user_ch_rated_max)]
            if self.user_ch_rated_min:
                self.size_constraints += [cvx.NonPos(self.user_ch_rated_min - self.ch_min_rated)]
            self.dis_max_rated = self.ch_max_rated
            if self.user_dis_rated_min:
                self.size_constraints += [cvx.NonPos(self.user_dis_rated_min - self.dis_min_rated)]
            if self.user_dis_rated_max:
                self.size_constraints += [cvx.NonPos(self.dis_max_rated - self.user_dis_rated_max)]
            if self.incl_charge_limits and self.limit_charge_max is not None:
                LogError.error(f'Ignoring charge max time series because {self.tag}-{self.name} sizing for power capacity')
                self.limit_charge_max = None
            if self.incl_discharge_limits and self.limit_discharge_max is not None:
                LogError.error(f'Ignoring discharge max time series because {self.tag}-{self.name} sizing for power capacity')
                self.limit_discharge_max = None

        elif not self.ch_max_rated:  # if the user inputted the discharge rating as 0, then size discharge rating
            self.ch_max_rated = cvx.Variable(name='charge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]
            if self.incl_charge_limits and self.limit_charge_max is not None:
                LogError.error(f'Ignoring charge max time series because {self.tag}-{self.name} sizing for power capacity')
                self.limit_charge_max = None
            if self.user_ch_rated_max:
                self.size_constraints += [cvx.NonPos(self.ch_max_rated-self.user_ch_rated_max)]
            if self.user_ch_rated_min:
                self.size_constraints += [cvx.NonPos(self.user_ch_rated_min-self.ch_min_rated)]

        elif not self.dis_max_rated:  # if the user inputted the charge rating as 0, then size for charge
            self.dis_max_rated = cvx.Variable(name='discharge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.dis_max_rated)]
            if self.incl_discharge_limits and self.limit_discharge_max is not None:
                LogError.error(f'Ignoring discharge max time series because {self.tag}-{self.name} sizing for power capacity')
                self.limit_discharge_max = None
            if self.user_dis_rated_min:
                self.size_constraints += [cvx.NonPos(self.user_dis_rated_min - self.dis_min_rated)]
            if self.user_dis_rated_max:
                self.size_constraints += [cvx.NonPos(self.dis_max_rated - self.user_dis_rated_max)]

        if self.user_duration:
            self.size_constraints += [cvx.NonPos((self.ene_max_rated / self.dis_max_rated) - self.user_duration)]

    def constraints(self, mask):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """

        constraint_list = BatteryTech.Battery.constraints(self, mask)
        constraint_list += self.size_constraints
        return constraint_list
