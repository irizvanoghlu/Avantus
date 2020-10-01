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
import pandas as pd

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
        TellUser.debug(f"Initializing {__name__}")
        super().__init__(params)  # BatteryTech.Battery->ESSizing->EnergyStorage->DER->Sizing
        self.user_duration = params['duration_max']
        # self.replacement_capacity = params['replacement_capacity']  TODO: implement this
        self.years_system_degraded = []

        if self.user_duration:
            self.size_constraints += [cvx.NonPos(self.ene_max_rated - self.user_duration*self.dis_max_rated)]

    def initialize_degredation(self, opt_agg):
        """

        Notes: Should be called once, after optimization levels are assigned, but before
        optimization loop gets called

        Args:
            opt_agg (DataFrame):

        Returns: None

        """
        super(Battery, self).initialize_degredation(opt_agg)
        if self.incl_cycle_degrade:
            # calculate current degrade_perc since installation
            step_before_optimziation_problems = opt_agg.sort_index().index[0] - pd.Timedelta(self.dt, unit='h')
            self.calc_degradation('Optimization Start', self.construction_year.to_timestamp(), step_before_optimziation_problems)

    def calc_degradation(self, opt_period, start_dttm, end_dttm):
        """ calculate degradation percent based on yearly degradation and cycle degradation

        Args:
            opt_period: the index of the optimization that occurred before calling this function, None if
                no optimization problem has been solved yet
            start_dttm (DateTime): Start timestamp to calculate degradation. ie. the first datetime in the optimization
                problem
            end_dttm (DateTime): End timestamp to calculate degradation. ie. the last datetime in the optimization
                problem

        A percent that represented the energy capacity degradation
        """
        super(Battery, self).calc_degradation(opt_period, start_dttm, end_dttm)
        if self.incl_cycle_degrade:
            if self.degraded_energy_capacity() == 0:
                # TODO record the year that the energy capacity reaches the point of replacement
                pass
                # TODO reset the energy capacity to its original nameplate if replaceable
            pass

    def constraints(self, mask, **kwargs):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """

        constraint_list = super().constraints(mask, **kwargs)  # BatteryTech.Battery->ESSSizing->EnergyStorage
        return constraint_list
