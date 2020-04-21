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
from .Sizing import Sizing


class ControllableLoad(Load, Sizing):
    """ An Load object. this object does not size.

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """
        # create generic technology object
        Load.__init__(self, params)
        # input params  UNITS ARE COMMENTED TO THE RIGHT
        self.rated_power = params['power_rating']  # kW
        self.duration = params['duration']  # hour
        self.energy_max = self.rated_power * self.duration
        self.variables_dict = {}
        if self.duration:  # if DURATION is not 0
            self.tag = 'ControllableLoad'
            self.variable_names = {'power', 'ene_load'}

    def initialize_variables(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            power (Variable): A cvxpy variable equivalent to dis and ch in batteries/CAES

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        # self.variables_dict = {}
        if self.duration:
            self.variables_dict = {'power': cvx.Variable(shape=size, name='power'),  # p_t = charge_t - discharge_t
                                   'ene_load': cvx.Variable(shape=size, name='ene', nonneg=True)}
        return self.variables_dict

    def get_charge(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the charge as a function of time for the

        """
        # load + (charge - discharge)
        effective_charge = cvx.Parameter(shape=sum(mask), value=super().get_charge(mask), name='OG Load')
        if self.duration:
            effective_charge += self.variables_dict['power']
        return effective_charge

    def get_state_of_energy(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the state of energy as a function of time for the

        """
        return self.variables_dict['ene_load']

    def objective_constraints(self, mask):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = []
        if self.duration:
            power = self.variables_dict['power']  # p_t = charge_t - discharge_t
            energy = self.variables_dict['ene_load']

            # constraints that keep the variables inside their limits
            constraint_list += [cvx.NonPos(power - self.rated_power)]
            constraint_list += [cvx.NonPos(-self.rated_power - power)]
            constraint_list += [cvx.NonPos(-energy)]
            constraint_list += [cvx.NonPos(energy - self.energy_max)]

            sub = mask.loc[mask]
            for day in sub.index.dayofyear.unique():
                day_mask = (day == sub.index.dayofyear)
                # general:  e_{t+1} = e_t + (charge_t - discharge_t) * dt = e_t + power_t * dt
                constraint_list += [cvx.Zero(energy[day_mask][:-1] + (power[day_mask][:-1] * self.dt) - energy[day_mask][1:])]
                # start of first timestep of the day
                constraint_list += [cvx.Zero(energy[day_mask][0] - self.energy_max)]
                # end of the last timestep of the day
                constraint_list += [cvx.Zero(energy[day_mask][-1] + (power[day_mask][-1] * self.dt) - self.energy_max)]

        return constraint_list

    def effective_load(self):
        """ Returns the load that is seen by the microgrid or point of interconnection

        """
        effective_load = super().effective_load()
        if self.duration:
            effective_load += self.variables_df.loc[:, 'power']
        return effective_load

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = super().timeseries_report()
        if self.duration:
            results["Site Load (kW)"] = self.site_load.loc[:]
            results["Load Offset (kW)"] = self.variables_df.loc[:, 'power']
        return results

    def sizing_summary(self):
        """

        Returns: A dictionary describe this DER's size and captial costs.

        """
        sizing_results = {
            'DER': self.name,
            'Power Capacity (kW)': self.rated_power,
            'Duration (hours)': self.duration
        }
        return sizing_results
