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
        self.rated_power = params['power_rating']  # kW
        self.duration = params['duration']  # hour
        self.energy_max = self.rated_power * self.duration
        self.variables_dict = {}
        if self.duration:  # if DURATION is not 0
            self.variable_names = {'power', 'ene_load'}

    def add_vars(self, size):
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
        # load + (charge - discharge)
        effective_charge = cvx.Parameter(shape=sum(mask), value=super().get_charge(mask), name='OG Load')
        if self.duration:
            effective_charge += self.variables_dict['power']
        return effective_charge

    def get_energy(self, mask):
        return self.variables_dict['ene_load']

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
            power = self.variables_dict['power']  # p_t = charge_t - discharge_t
            energy = self.variables_dict['ene_load']

            # constraints that keep the variables inside their limits
            constraint_list += [cvx.NonPos(power - self.rated_power)]
            constraint_list += [cvx.NonPos(-self.rated_power - power)]
            constraint_list += [cvx.NonPos(-energy)]
            constraint_list += [cvx.NonPos(energy - self.energy_max)]

            # SOE EVALUATION EQUATIONS
            # # general:  e_{t+1} = e_t + (charge_t - discharge_t) * dt = e_t + power_t * dt
            # constraint_list += [cvx.Zero(energy[:-1] + (power[:-1] * self.dt) - energy[1:])]
            # # start of first timestep of the day
            # constraint_list += [cvx.Zero(energy[0] - self.energy_max)]
            # # end of the last timestep of the day
            # constraint_list += [cvx.Zero(energy[-1] + (power[-1] * self.dt) - self.energy_max)]

            # day_i0 = 0  # index of the first timestep of the 1 hour of the opt window
            # day_i24 = 24 * self.dt  # index of last timestp of the 24th hour from the hour of DAY_INDEX_0
            # max_index = sum(mask)
            # while day_i24 < max_index:
            #     # general:  e_{t+1} = e_t + (charge_t - discharge_t) * dt = e_t + power_t * dt
            #     constraint_list += [cvx.Zero(energy[day_i0:day_i24-1] + (power[day_i0:day_i24-1] * self.dt) - energy[day_i0+1:day_i24])]
            #     # start of first timestep of the day
            #     constraint_list += [cvx.Zero(energy[day_i0] - self.energy_max)]
            #     # end of the last timestep of the day
            #     constraint_list += [cvx.Zero(energy[day_i24-1] + (power[day_i24-1] * self.dt) - self.energy_max)]
            #
            #     # update indexes to point to the next set of 24 hours
            #     day_i0 = day_i24
            #     day_i24 += 24 * self.dt

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

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
        """
        effective_load = super().effective_load()
        if self.duration:
            effective_load += self.variables['power']
        return effective_load

    def sizing_summary(self):
        """ load does not have a 'size' the same way other DER do. Instead you say a load
        has a shape, so it does not need to how up in the sizing summary

        Returns: None

        """

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = super().timeseries_report()
        if self.duration:
            results["Site Load (kW)"] = self.site_load
            results["Load Offset (kW)"] = self.variables['power']
        return results
