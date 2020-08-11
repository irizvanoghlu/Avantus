"""
CHP Sizing class

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Andrew Etringer'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

import cvxpy as cvx
from MicrogridDER.CombustionTurbine import CT
import storagevet.Library as Lib
from ErrorHandelling import *


class CHP(CT):
    """ Combined Heat and Power (CHP) Technology, with sizing optimization

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """
        TellUser.debug(f"Initializing {__name__}")
        # base class is CT
        super().__init__(params)

        # overrides
        self.tag = 'CHP'
        self.is_hot = True

        self.electric_ramp_rate = params['electric_ramp_rate']      # MW/min # TODO this is not being used? --AE
        self.electric_heat_ratio = params['electric_heat_ratio']    # elec/heat (generation)
        # time series inputs
        try:
            self.site_heating_load = params['site_heating_load']    # BTU/hr
        except KeyError:
            self.site_heating_load = None

    def grow_drop_data(self, years, frequency, load_growth):
        if self.site_heating_load is not None:
            self.site_heating_load = Lib.fill_extra_data(self.site_heating_load, years, 0, frequency)
            # TODO use a non-zero growth rate of heating load? --AE
            self.site_heating_load = Lib.drop_extra_data(self.site_heating_load, years)

    def initialize_variables(self, size):
        # rotating generation
        super().initialize_variables(size)
        # plus heat
        self.variables_dict.update({
            'heat': cvx.Variable(shape=size, name=f'{self.name}-heatP', nonneg=True),
        })

    def constraints(self, mask):
        constraint_list = super().constraints(mask)
        elec = self.variables_dict['elec']
        heat = self.variables_dict['heat']

        constraint_list += [cvx.Zero(heat * self.electric_heat_ratio - elec)]

        return constraint_list

    def get_heat_recovered(self, mask):
        # thermal power is recovered in a CHP plant whenever electric power is being generated
        # it is proportional to the electric power generated at a given time
        return self.variables_dict['heat']

    def timeseries_report(self):

        tech_id = self.unique_tech_id()
        results = super().timeseries_report()

        results[tech_id + ' Heat Generation (kW)'] = self.variables_df['heat']
        if self.site_heating_load is not None:
            results[tech_id + ' Site Heating Load (BTU/hr)'] = self.site_heating_load

        return results

    def objective_function(self, mask, annuity_scalar=1):

        costs = super().objective_function(mask, annuity_scalar)

#        # add startup objective costs
#        if self.startup:
#            # TODO this is NOT how you would calculate the start up cost of a CHP. pls look at formulation doc and revise --HN
#            # TODO This can be easily fixed, but let's do it some other time, when everything else works --AC
#            costs[self.name + 'startup': cvx.sum(self.variables_dict['on']) * self.p_startup * annuity_scalar]

        return costs
