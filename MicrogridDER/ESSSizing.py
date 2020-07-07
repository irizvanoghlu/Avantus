"""
ESSSizing.py

This file defines the ability for ESSes to be sized by DERVET
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

from MicrogridDER.Sizing import Sizing
from storagevet.Technology.EnergyStorage import EnergyStorage
from MicrogridDER.DERExtension import DERExtension


class ESSSizing(EnergyStorage, DERExtension, Sizing):
    """ Extended ESS class that can also be sized

    """

    def __init__(self, ess_type, params):
        """ Initialize all technology with the following attributes.

        Args:
            ess_type (str): A unique string name for the technology being added
            params (dict): Dict of parameters
        """
        EnergyStorage.__init__(self, ess_type, params)
        DERExtension.__init__(self, params)
        Sizing.__init__(self)

    def sizing_summary(self):
        """

        Returns: A dictionary describe this DER's size and captial costs.

        """
        sizing_dict = {
            'DER': self.name,
            'Energy Rating (kWh)': self.ene_max_rated,
            'Charge Rating (kW)': self.ch_max_rated,
            'Discharge Rating (kW)': self.dis_max_rated,
            'Round Trip Efficiency (%)': self.rte,
            'Lower Limit on SOC (%)': self.llsoc,
            'Upper Limit on SOC (%)': self.ulsoc,
            'Duration (hours)': self.ene_max_rated / self.dis_max_rated,
            'Capital Cost ($)': self.capital_cost_function[0],
            'Capital Cost ($/kW)': self.capital_cost_function[1],
            'Capital Cost ($/kWh)': self.capital_cost_function[2]
        }
        if (sizing_dict['Duration (hours)'] > 24).any():
            print('The duration of an Energy Storage System is greater than 24 hours!')

        return sizing_dict

    def update_for_evaluation(self, input_dict):
        """ Updates price related attributes with those specified in the input_dictionary

        Args:
            input_dict: hold input data, keys are the same as when initialized

        """
        super().update_for_evaluation(input_dict)
        fixed_om = input_dict.get('fixedOM')
        if fixed_om is not None:
            self.fixedOM_perKW = fixed_om

        variable_om = input_dict.get('OMexpenses')
        if variable_om is not None:
            self.variable_om = variable_om * 100

        heat_rate_high = input_dict.get('heat_rate_high')
        if heat_rate_high is not None:
            self.heat_rate_high = heat_rate_high

        if self.incl_startup:
            p_start_ch = input_dict.get('p_start_ch')
            if p_start_ch is not None:
                self.p_start_ch = p_start_ch

            p_start_dis = input_dict.get('p_start_dis')
            if p_start_dis is not None:
                self.p_start_dis = p_start_dis

