"""
Defines an class that extends DERs beyond their definition in StorageVET
for the purpose of DER-VET functionallty

"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = 'beta'

import numpy as np
import pandas as pd


class DERExtension:
    """ This class is to be inherited by DER classes that want to allow the DER our generic
    DER model to extend beyond that in StorageVET

    """

    def __init__(self, params):
        """

        """
        # try to look for DERVET specific user inputs that are shared by all DERs
        self.nsr_response_time = params['nsr_response_time']
        self.sr_response_time = params['sr_response_time']
        self.startup_time = params['startup_time']  # startup time, default value of 0, units in minutes

        self.macrs = params.get('macrs_term')
        self.construction_date = params.get('construction_date')
        self.operation_date = params.get('operation_date')
        self.decommission_cost = params['decommissioning_cost']
        self.salvage_value = params['salvage_value']
        self.expected_lifetime = params['expected_lifetime']
        self.replaceable = params['replaceable']

        self.replacement_cost_function = []
        rcost = params.get('rcost')
        if rcost is not None:
            self.replacement_cost_function.append(rcost)
        rcost_kW = params.get('rcost_kW')
        if rcost_kW is not None:
            self.replacement_cost_function.append(rcost_kW)
        rcost_kWh = params.get('rcost_kWh')
        if rcost_kWh is not None:
            self.replacement_cost_function.append(rcost_kWh)

    def update_for_evaluation(self, input_dict):
        """ Updates price related attributes with those specified in the input_dictionary

        Args:
            input_dict: hold input data, keys are the same as when initialized

        """
        marcs_term = input_dict.get('macrs_term')
        if marcs_term is not None:
            self.macrs = marcs_term

        ccost = input_dict.get('ccost')
        if ccost is not None:
            self.capital_cost_function[0] = ccost

        ccost_kw = input_dict.get('ccost_kw')
        if ccost_kw is not None:
            self.capital_cost_function[1] = ccost_kw

        ccost_kwh = input_dict.get('ccost_kwh')
        if ccost_kwh is not None:
            self.capital_cost_function[2] = ccost_kwh

    def decommissioning_cost(self, last_year):
        """ Returns the cost of decommissioning a DER and the year the cost will be incurred

        Returns: dataframe index by year that the cost applies to. if the year

        """
        cost = self.decommission_cost
        if self.replaceable:
            year = last_year
        else:
            year = pd.Period(self.operation_date.year + self.expected_lifetime)
        if year > last_year:
            cost = 0
            year = last_year

        return pd.DataFrame({f"{self.unique_tech_id()} Decommissioning Cost": cost}, index=[year])

    def calculate_salvage_value(self, start_year, last_year):
        """ Decode the user's input and return the salvage value
        (1) "Sunk Cost" this option means that there is no end of analysis value
            (salvage value = 0)
        (2) "Linear Salvage Value" which will calculate salvage value by multiplying
            the technology's capital cost by (remaining life/total life)
        (3) a number (in $) for the salvage value of the technology
            (User-specified Salvage Value)

        Args:
            last_year:

        Returns: the salvage value of the technology

        """
        if self.salvage_value == 'sunk cost':
            return 0
        decommission_year = start_year.year + self.expected_lifetime - 1

        # If the a technology has a life shorter than the analysis window with no replacement, then no salvage value applies.
        if decommission_year < last_year.year and not self.replaceable:
            return 0
        # else keep replacing, and update the DECOMMISSION_YEAR (assume installation occurs the year after)
        while self.replaceable and decommission_year < last_year.year:
            decommission_year += self.expected_lifetime

        # If it has a life shorter than the analysis window but is replaced, a salvage value will be applied.
        # If it has a life longer than the analysis window, then a salvage value will apply.
        years_beyond_project = last_year.year - decommission_year

        if years_beyond_project > 0:
            if self.salvage_value == "linear salvage value":
                try:
                    capex = self.get_capex().value
                except AttributeError:
                    capex = self.get_capex()
                return capex * (years_beyond_project/self.expected_lifetime)
            else:
                return self.salvage_value
        else:
            return 0
