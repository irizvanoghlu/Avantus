"""
Finances.py

This Python class contains methods and attributes vital for completing financial analysis given optimal dispathc.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

import logging
from storagevet.Finances import Financial
import numpy as np
import copy
import pandas as pd


SATURDAY = 5

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class CostBenefitAnalysis(Financial):
    """ This Cost Benefit Analysis Module

    """

    def __init__(self, financial_params):
        """ Initialize CBA model and edit any attributes that the user denoted a separate value
        to evaluate the CBA with

        Args:
            financial_params (dict): parameter dictionary as the Params class created
        """
        super().__init__(financial_params)
        self.horizon_mode = financial_params['analysis_horizon_mode']
        self.location = financial_params['location']
        self.ownership = financial_params['ownership']

        self.Scenario = financial_params['CBA']['Scenario']
        self.Finance = financial_params['CBA']['Finance']
        self.valuestream_values = financial_params['CBA']['valuestream_values']
        self.ders_values = financial_params['CBA']['ders_values']
        if 'Battery' in self.ders_values.keys():
            self.ders_values['Battery'] = self.ders_values.pop('Battery')
        if 'CAES' in self.ders_values.keys():
            self.ders_values['CAES'] = self.ders_values.pop('CAES')

        self.value_streams = {}
        self.ders = {}

    @staticmethod
    def annuity_scalar(start_year, end_year, opt_years, **kwargs):
        """Calculates an annuity scalar, used for sizing, to convert yearly costs/benefits
        this method is sometimes called before the class is initialized (hence it has to be
        static)

        Args:
            start_year (pd.Period): First year of project (from model parameter input)
            end_year (pd.Period): Last year of project (from model parameter input)
            opt_years (list): List of years that the user wants to optimize--should be length=1

        Returns: the NPV multiplier

        """
        n = end_year.year - start_year.year
        dollar_per_year = np.ones(n)
        base_year = min(opt_years)
        yr_index = base_year - start_year.year
        while yr_index < n - 1:
            dollar_per_year[yr_index + 1] = dollar_per_year[yr_index] * (1 + kwargs['inflation_rate'] / 100)
            yr_index += 1
        yr_index = base_year - start_year.year
        while yr_index > 0:
            dollar_per_year[yr_index - 1] = dollar_per_year[yr_index] * (100 / (1 + kwargs['inflation_rate']))
            yr_index -= 1
        lifetime_npv_alpha = np.npv(kwargs['npv_discount_rate']/100, [0] + dollar_per_year)
        return lifetime_npv_alpha

    def preform_cost_benefit_analysis(self, technologies, value_streams, results):
        """ this function calculates the proforma, cost-benefit, npv, and payback using the optimization variable results
        saved in results and the set of technology and service instances that have (if any) values that the user indicated
        they wanted to use when evaluating the CBA.

        Instead of using the technologies and services as they are passed in from the call in the Results class, we will pass
        the technologies and services with the values the user denoted to be used for evaluating the CBA.

        Args:
            technologies (Dict): all active technologies (provided access to ESS, generators, renewables to get capital and om costs)
            value_streams (Dict): Dict of all services to calculate cost avoided or profit
            results (DataFrame): DataFrame of all the concatenated timseries_report() method results from each DER
                and ValueStream

        """
        self.initiate_cost_benefit_analysis(technologies, value_streams)
        super().preform_cost_benefit_analysis(self.ders, self.value_streams, results)

    def initiate_cost_benefit_analysis(self, technologies, valuestreams):
        """ Prepares all the attributes in this instance of cbaDER with all the evaluation values.
        This function should be called before any finacial methods so that the user defined evaluation
        values are used

        Args:
            technologies (list): the management point of all active technology to access (needed to get capital and om costs)
            valuestreams (Dict): Dict of all services to calculate cost avoided or profit

        """
        # we deep copy because we do not want to change the original ValueStream objects
        self.value_streams = copy.deepcopy(valuestreams)
        self.ders = copy.deepcopy(technologies)

        self.place_evaluation_data()

    def place_evaluation_data(self):
        """ Place the data specified in the evaluation column into the correct places. This means all the monthly data,
        timeseries data, and single values are saved in their corresponding attributes within whatever ValueStream and DER
        that is active and has different values specified to evaluate the CBA with.

        """
        try:
            monthly_data = self.Scenario['monthly_data']
        except KeyError:
            monthly_data = None

        try:
            time_series = self.Scenario['time_series']
        except KeyError:
            time_series = None

        if time_series is not None or monthly_data is not None:
            for value_stream in self.value_streams.values():
                value_stream.update_price_signals(monthly_data, time_series)

        if 'customer_tariff' in self.Finance:
            self.tariff = self.Finance['customer_tariff']

        if 'User' in self.value_streams.keys():
            self.update_with_evaluation(self.value_streams['User'], self.valuestream_values['User'], self.verbose)

        for der_tag, instance_dict in self.ders_values.items():
            for id_str, der_instance in instance_dict.items():
                self.ders[der_tag][id_str].update_for_evaluation(der_instance)
                # self.update_with_evaluation(self.ders[der_tag][id_str], der_instance, self.verbose)

    @staticmethod
    def update_with_evaluation(param_object, evaluation_dict, verbose):
        """Searches through the class variables (which are dictionaries of the parameters with values to be used in the CBA)
        and saves that value

        Args:
            param_object (DER, ValueStream): the actual object that we want to edit
            evaluation_dict (dict, None): keys are the string representation of the attribute where value is saved, and values
                are what the attribute value should be
            verbose (bool): true or fla

        Returns: the param_object with attributes set to the evaluation values instead of the optimization values

        """
        if evaluation_dict:  # evaluates true if dict is not empty and the value is not None
            for key, value in evaluation_dict.items():
                try:
                    setattr(param_object, key, value)
                    print('attribute (' + param_object.name + ': ' + key + ') set: ' + str(value)) if verbose else None
                except KeyError:
                    print('No attribute ' + param_object.name + ': ' + key) if verbose else None
