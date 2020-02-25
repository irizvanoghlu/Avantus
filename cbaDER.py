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
from ParamsDER import ParamsDER
import numpy as np
import copy
import pandas as pd


SATURDAY = 5

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class CostBenefitAnalysis(Financial):

    @classmethod
    def initialize_evaluation(cls, schema, xml_tree):
        """
            Initialize the class variable of the Params class that will be used to create Params objects for the
            sensitivity analyses. Specifically, it will preload the needed CSV and/or time series data, identify
            sensitivity variables, and prepare so-called default Params values as a template for creating objects.

        """
        cls.schema_tree = schema
        cls.xml_tree = xml_tree
        cls.datasets = {"time_series": dict(),
                        "monthly_data": dict(),
                        "customer_tariff": dict(),
                        "yearly_data": dict()}

        cls.cba_input_error_raised = False

        # read in and validate XML
        cls.Scenario = cls.read_and_validate_cba('Scenario')
        cls.Finance = cls.read_and_validate_cba('Finance')

        # load in CBA values for DERs
        battery = cls.read_and_validate_cba('Battery')
        pv = cls.read_and_validate_cba('PV')
        ice = cls.read_and_validate_cba('ICE')
        caes = cls.read_and_validate_cba('CAES')

        # load in CBA values for predispatch services
        user = cls.read_and_validate_cba('User')

        if cls.cba_input_error_raised:
            raise AssertionError("The model parameter has some errors associated to it in the CBA column. Please fix and rerun.")
        # create dictionary for CBA values for DERs
        cls.ders_values = {'Storage': battery,
                           'PV': pv,  # cost_per_kW (and then recalculate capex)
                           'ICE': ice}  # fuel_price,

        # create dictionary for CBA values for all services (from data files)
        cls.valuestream_values = {'User': user}  # USER will only have one entry in it (key = price)

    @classmethod
    def read_and_validate_cba(cls, name):
        """ Read data from valuation XML file

        Args:
            name (str): name of root element in xml file

        Returns: Tuple (dict, list)
                A dictionary filled with values provided by user that will be used by the CBA class.
                A list of errors when reading in the values

        """
        tag = cls.xmlTree.find(name)
        schema_tag = cls.schema_tree.find(name)

        # check to see if user includes the tag within the provided xml
        if tag is None:
            return

        # This statement checks if the first character is 'y' or '1', if true it creates a dictionary.
        if tag.get('active')[0].lower() == "y" or tag.get('active')[0] == "1":
            # Check if tag is in schema
            if schema_tag is None:
                cls.report_warning("missing tag", tag=name, raise_input_error=False)
                # warn user that the tag given is not in the schema
                return
            dictionary = {}
            # iterate through each tag required by the schema
            for schema_key in schema_tag:
                # Check if attribute is in the schema
                try:
                    key = tag.find(schema_key.tag)
                    cba_value = key.find('Evaluation')
                except (KeyError, AttributeError):
                    cls.report_warning("missing key", tag=name, key=schema_key.tag)
                    continue
                # if we dont have a cba_value, skip to next key
                if cba_value is None:
                    continue
                # check if the first character is 'y' for the active value within each property
                if cba_value.get('active')[0].lower() == "y" or cba_value.get('active')[0] == "1":
                    # convert to correct data type
                    intended_type = key.find('Type').text
                    if key.get('analysis')[0].lower() == 'y' or key.get('analysis')[0].lower() == '1':
                        # if analysis, then convert each value and save as list
                        tag_key = (tag.tag, key.tag)
                        sensitivity_values = cls.extract_data(key.find('Evaluation').text, intended_type)
                        # validate each value
                        for values in sensitivity_values:
                            cls.checks_for_validate(values, schema_key, name)
                        #  check to make sure the length match with sensitivity analysis value set length
                        required_values = len(cls.sensitivity['attributes'][tag_key])
                        if required_values != len(sensitivity_values):
                            cls.report_warning('sa length', tag=name, key=key.tag, required_num=required_values)
                        cls.sensitivity['cba_values'][tag_key] = sensitivity_values
                    else:
                        dictionary[key.tag] = cls.convert_data_type(key.find('Evaluation').text, intended_type)
                else:
                    cls.report_warning('cba not allowed', tag=name, key=key.tag, raise_input_error=False)
            return dictionary

    @classmethod
    def report_warning(cls, warning_type, raise_input_error=True, **kwargs):
        """ Print a warning to the user log. Warnings are reported, but do not result in exiting.

        Args:
            warning_type (str): the classification of the warning to be reported to the user
            raise_input_error (bool): raise this warning as an error instead back to the user and stop running
                the program
            kwargs: elements about the warning that need to be reported to the user (like the tag and key that
                caused the error

        """

        if warning_type == 'cba not allowed':
            e_logger.error(f"INPUT: {kwargs['tag']}-{kwargs['key']} is not be used within the " +
                           "CBA module of the program. Value is ignored.")
        if warning_type == "sa length":
            e_logger.error(f"INPUT: {kwargs['tag']}-{kwargs['key']} has not enough CBA evaluatino values to "
                           f"successfully complete sensitivity analsysis. Please include {kwargs['required_num']} "
                           f"values, each corresponding to the Sensitivity Analysis value given")
        super().report_warning(warning_type, raise_input_error, **kwargs)
        cls.cba_input_error_raised = raise_input_error or cls.cba_input_error_raised

    @classmethod
    def read_referenced_data_for_cba(cls):
        """
            This function makes a unique set of filename(s) based on the results of grab_value_set function.
            It applies for time series filename(s), monthly data filename(s), customer tariff filename(s), and cycle
            life filename(s).
            For each set, the corresponding class dataset variable (ts, md, ct, cl) is loaded with the data.

            Returns: True after completing

            Notes: TODO: put try catch statements in this function around every read_from_file function
        """
        # READ IN EVALUATION VALUE FOR: TIME_SERIES_FILENAME
        ts_files = set(cls.determine_eval_values('Scenario', 'time_series_filename'))
        if None not in ts_files:
            for ts_file in ts_files:
                cls.datasets['time_series'][ts_file] = cls.read_from_file('time_series', ts_file, 'Datetime (he)')

        # READ IN EVALUATION VALUE FOR: MONTHLY_DATA_FILENAME
        md_files = set(cls.determine_eval_values('Scenario', 'monthly_data_filename'))
        if None not in md_files:
            for md_file in md_files:
                cls.datasets['monthly_data'][md_file] = cls.preprocess_monthly(cls.read_from_file('monthly_data', md_file, ['Year', 'Month']))

        # READ IN EVALUATION VALUE FOR: CUSTOMER_TARIFF_FILENAME
        ct_files = set(cls.determine_eval_values('Finance', 'customer_tariff_filename'))
        if None not in ct_files:
            for ct_file in ct_files:
                cls.datasets['customer_tariff'][ct_file] = cls.read_from_file('customer_tariff', ct_file, 'Billing Period')

    def __init__(self, financial_params):
        """ Initialize CBA model and edit any attributes that the user denoted a separate value
        to evaluate the CBA with

        Args:
            financial_params (dict): parameter dictionary as the Params class created
            dispatch_services (dict): Dict of services to calculate cost avoided or profit
            predispatch_services (dict): Dict of predispatch services to calculate cost avoided or profit
            technologies (dict): dictionary of all the DER subclasses that are active
        """
        Financial.__init__(self, financial_params)
        self.horizon_mode = financial_params['analysis_horizon_mode']
        self.location = financial_params['location']
        self.ownership = financial_params['ownership']

        self.value_streams = {}
        self.ders = {}
        # TODO: need to deal with the data obtained from CSVs

    def initiate_cost_benefit_analysis(self, technologies, valuestreams):
        """ Prepares all the attributes in this instance of cbaDER with all the evaluation values.
        This function should be called before any finacial methods so that the user defined evaluation
        values are used

        Args:
            technologies (Dict): Dict of technologies (needed to get capital and om costs)
            valuestreams (Dict): Dict of all services to calculate cost avoided or profit

        """
        # we deep copy because we do not want to change the original ValueStream objects
        self.value_streams = copy.deepcopy(valuestreams)
        self.ders = copy.deepcopy(technologies)
        self.load_data_sets()

        # TODO: need to save cba values and output them back to the user s.t. they know what values were used to get the CBA results
        self.update_with_evaluation('cbaDER', self, self.Scenario)
        self.update_with_evaluation('cbaDER', self, self.Finance)

        self.place_evaluation_data()

    def load_data_sets(self):
        """Loads data sets that are specified by the '_filename' parameters """
        # if self.Scenario is not None
        if self.Scenario:
            if 'time_series_filename' in self.Scenario.keys():
                time_series = self.datasets['time_series_filename'][self.Scenario['time_series_filename']]
                self.Scenario["time_series"], self.Scenario['frequency'] = self.preprocess_timeseries(time_series, self.dt)
            if 'monthly_data_filename' in self.Scenario.keys():
                self.Scenario["monthly_data"] = self.datasets["monthly_data"][self.Scenario["monthly_data_filename"]]

        # if self.Finance is not None
        if self.Finance:
            if 'yearly_data_filename' in self.Finance.keys():
                self.Finance["yearly_data"] = self.datasets["yearly_data"][self.Finance["yearly_data_filename"]]
            if 'customer_tariff_filename' in self.Finance.keys():
                self.Finance["customer_tariff"] = self.datasets["customer_tariff"][self.Scenario["customer_tariff_filename"]]

    @staticmethod
    def update_with_evaluation(param_name, param_object, evaluation_dict):
        """Searches through the class variables (which are dictionaries of the parameters with values to be used in the CBA)
        and saves that value

        Args:
            param_name (str): key of the ValueStream or DER as it is saved in the apporiate dictionary
            param_object (DER, ValueStream): the actual object that we want to edit
            evaluation_dict (dict, None): keys are the string representation of the attribute where value is saved, and values
                are what the attribute value should be

        Returns: the param_object with attributes set to the evaluation values instead of the optimization values

        """
        if evaluation_dict:  # evaluates true if dict is not empty and the value is not None
            for key, value in evaluation_dict.items():
                try:
                    setattr(param_object, key, value)
                    print('attribute (' + key + ': ' + param_name + ') set: ' + str(value))
                except KeyError:
                    print('No attribute: ' + key + 'in ' + param_name)

    def proforma_report(self, technologies, valuestreams, results, use_inflation=True):
        """ this function calculates the proforma, cost-benefit, npv, and payback using the optimization variable results
        saved in results and the set of technology and service instances that have (if any) values that the user indicated
        they wanted to use when evaluating the CBA.

        Instead of using the technologies and services as they are passed in from the call in the Results class, we will pass
        the technologies and services with the values the user denoted to be used for evaluating the CBA.

        Args:
            technologies (Dict): Dict of technologies (needed to get capital and om costs)
            valuestreams (Dict): Dict of all services to calculate cost avoided or profit
            results (DataFrame): DataFrame of all the concatenated timseries_report() method results from each DER
                and ValueStream
            use_inflation (bool): Flag to determine if using inflation rate to determine financials for extrapolation. If false, use extrapolation

        Returns: dataframe proforma

        """
        self.initiate_cost_benefit_analysis(technologies, valuestreams)
        proforma = Financial.proforma_report(self, self.ders, self.value_streams, results, use_inflation)
        return proforma

    def annuity_scalar(self, start_year, end_year, optimized_years):
        """Calculates an annuity scalar, used for sizing, to convert yearly costs/benefits


        Args:
            start_year (pd.Period): First year of project (from model parameter input)
            end_year (pd.Period): Last year of project (from model parameter input)
            optimized_years (list): List of years that the user wants to optimize--should be length=1

        Returns: the NPV multiplier

        """
        n = end_year.year - start_year.year
        dollar_per_year = np.ones(n)
        base_year = min(optimized_years)
        yr_index = base_year - start_year.year
        while yr_index < n - 1:
            dollar_per_year[yr_index + 1] = dollar_per_year[yr_index] * (1 + self.inflation_rate / 100)
            yr_index += 1
        yr_index = base_year - start_year.year
        while yr_index > 0:
            dollar_per_year[yr_index - 1] = dollar_per_year[yr_index] * (100 / (1 + self.inflation_rate))
            yr_index -= 1
        lifetime_npv_alpha = np.npv(self.npv_discount_rate/100, [0] + dollar_per_year)
        return lifetime_npv_alpha

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
            retail_prices = self.calc_retail_energy_price(self.Finance['customer_tariff'], self.frequency, self.opt_years)
            for value_stream in self.value_streams.values():
                value_stream.update_tariff_rate(self.Finance['customer_tariff'], retail_prices)

        if 'User' in self.value_streams.keys():
            self.update_with_evaluation('User', self.value_streams['User'], self.valuestream_values['User'])

        for key, value in self.ders.items():
            self.update_with_evaluation(key, value, self.ders_values[key])

    def grab_evaluation_value(self):
        pass
