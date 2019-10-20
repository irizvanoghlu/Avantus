"""
Finances.py

This Python class contains methods and attributes vital for completing financial analysis given optimal dispathc.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import logging
from storagevet.Finances import Financial
from storagevet.Params import Params
import numpy as np
import xml.etree.ElementTree as et


SATURDAY = 5

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')


class CostBenDER(Financial, Params):

    @classmethod
    def initialize_evaluation(cls):
        """
            Initialize the class variable of the Params class that will be used to create Params objects for the
            sensitivity analyses. Specifically, it will preload the needed CSV and/or time series data, identify
            sensitivity variables, and prepare so-called default Params values as a template for creating objects.

        """

        # read in and validate XML
        cls.Scenario = cls.read_evaluation_xml('Scenario')
        cls.Finance = cls.read_evaluation_xml('Finance')
        cls.Battery = cls.read_evaluation_xml('Battery')
        cls.PV = cls.read_evaluation_xml('PV')
        cls.Diesel = cls.read_evaluation_xml('Diesel')
        cls.User = cls.read_evaluation_xml('User')

    @classmethod
    def read_evaluation_xml(cls, name):
        """ Read data from valuation XML file

        Args:
            name (str): name of root element in xml file

        Returns: Tuple (dict, list)
                A dictionary filled with values provided by user that will be used by the CBA class.
                A list of errors when reading in the values

        """
        tag = cls.xmlTree.find(name)
        error_list = []

        # this catches a misspelling in the Params 'name' compared to the xml trees spelling of 'name'
        if tag is None:
            return None

        # This statement checks if the first character is 'y' or '1', if true it creates a dictionary.
        if tag.get('active')[0].lower() == "y" or tag.get('active')[0] == "1":
            dictionary = {}
            for key in tag:
                # check if the key can have a cba evaluation value
                cba_eval = key.find('Evaluation')
                if cba_eval is None:
                    continue

                # check if the first character is 'y' for the active value within each property
                if cba_eval.get('active')[0].lower() == "y" or cba_eval.get('active')[0] == "1":
                    # convert to correct data type
                    intended_type = key.find('Type').text

                    if key.get('analysis')[0].lower() == 'y' or key.get('analysis')[0].lower() == '1':
                        # if analysis, then convert each value and save as list
                        values = cls.extract_data(key.find('Evaluation').text, intended_type)
                    else:
                        values = cls.convert_data_type(key.find('Evaluation').text, intended_type)

                    # validate with schema
                    error = cls.validate_evaluation(tag.tag, key, values)

                    # if the error list is empty, then save the values within the dictionary
                    if not len(error):
                        dictionary[key.tag] = values

                    # else append the error to the list of errors to report to user
                    else:
                        error_list.append(error)
        else:
            # else returns None
            return None

        # if the list of errors is not empty, then report them to the user
        if len(error_list):
            cls.error(error_list)
        return dictionary

    @classmethod
    def validate_evaluation(cls, tag, key, value):
        """ validates the input data. A schema file is used to validate the inputs.
            if any errors are found they are saved to a dictionary and at the end of the method it will call the error
            method to print all the input errors

        Args:
            tag (str): value that corresponds to tag within model param CSV
            key (Element): key XML element
            value (:object): list of values that the user provided, which length should be the same as the sensitivity list

        Returns: list, length 1, of the error (if error) else return empty list

        """
        error_list = []
        # 1) check to see if key is in schema -- non-zero length then it exists, then continue validation
        prop = cls.schema_tree.findall(".//*[@name='" + tag + "']")[0].findall(".//*[@name='" + key.tag + "']")
        if len(prop):
            in_schema = prop[0].find('field')
            # 2) check to see if key is allowed to define cba values
            cba_allowed = in_schema.get('cba')
            if cba_allowed == 'y':
                intended_type = in_schema.get('type')
                intended_max = in_schema.get('max')
                intended_min = in_schema.get('min')
                if key.get('analysis')[0] == 'y':
                    # 3a) check to make sure length of values is the same as sensitivity if sensitivity
                    error_list.append(cls.validate_evaluation(tag, key.tag, value))
                    for val in value:
                        # 4) checks for validate: make type and range (if applicable) is correct
                        error_list = cls.checks_for_validate(val, prop[0], key.find('Type').text, intended_type, intended_min, intended_max, error_list)
                else:
                    # 3b) checks for validate: make type and range (if applicable) is correct
                    error_list = cls.checks_for_validate(value, prop[0], key.find('Type').text, intended_type, intended_min, intended_max, error_list)

        return error_list

    @classmethod
    def validate_value_length(cls, tag, key, lst):
        """ In the case that sensitivity analysis is turned on:
        This function makes sure that there are the same number of values to be evaluated within the cba, as there are
        sensitivity values.

        Args:
            tag (Element): value that corresponds to tag within model param CSV
            key (str): name of the key within model param CSV
            lst (list): list of values that the user provided, which length should be the same as the sensitivity list

        Returns: list of the error, if one exists. Else just an empty list

        """
        error_list = []
        try:
            sensitivity_att = cls.sensitivity['attributes'][(tag.tag, key)]
        except AttributeError:
            sensitivity_att = []
        if len(lst) != len(sensitivity_att):
            error_list.append((key, str(lst), "val_length", str(sensitivity_att)))
        return error_list

    def __init__(self, params):
        """ Initialized Financial object for case

         Args:
            params (Dict): input parameters
        """
        Financial.__init__(self, params)

    def annuity_scalar(self, start_year, end_year, optimized_years):
        """Calculates an annuity scalar, used for sizing, to convert yearly costs/benefits


        Args:
            start_year (pd.Period): First year of project (from model parameter input)
            end_year (pd.Period): Last year of project (from model parameter input)
            optimized_years (list): List of years that the user wants to optimize--should be length=1

        Returns: the NPV multiplier

        """
        n = end_year - start_year.year
        dollar_per_year = np.ones(n)
        base_year = optimized_years[0]
        yr_index = base_year - start_year.year
        while yr_index < n - 1:
            dollar_per_year[yr_index + 1] = dollar_per_year[yr_index] * (1 + self.inflation_rate / 100)
            yr_index += 1
        yr_index = base_year - start_year.year
        while yr_index > 0:
            dollar_per_year[yr_index - 1] = dollar_per_year[yr_index] * (100 / (1 + self.inflation_rate))
            yr_index -= 1
        lifetime_npv_alpha = np.npv(self.npv_discount_rate, dollar_per_year)
        return lifetime_npv_alpha
