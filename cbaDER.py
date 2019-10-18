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

    eval_xml_tree = None

    @classmethod
    def initialize_evaluation(cls, eval_filename, sensitivity):
        """
            Initialize the class variable of the Params class that will be used to create Params objects for the
            sensitivity analyses. Specifically, it will preload the needed CSV and/or time series data, identify
            sensitivity variables, and prepare so-called default Params values as a template for creating objects.

            Args:
                eval_filename (str):
                sensitivity (dict):
        """
        cls.eval_xml_tree = et.parse(eval_filename)
        cls.sensitivity = sensitivity


    @classmethod
    def read_evaluation_xml(cls, name):
        """ Read data from valuation XML file

        Args:
            name (str): name of root element in xml file

        Returns: Tuple (dict, list)
                A dictionary filled with values provided by user that will be used by the CBA class.
                A list of errors when reading in the values

        """
        tag = cls.eval_xml_tree.find(name)
        error_list = []

        # this catches a misspelling in the Params 'name' compared to the xml trees spelling of 'name'
        if tag is None:
            return None

        # This statement checks if the first character is 'y' or '1', if true it creates a dictionary.
        if tag.get('active')[0].lower() == "y" or tag.get('active')[0] == "1":
            dictionary = {}
            for key in tag:
                # check if the first character is 'y' for the active value within each property
                if key.get('active')[0].lower() == "y" or key.get('active')[0] == "1":

                    values = cls.extract_data(key.find('Value').text, key.find('Type').text)
                    error_list = cls.validate_evaluation(tag.tag, key, values)

                    if not len(error_list):
                        dictionary[key.tag] = values
                    else:
                        cls.error(error_list)

        else:
            return None

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
        # check to make sure the user can provide the evaluation value they provided

        # check to see if the schema
        prop = cls.schema_tree.findall(".//*[@name='" + tag + "']")[0].findall(".//*[@name='" + key + "']")[0]
        in_schema = prop.findall(".//*[@name='eval_value']")
        if len(in_schema):
            # non-zero length then it exists, then continue validation
            intended_type = in_schema[0].get('type')
            intended_max = in_schema[0].get('max')
            intended_min = in_schema[0].get('min')
            if key.get('analysis')[0] == 'y':
                # 1) check to make sure length of values is the same as sensitivity if sensitivity
                error_list.append(cls.validate_evaluation(tag, key.tag, value))
                for val in value:
                    error_list = cls.checks_for_validate(val, prop, key.find('Type'), intended_type, intended_min, intended_max, error_list)
            else:
                # 2) checks for validate
                error_list = cls.checks_for_validate(value, prop, key.find('Type'), intended_type, intended_min, intended_max, error_list)
        else:
            pass

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
