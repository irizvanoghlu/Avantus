"""
Params.py

"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']


import xml.etree.ElementTree as et
import logging
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from storagevet.Params import Params

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')
e_logger = logging.getLogger('Error')
fontP = FontProperties()
fontP.set_size('small')


class ParamsDER(Params):
    """
        Class attributes are made up of services, technology, and any other needed inputs. The attributes are filled
        by converting the xml file in a python object.

        Notes:
             Need to change the summary functions for pre-visualization every time the Params class is changed - TN
    """
    # Tag: key0, key1, key2 -- which a user can give evaluation values for
    EVALUATION = {'Scenario': ['monthly_data_filename', 'time_series_filename', 'customer_tariff_filename',
                               'start_year', 'end_year'],
                  'Finance': 'all',  # 'all' indicated every key under that tag
                  'Results': None,
                  'Battery': ['ccost', 'ccost_kw', 'ccost_kwh', 'startup', 'fixedOM', 'OMexpenses', 'install_date',
                              'p_start_ch', 'p_start_dis']}

    eval_xml_tree = None

    @staticmethod
    def csv_to_xml(csv_filename):
        """ converts csv to 2 xml files. One that contains values that correspond to optimization values and the other
        corresponds the values used to evaluate the CBA.

        Args:
            csv_filename (string): name of csv file

        Returns:
            opt_xml_filename (string): name of xml file with parameter values for optimization evaluation
            cba_xml_filename (string): name of xml file with parameter values for cba evaluation


        """
        opt_xml_filename = Params.csv_to_xml(csv_filename)

        # add '_cba' to end of filename, and find .csv in the filename and replace with .xml
        cba_xml_filename = csv_filename[:csv_filename.rfind('.')] + "_cba.xml"

        # open csv to read into dataframe and blank xml file to write to
        csv_data = pd.read_csv(csv_filename)
        xml_data = open(cba_xml_filename, 'w')

        # write the header of the xml file and specify columns to place in xml model parameters template
        xml_data.write('<?xml version="1.0" encoding="UTF-8"?>' + "\n")
        xml_data.write('\n<input>\n')

        # outer loop for each tag/object and active status, i.e. Scenario, Battery, DA, etc.
        for obj in csv_data.Tag.unique():
            mask = csv_data.Tag == obj
            xml_data.write('\n    <' + obj + ' active="' + csv_data[mask].Active.iloc[0] + '">\n')
            # middle loop for each object's elements and is sensitivity is needed: max_ch_rated, ene_rated, price, etc.
            for ind, row in csv_data[mask].iterrows():
                if row['Key'] is np.nan:
                    continue
                # skip adding to XML if no value is given
                if row['Evaluation Value'] == '.' or row['Evaluation Active'] == '.':
                    continue
                xml_data.write('        <' + str(row['Key']) + ' active="' + str(row['Evaluation Active']) + ' analysis="' + str(row['Sensitivity Analysis']) + '">\n')
                xml_data.write('            <Value>' + str(row['Evaluation Value']) + '</Value>\n')
                xml_data.write('        </' + str(row['Key']) + '>\n')
            xml_data.write('    </' + obj + '>\n')
        xml_data.write('\n</input>')
        xml_data.close()

        return opt_xml_filename, cba_xml_filename

    @classmethod
    def initialize(cls, filename, schema):
        """
            Initialize the class variable of the Params class that will be used to create Params objects for the
            sensitivity analyses. Specifically, it will preload the needed CSV and/or time series data, identify
            sensitivity variables, and prepare so-called default Params values as a template for creating objects.

            Args:
                filename (string): filename of XML file to load for StorageVET sensitivity analysis
                schema (string): schema file name
        """
        Params.initialize(filename, schema)
        cls.eval_xml_tree = None

    def __init__(self):
        """ Initialize these following attributes of the empty Params class object.
        """
        Params.__init__(self)
        self.Reliability = self.read_from_xml_object('Reliability')

    @classmethod
    def initialize_evaluation(cls, eval_filename):
        """
            Initialize the class variable of the Params class that will be used to create Params objects for the
            sensitivity analyses. Specifically, it will preload the needed CSV and/or time series data, identify
            sensitivity variables, and prepare so-called default Params values as a template for creating objects.

            Args:
                eval_filename (str):
        """
        cls.eval_xml_tree = et.parse(eval_filename)

    @classmethod
    def read_evaluation_xml(cls, name):
        """ Read data from valuation XML file

        Args:
            name (str): name of root element in xml file

        Returns:
                A dictionary filled with values provided by user that will be used by the CBA class.

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
                    error = cls.validate_evaluation(tag.tag, key.tag, values)

                    if not len(error):
                        dictionary[key.tag] = values
                    else:
                        error_list += error

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
            key (str): name of the key within model param CSV
            value (:object): list of values that the user provided, which length should be the same as the sensitivity list

        Returns: list, length 1, of the error (if error) else return empty list

        """
        # check to make sure the user can provide the evaluation value they provided
        attribute = cls.xmlTree.find(tag).find(key)

        # check to see if the schema
        in_schema = cls.schema_tree.findall(".//*[@name='"+ tag + "']")[0].findall(".//*[@name='" + key + "']")[0].findall(".//*[@name='eval_value']")
        if len(in_schema):
            # non-zero length then it exists, then continue validation
            pass
        else:
            return False


    @classmethod
    def validate_value_length(cls, tag, key, lst):
        """ In the case that sensitivity analysis is turned on:
        This function makes sure that there are the same number of values to be evaluated within the cba, as there are
        sensitivity values.

        Args:
            tag (str): value that corresponds to tag within model param CSV
            key (str): name of the key within model param CSV
            lst (list): list of values that the user provided, which length should be the same as the sensitivity list

        Returns: True iff the length of the optimization values and cba evaluation values are the same.

        """

        return True

    def prepare_services(self):
        """ Interprets user given data and prepares it for each ValueStream (dispatch and pre-dispatch).

        """
        Params.prepare_services(self)
        pre_dispatch_serv = self.active_components['pre-dispatch']

        if 'Reliability' in pre_dispatch_serv:
            self.Reliability["dt"] = self.Scenario["dt"]
            self.Reliability.update({'load': self.Scenario['time_series'].loc[:, 'Site Load (kW)']})

        dLogger.info("Successfully prepared the value-stream (services)")

    def prepare_scenario(self):
        """ Interprets user given data and prepares it for Scenario.

        """
        Params.prepare_scenario(self)

        if self.Scenario['binary']:
            e_logger.warning('Please note that the binary formulation will be used. If attemping to size, ' +
                             'there is a possiblity that the CVXPY will throw a "DCPError". This will resolve ' +
                             'by turning the binary formulation flag off.')
            uLogger.warning('Please note that the binary formulation will be used. If attemping to size, ' +
                            'there is a possiblity that the CVXPY will throw a "DCPError". This will resolve ' +
                            'by turning the binary formulation flag off.')

        dLogger.info("Successfully prepared the Scenario and some Finance")

    # def prepare_finance(self):
    #     """ Interprets user given data and prepares it for Finance.
    #
    #     """
    #     Params.prepare_finance(self)
    #
    #     dLogger.info("Successfully prepared the Finance")
