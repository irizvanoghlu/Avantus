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

    @staticmethod
    def csv_to_xml(csv_filename):
        """ converts csv to 2 xml files. One that contains values that correspond to optimization values and the other
        corresponds the values used to evaluate the CBA.

        Args:
            csv_filename (string): name of csv file

        Returns:
            opt_xml_filename (string): name of xml file with parameter values for optimization evaluation


        """
        opt_xml_filename = Params.csv_to_xml(csv_filename)

        # open and read xml file
        xml_tree = et.parse(opt_xml_filename)
        xml_root = xml_tree.getroot()

        # open csv to read into dataframe and blank xml file to write to
        csv_data = pd.read_csv(csv_filename)

        # outer loop for each tag/object and active status, i.e. Scenario, Battery, DA, etc.
        for obj in csv_data.Tag.unique():
            mask = csv_data.Tag == obj
            tag = xml_root.find(obj)
            # middle loop for each object's elements and is sensitivity is needed: max_ch_rated, ene_rated, price, etc.
            for ind, row in csv_data[mask].iterrows():
                # skip adding to XML if no value is given
                if row['Key'] is np.nan or row['Evaluation Value'] == '.' or row['Evaluation Active'] == '.':
                    continue
                key = tag.find(row['Key'])
                cba_eval = et.SubElement(key, 'Evaluation')
                cba_eval.text = str(row['Evaluation Value'])
                cba_eval.set('active', str(row['Evaluation Active']))
        xml_tree.write(opt_xml_filename)

        return opt_xml_filename

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

    def __init__(self):
        """ Initialize these following attributes of the empty Params class object.
        """
        Params.__init__(self)
        self.Reliability = self.read_from_xml_object('Reliability')

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
