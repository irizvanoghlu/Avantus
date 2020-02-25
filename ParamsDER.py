"""
Params.py

"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',  "Thien Nguyen"]
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

import xml.etree.ElementTree as et
import logging
import pandas as pd
import numpy as np
from storagevet.Params import Params
import os

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class ParamsDER(Params):
    """
        Class attributes are made up of services, technology, and any other needed inputs. The attributes are filled
        by converting the xml file in a python object.

        Notes:
             Need to change the summary functions for pre-visualization every time the Params class is changed - TN
    """
    # set schema loction based on the location of this file (this should override the global value within Params.py
    schema_location = os.path.abspath(__file__)[:-len('ParamsDER.py')] + "SchemaDER.xml"

    @staticmethod
    def csv_to_xml(csv_filename, verbose=False, ignore_cba_valuation=False):
        """ converts csv to 2 xml files. One that contains values that correspond to optimization values and the other
        corresponds the values used to evaluate the CBA.

        Args:
            csv_filename (string): name of csv file
            ignore_cba_valuation (bool): flag to tell whether to look at the evaluation columns provided (meant for
                testing purposes)
            verbose (bool): whether or not to print to console for more feedback


        Returns:
            opt_xml_filename (string): name of xml file with parameter values for optimization evaluation


        """
        xml_filename = Params.csv_to_xml(csv_filename, verbose)

        # open csv to read into dataframe and blank xml file to write to
        csv_data = pd.read_csv(csv_filename)
        # check to see if Evaluation rows are included
        if not ignore_cba_valuation and 'Evaluation Value' in csv_data.columns and 'Evaluation Active' in csv_data.columns:
            # then add values to XML

            # open and read xml file
            xml_tree = et.parse(xml_filename)
            xml_root = xml_tree.getroot()

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
            xml_tree.write(xml_filename)

        return xml_filename

    @classmethod
    def initialize(cls, filename, verbose):
        """ In addition to everything that initialize does in Params, this class will look at
        Evaluation Value to - 1) determine if cba value can be given and validate; 2) convert
        any referenced data into direct data 3) if sensitivity analysis, then make sure enough
        cba values are given 4) build a dictionary of CBA inputs that match with the instances
        that need to be run

            Args:
                filename (string): filename of XML or CSV model parameter
                verbose (bool): whether or not to print to console for more feedback

            Returns dictionary of instances of Params, each key is a number
        """
        cls.instances = super().initialize(filename, verbose)
        # 1) determine if cba value can be given and validate

        # 2) convert any referenced data into direct data

        # 3) if sensitivity analysis, then make sure enough cba values are given

        # 4) build a dictionary of CBA inputs that match with the instances that need to be run

        return cls.instances

    def __init__(self):
        """ Initialize these following attributes of the empty Params class object.
        """
        super().__init__()
        self.Reliability = self.read_and_validate('Reliability')

    def prepare_services(self):
        """ Interprets user given data and prepares it for each ValueStream (dispatch and pre-dispatch).

        """
        super().prepare_services()

        if self.Reliability is not None:
            self.Reliability["dt"] = self.Scenario["dt"]
            self.Reliability.update({'critical load': self.Scenario['time_series'].loc[:, 'Critical Load (kW)']})

        u_logger.info("Successfully prepared the value-stream (services)")

    def prepare_scenario(self):
        """ Interprets user given data and prepares it for Scenario.

        """
        Params.prepare_scenario(self)

        if self.Scenario['binary']:
            e_logger.warning('Please note that the binary formulation will be used. If attemping to size, ' +
                             'there is a possiblity that the CVXPY will throw a "DCPError". This will resolve ' +
                             'by turning the binary formulation flag off.')
            u_logger.warning('Please note that the binary formulation will be used. If attemping to size, ' +
                             'there is a possiblity that the CVXPY will throw a "DCPError". This will resolve ' +
                             'by turning the binary formulation flag off.')

        u_logger.info("Successfully prepared the Scenario and some Finance")

    def prepare_finance(self):
        """ Interprets user given data and prepares it for Finance.

        """
        super().prepare_finance()
        self.Finance.update({'location': self.Scenario['location'],
                             'ownership': self.Scenario['ownership']})

    def other_error_checks(self):
        """ Used to collect any other errors that was not specifically detected before.
            The errors is printed in the errors log.
            Including: errors in opt_years, opt_window, validation check for Battery and CAES's parameters.

        Returns (bool): True if there is no errors found. False if there is errors found in the errors log.

        """
        sizing_optimization = False
        if self.Battery:
            battery = self.Battery
            if not battery['ch_max_rated'] or not battery['dis_max_rated']:
                sizing_optimization = True
                # if sizing for power, with ...
                if self.Scenario['binary']:
                    # the binary formulation
                    e_logger.error('Params Error: trying to size the power of the battery with the binary formulation')
                    return False
                if self.DA or self.SR or self.NSR or self.FR:
                    # whole sale markets
                    e_logger.error('Params Error: trying to size the power of the battery to maximize profits in wholesale markets')
                    return False
            if not battery['ene_max_rated']:
                sizing_optimization = True

        if self.PV and not self.PV['rated_capacity']:
            sizing_optimization = True
        if self.ICE:
            if self.ICE['n_min'] != self.ICE['n_max']:
                sizing_optimization = True
                if self.ICE['n_min'] < self.ICE['n_max']:
                    e_logger.error('Params Error: ICE must have n_min < n_max')
                    return False
        if sizing_optimization and not self.Scenario['n'] == 'year':
            e_logger.error('Params Error: trying to size without setting the optimization window to \'year\'')
            return False
        return super().other_error_checks()
