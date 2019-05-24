"""
ParamsDER.py

"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import xml.etree.ElementTree as et
import logging
from dervet.storagevet.Params import Input
# from storagevet.Params import Input
import pandas as pd

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')


class ParamsDER(Input):
    """ Inherits from Input from storagevet. Takes user CSV or XML input and preforms validation/clean-up.
        Class attributes are made up of services, technology, and any other needed inputs. The attributes are filled
        by converting the xml file in a python object.
    """

    def __init__(self):
        """ Initialize all Input objects with the following attributes.
        """
        Input.__init__(self)
        self.Diesel = self.read_from_xml_object('Diesel')
        self.Reliability = self.read_from_xml_object('Reliability')

        self.Sizing = self.read_from_xml_object('Sizing')  # this is an empty dictionary
        self.Dispatch = self.read_from_xml_object('Dispatch')  # this is an empty dictionary

    def prepare_services(self):
        """ Interprets user given data and prepares it for each ValueStream (dispatch and pre-dispatch).

        Returns: collects required power timeseries

        """
        pre_dispatch_serv = self.active_components['pre-dispatch']

        required_power_series = Input.prepare_services(self)

        if 'Reliability' in pre_dispatch_serv:
            self.Reliability["dt"] = self.Scenario["dt"]

        return required_power_series
