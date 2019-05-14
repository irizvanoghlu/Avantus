"""
Input.py

"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']


import logging
from storagevet.Input import Input

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
        # tech
        dLogger.info('Adding info about Diesel...')
        self.Diesel = self.read_from_xml_object('Diesel')

        # predispatch
        dLogger.info('Adding info about Reliability')
        self.Reliability = self.read_from_xml_object('Reliability')

    def prepare_services(self):
        """ Interprets user given data and prepares it for each ValueStream (dispatch and pre-dispatch).

        Returns: collects required power timeseries

        """
        pre_dispatch_serv = self.active_components['pre-dispatch']

        required_power_series = Input.prepare_services(self)

        if self.Reliability in pre_dispatch_serv:
            self.Reliability["dt"] = self.Scenario["dt"]

        return required_power_series
