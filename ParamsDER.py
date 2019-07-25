"""
Params.py

"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']


import logging
from storagevet.Params import Params
from matplotlib.font_manager import FontProperties

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

    def __init__(self):
        """ Initialize these following attributes of the empty Params class object.
        """
        Params.__init__(self)
        self.Reliability = self.read_from_xml_object('Reliability')

    # def other_error_checks(self):
    #
    #     storagevet_checks = Params.other_error_checks(self)
    #
    #     if not storagevet_checks:
    #         return False
    #
    #     return True

    # def prepare_technology(self):
    #     """ Interprets user given data and prepares it for Storage/Storage.
    #
    #     Returns: collects required timeseries columns required + collects power growth rates
    #
    #     """
    #     Params.prepare_technology(self)
    #     dLogger.info("Successfully prepared the Technologies")

    def prepare_services(self):
        """ Interprets user given data and prepares it for each ValueStream (dispatch and pre-dispatch).

        """
        Params.prepare_services(self)
        pre_dispatch_serv = self.active_components['pre-dispatch']

        if 'Reliability' in pre_dispatch_serv:
            self.Reliability["dt"] = self.Scenario["dt"]

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
