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
import storagevet
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

    @classmethod
    def validateDER(cls):
        """ DERVET should have its own way to validate its ParamsDER
            A schema file is used to validate the inputs.
            if any errors are found they are saved to a dictionary and at the end of the method it will call the error
            method to print all the input errors

             Returns: a True when complete

             Note: this method and its implementation was initially brought from Storagevet validate method;
                   therefore, codes is commented out due to further required discussion and pending implementation
                   on validated errors for ParamsDER

        """
        # error_list = []
        #
        # val_root = cls.schema_tree.getroot()

        # for element in list(val_root):
        #
        #     if not element.get('name') in cls.active_components[element.get('type')]:
        #         continue

            # attribute = cls.xmlTree.find(element.get('name'))

            # DERVET can run both CAES and Battery at same time
            # ParamsDER don't need this for the overriden validate method or its own validateDER method
            # if cls.active_components['storage']:
            #     if 'CAES' in cls.active_components['storage'] and 'Battery' in cls.active_components['storage']:
            #         e_logger.error("Storage technology CAES and Battery should not be active together in StorageVET.")
            #         error_list.append(
            #             "Storage technology CAES and Battery should not be active together in StorageVET.")
            #         raise Exception("Storage technology CAES and Battery should not be active together in StorageVET.")

            # for properties in list(element):
            #
            #     # Check if attribute is in the schema
            #     try:
            #         value = attribute.find(properties.get('name')).find('Value').text
            #     except (KeyError, AttributeError):
            #         e_logger.error("Attribute Error in validate function: Missing inputs. Please check CSV inputs.")
            #         d_logger.error("Missing inputs. Please check CSV inputs.")
            #         error_list.append((properties.get('name'), None, "missing"))
            #         continue
            #
            #     obj = properties.findall("*")
            #     type_of_obj = obj[0].get('type')
            #     minimum = obj[0].get('min')
            #     maximum = obj[0].get('max')
            #
            #     elem = cls.xmlTree.findall(element.get('name'))
            #     elem = elem[0].find(properties.get('name'))
            #     type_of_input = elem.find('Type').text
            #     value = Params.convert_data_type(value, type_of_input)
            #     tups = element.get('name'), properties.get('name')
            #     if tups in list(cls.sensitivity['attributes'].keys()):
            #         sensitivity = attribute.find(properties.get('name')).find('Sensitivity_Parameters').text
            #         for values in Params.extract_data(sensitivity, type_of_input):
            #             error_list = Params.checks_for_validate(values, properties, type_of_input, type_of_obj, minimum,
            #                                                     maximum, error_list)
            #
            #     error_list = Params.checks_for_validate(value, properties, type_of_input, type_of_obj, minimum, maximum,
            #                                             error_list)

        # checks if error_list is not empty.
        # if error_list:
        #     Params.error(error_list)
