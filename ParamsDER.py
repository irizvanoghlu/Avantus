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
import pandas as pd
import sys
import ast
import itertools
import logging
import copy
import csv

#from tkinter import *
from prettytable import PrettyTable

import matplotlib.pyplot as plt
from pandas import read_csv

import numpy as np
import storagevet.Library as Lib


class ParamsDER:
    """
        # TODO note to self: make better class comment! - YY
        class attributes are made up of services, technology, and any other needed inputs. The attributes are filled
        by converting the xml file in a python object.
    """

    @classmethod
    def initialize(cls, filename, schema):
        """
            Initialize the class variable of the ParamsDER class that will be used to create ParamsDER objects for the
            sensitivity analyses. Specifically, it will preload the needed CSV and/or time series data, identify
            sensitivity variables, and prepare so-called default ParamsDER values as a template for creating objects.

            Args:
                filename (string): filename of XML file to load for StorageVET sensitivity analysis
        """
        logging.info('Initializing the XML tree with the provided file...')
        logging.info('Getting info for Tech of XML tree')
        cls.xmlTree = et.parse(filename)
        cls.schema_tree = et.parse(schema)

        cls.active_service = {"services": list(), "pre-dispatch": list(), 'technology': list(), 'other': list()}
        cls.Questionnaire = cls.read_from_xml_object('Questionnaire', False)

        cls.sensitivity = {"attributes": dict(), "coupled": list()} # contains all sensitivity variables as
                                                                    # keys and their values as lists
        cls.df_analysis = pd.DataFrame() # Each row specifies the value of each attribute
        cls.instances = dict()
        cls.datasets = {"time_series": dict(),
                       "monthly_data": dict(),
                       "customer_tariff": dict(),
                       "cycle_life": dict()}                               # for a scenario in the sensitivity analysis
         # holds the data of all the time series usd in sensitivity analysis

        logging.info('Updating data for Scenario and Sensitivity Analysis...')
        cls.fetch_sense()
        if cls.sensitivity["attributes"]:
            cls.build_dataframe()

        cls.data_prep()
        cls.template = cls()
        try:
            cls.template.load_data_sets()
        except:
            print("data cannot be loaded")

        if cls.sensitivity["attributes"]:
            cls.input_object_builder()
        else:
            cls.instances[0] = cls.template

        logging.info('Finished ParamsDER class initialization...!')

    def __init__(self):
        """ Initialize all ParamsDER objects with the following attributes.
        """
        #tech
        logging.info('Adding info about Battery...')
        self.Battery = self.read_from_xml_object('Battery')
        logging.info('Adding info about PV...')
        self.PV = self.read_from_xml_object('PV')
        logging.info('Adding info about Diesel...')
        self.Diesel = self.read_from_xml_object('Diesel')

        #service
        logging.info('Getting info for Service of XML tree')
        logging.info('Adding info about DCM...')
        self.DCM = self.read_from_xml_object('DCM')
        logging.info('Adding info about realTimeShift...')
        self.retailTimeShift = self.read_from_xml_object('retailTimeShift')

        #predispatch
        logging.info('Getting info for Predispatch of XML tree')
        logging.info('Adding info about Reliability')
        self.Reliability = self.read_from_xml_object('Reliability')

        #other
        logging.info('Getting info for Other of XML tree')
        logging.info('Adding info about incl_cycle_degrade...')
        self.incl_cycle_degrade = self.read_from_xml_object('incl_cycle_degrade')
        logging.info('Adding info about Finance...')
        self.Finance = self.read_from_xml_object('Finance')
        logging.info('Adding info about Scenario...')
        self.Scenario = self.read_from_xml_object('Scenario')

        logging.info('Adding info about Technology...')
        self.Technology = self.read_from_xml_object('Technology')

        if self.Reliability is not None:
            self.Reliability["dt"] = self.Scenario["dt"]

    @classmethod
    def read_from_xml_object(cls, name, flag=False):
        """ Read data from xml file.

            Args:
                name (str): name of the root element in the xml file
                tree (obj): a python obj of the xml file

             Returns:
                    A dictionary filled with needed given values of the properties of each service/technology
                    if the service or technology is not implemented in this case, then None is returned.
        """

        element = cls.xmlTree.find(name)

        # this catches a misspelling in the input 'name' compared to the xml trees spelling of 'name'
        if element is None:
            return None

        # This statement checks if the first character is 'y' or '1', if true it creates a dictionary.
        if element.get('active')[0].lower() == "y" or element.get('active')[0] == "1":
            dictionary = {}
            cls.fill_active(element.tag)
            for properties in element:
                # fills dictionary with the values from the xml file
                dictionary[properties.tag] = ParamsDER.convert_data_type(properties.find('value').text,
                                                                         properties.find('Data_Type').text)
                if flag and properties.get('analysis') is not None and properties.get('analysis')[0].lower() == "y":
                    temp = dictionary[properties.tag]
                    dictionary[properties.tag] = set(ast.literal_eval(properties.find('sensitivity').text))
                    dictionary[properties.tag].add(temp)
        else:
            return None

        return dictionary

    @classmethod
    def fill_active(cls, name):
        """ if given element is active it adds it to the class variable 'active_service'.
            checks to find the elements category (i.e. service and pre-dispatch) through the schema.

            Args:
                name (str): name of the root element in the input xml file

            Returns:

        """
        temp = cls.schema_tree.getroot().findall('*')

        for index in temp:
            if index.get('name') is not None and index.get('name') == name:
                category = index.get('type')

        if category == 'service':
            cls.active_service["services"].append(name)
        elif category == 'pre-dispatch':
            cls.active_service["pre-dispatch"].append(name)
        elif category == 'technology':
            cls.active_service["technology"].append(name)
        elif category == 'other':
            cls.active_service["other"].append(name)

    @classmethod
    def fetch_sense(cls):
        """

            Args: none


             Returns: True

        """
        root = cls.xmlTree.getroot()

        for element in list(root):
            if element.get('active')[0].lower() != "y" and element.get('active')[0] != "1":
                continue

            for properties in list(element):
                if properties.get('analysis') is None or \
                        properties.get('analysis')[0].lower() != "y" and properties.get('analysis')[0] != "1":
                    continue

                cls.sensitivity["attributes"][(element.tag, properties.tag)] = \
                    cls.determine_sensitivity_list(element.tag, properties.tag)

        return True

    @classmethod
    def determine_sensitivity_list(cls, element, property):
        """

            Args:

            Returns:

        """
        group = cls.xmlTree.find(element)
        attribute = group.find(property)
        if attribute.find('sensitivity').text is None or attribute.get('analysis')[0].lower() != "y" and attribute.get('analysis')[0] != "1":
            slist = cls.extract_data(attribute.find('value').text,attribute.find('Data_Type').text)
        else: slist = cls.extract_data(attribute.find('sensitivity').text,attribute.find('Data_Type').text)
        return slist

    @classmethod
    def extract_data(cls, expression, dataType):
        """

            Args:

            Returns:

        """
        result = []
        expression = expression.strip()
        if expression.startswith("[") and expression.endswith("]"):
            expression = expression[1:-1]
        sset = expression.split(',')
        for s in sset:
            data = cls.convert_data_type(s.strip(), dataType)
            result.append(data)
        return result

    @classmethod
    def build_dataframe(cls):
        """

            Args:

            Returns:

        """
        cls.df_analysis = pd.DataFrame(cls.sensitivity['attributes'])
        print(cls.df_analysis)

        sense = cls.sensitivity["attributes"]
        keys, values = zip(*sense.items())
        experiments = [dict(zip(keys,v)) for v in itertools.product(*values)]
        print(len(experiments))

        cls.df_analysis = pd.DataFrame(experiments)
        print(cls.df_analysis)

    @classmethod
    def input_object_builder(cls):
        """

            Args:

            Returns:

        """

        dictionary = {}
        for index, row in cls.df_analysis.iterrows():
            inputobject = copy.deepcopy(cls.template)
            for col in row.index:
                #TODO funny issue with pandas dataframe with all float gives numpy.float64
                # but float is wanted fixed with .astype(object) but might want to investigate -YY
                inputobject.modify_attribute(tupel=col, value=row[col])
            inputobject.load_data_sets()
            dictionary.update({index: inputobject})
        cls.instances = dictionary

    def modify_attribute(self, tupel, value):
        """

            Args:

            Returns:

        """

        attribute = getattr(self, tupel[0])
        attribute[tupel[1]] = value

    @classmethod
    def data_prep(cls):
        """ error.
        # TODO note to self: make better comment! - YY

            Args:
                list (list):

             Returns:

        """
        # i'm not sure about this function, a more general one might be better - YY
        # read data (the following TODO's were copied over form svet_inputs - YY)
        # TODO if these are the same across sim_cases only read in once
        # TODO use fill_gaps helper function (this should be dealt by preprocess)
        # TODO fill in nan, zeros
        # TODO change all times to UTC
        ts_files = set(cls.determine_sensitivity_list('Scenario', 'time_series_filename'))
        md_files = set(cls.determine_sensitivity_list('Scenario', 'monthly_data_filename'))
        ct_files = set(cls.determine_sensitivity_list('Scenario', 'customer_tariff_filename'))
        cl_files = set(cls.determine_sensitivity_list('Scenario', 'cycle_life_filename'))

        for ts_file in ts_files: cls.datasets['time_series'][ts_file] = cls.read_from_file('time_series', ts_file, 'Datetime', True)
        for md_file in md_files: cls.datasets['monthly_data'][md_file] = cls.preprocess_monthly(cls.read_from_file('monthly_data', md_file, ['Year', 'Month'], True))
        for ct_file in ct_files: cls.datasets['customer_tariff'][ct_file] = cls.read_from_file('customer_tariff', ct_file,  'billing_period', verbose=True)
        for cl_file in cl_files: cls.datasets['cycle_life'][cl_file] = cls.read_from_file('cycle_life', cl_file, None, True)

        return True

    @staticmethod
    def read_from_file(name, filename, ind_col=None, verbose=False):
        """ Read data from csv or excel file.

        Args:
            name (str): name of data to read
            filename (str): filename of file to read
            ind_col (str or list): column(s) to use as dataframe index
            verbose (bool): flag to display statements

         Returns:
                A pandas dataframe of file at FILENAME location
        TODO convert string to float where possible

        """

        raw = pd.DataFrame()

        if (filename is not None) and (not pd.isnull(filename)):

            # logic for time_series data
            parse_dates = name == 'time_series'
            infer_dttm = name == 'time_series'

            # select read function based on file type
            func = pd.read_csv if ".csv" in filename else pd.read_excel

            try:
                raw = func(filename, parse_dates=parse_dates, index_col=ind_col, infer_datetime_format=infer_dttm)
            except UnicodeDecodeError:
                try:
                    raw = func(filename, parse_dates=parse_dates, index_col=ind_col, infer_datetime_format=infer_dttm,
                               encoding="ISO-8859-1")
                except (ValueError, IOError):
                    print("Could not open: ", filename)
                else:
                    print("Successfully read in", filename) if verbose else None
            except (ValueError, IOError):
                # TODO improve error handling (i.e. if file is not there)
                print("Could not open: ", filename)
            else:
                print("Successfully read in", filename) if verbose else None

        return raw

    @staticmethod
    def preprocess_monthly(monthly_data):
        """ processing monthly data.
        Creates monthly Period index from year and month

        Args:
            monthly_data (DataFrame): Raw df

        Returns:
            monthly_data (DataFrame): edited df

        """
        if not monthly_data.empty:
            monthly_data.index = pd.PeriodIndex(year=monthly_data.index.get_level_values(0).values,
                                                month=monthly_data.index.get_level_values(1).values, freq='M')
            monthly_data.index.name = 'yr_mo'
        return monthly_data

    @staticmethod
    def csv_to_xml(filename):
        """ converts csv to xml

         Args:
             filename (string): name of csv file

         Returns: xmlFile (string): name of xml file


         """

        csvFile = filename
        xml_filename = filename[:-3] + "xml"
        xmlFile = xml_filename

        csvData = csv.reader(open(csvFile))
        xmlData = open(xmlFile, 'w')

        xmlData.write('<?xml version="1.0" encoding="UTF-8"?>' + "\n")
        xmlData.write("\n" + '<input>' + "\n")

        rowNum = 0
        for row in csvData:
            if rowNum == 0:
                tags = row
                tags.remove('Category')
                tags.remove('ParamsDER Name in UI')
                tags.remove('active')
                tags.remove('analysis')
                tags.remove('Validation Criteria')
                tags.remove('Options/Notes')
                tags.remove('Required? (0 = never required, 1 = Required by default, 2 = Not Required by default)')
                tags.remove('Proposed backend renaming')
                tags.remove('Description (Mouseover Text)')
                tags.remove('Parameter Name in Back End')

                # replace spaces w/ underscores in tag names
                for i in range(len(tags)):
                    tags[i] = tags[i].replace(' ', '_')
            else:

                if row[0] == '':
                    break

                if rowNum == 1 or temp != row[0]:
                    if rowNum != 1:
                        xmlData.write('    </' + temp + '>' + "\n")
                    xmlData.write('    <' + row[0] + ' active=' + '"' + str(row[9]) + '"' + '>' + "\n")
                    temp = row[0]

                if str(row[10]) != '.':
                    xmlData.write('        ' + '<' + row[1] + ' analysis=' + '"' + str(row[10]) + '"' + '>' + "\n")

                    for i in range(len(tags)):
                        xmlData.write('            ' + '<' + tags[i] + '>' + row[i + 4] + '</' + tags[i] + '>' + "\n")

                    xmlData.write('        </' + row[1] + '>' + "\n")

            rowNum += 1

        xmlData.write('    </' + temp + '>' + "\n")
        xmlData.write('</input>' + "\n")
        xmlData.close()

        return xmlFile

    @staticmethod
    def convert_data_type(value, desired_type):
        """ coverts data to a given type. takes in a value and a type

            Args:
                value (str): some data needed to be converted
                desired_type (str): the desired type of the given value

             Returns: Attempts to convert the type of 'value' to 'desired_type'. Returns value if it can,
             otherwise it returns a tuple (None, value) if desired_type is not a known type to the function or the value
             cannot be converted to the desired type.
        """

        if desired_type == "int":
            try:
                int(value)
            except ValueError:
                return None, value
            else:
                return int(value)
        elif desired_type == "float":
            try:
                float(value)
            except ValueError:
                return None, value
            else:
                return float(value)
        elif desired_type == 'tuple':
            try:
                tuple(value)
            except ValueError:
                return None, value
            else:
                return tuple(value)
        elif desired_type == 'list':
            return list(value)
        elif desired_type == 'list/int':
            return list(map(int, value.split()))
        elif desired_type == 'Timestamp':
            try:
                pd.Timestamp(value)
            except ValueError:
                return None, value
            else:
                return pd.Timestamp(value)
        elif desired_type == 'Period':
            try:
                pd.Period(value)
            except ValueError:
                return None, value
            else:
                return pd.Period(value)
        elif desired_type == 'string':
            return value
        elif desired_type == 'list/string':
            return value.split()
        else:
            return None, value

    @staticmethod
    def error(list):
        """ error. Prints errors generated in the validate method.

            Args:
                list (list): a list of errors.

        """
        #@TODO add the ability to generate an error log
        for value in list:

            if value[2] == "size":
                print("error in " + value[0] + ", the input value: " + str(value[1])
                      + " is out of bounds. The given bounds are: (" + str("-INF" if value[3] is None else value[3])
                      + ", " + str("INF" if value[4] is None else value[4]) + ")")
                logging.error("error in " + value[0] + ", the input value: " + str(value[1])
                      + " is out of bounds. The given bounds are: (" + str("-INF" if value[3] is None else value[3])
                      + ", " + str("INF" if value[4] is None else value[4]) + ")")
            elif value[2] == "type":
                print("error in " + value[0] + ", the input value: " + str(value[1])
                      + " is not of the correct type. The type should be: " + str(value[3]))
                logging.error("error in " + value[0] + ", the input value: " + str(value[1])
                      + " is not of the correct type. The type should be: " + str(value[3]))
            elif value[2] == "type_dec":
                print("error in " + value[0] + ", the input type: " + str(value[1])
                      + " is not of the correct type. The type should be: " + str(value[3]))
                logging.error("error in " + value[0] + ", the input type: " + str(value[1])
                      + " is not of the correct type. The type should be: " + str(value[3]))
            elif value[2] == "missing":
                print("error in " + value[0] + ", This input parameter is not in the schema")
                logging.error("error in " + value[0] + ", This input parameter is not in the schema")

        print('Exiting...\n')
        sys.exit()

    @classmethod
    def validate(cls):
        """ validates the input data. A schema file is used to validate the inputs.
            if any errors are found they are saved to a dictionary and at the end of the method it will call the error
            method to print all the input errors

            Args:
                schema_file_name(string):

             Returns: a True when complete

        """

        error_list = []

        val_root = cls.schema_tree.getroot()

        for element in list(val_root):

            attribute = None
            for key, value in cls.active_service.items():
                if element.get('name') in value:
                    attribute = key
            if attribute is None:
                continue

            attribute = cls.xmlTree.find(element.get('name'))

            for properties in list(element):

                # Check if attribute is in the schema
                try:
                    value = attribute.find(properties.get('name')).find('value').text
                except KeyError:
                    error_list.append((properties.get('name'), None, "missing"))
                    continue

                obj = properties.findall("*")
                type_of_obj = obj[0].get('type')
                minimum = obj[0].get('min')
                maximum = obj[0].get('max')

                elem = cls.xmlTree.findall(element.get('name'))
                elem = elem[0].find(properties.get('name'))
                type_of_input = elem.find('Data_Type').text
                value = ParamsDER.convert_data_type(value, type_of_input)
                tups = element.get('name'), properties.get('name')
                if tups in list(cls.sensitivity['attributes'].keys()):
                    sensitivity = attribute.find(properties.get('name')).find('sensitivity').text
                    for values in ParamsDER.extract_data(sensitivity, type_of_input):
                        error_list = ParamsDER.checks_for_validate(values, properties, type_of_input,
                                                                   type_of_obj, minimum, maximum, error_list)

                error_list = ParamsDER.checks_for_validate(value, properties, type_of_input,
                                                           type_of_obj, minimum, maximum, error_list)

        # checks if error_list is not empty.
        if error_list:
            ParamsDER.error(error_list)

        return True

    def other_error_checks(self):
        Scenario = self.Scenario
        Tech = self.Technology
        retailTimeShift = self.retailTimeShift
        DCM = self.DCM

        start_year = Scenario['start_year']
        dt = Scenario['dt']
        opt_years = Scenario['opt_years']
        time_series_dict = ParamsDER.datasets["time_series"]
        incl_site_load = Scenario['incl_site_load']

        for time_series_name in time_series_dict:
            time_series = time_series_dict[time_series_name]
            data_length = len(time_series.index)
            years_included = (time_series.index - pd.Timedelta('1s')).year.unique()
            leap_years = [Lib.is_leap_yr(year) for year in years_included]
            timeseries_dt = sum(8784 * leap_years) + sum(8760 * np.invert(leap_years))

            if any(value < years_included[0] for value in opt_years):
                print("Error: The 'opt_years' input starts before the given Time Series.")
                return False

            # quit if timestep size is not the same as indicated by timeseries
            if dt != timeseries_dt / data_length:
                print('Time Series not same granularity as dt parameter. Please check your inputs...')
                return False

        for value in opt_years:
            if value not in years_included.values:
                print("Error: One or more of the 'opt_years' inputs does not have a Time Series data for that year.")
                return False

        if Tech['ch_min_rated'] > Tech['ch_max_rated']:
            print('Error: ch_max_rated < ch_min_rated. ch_max_rated should be greater than ch_min_rated')
            return False

        if Scenario['end_year'] < start_year.year:
            print('Error: end_year < start_year. end_year should be later than start_year')
            return False

        if Tech['install_date'].year > Scenario['opt_years'][-1]:
            print('Error: install_date > opt_years. install_date should be before the last opt_years')
            return False

        if (DCM is not None or retailTimeShift is not None) and incl_site_load != 1:
            print('Error: incl_site_load should be = 1')
            return False

        if all(start_year.year > years_included.values):
            print('Error: "start_year" is set after the date of the time series file.')
            return False

        return True

    @staticmethod
    def checks_for_validate(value, properties, type_of_input, type_of_obj, minimum, maximum, error_list):
        """ Helper function to validate method. This runs the checks to validate the input data.

            Args:
                value ():
                properties ():
                type_of_input ():
                type_of_obj ():
                minimum (int or float):
                maximum (int or float):
                error_list (list):

            Returns: error_list (list):

        """
        # check if type of schema matches type in input file
        if type_of_obj != type_of_input:
            error_list.append((properties.get('name'), type_of_input, "type_dec", type_of_obj))
            return error_list

        # check if there was an error converting the given data in the convert_data_type method
        if type(value) == tuple and value[0] is None:
            error_list.append((properties.get('name'), value[1], "type", type_of_obj))
            return error_list

        # skip list, period, and string inputs before the range test
        if type_of_obj == 'list' or type_of_obj == 'Period' or type_of_obj == 'string':
            return error_list

        # check if data is in valid range
        if minimum is None and maximum is None:
            return error_list
        elif maximum is None and minimum is not None:
            minimum = ParamsDER.convert_data_type(minimum, type_of_obj)
        elif maximum is not None and minimum is None:
            maximum = ParamsDER.convert_data_type(maximum, type_of_obj)
        else:
            minimum = ParamsDER.convert_data_type(minimum, type_of_obj)
            maximum = ParamsDER.convert_data_type(maximum, type_of_obj)

        if minimum is not None and value < minimum:
            error_list.append((properties.get('name'), value, "size", minimum, maximum))
            return error_list

        if maximum is not None and value > maximum:
            error_list.append((properties.get('name'), value, "size", minimum, maximum))
            return error_list

        return error_list

    def load_data_sets(self):
        self.Scenario["time_series"] = self.datasets["time_series"][self.Scenario["time_series_filename"]]
        self.Scenario["monthly_data"] = self.datasets["monthly_data"][self.Scenario["monthly_data_filename"]]
        self.Scenario["customer_tariff"] = self.datasets["customer_tariff"][self.Scenario["customer_tariff_filename"]]
        self.Scenario["cycle_life"] = self.datasets["cycle_life"][self.Scenario["cycle_life_filename"]]

    @staticmethod
    def search_schema_type(root, attribute_name):
        for child in root:
            attributes = child.attrib
            if attributes.get('name') == attribute_name:
                if attributes.get('type') == None:
                    return "other"
                else:
                    return attributes.get('type')

    @classmethod
    def class_summary(cls):
        logging.warning('Asking whether pre-visualization should be included in the log file.')
        YesNo = input("Do you want to include the pre-visualization in the log file? ")
        tree = cls.xmlTree
        treeRoot = tree.getroot()
        schema = cls.schema_tree

        logging.info("Printing summary table for class ParamsDER")
        table = PrettyTable()
        table.field_names = ["Category", "Element", "Active?", "Property", "Analysis?",
                             "Value", "Value Type", "Sensitivity"]
        for element in treeRoot:
            schemaType = cls.search_schema_type(schema.getroot(), element.tag)
            activeness = element.attrib.get('active')
            for property in element:
                table.add_row([schemaType, element.tag, activeness, property.tag, property.attrib.get('analysis'),
                        property.find('value').text, property.find('type').text, property.find('sensitivity').text])

        print(table)
        if "y" in YesNo.lower():
            logging.info('\n' + str(table))
            logging.info("Successfully printed summary table for class ParamsDER in log file")

        print("Printing all scenarios available for Sensitivity Analysis...")
        df = cls.df_analysis
        headers = ["Scenario"]
        for v in df.columns.values:
            headers.append(v)
        table2 = PrettyTable()
        table2.field_names = headers

        for index, row in df.iterrows():
            entry = []
            entry.append(index)
            for col in row:
                entry.append(str(col))
            table2.add_row(entry)

        print(table2)
        if "y" in YesNo.lower():
            logging.info('\n' + str(table2))
            logging.info("Successfully printed all scenarios available in log file")

        logging.info("Building summary tables for the provided csv files...")

        logging.info("Making table for Monthly Data...")
        print("MONTHLY DATA")
        monthly_data_dict = cls.datasets["monthly_data"]
        for md in monthly_data_dict:
            logging.info("Building plots for Monthly Data: %s", md)
            cls.build_table_from_file(md, YesNo)

        logging.info("Making table for Customer Tariff...")
        print("CUSTOMER TARIFF")
        customer_tariff_dict = cls.datasets["customer_tariff"]
        for ct in customer_tariff_dict:
            logging.info("Building plots for Customer Tariff Data: %s", ct)
            cls.build_table_from_file(ct, YesNo)

        logging.info("Making table for Battery Cycle Life...")
        print("BATTERY CYCLE LIFE")
        cycle_life_dict = cls.datasets["cycle_life"]
        for cl in cycle_life_dict:
            logging.info("Building plots for Battery Cycle Life Data: %s", cl)
            cls.build_table_from_file(cl, YesNo)

        logging.info("Successfully building summary tables for csv table files...")

        time_series_dict = cls.datasets["time_series"]
        for ts in time_series_dict:
            logging.info("Building plots for Time Series Data: %s", ts)
            cls.build_plot(ts, YesNo)

        logging.info("Successfully plotting Time Series Data...")

    @staticmethod
    def build_plot(filename, log):

        fig = plt.figure('EPRI', figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
        series = read_csv(filename, header=0, parse_dates=[0], index_col=0, squeeze=True)
        x = series.iloc[1:, 0].values
        labels = list(series.columns.values)
        priceindex = []
        loadindex = []
        prices = []
        loads = []
        for column in range(len(labels)-1):
            if "price" in labels[column+1]:
                prices.append(series.iloc[1:, column+1].values)
                priceindex.append(column+1)
            else:
                loads.append(series.iloc[1:, column+1].values)
                loadindex.append(column+1)

        logging.debug("Finished loading data to plot the time series")

        ax1 = fig.add_subplot(211)
        ax1.set_title('Time Series Data for ' + filename)
        for i in range(len(prices)):
            ax1.plot(x, prices[i], label=labels[priceindex[i]])
        ax1.set_xlabel("DateTime")
        ax1.set_ylabel("Prices")
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax2 = fig.add_subplot(212)
        for i in range(len(loads)):
            ax2.plot(x, loads[i], label=labels[loadindex[i]])
        ax2.set_xlabel("DateTime")
        ax2.set_ylabel("Loads")
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        if 'y' in log:
            logging.info(fig)

        plt.close(fig)
        logging.info("Successfully plotting one Time Series Data...")

        return 0

    @staticmethod
    def build_table_from_file(filename, log):
        data = pd.read_csv(filename, sep=',', header=None)
        labels = data.iloc[0,:].values
        table = PrettyTable()

        for column in range(len(labels)):
            li = data.loc[1:len(data.index),column].values
            table.add_column(labels[column],li)
        print(table)

        if 'y' in log:
            logging.info('\n' + str(table))

        return 0


    # TODO: this summary is for a specific scenario/instance - TN
    def instance_summary(self):
        logging.info("Logging summary for an instance:")
        for key, value in self.active_service.items():
            logging.info("CATEGORY: %s", key)
            for v in value:
                element = getattr(self, v)
                c = copy.copy(element)
                c.pop('cycle_life', None)
                c.pop('monthly_data', None)
                c.pop('customer_tariff', None)
                c.pop('time_series', None)
                logging.info("Element %s: %s", v, str(c))
        return 0
