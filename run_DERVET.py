"""
run_DERVET.py

This Python script serves as the initial launch point executing the Python-based version of StorageVET
(AKA StorageVET 2.0 or SVETpy).
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import argparse
import Sizing
from ParamsDER import ParamsDER

from datetime import datetime
from pathlib import Path
import logging
import os
import time

try:
    os.chdir(os.path.dirname(__file__))
except:
    print('INFORMATION: Could not change the working directory.')

path = '.\logs'

# TODO: make multi-platform by using path combine functions
# TODO: set working directory path to be where the function is called (for results folder)

try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed. Possibly already created." % path)
else:
    print("Successfully created the directory %s " % path)

LOG_FILENAME1 = path + '\\developer_log_' + datetime.now().strftime('%H_%M_%S_%m_%d_%Y.log')
LOG_FILENAME2 = path + '\\user_log_' + datetime.now().strftime('%H_%M_%S_%m_%d_%Y.log')

handler = logging.FileHandler(Path(LOG_FILENAME1))
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
dLogger = logging.getLogger('Developer')
dLogger.setLevel(logging.DEBUG)
dLogger.addHandler(handler)

handler = logging.FileHandler(Path(LOG_FILENAME2))
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
uLogger = logging.getLogger('User')
uLogger.setLevel(logging.DEBUG)
uLogger.addHandler(handler)

dLogger.info('Started logging...')
uLogger.info('Started logging...')

TEMPLATE_HEADERS = ['Key', 'Units', 'Description', 'Options']


class RunSizing:

    def __init__(self, params):
        """
            Constructor to initialize the parameters and data needed to run StorageVET\

            Args:
                params (ParamsDER.ParamsDER): an initialized Input object for the class to initialize
        """

        self.i = params

        starts = time.time()

        for key, value in self.i.instances.items():
            if not value.other_error_checks():
                continue
            run = Sizing.Scenario(value)
            run.add_technology()
            run.add_services()
            run.add_control_constraints()
            run.optimize_problem_loop()
            run.post_optimization_analysis()
            run.save_results_csv(results_folder)

        ends = time.time()
        dLogger.info("run_StorageVET runtime: ")
        dLogger.info(ends - starts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='StorageVET.py',
                                     description='The Electric Power Research Institute\'s energy storage system ' +
                                                 'analysis, dispatch, modelling, optimization, and valuation tool' +
                                                 '. Should be used with Python 3.6.x, pandas 0.19+.x, and CVXPY' +
                                                 ' 0.4.x or 1.0.x.',
                                     epilog='Copyright 2018. Electric Power Research Institute (EPRI). ' +
                                            'All Rights Reserved.')
    parser.add_argument('parameters_filename', type=str,
                        help='specify the filename of the CSV file defining the PARAMETERS dataframe')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='specify this flag for verbose output during execution')
    arguments = parser.parse_args()

    dLogger.info('Finished basic configuration with the provided file: %s', arguments.parameters_filename)

    print(os.path.dirname(arguments.parameters_filename))
    if arguments.parameters_filename.endswith(".csv"):
        arguments.parameters_filename = ParamsDER.csv_to_xml(arguments.parameters_filename)

    # Initialize the ParamsDER Object from Model Parameters and Simulation Cases
    ParamsDER.initialize(arguments.parameters_filename, "Schema.xml")
    dLogger.info('Successfully initialized the ParamsDER class with the XML file.')

    # # Determine were model_parameters lives
    # results_folder = os.path.dirname(arguments.parameters_filename) + "\ "[0] + ParamsDER.Questionnaire['result_filename']
    # print(results_folder)

    active_commands = ParamsDER.active_components['command']  # TODO: replace with getter function --HN

    if not arguments.gitlab_ci:
        if "Previsualization" in active_commands:
            ParamsDER.class_summary()
            dLogger.info('Successfully ran the pre-visualization.')
            uLogger.info('Successfully ran the pre-visualization.')
        else:
            dLogger.info('No pre-visualization was done.')
            uLogger.info("No pre-visualization was done.")

        if "Validation" in active_commands:
            ParamsDER.validate()
            dLogger.info('Successfully ran validate.')
            uLogger.info('Successfully ran validate.')
        else:
            dLogger.info('No validation was done.')
            uLogger.info("No validation was done.")

        if "Simulation" in active_commands:
            RunSizing(ParamsDER)
            dLogger.info('Simulation ran successfully.')
            uLogger.info('Simulation ran successfully.')
        else:
            dLogger.info('No simulation was ran.')
            uLogger.info('No simulation was ran.')
    else:
        ParamsDER.class_summary()
        dLogger.info('Successfully ran the pre-visualization.')
        ParamsDER.validate()
        dLogger.info('Successfully ran validate.')
        RunSizing(ParamsDER)
        dLogger.info('Simulation ran successfully.')

    print("Program is done.")
