"""
runStorageVET.py

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
from dervet.storagevet.run_StorageVET import run_StorageVET

import logging
import os
import time
from datetime import datetime
from pathlib import Path
import sys

developer_path = '.\logs'
try:
    os.mkdir(developer_path)
except OSError:
    print("Creation of the developer_log directory %s failed. Possibly already created." % developer_path)
else:
    print("Successfully created the developer_log directory %s " % developer_path)

LOG_FILENAME1 = developer_path + '\\developer_log_' + datetime.now().strftime('%H_%M_%S_%m_%d_%Y.log')
handler = logging.FileHandler(Path(LOG_FILENAME1))
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
dLogger = logging.getLogger('Developer')
dLogger.setLevel(logging.DEBUG)
dLogger.addHandler(handler)
dLogger.info('Started logging...')


class RunSizing:

    def __init__(self, input):
        """
            Constructor to initialize the parameters and data needed to run StorageVET\

            Args:
                input (ParamsDER.ParamsDER): an initialized Input object for the class to initialize
        """

        self.i = input

        starts = time.time()

        for key, value in self.i.instances.items():

            if not value.other_error_checks():
                continue
            value.prepare_scenario()
            value.prepare_technology()
            value.prepare_services()
            value.prepare_finance()

            run = Sizing.Sizing(value)
            run.add_technology()
            run.add_services()
            run.add_control_constraints()
            run.optimize_problem_loop()
            # run.post_optimization_analysis()
            # run.save_results_csv()

            logging.debug('Successfully ran one simulation.')

        ends = time.time()
        dLogger.info("runStorageVET runtime: ")
        dLogger.info(ends - starts)


def main(model_params_path, schema_relative_path):
    """
                the Main section for DERVET that takes in a string path to the location
                of the model parameters defined by the user. Determines if Sizing or Dispatch
                is to be running, then calls the corresponding functions.
    """
    if model_params_path.endswith(".csv"):
        model_params_path = ParamsDER.csv_to_xml(model_params_path)

    ParamsDER.initialize(model_params_path, schema_relative_path)
    dLogger.info('Successfully initialized the Input class with the XML file.')

    active_commands = ParamsDER.active_components['command']
    if "Results_Directory" in active_commands:
        userLog_path = ParamsDER.csv_path
        try:
            os.makedirs(userLog_path)
        except OSError:
            print("Creation of the user_log directory %s failed. Possibly already created." % userLog_path)
        else:
            print("Successfully created the user_log directory %s " % userLog_path)
    else:
        userLog_path = developer_path

    LOG_FILENAME2 = userLog_path + '\\user_log_' + datetime.now().strftime('%H_%M_%S_%m_%d_%Y.log')

    handler = logging.FileHandler(Path(LOG_FILENAME2))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    uLogger = logging.getLogger('User')
    uLogger.setLevel(logging.DEBUG)
    uLogger.addHandler(handler)
    uLogger.info('Started logging...')

    if "Previsualization" in active_commands:
        ParamsDER.class_summary()
        uLogger.info('Successfully ran the pre-visualization.')

    if "Validation" in active_commands:
        ParamsDER.validate()
        uLogger.info('Successfully ran validate.')

    if "Simulation" in active_commands:
        if "Sizing" in active_commands:
            RunSizing(ParamsDER)
            uLogger.info('Sizing solution found.')
            print("Program is done.")
        elif "Dispatch" in active_commands:
            run_StorageVET(ParamsDER)
            uLogger.info('Dispatch solution found.')
            print("Program is done.")


if __name__ == '__main__':
    """
            the Main section for runStorageVET to run by itself without the GUI 
    """

    parser = argparse.ArgumentParser(prog='StorageVET.py',
                                     description='The Electric Power Research Institute\'s energy storage system ' +
                                                 'analysis, dispatch, modelling, optimization, and valuation tool' +
                                                 '. Should be used with Python 3.6.x, pandas 0.19+.x, and CVXPY' +
                                                 ' 0.4.x or 1.0.x.',
                                     epilog='Copyright 2018. Electric Power Research Institute (EPRI). ' +
                                            'All Rights Reserved.')
    parser.add_argument('parameters_filename', type=str,
                        help='specify the filename of the CSV file defining the PARAMETERS dataframe')
    # parser.add_argument('-v', '--verbose', action='store_true',
    #                     help='specify this flag for verbose output during execution')
    # parser.add_argument('--gitlab-ci', action='store_true',
    #                     help='specify this flag for gitlab-ci testing to skip user input')
    arguments = parser.parse_args()

    dLogger.info('Finished basic configuration with the provided file: %s', arguments.parameters_filename)

    # Initialize the Input Object from Model Parameters and Simulation Cases
    script_rel_path = sys.argv[0]
    dir_rel_path = script_rel_path[:-len('run_DERVET.py')]
    schema_rel_path = dir_rel_path + "SchemaDER.xml"

    main(arguments.parameters_filename, schema_rel_path)
