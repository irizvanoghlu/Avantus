"""
run_StorageVET.py

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
from ParamsDER import Input

import logging
import os
import time

try:
    os.chdir(os.path.dirname(__file__))
except:
    print('INFORMATION: Could not change the working directory.')


TEMPLATE_HEADERS = ['Key', 'Units', 'Description', 'Options']

if __name__ == '__main__':
    starts = time.time()
    logging.basicConfig(filename='logfile.log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w', level=logging.DEBUG)

    logging.info('Started basic configuration')

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

    logging.info('Finished basic configuration with the provided file: %s', arguments.parameters_filename)

    print(os.path.dirname(arguments.parameters_filename))
    if arguments.parameters_filename.endswith(".csv"):
        arguments.parameters_filename = Input.csv_to_xml(arguments.parameters_filename)

    # Initialize the Input Object from Model Parameters and Simulation Cases
    Input.initialize(arguments.parameters_filename, "SchemaDER.xml")
    logging.info('Successfully initialized the Input class with the XML file.')

    # Determine were model_parameters lives
    results_folder = os.path.dirname(arguments.parameters_filename) + "\ "[0] + Input.Questionnaire['result_filename']
    print(results_folder)

    for key, value in Input.instances.items():

        if not value.other_error_checks():
            continue
        run = Sizing.Scenario(value)
        run.add_technology()
        run.add_services()
        run.add_control_constraints()
        run.optimize_problem_loop()
        run.post_optimization_analysis()
        run.save_results_csv(results_folder)

        logging.debug('Successfully ran one simulation.')

    logging.debug('Successfully ran all simulations.')
    logging.debug('summary printed')

    ends = time.time()
    print("time time: ")
    print(ends - starts)
