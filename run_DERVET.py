"""
runDERVET.py

This Python script serves as the initial launch point executing the
Python-based version of DERVET.
"""
import argparse
from dervet.DERVET import DERVET


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='run_DERVET.py',
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
    parser.add_argument('--gitlab-ci', action='store_true',
                        help='specify this flag for gitlab-ci testing to skip user input')
    arguments = parser.parse_args()

    case = DERVET(arguments.parameters_filename, verbose=arguments.verbose, ignore_cba_valuation=True)
    case.solve()
