"""
runDERVET.py

This Python script serves as the initial launch point executing the
Python-based version of DERVET.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). ' \
                'All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta',
               'Halley Nathwani', "Thien Nguyen", 'Kunle Awojinrin']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = '0.1.1'

import argparse
import sys
import os
# # determine storagevet path (absolute path)
# storagevet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storagevet')
# # storagevet_path = os.path.join(storagevet_path, 'storagevet')
# print(storagevet_path)
# # add dervet (source root) to PYTHONPATH
# sys.path.insert(0, storagevet_path)
# print(sys.path)
from dervet.DERVET import DERVET


if __name__ == '__main__':
    """
        This section is run when the file is called from the command line.
    """

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
