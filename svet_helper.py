"""
svet_helper.py

Library of helper functions used in StorageVET.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import pyperclip
# import Case
import Technology.Storage as Tech
import ValueStreams.ValueStream as Srv
from pathlib import Path
import numpy as np
import pandas as pd
import time
import copy

# TODO: global variables should be capitalized -HN
bound_names = ['ch_max', 'ch_min', 'dis_max', 'dis_min', 'ene_max', 'ene_min']
fr_obj_names = ['regu_d_cap', 'regu_c_cap', 'regd_d_cap', 'regd_c_cap', 'regu_d_ene', 'regu_c_ene', 'regd_d_ene', 'regd_c_ene']


def yes_no(question):
    """ Helper function. That handles user yes no input question

    Args:
        question (str):

    Returns:
        bool
    """
    yes = {'yes', 'y', 'ye'}
    no = {'no', 'n'}

    while True:
        choice = input(question + '(y/n): ').lower().strip()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            print("Please respond with 'y' or 'n'\n")


def clips():
    """ Helper function: reads file path from clipboard and cleans up string

    Returns:
        path (string)

    Notes:
        TODO: Why do we need this function? Copy-pasting capabilities are handled by the UI -HN
    """

    raw = pyperclip.paste()
    path = Path(raw.replace('"', ''))
    return path


def update_df(df1, df2):
    """ Helper function: Updates elements of df1 based on df2. Will add new columns if not in df1 or insert elements at
    the corresponding index if existing column

    Args:
        df1 (Data Frame): original data frame to be editted
        df2 (Data Frame): data frame to be added

    Returns:
        df1 (Data Frame)
    """

    old_col = set(df2.columns).intersection(set(df1.columns))
    df1 = df1.join(df2[list(set(df2.columns).difference(old_col))], how='left')  # join new columns
    df1.update(df2[list(old_col)])  # update old columns
    return df1


def disagg_col(df, group, col):
    """ Helper function: Adds a disaggregated column of 'col' based on the count of group
    TEMP FUNCTION: assumes that column is merged into disagg dataframe at original resolution = Bad approach

    Args:
        df (Data Frame): original data frame to be
        group (list): columns to group on
        col (string): column to disagg

    Returns:
        df (Data Frame)

    Notes:
        It is not clear from the description above what this function does... -HN

    """
    # TODO this is a temp helper function until a more robust disagg function is built

    count_df = df.groupby(by=group).size()
    count_df.name = 'counts'
    df = df.reset_index().merge(count_df.reset_index(), on=group, how='left').set_index(df.index.names)
    df[col+'_disagg'] = df[col] / df['counts']
    return df


def create_outputs_df(index):
    """ Helper function: creates a template data frame from index of input time_series with necessary period columns

    Args:
        index (DatatimeIndex): time series index

    Returns:
        df (Data Frame)
    """
    SATURDAY = 5
    df = pd.DataFrame(index=index)
    outputs_df = pd.DataFrame(index=(df.index-pd.Timedelta('1s')).to_period('H'))
    df.index.name = 'Datetime'
    df['year'] = (df.index-pd.Timedelta('1s')).to_period('Y')

    df['yr_mo'] = (df.index - pd.Timedelta('1s')).to_period('M')
    df['date'] = (df.index-pd.Timedelta('1s')).to_period('D')

    # df['weekday'] = ((df.index).weekday < SATURDAY).astype('int64')
    df['weekday'] = ((df.index - pd.Timedelta('1s')).weekday < SATURDAY).astype('int64')
    # df['he'] = (df.index).hour
    df['he'] = (df.index - pd.Timedelta('1s')).hour + 1

    return df


def create_opt_agg(df, n, dt):
    """ Helper function: Add opt_agg column to df based on n

    Args:
        df (Data Frame)
        n (): optimization control length
        dt (float): time step

    Returns:
        df (Data Frame)
    """
    # TODO define what N is(maybe) --> used as both an int or a string in this function --HN**
    # optimization level
    if n == 'year':
        df['opt_agg'] = df['year']
    elif n == 'month':
        df['opt_agg'] = df['month']
    else:  # assume n number of days
        n = int(n)
        df['opt_agg'] = 0
        prev = 0
        # opt_agg period should not overlap multiple years
        for yr in df.year.unique():
            sub = copy.deepcopy(df[df.year == yr])
            sub['ind'] = range(len(sub))
            ind = (sub.ind//(n*24*dt)).astype(int)+1  # split year into groups of n days
            df.loc[df.year == yr, 'opt_agg'] = ind + prev  # continue counting from previous year opt_agg
            prev = max(df.opt_agg)

    return df


def apply_growth(source, rate, source_year, yr, dt):
    """ Applies linear growth rate to determine data for future year

    Args:
        source (Series): given data
        rate (float): yearly growth rate (%)
        source_year (Period): given data year
        yr (Period): future year to get data for
        dt (float): simulation time step

    Returns:
        new (Series)
    """

    years = yr - source_year  # difference in years between source and desired year
    new = source*(1+rate/100)**years  # apply growth rate to source data
    new.index = new.index + pd.DateOffset(years=years)  # create new index

    # deal with leap years
    source_leap = is_leap_yr(source_year.year)
    new_leap = is_leap_yr(yr.year)

    if (not source_leap) and new_leap:   # need to add leap day
        # if source is not leap year but desired year is, copy data from previous day
        leap_ind = pd.DatetimeIndex(start='02/29/'+str(yr), end='03/01/'+str(yr), freq=pd.Timedelta(dt, unit='h'), closed='left')
        leap = pd.Series(new[leap_ind - pd.DateOffset(days=1)].values, index=leap_ind, name=new.name)
        new = pd.concat([new, leap])
        new = new.sort_index()
    elif source_leap and (not new_leap):  # need to remove leap day
        leap_ind = pd.DatetimeIndex(start='02/29/'+str(yr), end='03/01/'+str(yr), freq=pd.Timedelta(dt, unit='h'), closed='left')
        new = new[~new.index.isin(leap_ind)]
    return new


def compare_class(c1, c2, compare_init=False, debug_flag=False):
    """ Determines whether class object equals another class object.
    Compare_init = True will do an initial comparison ignoring any attributes that are changed in the course of running a case.

    Args:
        c1 : Class object to compare
        c2 : Class object to compare
        compare_init (bool): Flag to ignore attributes that change after initialization. Default False
        debug_flag (bool): Flag to print or return not equal attributes. Default False


    Returns:
        bool: True if objects are close to equal, False if not equal. (only if debug_flag is False)
    """

    if c1.__class__ == c2.__class__:

        # all attributes from both classes
        attributes = list(set().union(c1.__dict__, c2.__dict__))

        # attributes that should not be compared
        always_ignore = ['inputs', 'results_path', 'verbose', 'verbose_opt', 'start_time', 'end_time']  # from case
        attributes = np.setdiff1d(attributes, always_ignore)

        # attributes that should not be compared if the class has only been initialized
        ignore_init_fields = ['opt_results', 'monthly_bill', 'technologies', 'services', 'predispatch_services', 'financials', 'results']  # from case
        ignore_init_fields += ['degrade_data', 'predispatch_services', 'services', 'expressions', 'physical_constraints', 'control_constraints']  # from tech
        ignore_init_fields += ['expressions', 'technologies', 'c_max', 'c_min', 'd_max', 'd_min', 'e', 'e_lower', 'e_upper', 'ene_results']  # from service
        ignore_init_fields += ['fin_summary', 'monthly_bill', 'obj_val']  # from financials
        ignore_init_fields += ['c_max', 'c_min', 'd_max', 'd_min', 'e', 'e_lower', 'e_upper']  # from services

        if compare_init:
            attributes = np.setdiff1d(attributes, ignore_init_fields)

        # for each attribute compare from class 1 and class 2 if possible
        for attr in attributes:
            print(('class attr:' + attr)) if debug_flag else None
            attr1 = None
            attr2 = None

            try:
                attr1 = getattr(c1, attr)
            except (AttributeError, KeyError):
                if debug_flag:
                    print('false')
                else:
                    return False
            try:
                attr2 = getattr(c2, attr)
            except (AttributeError, KeyError):
                if debug_flag:
                    print('false')
                else:
                    return False

            if (attr1 is not None) & (attr2 is not None):
                compare_attribute(attr1, attr2, parent_class=c1, debug_flag=debug_flag)

        return True
    else:
        print('Class types not equal') if debug_flag else None
        return False


def compare_attribute(attr1, attr2, parent_class=None, debug_flag=False):
    """ Determines whether class attribute equals another class attribute.

    Args:
        attr1 : Attribute object to compare
        attr2 : Attribute object to compare
        debug_flag (bool): Flag to print or return not equal attributes

    Returns:
        bool: True if objects are close to equal, False if not equal. (only if debug_flag is False)
    """

    import cvxpy as cvx

    # determines attribute type based on first attribute passed
    if isinstance(attr1, pd.core.frame.DataFrame):

        # loop over each column and compare if column is in both data frames
        for n in list(set().union(attr1, attr2)):
            print(('dataframe col:' + n)) if debug_flag else None
            col1 = None
            col2 = None
            try:
                col1 = attr1[n]
            except (AttributeError, KeyError):
                if debug_flag:
                    print('false')
                else:
                    return False
            try:
                col2 = attr2[n]
            except (AttributeError, KeyError):
                if debug_flag:
                    print('false')
                else:
                    return False
            if (col1 is not None) & (col2 is not None):
                compare_attribute(col1, col2, debug_flag=debug_flag)

        # # this is faster but cant do percent diff
        # test = pd.testing.assert_frame_equal(attr1, attr2,
        #                                      check_less_precise=4, check_like=True)
    elif isinstance(attr1, (np.ndarray, pd.core.series.Series, pd.core.indexes.datetimes.DatetimeIndex)):

        # just need for fin_results billing_period right now
        if attr1.dtype == object:
            if not attr1.equals(attr2):
                if debug_flag:
                    print('false')
                else:
                    return False
        else:
            compare_array(attr1, attr2, debug_flag=debug_flag)
    elif isinstance(attr1, list):
        for attr1a, attr2a in zip(list(attr1), list(attr2)):
            # print(('list item:' + str(item))) if debug_flag else None
            compare_attribute(attr1a, attr2a, debug_flag=debug_flag)  # recursive call here to loop over elements in list
    elif isinstance(attr1, dict):
        for item in list(attr1):
            # print(('dict item:' + str(item))) if debug_flag else None
            item1 = None
            item2 = None
            try:
                item1 = attr1[item]
            except (AttributeError, KeyError):
                if debug_flag:
                    print('false')
                else:
                    return False
            try:
                item2 = attr2[item]
            except (AttributeError, KeyError):
                if debug_flag:
                    print('false')
                else:
                    return False
            if (item1 is not None) & (item2 is not None):
                compare_attribute(item1, item2, parent_class=parent_class, debug_flag=debug_flag)  # recursive call here to loop over elements in dict
    elif isinstance(attr1, cvx.expressions.expression.Expression):
        if not attr1.__str__() == attr2.__str__():  # compare cvxpy expression as string
            if debug_flag:
                print('false')
            else:
                return False

    # this important to prevent infinite recursion (i.e srv -> tech ->srv)
    elif (isinstance(attr1, Tech.Technology)) & (isinstance(parent_class, (Srv.PreDispService, Srv.Service))):
        if debug_flag:
            print('skipping tech under service')
    else:
        if not attr1 == attr2:  # this could recall compare_class if a custom class
            if debug_flag:
                print('false')
            else:
                return False


def compare_array(arr1, arr2, rtol=1e-4, debug_flag=False):
    """ Determines whether class attribute equals another class attribute.

    Args:
        arr1 : Array to compare
        arr2 : Array to compare
        rtol: relative tolerance to compare
        debug_flag (bool): Flag to print or return not equal attributes

    Returns:
        bool: True if objects are close to equal, False if not equal. (only if debug_flag is False)
    """

    # relative tolerance is same as percent difference
    if not np.allclose(arr1, arr2, rtol=rtol, equal_nan=True):
        if debug_flag:
            print('false')
        else:
            return False


def is_leap_yr(year):
    """ Determines whether given year is leap year or not.

    Args:
        year (int): The year in question.

    Returns:
        bool: True for it being a leap year, False if not leap year.
    """
    return year % 4 == 0 and year % 100 != 0 or year % 400 == 0


def subset_time_series(time_series, year=None, month=None, date=None, start=None, end=None):
    """ Return a subset of given time series data frame based on parameters

    Args:
        time_series (DataFrame): Data
        year (int): year to subset data on
        month (int): month to subset data on
        date (str or Period): day to subset data on
        start (TimeStamp-like): start of subset time (exclusive)
        end (TimeStamp-like): end of subset time (inclusive)

    Returns:
         time_series (DataFrame): subset dataframe
    """
    time_series = copy.deepcopy(time_series)

    if start is not None:
        time_series = time_series[time_series.index > start]
    if end is not None:
        time_series = time_series[time_series.index <= end]

    if year is not None:
        time_series = time_series[(time_series.index - pd.Timedelta('1s')).year == year]

    if month is not None:
        time_series = time_series[(time_series.index - pd.Timedelta('1s')).month == month]

    if date is not None:
        date = pd.Period(date)
        time_series = time_series[(time_series.index-pd.Timedelta('1s')).to_period('D') == date]

    return time_series


def fill_gaps(df, start=None, end=None, dt=None, fill=None):
    """ Fill gaps in time series data frame.
    Will fill any gaps with fill parameter in range of time series index unless given start and end.


    Args:
        df (DataFrame): Data frame to edit
        start (datetime-like): start of dttms. Default is first in df index
        end (datetime-like): end of dttms. Default is last in df index
        dt (float): time series timestep (hours)
        fill (): value to fill in data over gaps

    Returns:
        df (DataFrame): data frame with gaps filled in

    """

    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]
    if dt is None:
        dt = (df.index[1] - df.index[0]).seconds / 3600

    new_index = pd.DatetimeIndex(start=start, end=end, freq=pd.Timedelta(dt, unit='h'))
    df = df.reindex(new_index, fill_value=fill)

    return df
