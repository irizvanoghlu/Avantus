"""
svet_helper.py

Library of helper functions to compare SVET runs. This is beginning of cost/benefit assessment or sensitivity analysis module (mostly untested/built out)
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import os
import svet_inputs as si
import svet_helper as sh
import pandas as pd
import plotly as py
import re
import copy
from functools import reduce
import matplotlib.pyplot as plt


def read_cases(filepath):
    """ Read in all pickle files in directory

    Args:
        filepath (str): directory to look for pickle files

    Returns:
        cases (list): list of case objects

    """
    case_files = [f for f in os.listdir(filepath) if f.endswith('.pickle')]
    cases = [si.read_case(filepath + f) for f in case_files]
    return cases


def compare_financials(cases, cols=None):
    """ Creates comparison df from case.financials.fin_summary

    Args:
        cases (list): list of case object
        cols (list): list of column names to include in comparison

    Returns:
        df (DataFrame): comparison dataframe

    """
    if cols is None:
        cols = ['net', 'net_npv']

    # if 'FR' in cols:
    #     cols.remove('FR')
    #     cols += sh.fr_obj_cols

    # list of dataframes from each case with columns names renames to include case name
    temp_list = [case.financials.fin_summary[list(set(cols) & set(list(case.financials.fin_summary)))].
                 rename(columns={old_col: new_col for (old_col, new_col) in list(zip(cols, [col+'_'+case.name for col in cols]))}) for case in cases]
    # if 'FR' in cols:
    #     df['FR'] = df[sh.fr_obj_names].sum(1)
    #     df = df.drop(columns=sh.fr_obj_names)

    # merge dataframes from all cases
    df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), temp_list)

    return df


def plotly_fin_comp(cases):
    """  Creates a summary plotly figure that compares financials for case files.
    Bar chart with yearly net income and line graph with npv values

    Args:
        cases (list): list of Case objects


    """
    comp = compare_financials(cases)

    npv_cols = list(filter(re.compile("(_*npv)").search, list(comp)))
    net_cols = list(set(list(comp)) - set(npv_cols))
    net_cols.sort()

    index = comp.index.to_series().astype(str).values

    data = []
    colors = py.colors.DEFAULT_PLOTLY_COLORS

    # line graph for npv series
    for i, col in enumerate(npv_cols):
        trace = py.graph_objs.Scatter(x=index, y=comp[col], name=col, mode='lines+markers', line=dict(color=colors[i]),
                                      marker=dict(color='rgb(0,0,0)'),hoverlabel=dict(bgcolor=colors[i]))
        data += [trace]

    # bar chart for income/costs
    for i, col in enumerate(net_cols):
        trace = py.graph_objs.Bar(x=index, y=comp[col], name=col, marker=dict(color=colors[i]))
        data += [trace]

    layout = py.graph_objs.Layout(
        xaxis=dict(title='Time'),
        yaxis=dict(title='$'),
        barmode='group')
    fig = py.graph_objs.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename='comp_plot.html')


def plot_fin_detailed_comp(cases, title, cols=None):
    """ Creates matplotlib figure with detailed financial comparison between cases.
    Plotly does not support grouped and stacked bars

    Args:
        cases (list): list of Case objects to compare
        title (str): Title of plot

    """

    labels = [case.name for case in cases]     # names of cases
    if cols is None:
        cols = ['DA', 'fixed_om', 'Battery_ccost', 'FR']  # names of columns to compare

    # create list
    dfall = []
    for case in cases:

        get_cols = copy.deepcopy(cols)

        # get all FR revenue columns
        if 'FR' in cols:
            get_cols.remove('FR')
            get_cols += sh.fr_obj_names

        df = copy.deepcopy(case.financials.fin_summary[list(set(get_cols) & set(list(case.financials.fin_summary)))])

        # if col not in fin_summary fill in with 0s
        fill_cols = list(set(get_cols) - set(list(df)))
        for col in fill_cols:
            df[col] = 0

        # combine all FR revenue into one column
        if 'FR' in cols:
            df['FR'] = df[sh.fr_obj_names].sum(1)
            df = df.drop(columns=sh.fr_obj_names)

        # set order
        df = df[cols]

        # add to list
        dfall += [df]

    # get NPV columns
    comp = compare_financials(cases, ['net_npv'])

    # create graph
    fig = plot_clustered_stacked(dfall, comp, labels, title=title)

    # change y axis labels to Millions of dollars
    if max(abs(fig.get_yticks())) > 1e6:
        lab = fig.get_yticklabels()
        new_lab = [str(int(l.get_text().replace("−", '-'))/1e6)+'M' for l in lab]
        fig.set_yticklabels(new_lab)


def plot_clustered_stacked(dfall, comp, labels=None, title="multiple stacked bar plot",  **kwargs):
    """ Helper function called in plot_fin_detailed_comp

    copied code from : https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas

    TODO: improve color

    Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
        labels is a list of the names of the dataframe, used for the legend
        title is a string for the title of the plot
        H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall: # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False)
                       #**kwargs)  # make bar plots

    h, l = axe.get_legend_handles_labels() # get the handles we want to modify


    H = ["", "\\", "/"]
    for ind, i in enumerate(range(0, n_df * n_col, n_col)): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H[ind] * 2) #edited part
                rect.set_width(1 / float(n_df + 1))

    # axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation=0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H[i]*2))

    colors = ["#006D2C", "#890000", "#08006b"] # TODO make this dynamic
    for i in range(n_df):
        axe.plot(comp.index.to_series().astype(str).values, comp.iloc[:, i], marker='.', color = colors[i])
    h, l = axe.get_legend_handles_labels() # get the handles we want to modify

    l1 = axe.legend(h[:n_col+n_df], l[:n_col+n_df], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1])
    axe.add_artist(l1)

    # h, l = fig.get_legend_handles_labels()
    # l1 = fig.legend(h[:n], l[:n], loc=[1.01, 0.8])
    # fig.add_artist(l1)
    axe.set_xlabel('Time')
    axe.set_ylabel('$')

    return axe

