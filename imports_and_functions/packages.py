import plotly.express as px
from datetime import datetime
import seaborn as sns
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import pmdarima as pm
import statsmodels.tsa.api as tsa
import pandas as pd
pd.set_option('display.max_columns', 0)
from ipywidgets import interact

# from jupyterthemes import jtplot
# # OG plot
# jtplot.reset()
# # jtplot.style(theme='monokai', context='notebook', ticks='True', grid='False')
# plt.style.use('ggplot')

# # https://matplotlib.org/stable/tutorials/introductory/customizing.html
# font = {'weight': 'normal', 'size': 8}
# text = {'color': 'white'}
# axes = {'labelcolor': 'white'}
# xtick = {'color': 'white'}
# ytick = {'color': 'white'}
# legend = {'facecolor': 'black', 'framealpha': 0.6}

# # mpl.rcParams['figure.facecolor'] = '#232323' # matplotlib over-writes this
# mpl.rc('legend', **legend)
# mpl.rc('text', **text)
# mpl.rc('xtick', **xtick)
# mpl.rc('ytick', **ytick)
# mpl.rc('axes', **axes)
# mpl.rc('font', **font)
