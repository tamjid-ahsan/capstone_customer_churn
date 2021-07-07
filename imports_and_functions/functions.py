# imports
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import r2_score
from pmdarima.arima.utils import ndiffs
from pmdarima.arima.utils import nsdiffs
import folium
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib as mpl
import statsmodels.tsa.api as tsa
from IPython.display import display, Markdown
import numpy as np
import pmdarima as pm
import warnings
import json
from statsmodels.tools.eval_measures import rmse, mse
from sklearn.metrics import mean_absolute_error as mae
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', ConvergenceWarning)


# functions


def forecast_to_df(forecast, zipcode):
    """
    Creates dataframe from statsmodels.tsa model obeject forecast

    ### Helper function ###

    Parameters:
    ===========
    forecast = output from model forecast;
    zipcode = str; zipcode name,
    """
    test_pred = forecast.conf_int()
    test_pred[zipcode] = forecast.predicted_mean
    test_pred.columns = ['lower', 'upper', 'prediction']
    return test_pred


def model_error_report(test, pred_df, show_report=False):
    """
    Generates model reports; rmse, mse, mae

    ### Helper function ###

    Parameters:
    ===========
    test = array like; no default; test y,
    pred_df = predicted y with confidance interval.
    """
    rmse_ = rmse(pred_df['prediction'], test)
    mse_ = mse(pred_df['prediction'], test)
    mae_ = mae(test, pred_df['prediction'])
    if show_report:
        print(f'Root Mean Squared Error of test and prediction: {rmse_}')
        print(f'Mean Squared Error: {mse_}')
        print(f'Mean Absolute Error: {mae_}')
    return rmse_, mse_, mae_


def plot_test_pred(test, pred_df, figsize=(15, 5), conf_int=True):
    """
    plots test and prediction

    ### Helper function ###

    returns matplotlib fig and ax object.

    Parameters:
    ===========
    test = array like; no default; test y,
    pred_df = predicted y with confidance interval.
    figsize = tuple of int or float; deafult = (15, 5), figure size control,
    conf_int = bool; default = True, plots confidance interval
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(test, label='Test', marker='o', color='#ff6961', lw=4)
    ax.plot(pred_df['prediction'], label='prediction',
            ls='--', marker='o', color='#639388', lw=4)
    if conf_int:
        ax.fill_between(
            x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'], color='#938863', alpha=.5)
    ax.legend()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.tight_layout()
    rmse_, mse_, mae_ = model_error_report(test, pred_df, show_report=True)

    return fig, ax


def plot_train_test_pred(train, test, pred_df, figsize=(15, 5), color_by_train_test=True):
    """
    plots test and prediction

    ### Helper function ###

    Parameters:
    ===========
    train = array like; no default; train y,
    test = array like; no default; test y,
    pred_df = pandas.DataFrame; no default; predicted y with confidance interval,
    figsize = tuple of int or float; deafult = (15, 5), figure size control,
    color_by_train_test = bool; default = True, seperates train and test data by color.

    returns matplotlib fig and ax object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # diffrentiate train test split by color
    if color_by_train_test:
        color_list = ['#886393', '#ff6961']
    else:
        color_list = ['#886393', '#886393']
    # train
    ax.plot(train, label='Train', marker='o', color=color_list[0])
    # test
    ax.plot(test, label='Test', marker='o', color=color_list[1])
    # prediction
    ax.plot(pred_df['prediction'], label='prediction',
            ls='--', marker='o', color='#639388')
    ax.fill_between(
        x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'], color='#938863', alpha=.5)
    ax.legend()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_train_test_pred_forecast(train, test, pred_df_test, pred_df, zipcode, figsize=(15, 5), color_by_train_test=True):
    """
    plots test and prediction

    ### Helper function ###

    Parameters:
    ===========
    train = array like; no default; train y,
    test = array like; no default; test y,
    pred_df_test = array like; no default; predicted y on train y with confidance interval,
    pred_df = array like; no default; predicted y on all data with confidance interval,
    figsize = tuple of int or float; deafult = (15, 5), figure size control,
    color_by_train_test = bool; default = True, seperates train and test data by color.

    returns matplotlib fig and ax object. and roi information as pandas.DataFrame object
    """
    fig, ax = plt.subplots(figsize=figsize)
    kws = dict(marker='*')
    if color_by_train_test:
        color_list = ['#886393', '#ff6961']
    else:
        color_list = ['#886393', '#886393']
    # train
    ax.plot(train, label='Train', **kws, color=color_list[0])
    # test
    ax.plot(test, label='Test', **kws, color=color_list[1])
    # prediction
    ax.plot(pred_df_test['prediction'],
            label='prediction',
            ls='--',
            **kws,
            color='#639388')
    ax.fill_between(x=pred_df_test.index,
                    y1=pred_df_test['lower'],
                    y2=pred_df_test['upper'],
                    color='#938863', alpha=.5)
    # forecast
    ax.plot(pred_df['prediction'],
            label='forecast',
            ls='--',
            **kws,
            color='#ffd700')
    ax.fill_between(x=pred_df.index,
                    y1=pred_df['lower'],
                    y2=pred_df['upper'],
                    color='#0028ff', alpha=.5)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title('train-test-pred-forecast plot', size=15)
    fig.tight_layout()
    plt.show()
    # ROI
    mean_roi = (pred_df[-1:]['prediction'][0] - test[-1:][0])/test[-1:][0]
    lower_roi = (pred_df[-1:]['lower'][0] - test[-1:][0])/test[-1:][0]
    upper_roi = (pred_df[-1:]['upper'][0] - test[-1:][0])/test[-1:][0]
    std_roi = np.std([lower_roi, upper_roi])
    roi_df = pd.DataFrame([{
        'zipcode': zipcode,
        'mean_forecasted_roi': round(mean_roi*100, 2),
        'lower_forecasted_roi': round(lower_roi*100, 2),
        'upper_forecasted_roi': round(upper_roi*100, 2),
        'std_forecasted_roi': round(std_roi*100, 2)
    }])
    display(roi_df)
    return fig, ax, roi_df


def plot_acf_pacf(ts, figsize=(15, 8), lags=24):
    """
    Plots acf and pacf of a time series

    Parameters:
    ===========
    ts = array like; no default; time series data, 
    figsize = tuple of int or float; deafult = (15, 8), figure size control,
    lags = int; default = 24, lag input of plot_acf() of statsmodels.tsa

    returns matplotlib fig and ax object.
    """
    fig, ax = plt.subplots(nrows=3, figsize=figsize)
    # Plot ts
    ts.plot(ax=ax[0], color='#886393', lw=5)
    # Plot acf, pacf
    plot_acf(ts, ax=ax[1], lags=lags, color='#886393', lw=5)
    plot_pacf(ts, ax=ax[2], lags=lags, method='ld', color='#886393', lw=5)
    fig.tight_layout()
    fig.suptitle(f"Zipcode: {ts.name}", y=1.02, fontsize=15)
    for a in ax[1:]:
        a.xaxis.set_major_locator(
            mpl.ticker.MaxNLocator(min_n_ticks=lags, integer=True))
        a.xaxis.grid()
    plt.show()
    return fig, ax


def adfuller_test_df(ts, index=['AD Fuller Results']):
    """
    Adapted from https://github.com/learn-co-curriculum/dsc-removing-trends-lab/tree/solution
    Returns the AD Fuller Test Results and p-values for the null hypothesis that there the 
    data is non-stationary (that there is a unit root in the data).

    ### helper function ###

    Parameters:
    ===========
    ts = array like; no default; time series data, 

    Returns:
    ========
    report of the test as pandas.DataFrame object.
    """

    df_res = tsa.stattools.adfuller(ts)

    names = ['Test Statistic', 'p-value',
             '#Lags Used', '# of Observations Used']
    res = dict(zip(names, df_res[:4]))

    res['p<.05'] = res['p-value'] < .05
    res['Stationary?'] = res['p<.05']

    if isinstance(index, str):
        index = [index]
    res_df = pd.DataFrame(res, index=index)
    res_df = res_df[['Test Statistic', '#Lags Used',
                     '# of Observations Used', 'p-value', 'p<.05',
                     'Stationary?']]
    return res_df


def stationarity_check(TS, window=8, plot=True, figsize=(15, 5), index=['ADF Result']):
    """
    Adapted from https://github.com/learn-co-curriculum/dsc-removing-trends-lab/tree/solution
    Checks stationarity of the time series.

    Parameters:
    ===========
    TS = array like; no default; time series data, , 
    window = int; default = 8, rolling window input.
    plot = bool; default = True, displays plot
    figsize = tuple of int or float; deafult = (15, 5), figure size control,
    index= list containing str; default = ['ADF Result']. dataframe index input.

    Returns:
    ========
    report of the test as pandas.DataFrame object.
    """
    # Calculate rolling statistics
    roll_mean = TS.rolling(window=window, center=False).mean()
    roll_std = TS.rolling(window=window, center=False).std()

    # Perform the Dickey Fuller Test
    dftest = adfuller_test_df(TS, index=index)

    if plot:

        # Building in contingency if not a series with a freq
        try:
            freq = TS.index.freq
        except:
            freq = 'N/A'

        # Plot rolling statistics:
        fig = plt.figure(figsize=figsize)
        plt.plot(TS, color='#886393', label=f'Original (freq={freq}')
        plt.plot(roll_mean,
                 color='#ff6961',
                 label=f'Rolling Mean (window={window})')
        plt.plot(roll_std,
                 color='#333333',
                 label=f'Rolling Std (window={window})')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation', size=15)
        # display(dftest)
        plt.show(block=False)

    return dftest


def melt_data(df):
    """
    Converts dataframe from wide form to long form.
    ### pre-defined function ###
    Returns:
    ========
    panda.DataFrame
    """
    melted = pd.melt(df,
                     id_vars=['RegionName', 'State',
                              'City', 'Metro', 'CountyName'],
                     var_name='date')
    melted['date'] = pd.to_datetime(melted['date'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted


def model_builder(train, test, order, seasonal_order, zipcode, figsize=(15, 8), show_summary=True, show_diagnostics=True, show_prediction=True):
    """
    builds a model with statsmodels.tsa.SARIMAX and prints information.

    Parameters:
    ===========
    train = array like; no default; train y, 
    test = array like; no default; test y, 
    order = tuple of three int; no default, p, d, q of SARIMAX model,
    seasonal_order = tuple of four int; no default, P, D, Q, m of SARIMAX model,
    zipcode = str; zipcode name, 
    figsize = tuple of int or float; deafult = (15, 8), figure size control, 
    show_summary = bool; default = True, show SARIMAX summary, 
    show_diagnostics = bool; default = True, show model diagonistics reports, 
    show_prediction = bool; default = True, show prediction,

    Returns:
    ========
    SARIMAX model object,
    prediction of the model for steps of length of test.
    """
    # model
    model = tsa.SARIMAX(train, order=order,
                        seasonal_order=seasonal_order).fit()
    if show_summary:
        display(model.summary())
    print('\033[1m \033[5;30;47m' +
          f'{" "*70}Model Diagonostics of {zipcode}{" "*70}'+'\033[0m')
    if show_diagnostics:
        model.plot_diagnostics(figsize=figsize)
        plt.tight_layout()
        plt.show()
    # forecast
    forecast = model.get_forecast(steps=len(test))
    pred_df = forecast_to_df(forecast, zipcode)
    if show_prediction:
        print('\033[1m \033[5;30;47m' +
              f'{" "*70}Performance on test data of {zipcode}{" "*70}'+'\033[0m')
        _, ax = plot_test_pred(
            test, pred_df, figsize=(figsize[0], figsize[1]/2))
        ax.set_title('test-pred plot', size=15)
        _, ax = plot_train_test_pred(
            train, test, pred_df, figsize=(figsize[0], figsize[1]-2))
        ax.set_title('train-test-pred plot', size=15)
        plt.show()

    return model, pred_df


def grid_search(ts, train, test, forecast_steps=36, figsize=(15, 5), trace=True, display_results=True, display_roi_results=True):
    """
    grid searching using pyramid arima for best p, d, q, P, D, Q, m for 
    a SARIMA model using predefined conditions and shows model performance
     for predicting in the future.

    ### predefined options ###
    # d and D is calculated using ndiffs using 'adf'(Augmented Dickey–Fuller test for Unit Roots)
     for d and 'ocsb' (Osborn, Chui, Smith, and Birchenhall Test for Seasonal Unit Roots) for D.
    # parameters for auto_arima model:
    start_p=0; The starting value of p, the order (or number of time lags) of the auto-regressive (“AR”) model.
    d=d; The order of first-differencing,
    start_q=0; order of the moving-average (“MA”) model,
    max_p=3, max value for p
    max_q=3, max value for q
    start_P=0; the order of the auto-regressive portion of the seasonal model,
    D=D; The order of the seasonal differencing,
    start_Q=0; the order of the moving-average portion of the seasonal model,
    max_P=3, max value of P
    max_Q=3, max value for Q
    m=12; The period for seasonal differencing, 
                refers to the number of periods in each season.,
    seasonal=True; this data is seasonal,
    stationary=False; data is not stationary,
    information_criterion='oob', optimizing on `out-of-bag` sample validation on a scoring metric, 
                other information criterias did not perform well
    out_of_sample_size=12, step hold out for validation,
    scoring='mse', validation metric,
    method='lbfgs'; limited-memory Broyden-Fletcher-Goldfarb-Shanno with optional box constraints, 
                BFGS is  in the family of quasi-Newton-Raphson methods that 
                approximates the `bfgs` using a limited amount of computer memory.

    Parameters:
    ===========
    ts = array like; no default; time series y, 
    train = array like; no default; train y, 
    test = array like; no default; test y, 
    forecast_steps = int; default = 36, steps to forecast into future, 
    figsize = tuple of int or float; deafult = (15, 8), figure size control, 
    trace = bool; default = True, 
    display_results = bool; default = True, 
    display_roi_results = bool; default = True,

    Returns:
    ========
    auto_model model object created by pmdautoarima, 
    predictions of test, 
    forecast of ts for selected steps.
    """

    d = ndiffs(train, alpha=0.05, test='adf', max_d=4)
    D = nsdiffs(train, m=12, test='ocsb', max_D=4)

    auto_model = pm.auto_arima(y=train,
                               X=None,
                               start_p=0,
                               d=d,
                               start_q=0,
                               max_p=3,
                               max_q=3,
                               start_P=0,
                               D=D,
                               start_Q=0,
                               max_P=3,
                               max_Q=3,
                               m=12,
                               seasonal=True,
                               stationary=False,
                               information_criterion='oob',
                               stepwise=True,
                               suppress_warnings=True,
                               error_action='warn',
                               trace=trace,
                               out_of_sample_size=12,
                               scoring='mse',
                               method='lbfgs',
                               )
    # display results of grid search
    zipcode = ts.name
    if display_results:
        print('\033[1m \033[5;30;47m' +
              f'{" "*70}Model Diagonostics of {zipcode}{" "*70}'+'\033[0m')
        display(auto_model.summary())
        auto_model.plot_diagnostics(figsize=figsize)
        plt.tight_layout()
        plt.show()
    # fitting model on train data with the best params found by grid search
    best_model = tsa.SARIMAX(train,
                             order=auto_model.order,
                             seasonal_order=auto_model.seasonal_order, maxiter=500,
                             enforce_invertibility=False).fit()
    forecast = best_model.get_forecast(steps=len(test))
    pred_df_test = pd.DataFrame([forecast.conf_int(
    ).iloc[:, 0], forecast.conf_int().iloc[:, 1], forecast.predicted_mean]).T
    pred_df_test.columns = ["lower", 'upper', 'prediction']
    if display_results:
        print('\033[1m \033[5;30;47m' +
              f'{" "*70}Performance on test data of {zipcode}{" "*70}'+'\033[0m')
        _, ax = plot_test_pred(
            test, pred_df_test, figsize=(figsize[0], figsize[1]/2))
        ax.set_title('test-pred plot', size=15)

        _, ax = plot_train_test_pred(
            train, test, pred_df_test, figsize=figsize)
        ax.set_title('train-test-pred plot', size=15)
        plt.show()

    # fitting on entire data
    best_model_all = tsa.SARIMAX(ts,
                                 order=auto_model.order,
                                 seasonal_order=auto_model.seasonal_order, maxiter=500,
                                 enforce_invertibility=False).fit()
    forecast = best_model_all.get_forecast(steps=forecast_steps)
    pred_df = pd.DataFrame([forecast.conf_int(
    ).iloc[:, 0], forecast.conf_int().iloc[:, 1], forecast.predicted_mean]).T
    pred_df.columns = ["lower", 'upper', 'prediction']
    if display_results:
        print('\033[1m \033[5;30;47m' +
              f'{" "*70}Forecast of {zipcode}{" "*70}'+'\033[0m')
        _, ax = plot_train_test_pred(train, test, pred_df, figsize=figsize)
        ax.set_title('train-test-forecast plot', size=15)
        plt.show()
    if display_roi_results:
        plot_train_test_pred_forecast(
            train, test, pred_df_test, pred_df, zipcode, figsize=figsize)

    return auto_model, pred_df_test, pred_df


def zip_code_map(roi_df):
    """
    Returns an interactive map of zip codes colorized to reflect
    expected return on investment using folium.
    """
    geojason = json.load(
        open('./data/ny_new_york_zip_codes_geo.min.json', 'r'))
    zip_code_map = folium.Map(
        location=[40.7027, -73.7890], width=1330, height=820, zoom_start=11)
    folium.Choropleth(
        geo_data=geojason,
        name='choropleth',
        data=roi_df,
        columns=['zipcode', 'mean_forecasted_roi'],
        key_on='feature.properties.ZCTA5CE10',
        fill_color='BuGn',
        fill_opacity=0.7,
        nan_fill_opacity=0
    ).add_to(zip_code_map)
    return zip_code_map


def map_zipcodes_return(df, plot_style='interactive', geojson_file_path='./data/ny_new_york_zip_codes_geo.min.json'):
    """
    GeoJson sourced from: 
    Returns an map of zip codes colorized to reflect
    expected return on investment using plotly express. 
    ### pre-defined function ###

    Parameters:
    ===========
    df = pandas.DataFrame; no default, return on investment dataframe, 
    plot_style = str; default = 'interactive', 
                available options:
                -  'interactive'
                - 'static'
                - 'dash'
    geojson_file_path = geojson; default = './data/ny_new_york_zip_codes_geo.min.json',
                    file path of geojson file.
    """
    import plotly.express as px
    geojason = json.load(open(geojson_file_path, 'r'))
    for feature in geojason['features']:
        feature['id'] = feature['properties']['ZCTA5CE10']
    fig = px.choropleth_mapbox(data_frame=df,
                               geojson=geojason,
                               locations='zipcode',
                               color='mean_forecasted_roi',
                               mapbox_style="stamen-terrain",
                               # range_color=[-20,20],
                               zoom=10.5, color_continuous_scale=['#FF3D70', '#FFE3E8', '#039E0F'],
                               color_continuous_midpoint=0, hover_name='Neighborhood',
                               hover_data=[
                                   'mean_forecasted_roi', 'lower_forecasted_roi',
                                   'upper_forecasted_roi', 'std_forecasted_roi'
                               ],
                               title='Zipcode by Average Price',
                               opacity=.85,
                               height=800,
                               center={
                                   'lat': 40.7027,
                                   'lon': -73.7890
                               })
    fig.update_geos(fitbounds='locations', visible=True)
    fig.update_layout(margin={"r": 0, "l": 0, "b": 0})
    if plot_style == 'interactive':
        fig.show()
    if plot_style == 'static':
        # import plotly.io as plyIo
        img_bytes = fig.to_image(format="png", width=1400, height=800, scale=1)
        from IPython.display import Image
        display(Image(img_bytes))
    if plot_style == 'dash':
        import dash
        import dash_core_components as dcc
        import dash_html_components as html
        app = dash.Dash()
        app.layout = html.Div([dcc.Graph(figure=fig)])
        app.run_server(debug=True, use_reloader=False)


def prediction_analysis(ts, test, forecast):
    """
    Creates forecast and time series data with 2 resistance and 5 support level.

    Parameters:
    ===========
    ts = array like; no default; time series y,
    test = array like; no default; test y,
    forecast = pandas.DataFrame; predicted y with confidance interval.
    """
    HIGH = test.max()
    LOW = test.min()
    CLOSE = test[-1]

    PP = (HIGH + LOW + CLOSE) / 3
    S1 = 2 * PP - HIGH
    S2 = PP - (HIGH - LOW)
    S3 = PP * 2 - (2 * HIGH - LOW)
    S4 = PP * 3 - (3 * HIGH - LOW)
    S5 = PP * 4 - (4 * HIGH - LOW)
    R1 = 2 * PP - LOW
    R2 = PP + (HIGH - LOW)

    fig, ax = plt.subplots(figsize=(15, 5))
    # TS
    ts.plot(ax=ax, color='#886393')
    # forecast
    forecast['prediction'].plot(ax=ax, color='#5d9db1')
    ax.fill_between(forecast.index,
                    forecast.lower,
                    forecast.upper,
                    alpha=.6,
                    color='#accdd7')
    # support and resistance
    kws = dict(color='#ff6961', xmin=.6, ls='dashed')
    kws_1 = dict(color='#008807', xmin=.6, ls='dashed')
    plt.axhline(S1, **kws, label='Support 1')
    plt.axhline(S2, **kws, label='Support 2', alpha=.8)
    plt.axhline(S3, **kws, label='Support 3', alpha=.7)
    plt.axhline(S4, **kws, label='Support 4', alpha=.6)
    plt.axhline(S5, **kws, label='Support 5', alpha=.5)
    plt.axhline(R1, **kws_1, label='Resistance 1')
    plt.axhline(R2, **kws_1, label='Resistance 2', alpha=.7)
    plt.title(f'Zipcode {ts.name} prediction analysis', fontsize=15)
    plt.legend()
    plt.show()
    return fig, ax


def show_py_file_content(file='./imports_and_functions/functions.py'):
    """
    displays content of a py file output formatted as python code in jupyter notebook.

    Parameter:
    ==========
    file = `str`; default: './imports_and_functions/functions.py',
                path to the py file.
    """
    with open(file, 'r', encoding="utf8") as f:
        x = f"""```python
{f.read()}```"""
        display(Markdown(x))


def map_based_on_zipcode(map_df, mapbox_style="open-street-map"):
    """ 
    plots map

    Parameters:
    ===========
    mapbox_style = str; options are following:
        > "white-bg" yields an empty white canvas which results in no external HTTP requests

        > "carto-positron", "carto-darkmatter", "stamen-terrain",
          "stamen-toner" or "stamen-watercolor" yield maps composed of raster tiles 
          from various public tile servers which do not require signups or access tokens

        > "open-street-map" does work
    """
    fig = px.scatter_mapbox(
        map_df,
        lat=map_df.lat,
        lon=map_df.long,
        color='Zipcode',
        zoom=11,
        size='values',
        height=1200,
        title='Zipcode location',
        center={
            'lat': map_df[map_df['Zipcode'] == '11418']['lat'].values[0],
            'lon': map_df[map_df['Zipcode'] == '11418']['long'].values[0]
        })
    # use "stamen-toner" or "carto-positron"
    fig.update_layout(mapbox_style=mapbox_style)
    fig.update_layout(margin={"r": 0, "l": 0, "b": 1})
    # fig.update_traces(marker=dict(size=20),
    #                   selector=dict(mode='markers'))
    fig.show()
    pass


def fig_ret(code, results):

    pred = results[code]['pred_df']['prediction']
    low = results[code]['pred_df']['lower']
    high = results[code]['pred_df']['upper']
    mergerd_train_test = results[code]['train'].combine_first(
        results[code]['test'])

    import plotly.io as pio
    pio.templates.default = 'presentation'  # plotly_dark' , ggplot2

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(y=mergerd_train_test,
                   x=mergerd_train_test.index,
                   mode='lines+markers',
                   line_color='#537d8d',
                   name='TS'))

    fig.add_trace(
        go.Scatter(y=low,
                   x=low.index,
                   mode='lines',
                   name='Lower CI',
                   line_color='#22c256'))
    fig.add_trace(
        go.Scatter(
            y=high,
            x=high.index,
            fill='tonexty',
            mode='lines',
            line_color='#22c256',
            opacity=.5,
            name='high CI',
        ))
    fig.add_trace(
        go.Scatter(y=pred,
                   x=pred.index,
                   mode='lines+markers',
                   name='Forecast',
                   line_color='#ff6961'))
    fig.update_layout(title=f"Time series plot with forecast, of zipcode: {code}",
                      xaxis_title="Years",
                      yaxis_title="Home values",
                      legend_title="Legends",
                      font=dict(family="Courier New, monospace",
                                size=12,
                                color="RebeccaPurple"))
    fig.show()
    return fig


def model_loop(ts_df,
               zipcode_list,
               train_size=.8, show_grid_search_steps=True,
               forecast_steps=36, figsize=(15, 5),
               display_details=False):
    """
    Loops through provided zipcodes as list with grid_search function and 
    stores output using provided train test split.

    - saves the model as it loops to `./model/ind_model/{zipcode}.joblib`
    - after looping saves results at `./model/all_models_output.joblib`
    - after looping saves roi information at `./model/roi.joblib`

    Parameters:
    ===========
    ts_df = pandas.DataFrame, all zipcode inforamation
    zipcode_list = list, zipcodes to loop
    train_size=.8, 
    show_grid_search_steps = bool; default = True, display gid search steps,
    display_details = bool; default = False, show forecast breakdown.
    forecast_steps = int; default = 36, steps to forecast into future. 
    figsize = tuple of int or float; deafult = (15, 8), figure size control,

    Returns: 
    ========
    Result dict, and ROI dataframe
    """
    # store results
    RESULTS = {}
    # store ROI information
    ROI = pd.DataFrame(columns=[
        'zipcode', 'mean_forecasted_roi', 'lower_forecasted_roi',
        'upper_forecasted_roi', 'std_forecasted_roi'
    ])

    # loop counter
    n = 0
    for zipcode in zipcode_list:
        MODEL_RESULTS = {}
        # loop counter
        n = n + 1
        len_ = len(zipcode_list)
        print(f"""Working on #{n} out of {len_} zipcodes.""")
        print('Working on:', zipcode)
        # make empty dicts for storing data
        temp_dict = {}
        # temp_dict_1 = {}

        # make a copy of time series data
        ts = ts_df[zipcode].dropna().copy()
        # train-test split
        train_size = train_size
        split_idx = round(len(ts) * train_size)
        # split
        train = ts.iloc[:split_idx]
        test = ts.iloc[split_idx:]

        # Get best params using auto_arima on train-test data
        if display_details:
            display_results_gs = True
            display_roi_results_gs = True
        else:
            display_results_gs = False
            display_roi_results_gs = False

        model, pred_df_test, pred_df = grid_search(ts,
                                                   train,
                                                   test,
                                                   forecast_steps=forecast_steps,
                                                   figsize=figsize,
                                                   trace=show_grid_search_steps,
                                                   display_results=display_results_gs,
                                                   display_roi_results=display_roi_results_gs)

        # storing data in RESULTS
        ORDER = {"order": model.order, "seasonal_order": model.seasonal_order}
        # temp_dict['model'] = model
        temp_dict['train'] = train
        temp_dict['test'] = test
        temp_dict['pred_df_test'] = pred_df_test
        temp_dict['pred_df'] = pred_df
        temp_dict['orders'] = ORDER

        MODEL_RESULTS[zipcode] = {'model': model}
        RESULTS[zipcode] = temp_dict
        # saving model
        joblib.dump(MODEL_RESULTS,
                    f'./model/ind_model/{zipcode}.joblib', compress=9)

        # storing data in ROI
        mean_roi = (pred_df[-1:]['prediction'][0] - test[-1:][0])/test[-1:][0]
        lower_roi = (pred_df[-1:]['lower'][0] - test[-1:][0])/test[-1:][0]
        upper_roi = (pred_df[-1:]['upper'][0] - test[-1:][0])/test[-1:][0]
        std_roi = np.std([lower_roi, upper_roi])

        roi_df = pd.DataFrame([{
            'zipcode': zipcode,
            'mean_forecasted_roi': mean_roi,
            'lower_forecasted_roi': lower_roi,
            'upper_forecasted_roi': upper_roi,
            'std_forecasted_roi': std_roi
        }])
        ROI = ROI.append(roi_df, ignore_index=True)
        print('-' * 90, end='\n')
    # saving results and ROI
    joblib.dump(RESULTS, f'./model/all_models_output.joblib')
    joblib.dump(ROI, './model/roi.joblib')
    print('Looping completed.')
    return RESULTS, ROI


def model_report(zipcode_list, results_, show_model_performance=True, show_train_fit=True, show_prediction=True, show_detailed_prediction=True, test_conf_int=True):
    """
    ### predefined funtion ###
    For visualizing reports by using results from model loop.


    #############################  OUTPUT CONTROL  ##############################
    #############################################################################
    show_model_performance = True  # Performance metrics & Diagonistics plots
    show_train_fit = True          # test and prediction fit
    show_prediction = True         # forecast in the future
    #---------------------------------------------------------------------------#
    show_detailed_prediction = True  # forecast in the future, whith prediction #
    #terminology: ###  prediction is to judge model performance   ###############
    ###################################   &   ###################################
    ################   forecast is prediction of unknown  #######################
    #############################################################################
    """
    for best_zipcode in zipcode_list:
        # display models
        print(f'{"-"*157}')
        print('\033[1m \033[5;30;47m' +
              f'{" "*70}Report of {best_zipcode}{" "*70}' + '\033[0m')
        print(f'{"-"*157}')
        print('\033[1m \033[91m' + 'Model Used:' + '\033[0m')

        if show_model_performance:
            # model = joblib.load(f'./model/ind_model/{best_zipcode}.joblib')
            # model = model[best_zipcode]['model']
            mergerd_train_test = results_[best_zipcode]['train'].combine_first(
                results_[best_zipcode]['test'])
            order = results_[best_zipcode]['orders']['order']
            seasonal_order = results_[best_zipcode]['orders']['seasonal_order']

            model = tsa.SARIMAX(mergerd_train_test, order=order,
                                seasonal_order=seasonal_order, enforce_invertibility=False).fit()
            # model performance
            print('order used: ', order)
            print('seasonal order used: ', seasonal_order)
            print(model.summary())
            model.plot_diagnostics(figsize=(10, 5))
            plt.tight_layout()
            plt.show()
        # extracting information from results dict
        train = results_[best_zipcode]['train']
        test = results_[best_zipcode]['test']
        pred_df_test = results_[best_zipcode]['pred_df_test']
        pred_df = results_[best_zipcode]['pred_df']
        if show_train_fit:
            print('\033[1m \033[1;33;40m' + 'Prediction:')
            print('\033[0m')
            # plot train fit
            _, ax = plot_test_pred(test, pred_df_test, conf_int=test_conf_int)
            ax.set_title(f'test-pred plot of {best_zipcode} [prediction reliability]',
                         size=15)
            plt.show()
        if show_prediction:
            # plot_train_test_pred
            _, ax = plot_train_test_pred(train, test, pred_df)
            ax.set_title(f'train-test-pred plot of {best_zipcode} [forecast]',
                         size=15)
            plt.show()
        if show_detailed_prediction:
            print('\033[1m \033[1;33;40m' + 'Insights:'+'\033[1m')
            # train_test_pred_forecast
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
            ax1 = tsa.seasonal_decompose(results_[best_zipcode]['test']).trend.plot(
                title='Most recent trend', ax=ax1)
            ax2 = tsa.seasonal_decompose(
                results_[best_zipcode]['test'][-36:]).seasonal.plot(
                    title='Last three year seasonality', ax=ax2)
            plt.show()
            print('\033[1m \033[1;33;40m' +
                  'Overall model performance and projected ROI:')
            _, ax, _ = plot_train_test_pred_forecast(train, test, pred_df_test,
                                                     pred_df, best_zipcode)
            ax.set_title(f'train-test-pred-forecast plot of {best_zipcode}',
                         size=15)
        plt.show()
    pass


def output_df(zipcode_list, results_):
    """ 
    # data processing function #
    creats a dataframe for decision making.
    """
    def roi(end, beg):
        """Return on investment calculation"""
        x = (end-beg)/beg
        x = (x*100).round(2)
        return x

    def relative_standard_deviation(lower, upper):
        """
        Standard deviation expressed in percent and is obtained by multiplying the standard 
        deviation by 100 and dividing this product by the average.
        _________________________________________________________________________
        Reference: https://www.chem.tamu.edu/class/fyp/keeney/stddev.pdf
        """
        x = abs(upper-lower)
        y = (np.std(x)/np.mean(x))*100
        return y

    output = {}
    for item in zipcode_list:
        loaded = joblib.load(f'./model/ind_model/{item}.joblib')

        model = loaded[item]['model']
        # print(model)
        temp_dict = {}
        temp_dict['aic'] = model.aic().round(2)
        temp_dict['bic'] = model.bic().round(2)
        temp_dict['oob'] = model.oob().round(2)
        rmse_o, mse_o, mae_o = model_error_report(results_[item]['test'], results_[
                                                  item]['pred_df_test'], show_report=False)
        temp_dict['rmse'] = rmse_o.round()
        temp_dict['mse'] = mse_o.round()
        temp_dict['mse'] = mae_o.round()
        temp_dict['r2'] = r2_score(results_[item]['test'], results_[
                                   item]['pred_df_test']['prediction']).round(3)
        temp_dict['test_roi'] = roi(results_[
                                    item]['test'][-1], results_[item]['test'][-len(results_[item]['pred_df'])])
        temp_dict['pred_roi'] = roi(results_[
                                    item]['pred_df']['prediction'][-1], results_[item]['pred_df']['prediction'][0])
        temp_dict['three_year_projected_mean_roi'] = roi(
            results_[item]['pred_df'][-1:]['prediction'][0], results_[item]['test'][-1:][0])
        temp_dict['risk'] = round(relative_standard_deviation(
            results_[item]['pred_df']['lower'], results_[item]['pred_df']['upper']), 2)
        temp_dict['three_year_projected_lower_roi'] = roi(
            results_[item]['pred_df'][-1:]['lower'][0], results_[item]['test'][-1:][0])
        temp_dict['three_year_projected_upper_roi'] = roi(
            results_[item]['pred_df'][-1:]['upper'][0], results_[item]['test'][-1:][0])
        output[item] = temp_dict
    out = pd.DataFrame(output).T
    out.index.name = "ZipCode"
    return out
