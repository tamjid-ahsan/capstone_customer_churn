import dash
import dash_core_components as dcc
import dash_html_components as html
import joblib
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
# from ../imports_and_functions.packages import *
from ..imports_and_functions.functions import *
import pandas as pd
# import ..imports_and_functions as fn


print(f'{"+"*30}')
print('press `CTRL+C` to exit')
print(f'{"+"*30}')

data = joblib.load('unscaled_data.joblib')
print(data)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], meta_tags=[{'name': 'viewport',
                                                                               'content': 'width=device-width, initial-scale=1.0'}])
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1(
            'Title')),
        dbc.Col([dcc.Dropdown(id='my-dpdn', className='bg-dark ', multi=False, value='Customer_Age', options=[{'label': x, 'value': x} for x in data.columns]),
            dcc.Graph(id='intracluster_viz', fig={})])
    )    
])

@app.callback(Output('line-fig', 'figure'), Input('my-dpdn', 'value'))
def intracluster_viz(x):
    fig = feature_analysis_intracluster(x, facet_col='Clusters', n_clusters=5, data_frame=data, title=None, nbins=None, marginal='box', histnorm='probability density', color_discrete_sequence=px.colors.qualitative.Pastel, template='presentation')
    fig.show()
# app.layout = dbc.Container([
#     dbc.Row(
#         dbc.Col(html.H1(
#             "House ROI after three years in Queens NY", className='text-center')
#         )
#     ),

#     dbc.Row(
#         dbc.Col(
#             dcc.Graph(
#                 figure=fig1
#             ))),
#     dbc.Row(
#         dbc.Col(html.H4(
#             "House Price in Queens NY, choose zipcode for forecast of three years", className='text-center')
#         )
#     ),

#     dbc.Row(
#         dbc.Col([
#             dcc.Dropdown(id='my-dpdn', className='bg-dark ', multi=False, value='11432',
#                          options=[{'label': x, 'value': x}
#                                   for x in sorted(results.keys())]
#                          ),
#             dcc.Graph(id='line-fig', figure={})
#         ])
#     )
# ])


# @app.callback(Output('line-fig', 'figure'), Input('my-dpdn', 'value'))



if __name__ == '__main__':
    app.run_server(debug=True)


# app.layout = dbc.Container([
#     dbc.Row(
#         dbc.Col(html.H1(
#             "House ROI after three years in Queens NY", className='text-center')
#         )
#     ),

#     dbc.Row(
#         dbc.Col(
#             dcc.Graph(
#                 figure=fig1
#             ))),
#     dbc.Row(
#         dbc.Col(html.H4(
#             "House Price in Queens NY, choose zipcode for forecast of three years", className='text-center')
#         )
#     ),

#     dbc.Row(
#         dbc.Col([
#             dcc.Dropdown(id='my-dpdn', className='bg-dark ', multi=False, value='11432',
#                          options=[{'label': x, 'value': x}
#                                   for x in sorted(results.keys())]
#                          ),
#             dcc.Graph(id='line-fig', figure={})
#         ])
#     )
# ])


# @app.callback(Output('line-fig', 'figure'), Input('my-dpdn', 'value'))
# def fig_ret(code):
#     pred = results[code]['pred_df']['prediction']
#     low = results[code]['pred_df']['lower']
#     high = results[code]['pred_df']['upper']
#     mergerd_train_test = results[code]['train'].combine_first(
#         results[code]['test'])

#     fig = go.Figure()

#     fig.add_trace(
#         go.Scatter(y=mergerd_train_test,
#                    x=mergerd_train_test.index,
#                    mode='lines+markers',
#                    line_color='#537d8d',
#                    name='TS'))

#     fig.add_trace(
#         go.Scatter(y=low,
#                    x=low.index,
#                    mode='lines',
#                    name='Lower CI',
#                    line_color='#22c256'))
#     fig.add_trace(
#         go.Scatter(
#             y=high,
#             x=high.index,
#             fill='tonexty',
#             mode='lines',
#             line_color='#22c256',
#             opacity=.5,
#             name='high CI',
#         ))
#     fig.add_trace(
#         go.Scatter(y=pred,
#                    x=pred.index,
#                    mode='lines+markers',
#                    name='Forecast',
#                    line_color='#ff6961'))
#     fig.update_layout(title=f"Time series plot with forecast of {code}",
#                       xaxis_title="Years",
#                       yaxis_title="Home values",
#                       legend_title="Legends",
#                       font=dict(family="Courier New, monospace",
#                                 size=12,
#                                 color="RebeccaPurple"))
#     return fig

# fig1 = joblib.load('viz.joblib')


# # load = joblib.load('model.joblib')
# # results = load['Results']
# results = joblib.load('all_models_output.joblib')