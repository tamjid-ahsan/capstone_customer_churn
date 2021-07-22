import dash
import dash_core_components as dcc
import dash_html_components as html
import joblib
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
# from ../imports_and_functions.packages import *
from imports_and_functions.functions import *
import numpy as np
import pandas as pd
# import ..imports_and_functions as fn
import seaborn as sns
import matplotlib.pyplot as plt

print(f'{"+"*30}')
print('press `CTRL+C` to exit')
print(f'{"+"*30}')

# Imports
# -------------------------------------------------------------------------
data_scaled = joblib.load('./model/scaled_data.joblib')

data = joblib.load('./model/unscaled_data.joblib')
data_cluster = data.groupby('Clusters').count()['target'].sort_index(
    ascending=False).reset_index()
preprocessor = joblib.load('./model/preprocessor.joblib')
nume_col = joblib.load('./model/nume_col.joblib')
cate_col = joblib.load('./model/cate_col.joblib')
prediction_model = joblib.load(
    './model/xgb_clf_churn_prediction_all_data.joblib')
kmeans_model = joblib.load('./model/kmeans_segmentation_model.joblib')

# functions
# -------------------------------------------------------------------------


def data_cluster_fig():
    fig = px.bar(y='target',
                 x='Clusters',
                 data_frame=data_cluster.astype('str'),
                 template='plotly_dark',
                 color='Clusters',
                 title='Cluster Size',
                 text=data_cluster['target'])
    fig.update_xaxes(showline=True,
                     linewidth=2,
                     linecolor='black',
                     mirror=True)
    return fig


def pca_fig(cluster_df):
    from sklearn.decomposition import PCA
    cluster_df_mod = cluster_df.drop(columns=['Clusters', 'target'])
    pca = PCA(n_components=3)
    pc_feature_names = [f"PC{x}" for x in range(1, pca.n_components + 1)]
    pca_data = pca.fit_transform(cluster_df_mod)
    pca_df = pd.DataFrame(pca_data, columns=pc_feature_names)
    pca_df['Clusters'] = cluster_df['Clusters'].astype('str')
    fig = px.scatter_3d(pca_df,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color='Clusters',
                        title='Cluster visualization with the help of PCA',
                        template='plotly_dark')
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(width=700, height=500, bargap=0.05)
    return fig


# APP
# -------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], meta_tags=[{'name': 'viewport',
                                                                               'content': 'width=device-width, initial-scale=1.0'}])
# Layout
# -------------------------------------------------------------------------
app.layout = dbc.Container([
    # row 1
    dbc.Row(
        # row 1 column 1
        dbc.Col(html.H1(
            'Consolidated Segmentation and Churn Analysis of Bank Clients Report')
        )
    ),
    # row 1 end
    dbc.Row([
        # row 1 column 1
        dbc.Col([
            dcc.Graph(id='cluster_dist_viz', figure=data_cluster_fig())
        ], width={'size': 7}),
        dbc.Col([
            dcc.Graph(id='cluster_pca_viz', figure=pca_fig(data_scaled))
        ], width={'size': 5}),
    ]),
    # row 3
    dbc.Row(dbc.Col(html.H2(
            'Intracluster analysis')
    )),
    dbc.Row(
        dbc.Col([
            dcc.Dropdown(
                id='my-dpdn', className='bg-dark ', multi=False, value='Customer_Age', options=[{'label': x, 'value': x} for x in data.columns], searchable=True),
            dcc.Graph(
                id='intracluster_viz', figure={})
        ], width={'size': 12})
    ),
    dbc.Row(dbc.Col(html.H2(
            'All the features with churning')
    )),
    dbc.Row(
        dbc.Col([
            dcc.Dropdown(
                id='my-dpdn-1', className='bg-dark ', multi=False, value='Customer_Age', options=[{'label': x, 'value': x} for x in data.columns], searchable=True),
            dcc.Graph(
                id='intracluster_viz_1', figure={})
        ], width={'size': 12})

    ),
    dbc.Row(dbc.Col(html.H2('Prediction Model'))),
    dbc.Row(
        dbc.Col([
            html.I("Please Input Features: [press ENTER after inputting]"),
            html.Br(),
            dcc.Dropdown(id='Gender',
                         placeholder='Input Gender',
                         options=[{
                             'label': x,
                             'value': x
                         } for x in data.Gender.unique()],
                         style={'marginRight': '10px'}),
            dcc.Dropdown(id='Dependent_count',
                         placeholder='Input Dependent count',
                         options=[{
                             'label': x,
                             'value': x
                         } for x in data.Dependent_count.unique()],
                         style={'marginRight': '10px'}),
            dcc.Dropdown(id='Education_Level',
                         placeholder='Input Education Level',
                         options=[{
                             'label': x,
                             'value': x
                         } for x in data.Education_Level.unique()],
                         style={'marginRight': '10px'}),
            dcc.Dropdown(id='Marital_Status',
                         placeholder='Input Marital Status',
                         options=[{
                             'label': x,
                             'value': x
                         } for x in data.Marital_Status.unique()],
                         style={'marginRight': '10px'}),
            dcc.Dropdown(id='Income_Category',
                         placeholder='Input Income Category',
                         options=[{
                             'label': x,
                             'value': x
                         } for x in data.Income_Category.unique()],
                         style={'marginRight': '10px'}),
            dcc.Dropdown(id='Card_Category',
                         placeholder='Input Card Category',
                         options=[{
                             'label': x,
                             'value': x
                         } for x in data.Card_Category.unique()],
                         style={'marginRight': '10px'}),
            dcc.Input(
                id="Months_on_book",
                type="number",
                style={'marginRight': '10px'},
                placeholder=f"Months on book max {data.Months_on_book.max()}",
                debounce=True,
                max=data.Months_on_book.max()),
            dcc.Input(id="Customer_Age",
                      type="number",
                      placeholder="Input Age",
                      style={'marginRight': '10px'}),
            dcc.Input(id="Total_Relationship_Count",
                      type="number",
                      placeholder="Total_Relationship_Count",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Months_Inactive_12_mon",
                      type="number",
                      placeholder="Months_Inactive_12_mon",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Contacts_Count_12_mon",
                      type="number",
                      placeholder="Contacts_Count_12_mon",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Credit_Limit",
                      type="number",
                      placeholder="Credit Limit",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Total_Revolving_Bal",
                      type="number",
                      placeholder="Total_Revolving_Bal",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Avg_Open_To_Buy",
                      type="number",
                      placeholder="Avg_Open_To_Buy",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Total_Amt_Chng_Q4_Q1",
                      type="number",
                      placeholder="Total_Amt_Chng_Q4_Q1",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Total_Trans_Amt",
                      type="number",
                      placeholder="Total_Trans_Amt",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Total_Trans_Ct",
                      type="number",
                      placeholder="Total_Trans_Ct",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Total_Ct_Chng_Q4_Q1",
                      type="number",
                      placeholder="Total_Ct_Chng_Q4_Q1",
                      debounce=True,
                      style={'marginRight': '10px'}),
            dcc.Input(id="Avg_Utilization_Ratio",
                      type="number",
                      placeholder="Avg_Utilization_Ratio",
                      debounce=True,
                      style={'marginRight': '10px'}),
            html.Br(),
            html.Br(),
            html.I('Prediction: '),
            html.Div(id="output")
        ])),
    dbc.Row(
        dbc.Col([html.Br(),
                 html.Br(),
                 html.H6(dcc.Markdown("""BY Tamjid Ahsan. Contact: 
    [LinkedIn](https://www.linkedin.com/in/tamjidahsan/), 
    [GitHub](https://github.com/tamjid-ahsan/capstone_customer_churn)"""))])),
], fluid=True)


# callbacks
# -------------------------------------------------------------------------
@app.callback(
    Output("output", "children"),
    Input('Gender', 'value'),
    Input('Dependent_count', 'value'),
    Input('Education_Level', 'value'),
    Input('Marital_Status', 'value'),
    Input('Income_Category', 'value'),
    Input('Card_Category', 'value'),
    Input("Months_on_book", 'value'),
    Input("Customer_Age", 'value'),
    Input("Total_Relationship_Count", 'value'),
    Input("Months_Inactive_12_mon", 'value'),
    Input("Contacts_Count_12_mon", 'value'),
    Input("Credit_Limit", 'value'),
    Input("Total_Revolving_Bal", 'value'),
    Input("Avg_Open_To_Buy", 'value'),
    Input("Total_Amt_Chng_Q4_Q1", 'value'),
    Input("Total_Trans_Amt", 'value'),
    Input("Total_Trans_Ct", 'value'),
    Input("Total_Ct_Chng_Q4_Q1", 'value'),
    Input("Avg_Utilization_Ratio", 'value'),
)
def update_output(Gender, Dependent_count, Education_Level, Marital_Status,
                  Income_Category, Card_Category, Months_on_book, Customer_Age,
                  Total_Relationship_Count, Months_Inactive_12_mon,
                  Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal,
                  Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt,
                  Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio):
    new_data = pd.DataFrame([{
        'Gender': Gender,
        'Dependent_count': Dependent_count,
        'Education_Level': Education_Level,
        'Marital_Status': Marital_Status,
        'Income_Category': Income_Category,
        'Card_Category': Card_Category,
        'Months_on_book': Months_on_book,
        'Customer_Age': Customer_Age,
        'Total_Relationship_Count': Total_Relationship_Count,
        'Months_Inactive_12_mon': Months_Inactive_12_mon,
        'Contacts_Count_12_mon': Contacts_Count_12_mon,
        'Credit_Limit': Credit_Limit,
        'Total_Revolving_Bal': Total_Revolving_Bal,
        'Avg_Open_To_Buy': Avg_Open_To_Buy,
        'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1,
        'Total_Trans_Amt': Total_Trans_Amt,
        'Total_Trans_Ct': Total_Trans_Ct,
        'Total_Ct_Chng_Q4_Q1': Total_Ct_Chng_Q4_Q1,
        'Avg_Utilization_Ratio': Avg_Utilization_Ratio
    }])
    new_data = new_data[[
        'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
        'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
    ]]
    try:
        processed_x = unseen_data_processor(new_data, preprocessor, nume_col,
                                            cate_col)
        processed_x['Clusters'] = kmeans_model.predict(processed_x)
        prediction = prediction_model.predict(processed_x)
        if processed_x['Clusters'][0] == 0:
            identified_cluster = 'Low value frequent user'
        elif processed_x['Clusters'][0] == 1:
            identified_cluster = 'High risk client'
        elif processed_x['Clusters'][0] == 2:
            identified_cluster = 'Regular client'
        elif processed_x['Clusters'][0] == 3:
            identified_cluster = 'Most loyal client'
        elif processed_x['Clusters'][0] == 4:
            identified_cluster = 'High value clients'

        if prediction == 0:
            prediction_label = 'Continu'
        if prediction == 1:
            prediction_label = 'Churn'
        return f'Client is identified as "{identified_cluster}" with a prediction of {prediction_label}ing.'
    except:
        return f'Please check input'


def feature_analysis_intracluster(
        x,
        facet_col,
        n_clusters, data_frame=None,
        title=None,
        nbins=None,
        marginal='box',
        histnorm='probability density',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template='presentation'):
    """produces plots for use in analysis intracluster
    Parameters follows conventional plotly express histogram options.
    """
    if title is None:
        if data_frame is None:
            title = f'{x.name.replace("_"," ")}'
        else:
            title = f'{data_frame[x].name.replace("_"," ")}'
    fig = px.histogram(
        data_frame=data_frame,
        x=x,
        facet_col=facet_col,
        marginal=marginal,
        histnorm=histnorm,
        nbins=nbins,
        color_discrete_sequence=color_discrete_sequence,
        template=template,
        title=f'{title} {histnorm}',
        facet_col_spacing=0.005,
        category_orders={'Clusters': list(np.arange(0, n_clusters))})
    fig.update_xaxes(showline=True, tickfont={'size': 8},
                     linewidth=1, title_font_size=2,
                     linecolor=color_discrete_sequence[0],
                     mirror=True,
                     title={'text': ''})
    fig.update_yaxes(showline=True, tickfont={'size': 8},
                     linewidth=1,
                     linecolor=color_discrete_sequence[0],
                     mirror=True)

    fig.update_yaxes(title={'font': {'size': 8}, 'text': ''})
    fig.for_each_annotation(
        lambda a: a.update(text=f'Cluster: {a.text.split("=")[1]}'))
    return fig


@app.callback(Output('intracluster_viz', 'figure'), Input('my-dpdn', 'value'))
def intracluster_viz(x):
    nbins = None
    color_discrete_sequence = px.colors.qualitative.Pastel
    if x == 'Customer_Age':
        nbins = 10
    if x == 'Credit_Limit':
        nbins = 25
        color_discrete_sequence = px.colors.qualitative.Dark2

    fig_ = feature_analysis_intracluster(x, facet_col='Clusters', n_clusters=5, data_frame=data, title=None, nbins=nbins,
                                         marginal='box', histnorm='probability density', color_discrete_sequence=color_discrete_sequence, template='presentation')

    if x == 'Customer_Age':
        fig_.update_xaxes(tickmode='linear', tick0=20, dtick=10)
    return fig_


@app.callback(Output('intracluster_viz_1', 'figure'), Input('my-dpdn-1', 'value'))
def intracluster_viz_2(x):
    fig = px.histogram(
        data_frame=data,
        x=x,
        marginal="box",
        template='presentation',
        color='target',
        facet_col='Clusters',
        color_discrete_sequence=px.colors.qualitative.Dark2,
        barmode='group',
        category_orders={'Clusters': list(
            np.arange(0, len(data.Clusters.unique())+1))},
        title=f'"{x.replace("_"," ")}" seperated by Clusters')
    fig.update_xaxes(showline=True, tickfont={'size': 8},
                     linewidth=1,
                     linecolor='black', title={'text': ''})
    fig.update_yaxes(tickfont={'size': 8}, title={
                     'font': {'size': 8}, 'text': ''})
    fig.for_each_annotation(
        lambda a: a.update(text=f'Cluster: {a.text.split("=")[1]}'))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
