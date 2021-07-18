# imports
import matplotlib.pyplot as plt
from sklearn import metrics
from IPython.display import display, HTML, Markdown
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from yellowbrick.classifier.rocauc import roc_auc
import seaborn as sns
import numpy as np
import plotly.express as px

# functions


def model_report(model,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 show_train_report=True,
                 show_test_report=True,
                 fitted_model=False,
                 cmap=['cool', 'copper_r'],
                 normalize='true',
                 figsize=(15, 5)):
    """
    Dispalys model report.


    """
    if fitted_model is False:
        model.fit(X_train, y_train)
    train = model.score(X_train, y_train)
    test = model.score(X_test, y_test)

    def str_model_(model):
        """Helper function to get model class display statement, this text conversion breaks code if 
        performed in ``model_report`` function's local space. This function is to isolate from the 
        previous function's local space."""
        str_model = str(model.__class__).split('.')[-1][:-2]
        display(
            HTML(
                f"""<strong>Report of {str_model} type model using train-test split dataset.</strong>"""
            ))

    str_model_(model)
    print(f"{'*'*90}")
    print(f"""Train accuracy score: {train.round(4)}""")
    print(f"""Test accuracy score: {test.round(4)}""")
    if abs(train - test) <= .05:
        print(
            f"    No over or underfitting detected, diffrence of scores did not cross 5% thresh hold."
        )
    elif (train - test) > .05:
        print(
            f"    Possible Overfitting, diffrence of scores {round(abs(train-test)*100,2)}% crossed 5% thresh hold."
        )
    elif (train - test) < -.05:
        print(
            f"    Possible Underfitting, diffrence of scores {round(abs(train-test)*100,2)}% crossed 5% thresh hold."
        )
    print(f"{'*'*90}")
    print("")

    if show_train_report:
        print(f'Train Report: ')
        print(f"{'*'*60}")
        # train report
        # classification report
        print(
            metrics.classification_report(y_train,
                                          model.predict(X_train)))
        print(f"{'*'*60}")
        # Confusion matrix
        fig, ax = plt.subplots(ncols=2, figsize=figsize)
        metrics.plot_confusion_matrix(model,
                                      X_train,
                                      y_train,
                                      cmap='cool',
                                      normalize='true',
                                      ax=ax[0])
        ax[0].title.set_text('Confusion Matrix')
        # ROC curve
        metrics.plot_roc_curve(model,
                               X_train,
                               y_train,
                               color='#0450E7',
                               ax=ax[1])
        ax[1].plot([0, 1], [0, 1], ls='-.', color='white')
        ax[1].title.set_text('ROC Curve')
        plt.grid()
        plt.tight_layout()
        plt.show()

    if show_test_report:
        # train report
        # classification report
        print(f'Test Report: ')
        print(f"{'*'*60}")
        print(metrics.classification_report(y_test,
                                            model.predict(X_test)))
        print(f"{'*'*60}")
        # Confusion matrix
        fig, ax = plt.subplots(ncols=2, figsize=figsize)
        metrics.plot_confusion_matrix(model,
                                      X_test,
                                      y_test,
                                      cmap='copper_r',
                                      normalize='true',
                                      ax=ax[0])
        ax[0].title.set_text('Confusion Matrix')
        # ROC curve
        metrics.plot_roc_curve(model,
                               X_test,
                               y_test,
                               color='gold',
                               ax=ax[1])
        ax[1].plot([0, 1], [0, 1], ls='-.', color='white')
        ax[1].title.set_text('ROC Curve')
        plt.grid()
        plt.tight_layout()
        plt.show()
    pass


def dataset_processor_segmentation(X, OHE_drop_option=None, verbose=0, scaler=None):
    """

    """
    # isolating numerical cols
    nume_col = list(X.select_dtypes('number').columns)
    if verbose > 0:
        print("Numerical columns: \n---------------------\n", nume_col)

    # isolating categorical cols
    cate_col = list(X.select_dtypes('object').columns)
    if verbose > 0:
        print('')
        print("Categorical columns: \n---------------------\n", cate_col)

    # pipeline for processing categorical features
    pipe_cate = Pipeline([('ohe',
                           OneHotEncoder(sparse=False, drop=OHE_drop_option))])
    # pipeline for processing numerical features
    if scaler is None:
        scaler = StandardScaler()
    pipe_nume = Pipeline([('scaler', scaler)])
    # transformer
    preprocessor = ColumnTransformer([('nume_feat', pipe_nume, nume_col),
                                      ('cate_feat', pipe_cate, cate_col)])
    # creating dataframes
    try:
        X_pr = pd.DataFrame(
            preprocessor.fit_transform(X),
            columns=nume_col +
            list(preprocessor.named_transformers_['cate_feat'].
                 named_steps['ohe'].get_feature_names(cate_col)))
        if verbose > 1:
            print("\n\n------")
            print(
                f"Scaler: {str(preprocessor.named_transformers_['nume_feat'].named_steps['scaler'].__class__)[1:-2].split('.')[-1]}, settings: {preprocessor.named_transformers_['nume_feat'].named_steps['scaler'].get_params()}"
            )
            print(
                f"Encoder: {str(preprocessor.named_transformers_['cate_feat'].named_steps['ohe'].__class__)[1:-2].split('.')[-1]}, settings: {preprocessor.named_transformers_['cate_feat'].named_steps['ohe'].get_params()}"
            )
            print("------")
    except:
        if verbose > 1:
            print("\n\n------")
            print(
                f"Scaler: {str(preprocessor.named_transformers_['nume_feat'].named_steps['scaler'].__class__)[1:-2].split('.')[-1]}, settings: {preprocessor.named_transformers_['nume_feat'].named_steps['scaler'].get_params()}"
            )
            print(
                f"Encoder: {str(preprocessor.named_transformers_['cate_feat'].named_steps['ohe'].__class__)[1:-2].split('.')[-1]}, settings: {preprocessor.named_transformers_['cate_feat'].named_steps['ohe'].get_params()}"
            )
            print("------")
            print("No Categorical columns found")
        X_pr = pd.DataFrame(preprocessor.fit_transform(X), columns=nume_col)
    return X_pr


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
{f.read()}
```"""
        display(Markdown(x))


def model_report_multiclass(model,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            show_train_report=True,
                            show_test_report=True,
                            fitted_model=False,
                            cmap=['cool', 'copper_r'],
                            normalize='true',
                            figsize=(15, 5)):
    """
    Dispalys model report.
    """
    if fitted_model is False:
        model.fit(X_train, y_train)
    train = model.score(X_train, y_train)
    test = model.score(X_test, y_test)

    def str_model_(model):
        """Helper function to get model class display statement, this text conversion breaks code if 
        performed in ``model_report`` function's local space. This function is to isolate from the 
        previous function's local space."""
        str_model = str(model.__class__).split('.')[-1][:-2]
        display(
            HTML(
                f"""<strong>Report of {str_model} type model using train-test split dataset.</strong>"""
            ))

    str_model_(model)
    print(f"{'*'*90}")
    print(f"""Train accuracy score: {train.round(4)}""")
    print(f"""Test accuracy score: {test.round(4)}""")
    if abs(train - test) <= .05:
        print(
            f"    No over or underfitting detected, diffrence of scores did not cross 5% thresh hold."
        )
    elif (train - test) > .05:
        print(
            f"    Possible Overfitting, diffrence of scores {round(abs(train-test)*100,2)}% crossed 5% thresh hold."
        )
    elif (train - test) < -.05:
        print(
            f"    Possible Underfitting, diffrence of scores {round(abs(train-test)*100,2)}% crossed 5% thresh hold."
        )
    print(f"{'*'*90}")
    print("")

    if show_train_report:
        print(f'Train Report: ')
        print(f"{'*'*60}")
        # train report
        # classification report
        print(
            metrics.classification_report(y_train,
                                          model.predict(X_train)))
        print(f"{'*'*60}")
        # Confusion matrix
        fig, ax = plt.subplots(ncols=2, figsize=figsize)
        metrics.plot_confusion_matrix(model,
                                      X_train,
                                      y_train,
                                      cmap='cool',
                                      normalize='true',
                                      ax=ax[0])
        ax[0].title.set_text('Confusion Matrix')
        # ROC curve
        _ = roc_auc(model,
                    X_train,
                    y_train,
                    classes=None,
                    is_fitted=True,
                    show=False,
                    ax=ax[1])

        ax[1].grid()
        ax[1].title.set_text('ROC Curve')
        plt.xlim([-.05, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.show()

    if show_test_report:
        # train report
        # classification report
        print(f'Test Report: ')
        print(f"{'*'*60}")
        print(metrics.classification_report(y_test,
                                            model.predict(X_test)))
        print(f"{'*'*60}")
        # Confusion matrix
        fig, ax = plt.subplots(ncols=2, figsize=figsize)
        metrics.plot_confusion_matrix(model,
                                      X_test,
                                      y_test,
                                      cmap='copper_r',
                                      normalize='true',
                                      ax=ax[0])
        ax[0].title.set_text('Confusion Matrix')
        # ROC curve
        _ = roc_auc(model,
                    X_test,
                    y_test,
                    classes=None,
                    is_fitted=True,
                    show=False,
                    ax=ax[1])
        plt.xlim([-.05, 1])
        plt.ylim([0, 1.05])
        ax[1].grid()
        ax[1].title.set_text('ROC Curve')
        plt.tight_layout()
        plt.show()
    pass


def plot_distribution(df,
                      color='gold',
                      figsize=(16, 26),
                      fig_col=3,
                      labelrotation=45,
                      plot_title='Histogram plots of the dataset'):
    def num_col_for_plotting(row, col=fig_col):
        """
        +++ formatting helper function +++
        __________________________________
        Returns number of rows to plot

        Parameters:
        ===========
        row = int;
        col = int; default col: 3
        """
        if row % col != 0:
            return (row // col) + 1
        else:
            return row // col

    fig, axes = plt.subplots(nrows=num_col_for_plotting(len(df.columns),
                                                        col=fig_col),
                             ncols=fig_col,
                             figsize=figsize,
                             sharey=False)
    for ax, column in zip(axes.flatten(), df):
        sns.histplot(x=column, data=df, color=color, ax=ax, kde=True)
        ax.set_title(f'Histplot of {column.title()}')
        ax.tick_params('x', labelrotation=labelrotation)
        sns.despine()
        plt.tight_layout()
        plt.suptitle(plot_title, fontsize=20, fontweight=3, va='bottom')
    plt.show()
    pass


def heatmap_of_features(df, figsize=(15, 15), annot_format='.1f'):
    """
    Return a masked heatmap of the given DataFrame

    Parameters:
    ===========
    df            = pandas.DataFrame object.
    annot_format  = str, for formatting; default: '.1f'

    Example of `annot_format`:
    --------------------------
    .1e = scientific notation with 1 decimal point (standard form)
    .2f = 2 decimal places
    .3g = 3 significant figures
    .4% = percentage with 4 decimal places

    Note:
    =====
    Rounding error can happen if '.1f' is used.

    -- version: 1.1 --
    """
    with plt.style.context('dark_background'):
        plt.figure(figsize=figsize, facecolor='k')
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        cmap = sns.diverging_palette(3, 3, as_cmap=True)
        ax = sns.heatmap(df.corr(),
                         mask=mask,
                         cmap=cmap,
                         annot=True,
                         fmt=annot_format,
                         linecolor='k',
                         annot_kws={"size": 9},
                         square=False,
                         linewidths=.5,
                         cbar_kws={"shrink": .5})
        plt.title(f'Features heatmap', fontdict={"size": 20})
        plt.show()
        return ax


def drop_features_based_on_correlation(df, threshold=0.75):
    """
    Returns features with high collinearity.

    Parameters:
    ===========
    df = pandas.DataFrame; no default.
            data to work on.
    threshold = float; default: .75.
            Cut off value of check of collinearity.

    -- ver: 1.0 --
    """
    # Set of all the names of correlated columns
    feature_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # absolute coeff value
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # getting the name of column
                colname = corr_matrix.columns[i]
                feature_corr.add(colname)
    return feature_corr


def cluster_insights(df, color=px.colors.qualitative.Pastel):
    """

    """
    # fig 1 Age
    financials = [
        'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
    ]
    fig = px.histogram(df,
                       x='Customer_Age',
                       marginal="box",
                       template='presentation',
                       nbins=10,
                       color='Gender',
                       barmode='group', color_discrete_sequence=color,
                       title='Customer Demographics',
                       hover_data=df)
    fig.update_traces(opacity=0.8)
    fig.update_layout(bargap=0.05)
    fig.show()
    # fig 2 Education
    fig = px.histogram(df,
                       color='Education_Level',
                       marginal="box",
                       template='presentation', color_discrete_sequence=color,
                       category_orders=dict(Income_Category=[
                           'Unknown', 'Less_than_40K', '40K_to_60K',
                           '60K_to_80K', '80K_to_120K', 'Above_120K'
                       ]),
                       title='Education Level & Income Category',
                       x='Income_Category',
                       barmode='group',
                       hover_data=df)
    # fig.update_layout(width=700, height=500, bargap=0.05)
    fig.show()
    # fig 4 dependent count
    fig = px.histogram(df,
                       x='Dependent_count',
                       marginal="box",
                       template='presentation', color_discrete_sequence=color,
                       title='Marital Status & Dependent count',
                       color='Marital_Status',
                       barmode='group',
                       hover_data=df)
    fig.update_traces(opacity=0.8)
    fig.update_layout(width=700, height=500, bargap=0.05)
    fig.show()
    # fig 5 Card category
    fig = px.bar(x='Card_Category',
                 color='Card_Category',
                 data_frame=df,
                 template='presentation',
                 title='Card Category',
                 color_discrete_sequence=["blue", "gold", "silver", "#c1beba"])
    fig.update_layout(width=700, height=500, bargap=0.05)
    fig.show()
    # fig 6
    plot_distribution(df[financials], color='silver', figsize=(
        16, 16), plot_title='Histogram of Numreical features')
    plt.show()
    pass


def describe_dataframe(df):
    """

    """
    left = df.describe(include='all').round(2).T
    right = pd.DataFrame(df.dtypes)
    right.columns = ['dtype']
    ret_df = pd.merge(left=left, right=right,
                      left_index=True, right_index=True)
    na_df = pd.DataFrame(df.isna().sum())
    na_df.columns = ['nulls']
    ret_df = pd.merge(left=ret_df, right=na_df,
                      left_index=True, right_index=True)
    ret_df.fillna('', inplace=True)
    return ret_df


def check_duplicates(df, verbose=0, limit_output=True, limit_num=150):
    """
    Checks for duplicates in the pandas DataFrame and return a Dataframe of report.

    Parameters:
    ===========
    df = pandas.DataFrame
    verbose = `int` or `boolean`; default: `False`
    limit_output = `int` or `boolean`; default: `True`
                `True` limits featurs display to 150.
                `False` details of unique features.
    limit_num = `int`, limit number of uniques; default: 150,

    Returns:
    ========
    pandas.DataFrame, if verbose = 1.

    ---version 1.3---
    """
    dup_checking = []
    for column in df.columns:
        not_duplicated = df[column].duplicated().value_counts()[0]
        try:
            duplicated = df[column].duplicated().value_counts()[1]
        except:
            duplicated = 0
        temp_dict = {
            'name': column,
            'duplicated': duplicated,
            'not_duplicated': not_duplicated
        }
        dup_checking.append(temp_dict)
    df_ = pd.DataFrame(dup_checking)

    if verbose > 0:
        if limit_output:
            for col in df:
                if (len(df[col].unique())) <= limit_num:
                    print(
                        f"{col} >> number of uniques: {len(df[col].unique())}\nValues:\n{df[col].unique()}")
                else:
                    print(
                        f"{col} >> number of uniques: {len(df[col].unique())}, showing top {limit_num} values\nTop {limit_num} Values:\n{df[col].unique()[:limit_num]}\n")
                print(f"{'_'*60}\n")
        else:
            for col in df:
                print(
                    f"{col} >> number of uniques: {len(df[col].unique())}\nValues:\n{df[col].unique()}")
    if 1 > verbose >= 0:
        return df_


def unseen_data_processor(X, preprocessor, nume_col, cate_col):
    ret_df = pd.DataFrame(preprocessor.transform(X),
                          columns=nume_col +
                          list(preprocessor.named_transformers_['cate_feat'].
                               named_steps['ohe'].get_feature_names(cate_col)))
    return ret_df


def show_px_color_options(type='qualitative'):
    if type == 'qualitative':
        display(dir(px.colors.qualitative))
    elif type == 'sequential':
        display(dir(px.colors.sequential))
    pass


def dataset_processor(X, y, train_size=.8, scaler=None,  OHE_drop_option=None, oversample=True, random_state=None, verbose=0, output='default'):
    """All data processing steps in one. Train test split, scale, OHE, Oversample."""
    from sklearn.model_selection import train_test_split
    # isolating numerical cols
    nume_col = list(X.select_dtypes('number').columns)
    if verbose > 0:
        print("Numerical columns: \n---------------------\n", nume_col)

    # isolating categorical cols
    cate_col = list(X.select_dtypes('object').columns)
    if verbose > 0:
        print('')
        print("Categorical columns: \n---------------------\n", cate_col)
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state)

    # pipeline for processing categorical features
    pipe_cate = Pipeline([('ohe',
                           OneHotEncoder(sparse=False, drop=OHE_drop_option))])
    # pipeline for processing numerical features
    if scaler is None:
        scaler = StandardScaler()
    pipe_nume = Pipeline([('scaler', scaler)])
    # transformer
    preprocessor = ColumnTransformer([('nume_feat', pipe_nume, nume_col),
                                      ('cate_feat', pipe_cate, cate_col)])
    # creating dataframes
    try:
        X_train = pd.DataFrame(
            preprocessor.fit_transform(X_train),
            columns=nume_col +
            list(preprocessor.named_transformers_['cate_feat'].
                 named_steps['ohe'].get_feature_names(cate_col)))
        X_test = pd.DataFrame(
            preprocessor.transform(X_test),
            columns=nume_col +
            list(preprocessor.named_transformers_['cate_feat'].
                 named_steps['ohe'].get_feature_names(cate_col)))
        if verbose > 2:
            print("\n\n------")
            print(
                f"Scaler: {str(preprocessor.named_transformers_['nume_feat'].named_steps['scaler'].__class__)[1:-2].split('.')[-1]}, settings: {preprocessor.named_transformers_['nume_feat'].named_steps['scaler'].get_params()}"
            )
            print(
                f"Encoder: {str(preprocessor.named_transformers_['cate_feat'].named_steps['ohe'].__class__)[1:-2].split('.')[-1]}, settings: {preprocessor.named_transformers_['cate_feat'].named_steps['ohe'].get_params()}"
            )
            print("------")
    except:
        # if no categorical cols found
        if verbose > 2:
            print("\n\n------")
            print(
                f"Scaler: {str(preprocessor.named_transformers_['nume_feat'].named_steps['scaler'].__class__)[1:-2].split('.')[-1]}, settings: {preprocessor.named_transformers_['nume_feat'].named_steps['scaler'].get_params()}"
            )
            print(
                f"Encoder: {str(preprocessor.named_transformers_['cate_feat'].named_steps['ohe'].__class__)[1:-2].split('.')[-1]}, settings: {preprocessor.named_transformers_['cate_feat'].named_steps['ohe'].get_params()}"
            )
            print("------")
        print("No Categorical columns found")
        X_train = pd.DataFrame(
            preprocessor.fit_transform(X_train), columns=nume_col)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns=nume_col)
    if oversample:
        from imblearn.over_sampling import SMOTENC
        if verbose > 1:
            print("\n----------------------")
            print("oversampled train data")
            print("----------------------")
        smotenc_features = [
            False] * len(nume_col) + [True] * (len(X_train.columns) - len(nume_col))
        if verbose > 3:
            print(
                f'debug mode: oversampling, based on X_train, check dtype of oversampled data')
            print(f'smotenc_features: {smotenc_features}')
        oversampling = SMOTENC(
            categorical_features=smotenc_features, random_state=random_state, n_jobs=-1)
        X_train, y_train = oversampling.fit_sample(X_train, y_train)

    if output == 'default':
        return X_train, y_train, X_test, y_test
    elif output == 'all':
        return X_train, y_train, X_test, y_test, preprocessor


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
        title=title,
        facet_col_spacing=0.005,
        category_orders={'Clusters': list(np.arange(0, n_clusters))})
    fig.update_xaxes(showline=True,
                     linewidth=1,
                     linecolor=color_discrete_sequence[0],
                     mirror=True,
                     title={'text': ''})
    fig.update_yaxes(showline=True,
                     linewidth=1,
                     linecolor=color_discrete_sequence[0],
                     mirror=True)
    fig.for_each_annotation(
        lambda a: a.update(text=f'Cluster: {a.text.split("=")[1]}'))
    return fig


def get_variable_name(*args):
    """ modified from: https://stackoverflow.com/questions/32000934/python-print-a-variables-name-and-value """
    import inspect
    import re
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    vnames = r.split(", ")
    for i, (var, val) in enumerate(zip(vnames, args)):
        x = f"{var}"
        return x
