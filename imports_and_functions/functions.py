# imports
import plotly.graph_objs as go
import matplotlib.pyplot as plt
# import plotly
# from plotly import graph_objs
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
    Dispalys classification model report.
    Report of model performance using train-test split dataset.
    Shows train and test score, Confusion Matrix and, ROC Curve of performane of test data.
    Uses sklearn for plotting.
    
    Intended to work ONLY on model where target has properly encoded binomial class value.
    
    Parameters:
    -----------
    model : object, scikit-learn model object; no default.
    X_train : pandas.DataFrame, predictor variable training data split; no default,
    y_train : pandas.DataFrame, target variable training data split; no default,
    X_test : pandas.DataFrame, predictor variable test data split; no default,
    y_test : pandas.DataFrame, target variable test data split; no default,
    cmap : {NOT IMPLIMENTED} list of str, colormap of Confusion Matrix; default: ['cool', 'copper_r'],
        cmap of train and test data
    normalize : {NOT IMPLIMENTED} str, normalize count of Confusion Matrix; default: 'true',
        - `true` to normalize counts.
        - `false` to show raw counts.
    figsize : tuple ``(lenght, height) in inchs``, figsize of output; default: (16, 6),
    show_train_report : boolean; default: False,
        - True, to show report.
        - False, to turn off report.
    fitted_model : bool; default: False,
        - if True, fits model to train data and generates report.
        - if False, does not fits model and generates report.
        Use False for previously fitted model.

    ---version 0.9.14---
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
    """Prepares data for use in Kmeans clustering algorithm.
    
    +++++++++++++++++++++
     predefined function
    +++++++++++++++++++++
    
    Parameters:
    -----------
    X : pandas.core.frame.DataFrame; no defalut, independent variables, 
    scaler : sklearn.preprocessing; default = None,
        None uses ```StandardScaler```
    OHE_drop_option : str; default = None,
        for use in sklearn.preprocessing._encoders.OneHotEncoder
        drop : {'first', 'if_binary'} or a array-like of shape (n_features,),             
        default=None, Specifies a methodology to use to drop one of the 
        categories per feature. This is useful in situations where perfectly 
        collinear features cause problems, such as when feeding the resulting 
        data into a neural network or an unregularized regression.

        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

            - None : retain all features (the default).
            - 'first' : drop the first category in each feature. If only one
            category is present, the feature will be dropped entirely.
            - 'if_binary' : drop the first category in each feature with two
            categories. Features with 1 or more than 2 categories are
            left intact.
            - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
            should be dropped.
    verbose : int; default = 0, 
        verbosity control. Larger value means more report. 
    
    Returns:
    --------
    X  : pandas.core.frame.DataFrame, 
    
    --- version 0.1 ---
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
    file : `str`; default: './imports_and_functions/functions.py',
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
    Dispalys model report of multiclass classification model.
    Report of model performance using train-test split dataset.
    Shows train and test score, Confusion Matrix and, ROC Curve of performane of test data. 
    Uses sklearn and yellowbrick for plotting.
    
    Parameters:
    -----------
    model : object, scikit-learn model object; no default.
    X_train : pandas.DataFrame, predictor variable training data split; no default,
    y_train : pandas.DataFrame, target variable training data split; no default,
    X_test : pandas.DataFrame, predictor variable test data split; no default,
    y_test : pandas.DataFrame, target variable test data split; no default,
    cmap : {NOT IMPLIMENTED} list of str, colormap of Confusion Matrix; default: ['cool', 'copper_r'],
        cmap of train and test data
    normalize : {NOT IMPLIMENTED} str, normalize count of Confusion Matrix; default: 'true',
        - `true` to normalize counts.
        - `false` to show raw counts.
    figsize : tuple ``(lenght, height) in inchs``, figsize of output; default: (16, 6),
    show_train_report : boolean; default: False,
        - True, to show report.
        - False, to turn off report.
    fitted_model : bool; default: False,
        - if True, fits model to train data and generates report.
        - if False, does not fits model and generates report.
        Use False for previously fitted model.

    ---version 0.9.14---
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
    """Plots distribution of features
    
    +++++++++++++++++
     Helper function
    +++++++++++++++++

    Parameters:
    -----------
    df : pandas.DataFrame, predictor variable training data split; no default,
    color : str, default = 'gold', 
        color of bars, takes everything that seaborn takes as color option,
    figsize : tuple ``(lenght, height) in inchs``, figsize of output; default: (16, 26),
    fig_col : int; defalut = 3, Controls how many colums to plot in one row,
    labelrotation : int; default = 45, xlabel tick rotation,
    plot_title : str; default = 'Histogram plots of the dataset',
    """
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
    -----------
    df : pandas.DataFrame object.
    annot_format : str, for formatting; default: '.1f'

    Example of `annot_format`:
    --------------------------
    .1e = scientific notation with 1 decimal point (standard form)
    .2f = 2 decimal places
    .3g = 3 significant figures
    .4% = percentage with 4 decimal places

    Note:
    -----
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
    """Plots plotly plots.
    
    +++++++++++++++++
     Helper function
    +++++++++++++++++
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
                       title='Education Level by Income Category',
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
                 color_discrete_sequence=["#4169e1", "#fdff00", "#797979", "#e5e5e5"])
    fig.update_layout(width=700, height=500, bargap=0.05)
    fig.show()
    # fig 6
    plot_distribution(df[financials], color='silver', figsize=(
        16, 16), plot_title='Histogram of Numreical features')
    plt.show()
    pass


def describe_dataframe(df):
    """Statistical description of the pandas.DataFrame."""
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
    """    
    +++++++++++++++++
     Helper function
    +++++++++++++++++ 
    """
    ret_df = pd.DataFrame(preprocessor.transform(X),
                          columns=nume_col +
                          list(preprocessor.named_transformers_['cate_feat'].
                               named_steps['ohe'].get_feature_names(cate_col)))
    return ret_df


def show_px_color_options(type='qualitative'):
    """Shows available options for plotly express."""
    if type == 'qualitative':
        display(dir(px.colors.qualitative))
    elif type == 'sequential':
        display(dir(px.colors.sequential))
    pass


def dataset_processor(X, y, train_size=.8, scaler=None,  OHE_drop_option=None, oversample=True, random_state=None, verbose=0, output='default'):
    """All data processing steps in one. Train test split, scale, OHE, Oversample.

    Parameters:
    -----------
    X : pandas.core.frame.DataFrame; no defalut, independent variables, 
    y : pandas.core.series.Series, no defalut, dependent variables,
    train_size : float or int; default = .8, 
        For use in train_test_split module from sklearn.model_selection 
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    scaler : sklearn.preprocessing; default = None,
        None uses ```StandardScaler```
    OHE_drop_option : str; default = None,
        for use in sklearn.preprocessing._encoders.OneHotEncoder
        drop : {'first', 'if_binary'} or a array-like of shape (n_features,),             
        default=None, Specifies a methodology to use to drop one of the 
        categories per feature. This is useful in situations where perfectly 
        collinear features cause problems, such as when feeding the resulting 
        data into a neural network or an unregularized regression.

        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

            - None : retain all features (the default).
            - 'first' : drop the first category in each feature. If only one
            category is present, the feature will be dropped entirely.
            - 'if_binary' : drop the first category in each feature with two
            categories. Features with 1 or more than 2 categories are
            left intact.
            - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
            should be dropped.
    oversample : bool; default = True,
        - ```True``` oversamples train data
        - ```False``` does not oversample train data
    random_state : int; defult = None,
        for use in ```train_test_split``` and ```SMOTENC```
    verbose : int; default = 0, 
        verbosity control. Larger value means more report. 
    output : str; default = 'default',
        output control, options == ```'default' , 'all'```
        - 'default' returns {X_train, y_train, X_test, y_test}
        - 'all' returns {X_train, y_train, X_test, y_test, preprocessor, nume_col, cate_col}
    
    Returns:
    --------
    --- depending on output control ---
    X_train : pandas.core.frame.DataFrame, 
    y_train : pandas.core.series.Series, 
    X_test : pandas.core.frame.DataFrame, 
    y_test : pandas.core.series.Series, 
    preprocessor : ColumnTransformer object,
    nume_col : list,
    cate_col : list,

    --- version 0.1 ---
    """
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
        return X_train, y_train, X_test, y_test, preprocessor, nume_col, cate_col


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

    +++++++++++++++++
     Helper function
    +++++++++++++++++

    Parameters:
    -----------
    data_frame: DataFrame or array-like or dict
        This argument needs to be passed for column names (and not keyword
        names) to be used. Array-like and dict are tranformed internally to a
        pandas DataFrame. Optional: if missing, a DataFrame gets constructed
        under the hood using the other arguments.
    x: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        position marks along the x axis in cartesian coordinates. If
        `orientation` is `'h'`, these values are used as inputs to `histfunc`.
        Either `x` or `y` can optionally be a list of column references or
        array_likes,  in which case the data will be treated as if it were
        'wide' rather than 'long'.
    facet_col: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like are used to
        assign marks to facetted subplots in the horizontal direction.
    color_discrete_sequence: list of str
        Strings should define valid CSS-colors. When `color` is set and the
        values in the corresponding column are not numeric, values in that
        column are assigned colors by cycling through `color_discrete_sequence`
        in the order described in `category_orders`, unless the value of
        `color` is a key in `color_discrete_map`. Various useful color
        sequences are available in the `plotly.express.colors` submodules,
        specifically `plotly.express.colors.qualitative`.
    marginal: str
        One of `'rug'`, `'box'`, `'violin'`, or `'histogram'`. If set, a
        subplot is drawn alongside the main plot, visualizing the distribution.
    histnorm: str (default `None`)
        One of `'percent'`, `'probability'`, `'density'`, or `'probability
        density'` If `None`, the output of `histfunc` is used as is. If
        `'probability'`, the output of `histfunc` for a given bin is divided by
        the sum of the output of `histfunc` for all bins. If `'percent'`, the
        output of `histfunc` for a given bin is divided by the sum of the
        output of `histfunc` for all bins and multiplied by 100. If
        `'density'`, the output of `histfunc` for a given bin is divided by the
        size of the bin. If `'probability density'`, the output of `histfunc`
        for a given bin is normalized such that it corresponds to the
        probability that a random event whose distribution is described by the
        output of `histfunc` will fall into that bin.
    nbins: int
        Positive integer. Sets the number of bins.
    title: str
        The figure title.
    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name (must be a key in plotly.io.templates) or
        definition.
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
        # labels={'count':histnorm},
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

    fig.update_yaxes(title={'font': {'size': 8}, 'text': ''})
    fig.for_each_annotation(
        lambda a: a.update(text=f'Cluster: {a.text.split("=")[1]}'))

    fig.update_layout(
        # keep the original annotations and add a list of new annotations:
        annotations=list(fig.layout.annotations) +
        [go.layout.Annotation(x=-0.06, y=0.5, font=dict(size=12),
                              showarrow=False,
                              text=histnorm,
                              textangle=-90,
                              xref="paper",
                              yref="paper")])
    return fig


def save_plotly_image(fig, filename=None, ext='.png', width=1400, height=700):
    """Saves plotly image as png in assets folder

    Parameter:
    ----------
    fig : plotly figure object; no default, 
    filename : str; default = None, 
    ext : str; default = '.png', extension of the file to save. options == ``'pdf', 'png', 'jpg'``,
    width : int; default = 1400, width in pixels, 
    height : int: default = 700, height in pixels,
    """
    import plotly.io as pio
    pio.write_image(
        fig, f'./assets/{filename}{ext}', width=width, height=height)
    pass


def get_variable_name(*args):
    """modified from: https://stackoverflow.com/questions/32000934/python-print-a-variables-name-and-value 

    +++++++++++++++++
     Helper function
    +++++++++++++++++

    Gets variable name for use in function (with eval()).

    Parameter:
    ----------
    *args : vairable

    Returns:
    --------
    str

    +++ version: 0.0.1 +++
    """
    import inspect
    import re
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    vnames = r.split(", ")
    for i, (var, val) in enumerate(zip(vnames, args)):
        x = f"{var}"
        return x
