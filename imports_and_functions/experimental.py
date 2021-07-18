import plotly.express as px
from sklearn import linear_model
import numpy as np
import scipy.stats as stat
from scipy import stats


class LogisticReg:
    """
    from: https://gist.github.com/brentp/5355925
    __________________________________________________________________________
    Wrapper Class for Logistic Regression which has the usual sklearn instance 
    in an attribute self.model, and pvalues, z scores and estimated 
    errors for each coefficient in 

    self.z_scores
    self.p_values
    self.sigma_estimates

    as well as the negative hessian of the log Likelihood (Fisher information)

    self.F_ij
    """

    def __init__(self, *args, **kwargs):  # ,**kwargs):
        self.model = linear_model.LogisticRegression(
            *args, **kwargs)  # ,**args)

    def fit(self, X, y):
        self.model.fit(X, y)
        #### Get p-values for the fitted model ####
        denom = (2.0*(1.0+np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X/denom).T, X)  # Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)  # Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        # z-score for eaach model coefficient
        z_scores = self.model.coef_[0]/sigma_estimates
        # two tailed test for p-values
        p_values = [stat.norm.sf(abs(x))*2 for x in z_scores]

        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij


# class LinearRegression(linear_model.LinearRegression):

#     def __init__(self, *args, **kwargs):
#         # *args is the list of arguments that might go into the LinearRegression object
#         # that we don't know about and don't want to have to deal with. Similarly, **kwargs
#         # is a dictionary of key words and values that might also need to go into the orginal
#         # LinearRegression object. We put *args and **kwargs so that we don't have to look
#         # these up and write them down explicitly here. Nice and easy.

#         if not "fit_intercept" in kwargs:
#             kwargs['fit_intercept'] = False

#         super(LinearRegression, self).__init__(*args, **kwargs)

#     # Adding in t-statistics for the coefficients.
#     def fit(self, x, y):
#         # This takes in numpy arrays (not matrices). Also assumes you are leaving out the column
#         # of constants.

#         # Not totally sure what 'super' does here and why you redefine self...
#         self = super(LinearRegression, self).fit(x, y)
#         n, k = x.shape
#         yHat = np.matrix(self.predict(x)).T

#         # Change X and Y into numpy matricies. x also has a column of ones added to it.
#         x = np.hstack((np.ones((n, 1)), np.matrix(x)))
#         y = np.matrix(y).T

#         # Degrees of freedom.
#         df = float(n-k-1)

#         # Sample variance.
#         sse = np.sum(np.square(yHat - y), axis=0)
#         self.sampleVariance = sse/df

#         # Sample variance for x.
#         self.sampleVarianceX = x.T*x

#         # Covariance Matrix = [(s^2)(X'X)^-1]^0.5. (sqrtm = matrix square root.  ugly)
#         self.covarianceMatrix = sc.linalg.sqrtm(
#             self.sampleVariance[0, 0]*self.sampleVarianceX.I)

#         # Standard erros for the difference coefficients: the diagonal elements of the covariance matrix.
#         self.se = self.covarianceMatrix.diagonal()[1:]

#         # T statistic for each beta.
#         self.betasTStat = np.zeros(len(self.se))
#         for i in xrange(len(self.se)):
#             self.betasTStat[i] = self.coef_[0, i]/self.se[i]

#         # P-value for each beta. This is a two sided t-test, since the betas can be
#         # positive or negative.
#         self.betasPValue = 1 - t.cdf(abs(self.betasTStat), df)


def feature_analysis_intracluster(
        df, cluster_df, n_clusters, title=None,
        nbins=None, marginal='box', histnorm='probability density',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template='presentation'):
    if title is None:
        title = f'{df.name.replace("_"," ")}'
    fig = px.histogram(
        data_frame=df,
        facet_col=cluster_df,
        marginal=marginal,
        histnorm=histnorm,
        nbins=nbins,
        color_discrete_sequence=color_discrete_sequence,
        template=template,
        title=title,
        category_orders={
            'Clusters': list(np.arange(0, n_clusters))}
    ).update_layout(showlegend=False)
    fig.for_each_annotation(lambda a: a.update(
        text=f'Cluster: {a.text.split("=")[1]}'))
    return fig
