import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import KFold


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        return X.dot(self.weights_)

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)
        
        regularization = self.reg_lambda * len(y) * np.eye(X.shape[-1])
        regularization[0][0] = 0 # bias shtick...
        self.weights_ = np.dot(np.linalg.inv(np.matmul(X.T, X) + regularization), X.T.dot(y))

        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    x = df.drop(target_name, axis=1) if feature_names is None else df[feature_names]
    y = df[target_name]

    return model.fit_predict(np.array(x), np.array(y))


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        return np.hstack((np.ones((X.shape[0],1)), X))


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=3):
        self.degree = degree
        self.to_drop_features = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        pol_trans = sklearn.preprocessing.PolynomialFeatures(degree=self.degree)
        
        return pol_trans.fit_transform(X)



def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """
    curr_vector = df.corr()[target_feature]
    curr_vector.drop(target_feature, inplace=True)
    top_n_features = curr_vector.sort_values(ascending=False, key=abs).index[0:n]

    return top_n_features, curr_vector.loc[top_n_features]


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """
    return np.sum((y - y_pred) ** 2) / len(y)

def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """
    t = np.sum((y - y_pred) ** 2)
    k = np.sum((y - np.mean(y)) ** 2)

    return 1 - (t / k)


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.
    k = KFold(k_folds, shuffle=True)
    mse_matrix = np.zeros(shape=(len(degree_range),len(lambda_range)))
    
    for i, deg in enumerate(degree_range):
        for j, lmda in enumerate(lambda_range):
            mse = 0
            model.set_params(bostonfeaturestransformer__degree = deg, linearregressor__reg_lambda = lmda)
            
            for idxs_valid, idxs_train in k.split(X):
                X_train, y_train = X[idxs_train], y[idxs_train]
                X_valid, y_valid = X[idxs_valid], y[idxs_valid]
                
                y_pred = model.fit(X_train, y_train).predict(X_valid)
                mse += mse_score(y_valid, y_pred)
            
            mse_matrix[i][j] = mse

    idxs = np.unravel_index(np.argmin(mse_matrix),mse_matrix.shape)
    best_params = {'bostonfeaturestransformer__degree': degree_range[idxs[0]], 'linearregressor__reg_lambda': lambda_range[idxs[1]]}

    return best_params
