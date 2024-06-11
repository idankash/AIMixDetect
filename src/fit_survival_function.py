"""
Script to read log-loss data of many sentences and characterize the empirical distribution.
We also report the mean log-loss as a function of sentence length
"""
from scipy.interpolate import RectBivariateSpline, interp1d
import numpy as np

def fit_survival_func(xx, log_space=True):
    """
    Returns an estimated survival function to the data in :xx: using
    interpolation.

    Args:
        :xx:  data
        :log_space:  indicates whether fitting is in log space or not.

    Returns:
         univariate function
    """
    assert len(xx) > 0

    eps = 1 / len(xx)
    inf = 1 / eps

    sxx = np.sort(xx)
    qq = np.mean(np.expand_dims(sxx,1) >= sxx, 0)

    if log_space:
        qq = -np.log(qq)


    if log_space:
        return interp1d(sxx, qq, fill_value=(0 , np.log(inf)), bounds_error=False)
    else:
        return interp1d(sxx, qq, fill_value=(1 , 0), bounds_error=False)


def fit_per_length_survival_function(lengths, xx, G=501, log_space=True):
    """
    Returns a survival function for every sentence length in tokens.
    Use 2D interpolation over the empirical survival function of the pairs (length, x)
    
    Args:
        :lengths:, :xx:, 1-D arrays
        :G:  number of grid points to use in the interpolation in the xx dimension
        :log_space:  indicates whether result is in log space or not.

    Returns:
        bivariate function (length, x) -> [0,1]
    """

    assert len(lengths) == len(xx)

    min_tokens_per_sentence = lengths.min()
    max_tokens_per_sentence = lengths.max()
    ll = np.arange(min_tokens_per_sentence, max_tokens_per_sentence)

    ppx_min_val = xx.min()
    ppx_max_val = xx.max()
    xx0 = np.linspace(ppx_min_val, ppx_max_val, G)

    ll_valid = []
    zz = []
    for l in ll:
        xx1 = xx[lengths == l]
        if len(xx1) > 1:
            univariate_survival_func = fit_survival_func(xx1, log_space=log_space)
            ll_valid.append(l)
            zz.append(univariate_survival_func(xx0))

    func = RectBivariateSpline(np.array(ll_valid), xx0, np.vstack(zz))
    if log_space:
        def func2d(x, y):
            return np.exp(-func(x,y))
        return func2d
    else:
        return func
    

# import pickle
# import pandas as pd
# df = pd.read_csv('D:\\.Idan\\תואר שני\\תזה\\detectLM\\article_null.csv')
# LOGLOSS_PVAL_FUNC_FILE = 'D:\.Idan\תואר שני\תזה\detectLM\example\logloss_pval_function.pkl'
# LOGLOSS_PVAL_FUNC_FILE_TEST = 'D:\.Idan\תואר שני\תזה\detectLM\example\logloss_pval_function_test.pkl'
# with open(LOGLOSS_PVAL_FUNC_FILE, 'wb') as handle:
#     pickle.dump(fit_per_length_survival_function(df['length'].values, df['response'].values), handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(LOGLOSS_PVAL_FUNC_FILE, 'rb') as f:
#     data = pickle.load(f)
#     print(data)

# with open(LOGLOSS_PVAL_FUNC_FILE_TEST, 'rb') as f:
#     data = pickle.load(f)
#     print(data)
