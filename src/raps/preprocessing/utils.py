import numpy as np
from scipy.stats import mannwhitneyu

def calc_median_iqr(series):
    """
    Calculate the median and interquartile range (IQR) of a numeric series.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        The input array or series, possibly containing NaN values.

    Returns
    -------
    tuple
        A tuple containing:
        - median (float)
        - first quartile (25th percentile)
        - third quartile (75th percentile)
    """
    return np.nanmedian(series), np.nanpercentile(series, 25), np.nanpercentile(series, 75)

def calculate_p_value_numerical(before, after):
    """
    Compute the p-value from the Mann-Whitney U test between two samples.

    Parameters
    ----------
    before : pd.Series or np.ndarray
        Original sample.
    after : pd.Series or np.ndarray
        Imputed sample.

    Returns
    -------
    float
        P-value from the non-parametric Mann-Whitney test, or NaN if inputs are invalid.
    """
    before_cleaned, after_cleaned = before.dropna(), after.dropna()
    if len(before_cleaned) > 0 and len(after_cleaned) > 0:
        _, p_value = mannwhitneyu(before_cleaned, after_cleaned)
    else:
        p_value = np.nan
    return p_value

def is_binary(series):
    """
    Check if a series contains only binary values (0 or 1), ignoring NaNs.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input data.

    Returns
    -------
    bool
        True if the series consists only of binary values, otherwise False.
    """
    return series.dropna().isin([0, 1]).all()

def extract_stat(stat_name, stats_dict):
    """
    Extract a specific statistic from a dictionary of statistical outputs.

    Parameters
    ----------
    stat_name : str
        Key to extract (e.g., 'MAE', 'MAPE').
    stats_dict : dict
        Dictionary where values are themselves dicts containing multiple statistics.

    Returns
    -------
    list
        List of values for the requested statistic.
    """
    return [v[stat_name] for v in stats_dict.values()]