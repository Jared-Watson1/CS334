import pandas as pd
import numpy as np


def load_csv(inputfile):
    """
    Load the csv as a pandas data frame

    Parameters
    ----------
    inputfile : string
        filename of the csv to load

    Returns
    -------
    csvdf : pandas.DataFrame
        return the pandas dataframe with the contents
        from the csv inputfile
    """
    return pd.read_csv(inputfile)


def remove_na(inputdf, colname):
    """
    Remove the rows in the dataframe with NA as values
    in the column specified.

    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to check and remove rows with NA

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """
    return inputdf.dropna(subset=[colname])


def onehot(inputdf, colname):
    """
    Convert the column in the dataframe into a one hot encoding.
    The newly converted columns should be at the end of the data
    frame and you should also drop the original column.

    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to one-hot encode

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """
    one_hot = pd.get_dummies(inputdf[colname], prefix=colname)
    outputdf = inputdf.drop(colname, axis=1)
    outputdf = pd.concat([outputdf, one_hot], axis=1)
    return outputdf


def to_numeric(inputdf):
    """
    Extract all the

    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe

    Returns
    -------
    outputnp : numpy.ndarray
        return the numeric contents of the input dataframe as a
        numpy array
    """
    numeric_df = inputdf.select_dtypes(include=np.number)
    return numeric_df.to_numpy()
