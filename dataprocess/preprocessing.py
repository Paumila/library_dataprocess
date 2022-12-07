import pandas as pd
from typing import Sequence, Dict, Optional, Tuple
from enum import Enum
from sklearn import preprocessing

class method(str, Enum):
    drop = "drop"
    mean = "mean"

def split_columns(all_columns: Sequence[str], columns_to_subtract: Sequence[str]) -> Sequence[str]:
    """Subtract a set of columns form another set

    Args:
        all_columns (Sequence[str]): Complete list of columns
        columns_to_subtract (Sequence[str]): List of columns to subtract

    Returns:
        Sequence[str]: List of all_columns - columns_to_subtract
    """    
    all_columns_set = set(all_columns)
    dummy_columns_set = set(columns_to_subtract)
    return list(all_columns_set - dummy_columns_set)

def one_hot_encoder(df: pd.DataFrame, column_list: Sequence[str]) -> pd.DataFrame:
    """Takes in a DataFrame and a list of columns
    for pre-processing via one hot encoding

    Args:
        df (pd.DataFrame): Input DataFrame
        column_list (Sequence[str]): List of columns to process

    Returns:
        pd.DataFrame: Full Dataframe with one hot encoded columns
    """ 
    non_dummy_columns_list = split_columns(list(df.columns), column_list)
    
    df_to_encode = df[column_list]
    df_dummies = pd.get_dummies(df_to_encode)
    df = df[non_dummy_columns_list].join(df_dummies)
    return df[df.columns]

def scale_data(df: pd.DataFrame, range: Tuple[float, float], column_list: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Takes in a DataFrame and a list of column names to transform
    returns a DataFrame of scaled values

    Args:
        df (pd.DataFrame): Input DataFrame
        range (Tuple[float, float]): Range of minimum and maximun elements: [min, max]
        column_list (Optional[Sequence[str]], optional): _description_. Defaults to [].

    Returns:
        pd.DataFrame: Full DataFrame with scaled columns
    """
    if column_list == None:
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        column_list = list(df.select_dtypes(include=numerics).columns)
    
    df_to_scale = df[column_list]
    min_max_scaler = preprocessing.MinMaxScaler(range)
    x_scaled = min_max_scaler.fit_transform(df_to_scale)
    df_to_scale = pd.DataFrame(x_scaled, columns=df_to_scale.columns)
    
    non_dummy_columns_list = split_columns(list(df.columns), column_list)

    return df[non_dummy_columns_list].join(df_to_scale)[df.columns]
        
def fill_na(df: pd.DataFrame, mode: method, column_list: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Takes in a DataFrame and a list of column names to process null values
    returns a DataFrame without null values

    Args:
        df (pd.DataFrame): Input DataFrame
        # mode (method): Can be `drop` or `mean`
        column_list (Optional[Sequence[str]], optional): List of columns to process. If not specified, checks all columns

    Returns:
        pd.DataFrame: Full DataFrame without null values
    """
    if column_list == None:
        column_list = list(df.columns)

    if mode == "drop":
        df_dropped = df[column_list].dropna()
        non_dummy_columns_list = split_columns(list(df.columns), column_list)
        return df_dropped.join(df[non_dummy_columns_list])[df.columns]
    elif mode == 'mean':
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        column_list = list(df[column_list].select_dtypes(include=numerics).columns)
        mean_value=df[column_list].mean()
        df.fillna(value=mean_value, inplace=True)
        return df[df.columns]
    else:
        raise Exception('Method not allowed')