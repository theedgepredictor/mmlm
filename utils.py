import pandas as pd

DATA_PATH = './data/march-machine-learning-mania-2024/'



def df_rename_fold(df, t1_prefix, t2_prefix):
    """
    Fold two prefixed column types into one generic type in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        t1_prefix (str): Prefix for the first type of columns.
        t2_prefix (str): Prefix for the second type of columns.

    Returns:
        pd.DataFrame: DataFrame with folded columns.
    """
    try:
        t1_all_cols = [i for i in df.columns if t2_prefix not in i]
        t2_all_cols = [i for i in df.columns if t1_prefix not in i]

        t1_cols = [i for i in df.columns if t1_prefix in i]
        t2_cols = [i for i in df.columns if t2_prefix in i]
        t1_new_cols = [i.replace(t1_prefix, '') for i in df.columns if t1_prefix in i]
        t2_new_cols = [i.replace(t2_prefix, '') for i in df.columns if t2_prefix in i]

        t1_df = df[t1_all_cols].rename(columns=dict(zip(t1_cols, t1_new_cols)))
        t2_df = df[t2_all_cols].rename(columns=dict(zip(t2_cols, t2_new_cols)))

        df_out = pd.concat([t1_df, t2_df]).reset_index().drop(columns='index')
        return df_out
    except Exception as e:
        print("--df_rename_fold-- " + str(e))
        print(f"columns in: {df.columns}")
        print(f"shape: {df.shape}")
        return df


def is_pandas_none(val):
    """
    Check if a value represents a "None" in pandas.

    Args:
        val: Value to check.

    Returns:
        bool: True if the value represents a "None," False otherwise.
    """
    return str(val) in ["nan", "None", "", "none", " ", "<NA>", "NaT", "NaN"]

