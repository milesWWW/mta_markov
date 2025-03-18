from collections import defaultdict
import numpy as np
import pandas as pd

RESULT_RATIO_COL = "removal_effect_ratio"
RESULT_ATTRIBUTION_COL = "removal_effect_value"

def sherman_morrison_inverse(transition_matrix_df):
    """
    Compute the inverse of submatrices incrementally using Sherman-Morrison formula
    when removing each row and column.

    Parameters:
    matrix_A (numpy.ndarray): The original n x n matrix.

    Returns:
    list: A list of inverse matrices for each submatrix after removing one row and column.
    """
    
    matrix_A = transition_matrix_df.values[:-2, :-2]
    
    
    n = len(matrix_A)
    m = len(transition_matrix_df)
    A_inv = np.linalg.inv(np.eye(len(matrix_A)) - matrix_A)  # Precompute the full inverse
    results = []

    for i in range(1,n):
        # Mask to exclude the i-th row and column
        mask = np.arange(n) != i
        A_sub_inv = A_inv[np.ix_(mask, mask)]
        
        # Sherman-Morrison update: Remove row and column `i`
        u = A_inv[mask, i]  # Column `i` without diagonal
        v = A_inv[i, mask]  # Row `i` without diagonal
        scalar = A_inv[i, i]  # Diagonal element
        
        # Update formula for inverse
        delta_inv = np.outer(u, v) / scalar
        A_updated_inv = A_sub_inv - delta_inv
        matrix_R_mask = np.ones(m, dtype=bool)
        matrix_R_mask[[i, m - 1, m - 2]] = False
        matrix_R = transition_matrix_df.values[matrix_R_mask, :][:,[i,-2,-1]]
        matrix_Result = np.einsum('i,i->', A_updated_inv[0, :], matrix_R[:, 1])
        results.append(matrix_Result)
    return results

def get_channel_ratio_sherman_morrison(transition_matrix_df, conversion_rate=1):
    """
    Calculate the channel ratios using the removal effect for each channel in the transition matrix.

    Parameters:
    transition_matrix_df (pandas.DataFrame): The transition matrix as a DataFrame.
    conversion_rate (float): The overall conversion rate.

    Returns:
    pandas.DataFrame: A DataFrame containing the channel names and their respective removal effect ratios.
    """
    sherman_morrison_results = sherman_morrison_inverse(transition_matrix_df)
    results = [1-result/conversion_rate for result in sherman_morrison_results]
    
    result_df = pd.DataFrame(
        {"channel_name": transition_matrix_df.columns[1:-2], RESULT_RATIO_COL: results}
    )
    result_df[RESULT_RATIO_COL] = result_df[RESULT_RATIO_COL] / sum(results)

    return result_df

def markov_chain_attribution(
    df, var_path, var_conv, var_value, var_null, use_value=False, max_rows=1000
):
    # df = df[df[var_conv]>0]
    if len(df)<=max_rows:
        return markov_chain_attribution_single(df, var_path, var_conv, var_value, var_null, use_value)
    batch = [df.iloc[i:i + max_rows] for i in range(0, len(df), max_rows)]
    result=[]
    for dataframe in batch:
        result.append(markov_chain_attribution_single(dataframe, var_path, var_conv, var_value, var_null, use_value))
    return pd.concat(result).groupby('channel_name').agg({'removal_effect_ratio':np.mean, 'removal_effect_value':np.sum}).reset_index()

def gen_final_matrix(matrix_A, ids):
    """
    Calculate the removal effect for a certain list of ids in the transition matrix, resulting in new probabilities
    from the start to conversion after removing the specified ids.

    Parameters:
    matrix_A (numpy.ndarray): The original transition matrix.
    ids (list): List of indices to be removed from the transition matrix.

    Returns:
    float: The resulting probability from start to conversion after removing the specified ids.
    """
    # epsilon = 1e-8  # 小的正则化项
    # matrix_IQ = np.eye(matrix_A.shape[0] - len(ids)) - np.delete(np.delete(matrix_A, ids, axis=0), ids, axis=1) + epsilon * np.eye(matrix_A.shape[0] - len(ids))
    matrix_IQ = np.eye(matrix_A.shape[0] - len(ids)) - np.delete(
        np.delete(matrix_A, ids, axis=0), ids, axis=1
    )
    matrix_R = np.delete(matrix_A[:, ids], ids, axis=0)
    matrix_Result = np.dot(np.linalg.inv(matrix_IQ), matrix_R)[0, 1]

    return matrix_Result


def gen_matrix_from_df(df, click_paths, total_conversions, non_conversion):
    """
    Generate a transition matrix from a DataFrame containing click paths, total conversions, and non-conversions.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    click_paths (str): The column name in df that contains the click paths.
    total_conversions (str): The column name in df that contains the total conversions.
    non_conversion (str): The column name in df that contains the non-conversions.

    Returns:
    pandas.DataFrame: The transition matrix as a DataFrame.
    """
    transition_counts = defaultdict(lambda: defaultdict(int))

    df["paths"] = df[click_paths].map(lambda x: ["S"] + x)
    paths = df["paths"]

    conversion_counts = df[total_conversions].values
    non_conversion_counts = df[non_conversion].values

    states = set()
    for path, incre in zip(paths.values, conversion_counts):
        states.update(path)
        for i in range(len(path) - 1):
            transition_counts[path[i]][path[i + 1]] += incre
        transition_counts[path[-1]][total_conversions] += incre

    for path, incre in zip(paths.values, non_conversion_counts):
        for i in range(len(path) - 1):
            transition_counts[path[i]][path[i + 1]] += incre
        transition_counts[path[-1]][non_conversion] += incre

    states.remove("S")
    states = list(sorted(states))
    states.insert(0, "S")
    states.append(total_conversions)
    states.append(non_conversion)

    n = len(states)
    transition_matrix = np.zeros((n, n))
    state_index = {state: i for i, state in enumerate(states)}

    for from_state, to_dict in transition_counts.items():
        for to_state, count in to_dict.items():
            i = state_index[from_state]
            j = state_index[to_state]
            transition_matrix[i, j] = count

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
    zero_sum_rows = row_sums == 0
    transition_matrix[zero_sum_rows.squeeze()] = 0

    transition_matrix_df = pd.DataFrame(transition_matrix, index=states, columns=states)
    return transition_matrix_df


def get_channel_ratio(transition_matrix_df, conversion_rate=1):
    """
    Calculate the channel ratios using the removal effect for each channel in the transition matrix.

    Parameters:
    transition_matrix_df (pandas.DataFrame): The transition matrix as a DataFrame.
    conversion_rate (float): The overall conversion rate.

    Returns:
    pandas.DataFrame: A DataFrame containing the channel names and their respective removal effect ratios.
    """
    results = []
    for i in range(1, transition_matrix_df.shape[0] - 2):
        result = gen_final_matrix(transition_matrix_df.values, [i, -2, -1])
        results.append(1 - result / conversion_rate)

    result_df = pd.DataFrame(
        {"channel_name": transition_matrix_df.columns[1:-2], RESULT_RATIO_COL: results}
    )
    result_df[RESULT_RATIO_COL] = result_df[RESULT_RATIO_COL] / sum(results)

    return result_df


def markov_chain_attribution_single(
    df, var_path, var_conv, var_value, var_null, use_value=False
):
    """
    Calculate the Markov chain attribution for channels based on the click paths, total conversions, and non-conversions.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    var_path (str): The column name in df that contains the click paths.
    var_conv (str): The column name in df that contains the total conversions.
    var_value (str): The column name in df that contains the value for attribution.
    var_null (str): The column name in df that contains the non-conversions.

    Returns:
    pandas.DataFrame: A DataFrame containing the channel names, their removal effect ratios, and their attributed values.
    """
    df = df[df[var_path].map(lambda x: len(x) > 0)]

    df["avg_sales"] = 1
    if use_value:
        df["avg_sales"] = (df[var_value] / (df[var_conv].replace(0, np.nan))).fillna(0)
    df["var_null_value"] = df[var_null] * df["avg_sales"]
    # var_conv = var_value

    conversion_rate = (df[var_conv] * df["avg_sales"]).sum() / (
        (df[var_conv] * df["avg_sales"]).sum() + (df["var_null_value"]).sum()
    )

    df["click_paths"] = df[var_path].map(lambda x: x.split(" > "))
    if use_value:
        transition_matrix_df = gen_matrix_from_df(
            df,
            click_paths="click_paths",
            total_conversions=var_value,
            non_conversion="var_null_value",
        )
    else:
        transition_matrix_df = gen_matrix_from_df(
            df,
            click_paths="click_paths",
            total_conversions=var_conv,
            non_conversion=var_null,
        )

    results = get_channel_ratio_sherman_morrison(transition_matrix_df, conversion_rate=conversion_rate)
    # results = get_channel_ratio(transition_matrix_df, conversion_rate=conversion_rate)
    results[RESULT_ATTRIBUTION_COL] = results[RESULT_RATIO_COL] * df[var_value].sum()

    return results
