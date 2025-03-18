from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class AttributionResult:
    channel_name: str
    removal_effect_ratio: float
    removal_effect_value: float

class MarkovAttribution:
    """Markov Chain based Multi-Touch Attribution model"""
    
    def __init__(self):
        self._transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._states: List[str] = []
        self._state_index: Dict[str, int] = {}
        self._result_ratio_col = "removal_effect_ratio"
        self._result_attribution_col = "removal_effect_value"

    def calculate_attribution(
        self,
        df: pd.DataFrame,
        var_path: str,
        var_conv: str,
        var_value: str,
        var_null: str,
        use_value: bool = False,
        max_rows: int = 1000
    ) -> pd.DataFrame:
        """
        Perform Markov chain attribution analysis on marketing channel data.

        Args:
            df: Input DataFrame containing marketing data
            var_path: Column name for the path sequence
            var_conv: Column name for conversion counts
            var_value: Column name for conversion values
            var_null: Column name for non-conversion counts
            use_value: Whether to use conversion values for attribution
            max_rows: Maximum number of rows to process at once

        Returns:
            DataFrame with channel attribution results

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Validate input columns
        required_cols = {var_path, var_conv, var_value, var_null}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Filter empty paths
        if df[var_path].map(lambda x: len(x) == 0).any():
            raise ValueError("Empty paths found in input data")
        df = df[df[var_path].map(lambda x: len(x) > 0)]

        # Process in batches if needed
        if len(df) <= max_rows:
            return self._calculate_single_attribution(df, var_path, var_conv, var_value, var_null, use_value)
        
        batch = [df.iloc[i:i + max_rows] for i in range(0, len(df), max_rows)]
        results = []
        for batch_df in batch:
            try:
                result = self._calculate_single_attribution(batch_df, var_path, var_conv, var_value, var_null, use_value)
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"Error processing batch: {str(e)}")

        # Aggregate results
        return (
            pd.concat(results)
            .groupby('channel_name')
            .agg({
                self._result_ratio_col: lambda x: x.iloc[0],  # Take first value
                self._result_attribution_col: np.sum
            })
            .reset_index()
            .sort_values(self._result_ratio_col, ascending=False)
        )

    def _calculate_single_attribution(
        self,
        df: pd.DataFrame,
        var_path: str,
        var_conv: str,
        var_value: str,
        var_null: str,
        use_value: bool
    ) -> pd.DataFrame:
        """Calculate attribution for a single batch of data"""
        df = df[df[var_path].map(lambda x: len(x) > 0)]

        df["avg_sales"] = 1
        if use_value:
            df["avg_sales"] = (df[var_value] / (df[var_conv].replace(0, np.nan))).fillna(0)
        df["var_null_value"] = df[var_null] * df["avg_sales"]

        conversion_rate = (df[var_conv] * df["avg_sales"]).sum() / (
            (df[var_conv] * df["avg_sales"]).sum() + (df["var_null_value"]).sum()
        )

        df["click_paths"] = df[var_path].map(lambda x: x.split(" > "))
        transition_matrix_df = self._generate_transition_matrix(
            df,
            click_paths="click_paths",
            total_conversions=var_value if use_value else var_conv,
            non_conversion="var_null_value"
        )

        results = self._get_channel_ratios(transition_matrix_df, conversion_rate)
        results[self._result_attribution_col] = results[self._result_ratio_col] * df[var_value].sum()

        return results

    def _sherman_morrison_inverse(self, transition_matrix_df: pd.DataFrame) -> List[float]:
        """
        Compute the inverse of submatrices incrementally using Sherman-Morrison formula
        when removing each row and column.

        Args:
            transition_matrix_df: The transition matrix as a DataFrame

        Returns:
            A list of inverse matrices for each submatrix after removing one row and column
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

    def _get_channel_ratios(self, transition_matrix_df: pd.DataFrame, conversion_rate: float) -> pd.DataFrame:
        """
        Calculate the channel ratios using the removal effect for each channel in the transition matrix.

        Args:
            transition_matrix_df: The transition matrix as a DataFrame
            conversion_rate: The overall conversion rate

        Returns:
            A DataFrame containing the channel names and their respective removal effect ratios
        """
        sherman_morrison_results = self._sherman_morrison_inverse(transition_matrix_df)
        results = [1-result/conversion_rate for result in sherman_morrison_results]
        
        result_df = pd.DataFrame(
            {"channel_name": transition_matrix_df.columns[1:-2], self._result_ratio_col: results}
        )
        result_df[self._result_ratio_col] = result_df[self._result_ratio_col] / sum(results)

        return result_df

    @staticmethod
    def calculate_attribution_legacy(
        df: pd.DataFrame,
        var_path: str,
        var_conv: str,
        var_value: str,
        var_null: str,
        use_value: bool = False,
        max_rows: int = 1000
    ) -> pd.DataFrame:
        """
        Perform Markov chain attribution analysis on marketing channel data using legacy method.

        Args:
            df: Input DataFrame containing marketing data
            var_path: Column name for the path sequence
            var_conv: Column name for conversion counts
            var_value: Column name for conversion values
            var_null: Column name for non-conversion counts
            use_value: Whether to use conversion values for attribution
            max_rows: Maximum number of rows to process at once

        Returns:
            DataFrame with channel attribution results

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Validate input columns
        required_cols = {var_path, var_conv, var_value, var_null}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Filter empty paths
        if df[var_path].map(lambda x: len(x) == 0).any():
            raise ValueError("Empty paths found in input data")
        df = df[df[var_path].map(lambda x: len(x) > 0)]

        # Process in batches if needed
        if len(df) <= max_rows:
            return MarkovAttribution()._calculate_single_attribution_legacy(df, var_path, var_conv, var_value, var_null, use_value)
        
        batch = [df.iloc[i:i + max_rows] for i in range(0, len(df), max_rows)]
        results = []
        for batch_df in batch:
            try:
                result = MarkovAttribution()._calculate_single_attribution_legacy(batch_df, var_path, var_conv, var_value, var_null, use_value)
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"Error processing batch: {str(e)}")

        # Aggregate results
        return (
            pd.concat(results)
            .groupby('channel_name')
            .agg({
                'removal_effect_ratio': lambda x: x.iloc[0],  # Take first value
                'removal_effect_value': np.sum
            })
            .reset_index()
            .sort_values('removal_effect_ratio', ascending=False)
        )

    def _generate_transition_matrix(
        self,
        df: pd.DataFrame,
        click_paths: str,
        total_conversions: str,
        non_conversion: str
    ) -> pd.DataFrame:
        """
        Generate a transition matrix from a DataFrame containing click paths, total conversions, and non-conversions.

        Args:
            df: Input DataFrame
            click_paths: Column name containing click paths
            total_conversions: Column name containing total conversions
            non_conversion: Column name containing non-conversions

        Returns:
            Transition matrix as a DataFrame
        """
        self._transition_counts = defaultdict(lambda: defaultdict(int))
        self._states = []
        self._state_index = {}

        df["paths"] = df[click_paths].map(lambda x: ["S"] + x)
        paths = df["paths"]

        conversion_counts = df[total_conversions].values
        non_conversion_counts = df[non_conversion].values

        states = set()
        for path, incre in zip(paths.values, conversion_counts):
            states.update(path)
            for i in range(len(path) - 1):
                self._transition_counts[path[i]][path[i + 1]] += incre
            self._transition_counts[path[-1]][total_conversions] += incre

        for path, incre in zip(paths.values, non_conversion_counts):
            for i in range(len(path) - 1):
                self._transition_counts[path[i]][path[i + 1]] += incre
            self._transition_counts[path[-1]][non_conversion] += incre

        states.remove("S")
        self._states = list(sorted(states))
        self._states.insert(0, "S")
        self._states.append(total_conversions)
        self._states.append(non_conversion)

        n = len(self._states)
        transition_matrix = np.zeros((n, n))
        self._state_index = {state: i for i, state in enumerate(self._states)}

        for from_state, to_dict in self._transition_counts.items():
            for to_state, count in to_dict.items():
                i = self._state_index[from_state]
                j = self._state_index[to_state]
                transition_matrix[i, j] = count

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
        zero_sum_rows = row_sums == 0
        transition_matrix[zero_sum_rows.squeeze()] = 0

        return pd.DataFrame(transition_matrix, index=self._states, columns=self._states)

    def _calculate_removal_effect(
        self,
        matrix_A: np.ndarray,
        ids: List[int]
    ) -> float:
        """
        Calculate the removal effect for a certain list of ids in the transition matrix.

        Args:
            matrix_A: The original transition matrix
            ids: List of indices to be removed

        Returns:
            The resulting probability from start to conversion after removal
        """
        matrix_IQ = np.eye(matrix_A.shape[0] - len(ids)) - np.delete(
            np.delete(matrix_A, ids, axis=0), ids, axis=1
        )
        matrix_R = np.delete(matrix_A[:, ids], ids, axis=0)
        return np.dot(np.linalg.inv(matrix_IQ), matrix_R)[0, 1]


    def _get_channel_ratios_legacy(self, transition_matrix_df: pd.DataFrame, conversion_rate: float) -> pd.DataFrame:
        """
        Calculate the channel ratios using the removal effect for each channel in the transition matrix.

        Args:
            transition_matrix_df: The transition matrix as a DataFrame
            conversion_rate: The overall conversion rate

        Returns:
            A DataFrame containing the channel names and their respective removal effect ratios
        """
        results = []
        for i in range(1, transition_matrix_df.shape[0] - 2):
            result = self._calculate_removal_effect(transition_matrix_df.values, [i, -2, -1])
            results.append(1 - result / conversion_rate)

        result_df = pd.DataFrame(
            {"channel_name": transition_matrix_df.columns[1:-2], self._result_ratio_col: results}
        )
        result_df[self._result_ratio_col] = result_df[self._result_ratio_col] / sum(results)

        return result_df

    def _calculate_single_attribution_legacy(
        self,
        df: pd.DataFrame,
        var_path: str,
        var_conv: str,
        var_value: str,
        var_null: str,
        use_value: bool
    ) -> pd.DataFrame:
        """
        Calculate attribution for a single batch of data using legacy method

        Args:
            df: Input DataFrame
            var_path: Column name for path sequence
            var_conv: Column name for conversion counts
            var_value: Column name for conversion values
            var_null: Column name for non-conversion counts
            use_value: Whether to use conversion values

        Returns:
            DataFrame with attribution results
        """
        df = df[df[var_path].map(lambda x: len(x) > 0)]

        df["avg_sales"] = 1
        if use_value:
            df["avg_sales"] = (df[var_value] / (df[var_conv].replace(0, np.nan))).fillna(0)
        df["var_null_value"] = df[var_null] * df["avg_sales"]

        conversion_rate = (df[var_conv] * df["avg_sales"]).sum() / (
            (df[var_conv] * df["avg_sales"]).sum() + (df["var_null_value"]).sum()
        )

        df["click_paths"] = df[var_path].map(lambda x: x.split(" > "))
        transition_matrix_df = self._generate_transition_matrix(
            df,
            click_paths="click_paths",
            total_conversions=var_value if use_value else var_conv,
            non_conversion="var_null_value"
        )

        results = self._get_channel_ratios(transition_matrix_df, conversion_rate)
        results[self._result_attribution_col] = results[self._result_ratio_col] * df[var_value].sum()

        return results
