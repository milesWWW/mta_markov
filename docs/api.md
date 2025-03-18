# API Documentation

## markov_chain_attribution

```python
markov_chain_attribution(
    df: pd.DataFrame,
    var_path: str,
    var_conv: str,
    var_value: str,
    var_null: str,
    use_value: bool = False,
    max_rows: int = 1000
) -> pd.DataFrame
```

Perform Markov chain attribution analysis on marketing channel data.

### Parameters

- `df`: Input DataFrame containing marketing data
- `var_path`: Column name for the path sequence
- `var_conv`: Column name for conversion counts
- `var_value`: Column name for conversion values
- `var_null`: Column name for non-conversion counts
- `use_value`: Whether to use conversion values for attribution (default: False)
- `max_rows`: Maximum number of rows to process at once (default: 1000)

### Returns

DataFrame with the following columns:
- `channel_name`: Name of the marketing channel
- `removal_effect_ratio`: Attribution ratio for the channel
- `removal_effect_value`: Attributed value for the channel

### Example

```python
import pandas as pd
from mta_markov import markov_chain_attribution

data = {
    'path': ['A > B > C', 'A > D', 'B > C'],
    'conversions': [1, 0, 1],
    'value': [100, 0, 50],
    'null': [0, 1, 0]
}
df = pd.DataFrame(data)

result = markov_chain_attribution(
    df,
    var_path='path',
    var_conv='conversions',
    var_value='value',
    var_null='null',
    use_value=True
)
```

## get_channel_ratio_sherman_morrison

```python
get_channel_ratio_sherman_morrison(
    transition_matrix_df: pd.DataFrame,
    conversion_rate: float = 1
) -> pd.DataFrame
```

Calculate channel attribution ratios using Sherman-Morrison formula.

## gen_matrix_from_df

```python
gen_matrix_from_df(
    df: pd.DataFrame,
    click_paths: str,
    total_conversions: str,
    non_conversion: str
) -> pd.DataFrame
```

Generate transition matrix from click path data.
