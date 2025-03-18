import pytest
import pandas as pd
import numpy as np
from mta_markov import MarkovAttribution

def test_basic_attribution():
    data = {
        'path': ['A > B > C', 'A > D', 'B > C'],
        'conversions': [1, 0, 1],
        'value': [100, 0, 50],
        'null': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    model = MarkovAttribution()
    result = model.calculate_attribution(
        df,
        var_path='path',
        var_conv='conversions',
        var_value='value',
        var_null='null',
        use_value=True
    )
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'channel_name' in result.columns
    assert 'removal_effect_ratio' in result.columns
    assert 'removal_effect_value' in result.columns
    assert np.isclose(result['removal_effect_ratio'].sum(), 1.0)

def test_missing_columns():
    data = {
        'path': ['A > B > C'],
        'conversions': [1]
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(ValueError):
        model = MarkovAttribution()
        model.calculate_attribution(
            df,
            var_path='path',
            var_conv='conversions',
            var_value='value',
            var_null='null'
        )

def test_empty_paths():
    data = {
        'path': ['', 'A > B'],
        'conversions': [1, 0],
        'value': [100, 0],
        'null': [0, 1]
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(ValueError):
        model = MarkovAttribution()
        model.calculate_attribution(
            df,
            var_path='path',
            var_conv='conversions',
            var_value='value',
            var_null='null'
        )

def test_batch_processing():
    data = {
        'path': ['A > B > C'] * 1500,
        'conversions': [1] * 1500,
        'value': [100] * 1500,
        'null': [0] * 1500
    }
    df = pd.DataFrame(data)
    
    model = MarkovAttribution()
    result = model.calculate_attribution(
        df,
        var_path='path',
        var_conv='conversions',
        var_value='value',
        var_null='null',
        use_value=True,
        max_rows=1000
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Channels A, B and C
