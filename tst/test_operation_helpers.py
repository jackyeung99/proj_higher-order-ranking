import os 
import sys
import pytest
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
sys.path.append(repo_root)

from src.utils.operation_helpers import calculate_percentages, calculate_column_means 

@pytest.fixture

def sample_df():
    # Set up a sample DataFrame for testing
    data = { 
        'test_comparison': [-1, -2, -3],
        'value_x': [-2, -3, -4],
        'value_y': [0, -1, -2]
    }
    return pd.DataFrame(data)

def test_calculate_percentages(sample_df):
    df = sample_df
    compared_axis = 0

    # Test with valid compared_axis
    percentages = calculate_percentages(df, compared_axis)
    assert isinstance(percentages, dict)
    assert all(isinstance(v, float) for v in percentages.values())
    assert all(0 <= v <= 1 for v in percentages.values())

    assert percentages['value_x'] == 0.0
    assert percentages['value_y'] == 1.0

    # Test with invalid compared_axis
    with pytest.raises(ValueError):
        calculate_percentages(df, -1)
    with pytest.raises(ValueError):
        calculate_percentages(df, df.shape[1])

def test_calculate_column_means(sample_df):
    df = sample_df
    compared_axis = 0  

    # Test with valid compared_axis
    means = calculate_column_means(df, compared_axis)
    assert isinstance(means, dict)
    assert all(isinstance(v, float) for v in means.values())

    assert means['value_x'] == -1
    assert means['value_y'] == 1

    # Test with invalid compared_axis
    with pytest.raises(ValueError):
        calculate_column_means(df, -1)
    with pytest.raises(ValueError):
        calculate_column_means(df, df.shape[1])

