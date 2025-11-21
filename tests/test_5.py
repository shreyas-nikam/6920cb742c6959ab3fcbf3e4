import pytest
import numpy as np
import pandas as pd
from definition_357c40bb9f244a40812aec6a48f7321f import calculate_individual_prediction_impact

@pytest.mark.parametrize("baseline_predictions, stressed_predictions, expected_output, output_type", [
    # Test Case 1: Standard numpy arrays with positive and negative changes
    (np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 35.0]), np.array([2.0, -2.0, 5.0]), 'numpy'),

    # Test Case 2: Standard pandas Series with mixed changes
    (pd.Series([100, 110, 120]), pd.Series([90, 115, 130]), pd.Series([-10, 5, 10]), 'pandas'),

    # Test Case 3: Empty numpy arrays
    (np.array([]), np.array([]), np.array([]), 'numpy'),

    # Test Case 4: Mismatched lengths (should raise ValueError)
    (np.array([1, 2]), np.array([1, 2, 3]), ValueError, None),

    # Test Case 5: Invalid input type (Python lists instead of pd.Series or np.ndarray)
    ([1, 2, 3], [4, 5, 6], TypeError, None),
])
def test_calculate_individual_prediction_impact(baseline_predictions, stressed_predictions, expected_output, output_type):
    if expected_output in (ValueError, TypeError):
        with pytest.raises(expected_output):
            calculate_individual_prediction_impact(baseline_predictions, stressed_predictions)
    else:
        result = calculate_individual_prediction_impact(baseline_predictions, stressed_predictions)
        if output_type == 'numpy':
            np.testing.assert_array_equal(result, expected_output)
            assert isinstance(result, np.ndarray)
        elif output_type == 'pandas':
            pd.testing.assert_series_equal(result, expected_output)
            assert isinstance(result, pd.Series)